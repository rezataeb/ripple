/**
 * Record the demo video (spec Section 14).
 *
 *   node scripts/demo/record.ts [baseUrl] [--out dir] [--headed]
 *
 * Defaults to http://127.0.0.1:3000; pass the Vercel URL to record against the
 * deployed site instead. 1440x900, video on, Chromium.
 *
 * Every scene comes from narration.txt and is held for exactly as long as its
 * spoken line takes (measured by `make voice` into timings.json), so the picture
 * and the voice line up without any drift correction in ffmpeg.
 *
 * Session B still owns and is changing web/, so every interaction is BEST EFFORT:
 * a selector that no longer matches logs a warning and the scene still holds for
 * its full duration. A UI change can make the recording less interesting; it
 * cannot make it the wrong length or crash it. The run prints a bind report at
 * the end so you know exactly which cues did nothing.
 *
 * IMPORTANT: record against a production build (`next build && next start`).
 * In `next dev` the home page hangs on its loading skeleton and never fetches
 * the posterior.
 */
import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { type Browser, type Page, chromium } from "playwright";
import { DEMO_DIR, type Scene, scenesWithTimings, totalSeconds } from "./narration.ts";

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const positional = args.filter((a) => !a.startsWith("--"));
const outIdx = args.indexOf("--out");

const BASE = (positional[0] ?? process.env.RIPPLE_BASE_URL ?? "http://127.0.0.1:3000").replace(
  /\/$/,
  "",
);
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(DEMO_DIR, "recording");
const HEADED = flags.has("--headed");
const SUBTITLES = flags.has("--subtitles");

const VIEWPORT = { width: 1440, height: 900 };
/** Short on purpose: a missing element must fail fast, not eat the scene. */
const ELEMENT_TIMEOUT = 4000;
const bound: { cue: string; ok: boolean; note: string }[] = [];

/** Run an interaction, but never let a missing selector end the recording. */
async function act(cue: string, fn: () => Promise<string>): Promise<void> {
  try {
    bound.push({ cue, ok: true, note: await fn() });
  } catch (err) {
    const note = String(err).split("\n")[0].slice(0, 110);
    bound.push({ cue, ok: false, note });
    console.warn(`  ! "${cue}" did not bind: ${note}`);
  }
}

const sleep = (s: number) => new Promise((r) => setTimeout(r, Math.max(0, s) * 1000));

const CAPTION_ID = "__ripple_caption__";
/** The line currently on screen, so a navigation can restore it immediately. */
let currentCaption = "";

/**
 * Draw one subtitle line over the page.
 *
 * Re-injected per scene rather than installed once, because a full navigation
 * wipes the DOM and the caption would silently vanish for the rest of the video.
 */
async function showCaption(page: Page, text: string): Promise<void> {
  if (!SUBTITLES) return;
  currentCaption = text;
  await page.evaluate(
    ({ id, text }) => {
      let el = document.getElementById(id);
      if (!el) {
        el = document.createElement("div");
        el.id = id;
        el.setAttribute("aria-hidden", "true");
        Object.assign(el.style, {
          position: "fixed",
          left: "0",
          right: "0",
          bottom: "0",
          zIndex: "2147483647",
          display: "flex",
          justifyContent: "center",
          padding: "0 64px 38px",
          pointerEvents: "none",
          font: '500 25px/1.42 -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
          letterSpacing: "-0.01em",
        } as Partial<CSSStyleDeclaration>);
        const inner = document.createElement("span");
        inner.id = id + "_t";
        Object.assign(inner.style, {
          background: "rgba(10,10,10,0.92)",
          color: "#FFFFFF",
          padding: "15px 26px",
          borderRadius: "14px",
          maxWidth: "1120px",
          textAlign: "center",
          boxShadow: "0 8px 30px rgba(0,0,0,0.28)",
          textWrap: "balance",
        } as Partial<CSSStyleDeclaration>);
        el.appendChild(inner);
        document.body.appendChild(el);
      }
      const t = document.getElementById(id + "_t");
      if (t) t.textContent = text;
    },
    { id: CAPTION_ID, text },
  );
}

/**
 * Abandon an action that overruns its scene.
 *
 * Without this one slow selector pushes every later scene past its spoken line and
 * the whole voice track drifts out of sync. A scene that shows nothing is recoverable;
 * a video whose audio no longer matches the picture is not.
 */
function withDeadline<T>(p: Promise<T>, seconds: number, what: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`overran its ${seconds.toFixed(1)}s scene: ${what}`)),
        seconds * 1000),
    ),
  ]);
}

/** Wait for the posterior to be fetched and the simulator to actually render. */
async function waitForApp(page: Page): Promise<void> {
  // the app fetches a ~400KB posterior before it can render anything, so this one
  // wait is deliberately generous; every other wait is short.
  await page
    .getByText("Incremental profit vs. no change", { exact: false })
    .first()
    .waitFor({ timeout: 30_000 });
}

/**
 * Move a range input in visible steps rather than setting its value.
 * Keyboard stepping fires real React change events AND shows motion on camera,
 * which `fill()` does not.
 */
async function rampSlider(
  page: Page,
  label: string,
  target: number,
  seconds: number,
): Promise<string> {
  const slider = page.getByLabel(label, { exact: false }).first();
  await slider.waitFor({ timeout: ELEMENT_TIMEOUT });
  await slider.focus();
  const read = async () => Number(await slider.inputValue());
  const step = Number((await slider.getAttribute("step")) || 0) || undefined;
  let value = await read();
  const from = value;

  // how many presses, and how long to pause between them so the ramp fills the scene
  const guessStep = step ?? (Math.abs(target - value) / 10 || 1);
  const presses = Math.max(1, Math.min(60, Math.round(Math.abs(target - value) / guessStep)));
  const gap = Math.min(0.25, (seconds * 0.7) / presses);
  const key = target > value ? "ArrowRight" : "ArrowLeft";

  for (let i = 0; i < presses; i++) {
    const before = value;
    await page.keyboard.press(key);
    await sleep(gap);
    value = await read();
    if (value === before) break; // hit an end stop
    if ((key === "ArrowRight" && value >= target) || (key === "ArrowLeft" && value <= target)) break;
  }
  return `${label}: ${from} -> ${value}`;
}

/**
 * Click by visible text, whatever role it carries.
 *
 * The launch pickers are <button> elements with role="tab", and an explicit ARIA
 * role beats the implicit one — so getByRole("button") silently misses them.
 * Try the roles the UI actually uses, then fall back to plain text.
 */
async function clickText(page: Page, text: string, seconds = 0.4): Promise<string> {
  const candidates = [
    page.getByRole("tab", { name: text, exact: false }),
    page.getByRole("button", { name: text, exact: false }),
    page.getByRole("link", { name: text, exact: false }),
    page.getByText(text, { exact: false }),
  ];
  for (const c of candidates) {
    const el = c.first();
    if ((await el.count()) === 0) continue;
    try {
      await el.scrollIntoViewIfNeeded({ timeout: ELEMENT_TIMEOUT });
      await el.click({ timeout: ELEMENT_TIMEOUT });
      await sleep(seconds);
      return `clicked "${text}"`;
    } catch {
      /* try the next role */
    }
  }
  throw new Error(`no clickable element matching "${text}"`);
}

async function showHeading(page: Page, text: string): Promise<string> {
  const el = page.getByText(text, { exact: false }).first();
  await el.waitFor({ timeout: ELEMENT_TIMEOUT });
  await el.scrollIntoViewIfNeeded();
  await sleep(0.6);
  return `scrolled to "${text}"`;
}

/** cue (from narration.txt) -> what the camera does. Unmapped cues just hold. */
function actionFor(cue: string): ((page: Page, scene: Scene) => Promise<void>) | null {
  const c = cue.toLowerCase();

  if (c === "home")
    return async (page) => {
      await act(cue, async () => {
        await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
        await waitForApp(page);
        return "home loaded";
      });
    };

  if (c.includes("verdict"))
    return async (page) => {
      await act(cue, () => showHeading(page, "Weekly units, all cities and groups"));
    };

  if (c.includes('preset "student coupon in boston"'))
    return async (page) => {
      await act(cue, async () => {
        await page.mouse.wheel(0, -400);
        return clickText(page, "20% students coupon in Boston", 1.2);
      });
    };

  if (c.includes("drag discount"))
    return async (page, scene) => {
      // the narration says 10% -> 30%, so step down first, then up across the scene
      await act(cue, async () => {
        const down = await rampSlider(page, "Discount", 0.1, scene.seconds * 0.35);
        const up = await rampSlider(page, "Discount", 0.3, scene.seconds * 0.5);
        return `${down}; ${up}`;
      });
    };

  if (c.includes("dip"))
    return async (page) => {
      await act(cue, () => showHeading(page, "Weekly change in units vs. no change"));
    };

  if (c.includes("austin launch, slider at 0"))
    return async (page) => {
      await act(cue, async () => {
        await page.goto(`${BASE}/launch`, { waitUntil: "networkidle" });
        await clickText(page, "Austin launch", 0.8);
        return "launch page, Austin selected";
      });
    };

  if (c.includes("week slider"))
    return async (page, scene) => {
      await act(cue, () => rampSlider(page, "Weeks of real launch data", 8, scene.seconds * 0.85));
    };

  if (c.includes("truth toggle"))
    return async (page) => {
      await act(cue, async () => {
        const box = page.locator('input[type="checkbox"]').first();
        await box.waitFor({ timeout: ELEMENT_TIMEOUT });
        await box.scrollIntoViewIfNeeded();
        await box.click();
        return "truth toggled on";
      });
    };

  if (c.includes("gate badge"))
    return async (page) => {
      await act(cue, () => showHeading(page, "prior vs. what the data says"));
    };

  if (c.includes("cmd-k"))
    return async (page) => {
      await act(cue, async () => {
        await page.keyboard.press("Meta+KeyK");
        const box = page.getByLabel("Your question", { exact: false }).first();
        await box.waitFor({ timeout: ELEMENT_TIMEOUT });
        await box.click();
        await box.type("Is Miami a good next city?", { delay: 55 });
        await sleep(0.5);
        await page.getByRole("button", { name: "Ask", exact: true }).first().click();
        return "asked: Is Miami a good next city?";
      });
    };

  if (c.includes("ask panel"))
    return async (page, scene) => {
      await act(cue, async () => {
        await sleep(Math.min(3, scene.seconds * 0.4));
        await page.mouse.wheel(0, 260);
        return "answer shown";
      });
    };

  if (c.includes("/data"))
    return async (page) => {
      await act(cue, async () => {
        await page.keyboard.press("Escape");
        await page.goto(`${BASE}/data`, { waitUntil: "networkidle" });
        await sleep(1.0);
        return "data page";
      });
    };

  if (c.includes("recovery"))
    return async (page) => {
      await act(cue, async () => {
        await page.goto(`${BASE}/method`, { waitUntil: "networkidle" });
        return showHeading(page, "Parameter recovery");
      });
    };

  if (c.includes("stress arm"))
    return async (page) => {
      await act(cue, () => showHeading(page, "Does the launch tracker tell the truth?"));
    };

  if (c === "end")
    return async (page) => {
      await act(cue, async () => {
        await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
        await waitForApp(page);
        return "back home";
      });
    };

  return null;
}

// --------------------------------------------------------------------------
async function main(): Promise<void> {
  const scenes = scenesWithTimings();
  const planned = totalSeconds(scenes);
  console.log(
    `recording ${scenes.length} scenes, ${planned.toFixed(1)}s, against ${BASE}` +
      (SUBTITLES ? " [subtitles burned in]" : ""),
  );

  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  let browser: Browser | undefined;
  try {
    browser = await chromium.launch({ headless: !HEADED });
    const context = await browser.newContext({
      viewport: VIEWPORT,
      deviceScaleFactor: 1,
      reducedMotion: "no-preference",
      recordVideo: { dir: OUT, size: VIEWPORT },
    });
    const page = await context.newPage();
    if (SUBTITLES) {
      // a goto wipes the overlay; without this the caption is missing for the
      // first seconds of every scene that navigates, including scene 1
      page.on("load", () => {
        if (currentCaption) void showCaption(page, currentCaption).catch(() => {});
      });
      page.on("domcontentloaded", () => {
        if (currentCaption) void showCaption(page, currentCaption).catch(() => {});
      });
    }

    // warm the cache so the first scene is not a loading skeleton on camera
    await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
    await waitForApp(page);

    const started = Date.now();
    for (const [i, scene] of scenes.entries()) {
      const sceneStart = Date.now();
      console.log(
        `[${((sceneStart - started) / 1000).toFixed(1).padStart(6)}s] ` +
          `${String(i).padStart(2)}  ${scene.seconds.toFixed(1)}s  ${scene.cue}`,
      );
      const action = actionFor(scene.cue);
      if (action) {
        await showCaption(page, scene.text).catch(() => {});
        // keep a slice of the scene for the picture to settle before the cut
        await withDeadline(action(page, scene), scene.seconds * 0.85, scene.cue).catch((e) => {
          console.warn(`  ! ${String(e).split("\n")[0].slice(0, 110)}`);
        });
      } else {
        bound.push({ cue: scene.cue, ok: true, note: "hold (no action mapped)" });
      }

      // an action may have navigated, which wipes the overlay: put it back
      await showCaption(page, scene.text).catch(() => {});
      // hold whatever is left of the scene so the cut lands on the spoken line
      await sleep(scene.seconds - (Date.now() - sceneStart) / 1000);
    }

    const elapsed = (Date.now() - started) / 1000;
    await context.close(); // flushes the video file
    await browser.close();

    console.log(`\nrecorded ${elapsed.toFixed(1)}s (planned ${planned.toFixed(1)}s)`);
    const failed = bound.filter((b) => !b.ok);
    console.log(`\ncue report — ${bound.length - failed.length}/${bound.length} bound`);
    for (const b of bound) console.log(`  ${b.ok ? "ok  " : "MISS"} ${b.cue}\n       ${b.note}`);
    if (failed.length) {
      console.log(
        `\n${failed.length} cue(s) did not bind. The video is the right length but those ` +
          "moments show nothing. Re-run scripts/demo/probe.ts to see what the UI renamed.",
      );
    }
    console.log(`\nvideo in ${OUT}/ — run \`make video\` to merge it with the voice track.`);
  } finally {
    await browser?.close().catch(() => {});
  }
}

await main();
