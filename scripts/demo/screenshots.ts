/**
 * The four README screenshots (spec Section 13, item 3).
 *
 *   node scripts/demo/screenshots.ts [baseUrl] [--out dir]
 *
 * Each shot is set up the same way the video sets it up, so the README and the
 * video show the same product. Retina scale, then written to assets/screenshots/.
 *
 * Record against a production build: in `next dev` the home page hangs on its
 * loading skeleton and never fetches the posterior.
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { type Page, chromium } from "playwright";
import { DEMO_DIR } from "./narration.ts";

const args = process.argv.slice(2);
const outIdx = args.indexOf("--out");
const BASE = (args.find((a) => !a.startsWith("--")) ?? process.env.RIPPLE_BASE_URL ??
  "http://127.0.0.1:3000").replace(/\/$/, "");
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(DEMO_DIR, "..", "..", "assets", "screenshots");

const VIEWPORT = { width: 1440, height: 900 };
const sleep = (s: number) => new Promise((r) => setTimeout(r, s * 1000));

async function waitForApp(page: Page): Promise<void> {
  await page
    .getByText("Incremental profit vs. no change", { exact: false })
    .first()
    .waitFor({ timeout: 30_000 });
}

async function click(page: Page, text: string): Promise<void> {
  for (const role of ["tab", "button", "link"] as const) {
    const el = page.getByRole(role, { name: text, exact: false }).first();
    if ((await el.count()) > 0) {
      await el.click({ timeout: 4000 });
      return;
    }
  }
  throw new Error(`no clickable "${text}"`);
}

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 2 });
const page = await context.newPage();
mkdirSync(OUT, { recursive: true });

const shots: { name: string; setup: () => Promise<void> }[] = [
  {
    name: "simulator",
    setup: async () => {
      await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
      await waitForApp(page);
      await click(page, "20% students coupon in Boston");
      await sleep(1.5);
    },
  },
  {
    name: "launch",
    setup: async () => {
      await page.goto(`${BASE}/launch`, { waitUntil: "networkidle" });
      await click(page, "Austin launch");
      await sleep(0.8);
      // mid-slider: the band has narrowed but the verdict is still moving
      const slider = page.getByLabel("Weeks of real launch data", { exact: false }).first();
      await slider.focus();
      for (let i = 0; i < 5; i++) {
        await page.keyboard.press("ArrowRight");
        await sleep(0.15);
      }
      await sleep(1.5);
    },
  },
  {
    name: "ask",
    setup: async () => {
      await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
      await waitForApp(page);
      await page.keyboard.press("Meta+KeyK");
      const box = page.getByLabel("Your question", { exact: false }).first();
      await box.waitFor({ timeout: 6000 });
      await box.click();
      await box.type("Is Miami a good next city?", { delay: 20 });
      await page.getByRole("button", { name: "Ask", exact: true }).first().click();
      await sleep(6); // let the agent finish and the sources render
    },
  },
  {
    name: "method",
    setup: async () => {
      await page.goto(`${BASE}/method`, { waitUntil: "networkidle" });
      const el = page.getByText("Does the launch tracker tell the truth?", { exact: false }).first();
      await el.waitFor({ timeout: 8000 });
      await el.scrollIntoViewIfNeeded();
      await sleep(1.5);
    },
  },
];

for (const shot of shots) {
  try {
    await shot.setup();
    const path = join(OUT, `${shot.name}.png`);
    await page.screenshot({ path });
    console.log(`  ${shot.name.padEnd(10)} -> ${path}`);
  } catch (err) {
    console.warn(`  ${shot.name.padEnd(10)} FAILED: ${String(err).split("\n")[0].slice(0, 110)}`);
  }
}

await browser.close();
