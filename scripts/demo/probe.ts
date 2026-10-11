/**
 * Dump the real selectors the live app exposes, so record.ts binds to what exists
 * instead of to what I guessed. Session B owns web/, so this is read-only
 * reconnaissance: run it after any UI change to see what moved.
 *
 *   node scripts/demo/probe.ts [baseUrl]
 */
import { chromium } from "playwright";

const base = process.argv[2] ?? "http://127.0.0.1:3000";

const page_of = async (browser: Awaited<ReturnType<typeof chromium.launch>>) =>
  browser.newPage({ viewport: { width: 1440, height: 900 } });

const browser = await chromium.launch();
const page = await page_of(browser);

for (const [path, label] of [
  ["/", "home"],
  ["/launch", "launch"],
  ["/method", "method"],
] as const) {
  await page.goto(base + path, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  console.log(`\n===== ${label}  ${path} =====`);
  const info = await page.evaluate(() => {
    const txt = (e: Element) => (e.textContent || "").trim().replace(/\s+/g, " ").slice(0, 70);
    const out = {
      buttons: [] as string[],
      inputs: [] as unknown[],
      widgets: [] as unknown[],
      testids: [] as string[],
      headings: [] as string[],
    };
    document.querySelectorAll("button").forEach((e) => out.buttons.push(txt(e)));
    document.querySelectorAll("input").forEach((e) =>
      out.inputs.push({
        type: (e as HTMLInputElement).type,
        aria: e.getAttribute("aria-label"),
        min: (e as HTMLInputElement).min,
        max: (e as HTMLInputElement).max,
        value: (e as HTMLInputElement).value,
        id: e.id,
      }),
    );
    document
      .querySelectorAll('[role="slider"],[role="switch"],[role="tab"],[role="radio"],[role="dialog"]')
      .forEach((e) =>
        out.widgets.push({
          role: e.getAttribute("role"),
          label: e.getAttribute("aria-label") || txt(e),
          min: e.getAttribute("aria-valuemin"),
          max: e.getAttribute("aria-valuemax"),
          now: e.getAttribute("aria-valuenow"),
        }),
      );
    document.querySelectorAll("[data-testid]").forEach((e) => {
      const v = e.getAttribute("data-testid");
      if (v) out.testids.push(v);
    });
    document.querySelectorAll("h1,h2,h3").forEach((e) => out.headings.push(txt(e)));
    return out;
  });
  console.log("headings:", JSON.stringify(info.headings.filter(Boolean)));
  console.log("buttons :", JSON.stringify(info.buttons.filter(Boolean)));
  console.log("inputs  :", JSON.stringify(info.inputs));
  console.log("widgets :", JSON.stringify(info.widgets));
  console.log("testids :", JSON.stringify([...new Set(info.testids)]));
}

await browser.close();
