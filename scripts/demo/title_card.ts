/**
 * Render the 2-second title card as a PNG.
 *
 * Done in the browser rather than with ffmpeg's drawtext: drawtext needs
 * libfreetype and a resolvable font, which varies by ffmpeg build, and this way
 * the card uses the same typography as the product.
 *
 *   node scripts/demo/title_card.ts [--out path]
 */
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { chromium } from "playwright";
import { DEMO_DIR } from "./narration.ts";

const args = process.argv.slice(2);
const outIdx = args.indexOf("--out");
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(DEMO_DIR, "title.png");

const HTML = `<!doctype html><meta charset="utf-8">
<style>
  :root { --ink:#0A0A0A; --muted:#737373; --accent:#4F46E5; }
  * { margin:0; padding:0; box-sizing:border-box; }
  html,body { width:1440px; height:900px; }
  body {
    background:#FFFFFF; color:var(--ink);
    font-family: "Geist", "Inter", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:26px; letter-spacing:-0.02em;
  }
  .word { font-size:104px; font-weight:600; }
  .word::after { content:""; display:block; width:120px; height:4px; margin:18px auto 0;
                 background:var(--accent); border-radius:2px; }
  .pitch { font-size:34px; color:var(--ink); font-weight:500; }
  .sub { font-size:21px; color:var(--muted); max-width:900px; text-align:center; line-height:1.5; }
  .foot { position:absolute; bottom:56px; font-size:17px; color:var(--muted); }
</style>
<div class="word">ripple</div>
<div class="pitch">&ldquo;What happens to sales if&hellip;?&rdquo;</div>
<div class="sub">A Bayesian what-if simulator, a launch tracker that learns from real data,
and a market-aware AI agent.</div>
<div class="foot">AI Engineer Hackathon NYC &middot; 2026-10-11</div>`;

mkdirSync(dirname(OUT), { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
await page.setContent(HTML, { waitUntil: "networkidle" });
await page.screenshot({ path: OUT });
await browser.close();
console.log(`wrote ${OUT}`);
