// Copies docs/method/*.md into public/method/docs.json (RAG source for explain_method).
import { readdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, "../../docs/method");
if (!existsSync(src)) {
  console.log("docs/method not found; keeping the committed docs.json");
  process.exit(0);
}
const docs = readdirSync(src)
  .filter((f) => f.endsWith(".md"))
  .sort()
  .map((f) => {
    const text = readFileSync(path.join(src, f), "utf8");
    const title = (text.match(/^#\s+(.+)$/m) ?? [null, f])[1];
    return { file: f, title, text };
  });
writeFileSync(path.resolve(here, "../public/method/docs.json"), JSON.stringify(docs));
console.log(`wrote ${docs.length} docs`);
