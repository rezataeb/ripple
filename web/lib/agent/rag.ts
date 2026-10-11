// Small RAG over docs/method/*.md for explain_method. Gemini embeddings when available,
// otherwise a deterministic lexical (TF-IDF cosine) ranking, so the tool works with no key.
import { GoogleGenAI } from "@google/genai";
import type { AgentContext, DocChunk } from "./types";

const MAX_CHUNK = 700;

export function chunkDocs(docs: AgentContext["docs"]): DocChunk[] {
  const chunks: DocChunk[] = [];
  for (const d of docs) {
    let heading = d.title;
    let buf = "";
    const flush = () => {
      const t = buf.trim();
      if (t.length > 40) chunks.push({ id: `${d.file}#${chunks.length}`, file: d.file, title: heading, text: t });
      buf = "";
    };
    for (const line of d.text.split("\n")) {
      const h = line.match(/^#{1,3}\s+(.+)$/);
      if (h) {
        flush();
        heading = h[1];
        continue;
      }
      if (line.trim() === "" && buf.length > MAX_CHUNK / 2) flush();
      else buf += line + "\n";
      if (buf.length > MAX_CHUNK) flush();
    }
    flush();
  }
  return chunks;
}

const STOP = new Set("a an the and or of to in on for is are was be it this that with as at by from how why what do does can i we you your our not no".split(" "));
const tokens = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter((w) => w.length > 1 && !STOP.has(w));

function lexicalRank(question: string, chunks: DocChunk[]): { chunk: DocChunk; score: number }[] {
  const docsTokens = chunks.map((c) => tokens(`${c.title} ${c.text}`));
  const df = new Map<string, number>();
  docsTokens.forEach((t) => new Set(t).forEach((w) => df.set(w, (df.get(w) ?? 0) + 1)));
  const N = chunks.length;
  const idf = (w: string) => Math.log(1 + N / (1 + (df.get(w) ?? 0)));
  const q = tokens(question);
  return chunks
    .map((chunk, i) => {
      const tf = new Map<string, number>();
      docsTokens[i].forEach((w) => tf.set(w, (tf.get(w) ?? 0) + 1));
      let score = 0;
      for (const w of q) score += ((tf.get(w) ?? 0) / Math.sqrt(docsTokens[i].length + 1)) * idf(w);
      return { chunk, score };
    })
    .sort((a, b) => b.score - a.score);
}

const embedCache = new Map<string, number[]>();

async function embed(ai: GoogleGenAI, model: string, texts: string[]): Promise<number[][]> {
  const need = texts.filter((t) => !embedCache.has(t));
  if (need.length > 0) {
    const res = await ai.models.embedContent({ model, contents: need });
    (res.embeddings ?? []).forEach((e, i) => embedCache.set(need[i], e.values ?? []));
  }
  return texts.map((t) => embedCache.get(t) ?? []);
}

const cosine = (a: number[], b: number[]) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
};

export async function topChunks(
  ctx: AgentContext,
  question: string,
  k = 3,
): Promise<{ chunks: DocChunk[]; method: "embeddings" | "lexical" }> {
  const chunks = chunkDocs(ctx.docs);
  if (chunks.length === 0) return { chunks: [], method: "lexical" };
  const key = process.env.GEMINI_API_KEY;
  if (ctx.live.gemini && key) {
    try {
      const ai = new GoogleGenAI({ apiKey: key });
      const vecs = await embed(ai, ctx.embedModel, [question, ...chunks.map((c) => c.text)]);
      const ranked = chunks
        .map((chunk, i) => ({ chunk, score: cosine(vecs[0], vecs[i + 1]) }))
        .sort((a, b) => b.score - a.score);
      if (ranked[0].score > 0) return { chunks: ranked.slice(0, k).map((r) => r.chunk), method: "embeddings" };
    } catch {
      // fall through to lexical
    }
  }
  return { chunks: lexicalRank(question, chunks).slice(0, k).map((r) => r.chunk), method: "lexical" };
}
