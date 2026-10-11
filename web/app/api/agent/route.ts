import type { NextRequest } from "next/server";
import { loadContext } from "@/lib/agent/context";
import { runAgent } from "@/lib/agent/run";
import type { Posterior } from "@/lib/types";

export const maxDuration = 60;

/**
 * POST { question: string, posterior?: Posterior }  ->  NDJSON stream:
 *   {"type":"step","step":{id,label,status}} ... {"type":"answer","answer":AgentAnswer}
 * No key: rule-based planner + template answers + cached market scan (demo mode).
 */
export async function POST(req: NextRequest) {
  let body: { question?: unknown; posterior?: Posterior };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Send JSON: { question: string }" }, { status: 400 });
  }
  const question = typeof body.question === "string" ? body.question.trim().slice(0, 500) : "";
  if (!question) return Response.json({ error: "question is required" }, { status: 400 });

  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (o: unknown) => controller.enqueue(enc.encode(JSON.stringify(o) + "\n"));
      try {
        const ctx = await loadContext({ origin: req.nextUrl.origin, posterior: body.posterior });
        send({ type: "mode", live: ctx.live });
        const answer = await runAgent(ctx, question, (step) => send({ type: "step", step }));
        send({ type: "answer", answer });
      } catch (e) {
        send({ type: "error", message: (e as Error).message });
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson", "Cache-Control": "no-store" } });
}
