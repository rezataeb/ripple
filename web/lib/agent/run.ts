import { numberGuard } from "./guard";
import { runGemini } from "./gemini";
import { plan } from "./intent";
import { composeTemplate, type TemplateInput } from "./template";
import {
  buildScenario,
  explainMethod,
  marketScan,
  modelFacts,
  runScenario,
  scenarioSpec,
  updateTool,
} from "./tools";
import type { AgentAnswer, AgentContext, Claim, MarketScan, Source, Step, ToolOutput } from "./types";
import type { Scenario, ScenarioResult, UpdateResult } from "@/lib/types";

const LABELS: Record<string, string> = {
  build_scenario: "building scenario",
  run_scenario: "running model",
  model_facts: "reading model facts",
  update_with_launch: "updating with launch weeks",
  market_scan: "scanning market",
  explain_method: "searching the method docs",
};

interface Session {
  claims: Claim[];
  sources: Source[];
  scenario?: Scenario;
  result?: ScenarioResult;
  update?: UpdateResult;
  scan?: MarketScan;
  openQuestion?: string;
  facts?: ToolOutput;
  method?: ToolOutput;
  launchLabel?: string;
  calls: number;
}

type Emit = (s: Step) => void;

async function execTool(
  ctx: AgentContext,
  sess: Session,
  name: string,
  args: Record<string, unknown>,
  emit: Emit,
): Promise<ToolOutput> {
  sess.calls += 1;
  const id = `call_${sess.calls}`;
  emit({ id, label: LABELS[name] ?? name, status: "running" });
  let out: ToolOutput;
  try {
    switch (name) {
      case "build_scenario":
        out = buildScenario(ctx, args);
        break;
      case "run_scenario": {
        const built = buildScenario(ctx, (args.scenario ?? args) as unknown);
        out = built.scenario ? runScenario(ctx, built.scenario) : built;
        break;
      }
      case "model_facts":
        out = modelFacts(ctx);
        break;
      case "update_with_launch":
        out = updateTool(ctx, String(args.launch_id ?? ""), Number(args.weeks ?? 0));
        break;
      case "market_scan":
        out = await marketScan(ctx, {
          product: typeof args.product === "string" ? args.product : undefined,
          city: typeof args.city === "string" ? args.city : undefined,
          your_price: typeof args.your_price === "number" ? args.your_price : undefined,
        });
        break;
      case "explain_method":
        out = await explainMethod(ctx, String(args.question ?? ""));
        break;
      default:
        out = { output: { error: `unknown tool ${name}` }, claims: [] };
    }
  } catch (e) {
    emit({ id, label: LABELS[name] ?? name, status: "error" });
    return { output: { error: (e as Error).message }, claims: [] };
  }
  for (const c of out.claims) sess.claims.push({ ...c, tool_call_id: id });
  if (out.sources) sess.sources.push(...out.sources);
  if (out.result) {
    sess.result = out.result;
    sess.scenario = out.scenario;
  } else if (out.scenario && !sess.scenario) sess.scenario = out.scenario;
  if (out.update) {
    sess.update = out.update;
    sess.launchLabel = ctx.launches.find((l) => l.id === out.update!.launch_id)?.label;
  }
  if (out.scan) sess.scan = out.scan;
  if (out.openQuestion) sess.openQuestion = out.openQuestion;
  if (name === "model_facts") sess.facts = out;
  if (name === "explain_method") sess.method = out;
  emit({ id, label: LABELS[name] ?? name, status: "done" });
  return out;
}

function firstSentences(text: string, n: number): string[] {
  return text
    .replace(/[#*`|>]/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.replace(/\s+/g, " ").trim())
    .filter((s) => s.length > 30 && s.length < 240 && !/[=→_\[\]\/]/.test(s) && !/\.json|\.md\b/.test(s))
    .slice(0, n);
}

function templateFromSession(ctx: AgentContext, sess: Session, question: string): string {
  const base = { claims: sess.claims, question };
  if (sess.openQuestion && !sess.result) return composeTemplate({ ...base, kind: "clarify", question: sess.openQuestion });
  if (sess.result && sess.scenario) {
    const sc = sess.scenario;
    const input: TemplateInput = {
      ...base,
      kind: "scenario",
      scenarioType: sc.type,
      cityLabel: sc.candidate ?? sc.cities?.join(", "),
      scan: sess.scan,
      warnings: sess.result.flags.warnings,
      extrapolation: sess.result.flags.extrapolation,
      pPays: sess.result.p_profit_positive,
      segmentLines: sess.result.segments?.filter((s) => s.incremental.units.mean !== 0),
    };
    return composeTemplate(input);
  }
  if (sess.update) {
    return composeTemplate({
      ...base,
      kind: "launch",
      launchLabel: sess.launchLabel,
      launchUnknownLabel: sess.update.unknown.label,
    });
  }
  if (sess.facts || sess.method) {
    const checks = (ctx.trust?.checks ?? []).slice(0, 4).map((c) => `${c.name}: ${c.plain_english}`);
    const passages = (sess.method?.output.passages as { text: string }[] | undefined) ?? [];
    return composeTemplate({
      ...base,
      kind: "trust",
      checkSentences: checks,
      docSentences: passages[0] ? firstSentences(passages[0].text, 1) : [],
    });
  }
  return composeTemplate({ ...base, kind: "clarify", question: "Ask me about a promotion, a price change, a new city, or why to trust the model." });
}

/** Rule-based tool plan, used when there is no model. */
async function runTemplatePlan(ctx: AgentContext, sess: Session, question: string, emit: Emit) {
  for (const step of plan(question, ctx)) {
    if (step.tool === "clarify") {
      sess.openQuestion = step.args.question;
      return;
    }
    const built = await execTool(ctx, sess, step.tool, step.args as Record<string, unknown>, emit);
    if (step.tool === "build_scenario") {
      if (!built.scenario) return;
      const run = await execTool(ctx, sess, "run_scenario", { scenario: built.scenario }, emit);
      const sc = built.scenario;
      const net = run.result?.meta.net_price ?? ctx.posterior.meta.base_price;
      const city = sc.type === "new_city" ? sc.candidate : sc.cities?.[0];
      await execTool(ctx, sess, "market_scan", { city: city ?? "", your_price: net }, emit);
    }
  }
}

export async function runAgent(
  ctx: AgentContext,
  question: string,
  emit: Emit = () => {},
): Promise<AgentAnswer> {
  const sess: Session = { claims: [], sources: [], calls: 0 };
  let mode: AgentAnswer["mode"] = "template";
  let text = "";

  if (ctx.live.gemini && process.env.GEMINI_API_KEY) {
    try {
      text = await runGemini({
        apiKey: process.env.GEMINI_API_KEY,
        model: ctx.geminiModel,
        question,
        exec: async (name, args) => (await execTool(ctx, sess, name, args, emit)).output,
      });
      mode = "gemini";
    } catch {
      text = "";
      sess.calls = 0;
      sess.claims = [];
      sess.sources = [];
      sess.scenario = sess.result = sess.update = sess.scan = undefined;
    }
  }
  if (mode === "template") await runTemplatePlan(ctx, sess, question, emit);

  let guard = { passed: true, offending: [] as string[] };
  if (mode === "gemini") {
    guard = numberGuard(text, sess.claims);
    if (!guard.passed || text.length === 0) {
      // A number the tools did not return: discard the model's wording, keep the tool results.
      text = templateFromSession(ctx, sess, question);
      mode = "template";
    }
  } else {
    text = templateFromSession(ctx, sess, question);
    guard = numberGuard(text, sess.claims);
  }

  // dedupe sources
  const seen = new Set<string>();
  const sources = sess.sources.filter((s) => (seen.has(s.url) ? false : (seen.add(s.url), true)));
  return {
    text,
    claims: sess.claims,
    sources,
    scenario: sess.result ? sess.scenario : undefined,
    open_question: sess.openQuestion && !sess.result ? sess.openQuestion : undefined,
    mode,
    guard,
    result_summary: sess.result
      ? { profit: sess.result.incremental.profit, p_profit_positive: sess.result.p_profit_positive }
      : undefined,
  };
}

export { scenarioSpec };
