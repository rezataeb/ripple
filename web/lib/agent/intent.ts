// Rule-based planner: stands in for the model when there is no Gemini key (demo mode).
// It only extracts WHAT the user asked (type, cities, segments, percentages); it never makes numbers.
import type { AgentContext } from "./types";

export type PlanStep =
  | { tool: "build_scenario"; args: Record<string, unknown> }
  | { tool: "model_facts"; args: Record<string, never> }
  | { tool: "explain_method"; args: { question: string } }
  | { tool: "update_with_launch"; args: { launch_id: string; weeks: number } }
  | { tool: "clarify"; args: { question: string } };

const pctIn = (q: string): number | null => {
  const m = q.match(/(-|−|\+)?\s?(\d{1,3}(?:\.\d+)?)\s?(?:%|percent|pct)/i);
  return m ? Number(m[2]) / 100 : null;
};

export function plan(question: string, ctx: AgentContext): PlanStep[] {
  const q = question.toLowerCase();
  const { cities, segments } = ctx.posterior.meta;
  const cands = ctx.posterior.candidates.map((c) => c.name);
  const mentioned = (xs: string[]) => xs.filter((x) => q.includes(x.toLowerCase()));
  const segs = segments.filter((s) => q.includes(s.toLowerCase()) || q.includes(s.toLowerCase().replace(/s$/, "")));
  const trainCities = mentioned(cities);
  const candHit = mentioned(cands);

  // 1. trust / method
  if (/\b(trust|reliab|valid|accura|calibrat|check|how does|how do|method|why should|wrong|limit)/.test(q)) {
    return [
      { tool: "model_facts", args: {} as Record<string, never> },
      { tool: "explain_method", args: { question } },
    ];
  }

  // 2. launch tracker
  const launchWord = /\b(on track|so far|real data|launched|after \d+ weeks?|weeks? in)\b/.test(q);
  if (launchWord || /\bis the launch\b/.test(q)) {
    const wk = q.match(/(\d+)\s*weeks?/);
    const key = candHit[0] ?? trainCities[0] ?? (q.includes("coupon") || segs.length ? "student" : q.includes("gift") || q.includes("feature") ? "gift" : "austin");
    return [{ tool: "update_with_launch", args: { launch_id: key, weeks: wk ? Number(wk[1]) : 4 } }];
  }

  // 3. new city
  if (candHit.length > 0 || /\b(next city|new city|expand|open in|launch in|good city|enter)\b/.test(q)) {
    if (candHit.length === 0) {
      return [{ tool: "clarify", args: { question: `Which city: ${cands.join(", ")}?` } }];
    }
    return [{ tool: "build_scenario", args: { type: "new_city", candidate: candHit[0] } }];
  }

  const pct = pctIn(q);
  // 4. price change
  if (/\b(price|prices|raise|increase|lower|cut)\b/.test(q) && !/\b(coupon|discount|promo)/.test(q)) {
    const down = /\b(lower|cut|drop|reduce|decrease)\b/.test(q);
    if (pct === null) return [{ tool: "clarify", args: { question: "By how much should the price change, for example +8% or −5%?" } }];
    return [
      {
        tool: "build_scenario",
        args: { type: "price", price_change: down ? -pct : pct, cities: trainCities, segments: segs },
      },
    ];
  }

  // 5. promotion
  if (/\b(coupon|discount|promo|promotion|off|deal|sale)\b/.test(q)) {
    if (pct === null) return [{ tool: "clarify", args: { question: "How big is the discount, for example 10%, 20% or 30%?" } }];
    const wk = q.match(/(\d+)\s*(?:weeks?|wks?)/);
    return [
      {
        tool: "build_scenario",
        args: { type: "promo", discount: pct, weeks: wk ? Number(wk[1]) : 4, cities: trainCities, segments: segs },
      },
    ];
  }

  return [
    {
      tool: "clarify",
      args: {
        question:
          "Do you want to test a promotion (for example 20% off for students in Boston), a price change (for example +8%), or a new city?",
      },
    },
  ];
}
