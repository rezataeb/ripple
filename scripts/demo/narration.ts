/**
 * Parse scripts/demo/narration.txt into ordered scenes.
 *
 * narration.txt is the single source of truth for both the voice track and the
 * recording: `[cue]` lines name a scene, the line under it is what gets spoken,
 * and `#` lines are comments. Keeping one file means the video cannot drift out of
 * sync with what the narrator says.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";

export interface Scene {
  /** Scene cue from the [brackets], e.g. "home" or "/launch — drag the week slider 0 to 8". */
  cue: string;
  /** The spoken line. Never includes the cue or a comment. */
  text: string;
  /** Seconds to hold this scene. Measured from the voice track when available. */
  seconds: number;
}

export const DEMO_DIR = dirname(new URL(import.meta.url).pathname);
export const NARRATION = join(DEMO_DIR, "narration.txt");
export const TIMINGS = join(DEMO_DIR, "timings.json");

/** Fallback when the voice track has not been measured: macOS `say` runs ~2.6 words/sec. */
const WORDS_PER_SECOND = 2.6;
/** Silence between scenes, so the video does not cut on the last syllable. */
export const TAIL_PAD_SECONDS = 0.8;

/** No subtitle should flash past before it can be read. */
export const MIN_SCENE_SECONDS = 2.5;

export function parseNarration(path: string = NARRATION): Scene[] {
  const lines = readFileSync(path, "utf8").split("\n");
  const scenes: Scene[] = [];
  let cue: string | null = null;

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("[") && line.endsWith("]")) {
      cue = line.slice(1, -1).trim();
      continue;
    }
    if (cue === null) {
      throw new Error(`narration.txt: spoken line before any [cue]: ${line}`);
    }
    scenes.push({
      cue,
      text: line,
      seconds: line.split(/\s+/).length / WORDS_PER_SECOND + TAIL_PAD_SECONDS,
    });
    cue = null; // one spoken line per cue
  }
  if (!scenes.length) throw new Error("narration.txt produced no scenes");
  return scenes;
}

/** Scenes with real durations from timings.json, when `make voice` has been run. */
export function scenesWithTimings(path: string = NARRATION): Scene[] {
  const scenes = parseNarration(path);
  try {
    const measured = JSON.parse(readFileSync(TIMINGS, "utf8")) as {
      seconds: number[];
      pad: number;
    };
    if (measured.seconds.length !== scenes.length) {
      console.warn(
        `timings.json has ${measured.seconds.length} entries for ${scenes.length} scenes — ` +
          "run `make voice` again. Falling back to estimates.",
      );
      return scenes;
    }
    const pad = measured.pad ?? TAIL_PAD_SECONDS;
    return scenes.map((s, i) => ({
      ...s,
      seconds: Math.max(measured.seconds[i] + pad, MIN_SCENE_SECONDS),
    }));
  } catch {
    return scenes; // no voice track yet
  }
}

export function spokenText(scenes: Scene[]): string {
  return scenes.map((s) => s.text).join("\n");
}

export function totalSeconds(scenes: Scene[]): number {
  return scenes.reduce((a, s) => a + s.seconds, 0);
}

if (process.argv[2] === "--start") {
  // print the start time of the first scene whose cue matches, for the GIF window
  const needle = (process.argv[3] ?? "").toLowerCase();
  const scenes = scenesWithTimings();
  let t = 0;
  for (const s of scenes) {
    if (s.cue.toLowerCase().includes(needle)) break;
    t += s.seconds;
  }
  process.stdout.write(t.toFixed(2) + "\n");
} else if (process.argv[2] === "--total") {
  process.stdout.write(totalSeconds(scenesWithTimings()).toFixed(2) + "\n");
} else if (process.argv[2] === "--spoken") {
  // used by `make voice` to feed `say` only the words, never the cues or comments
  process.stdout.write(spokenText(parseNarration()) + "\n");
} else if (process.argv[2] === "--plan") {
  const scenes = scenesWithTimings();
  let t = 0;
  for (const s of scenes) {
    process.stdout.write(
      `${t.toFixed(1).padStart(6)}s  ${s.seconds.toFixed(1).padStart(5)}s  ${s.cue}\n`,
    );
    t += s.seconds;
  }
  process.stdout.write(`\ntotal ${t.toFixed(1)}s (${scenes.length} scenes)\n`);
}
