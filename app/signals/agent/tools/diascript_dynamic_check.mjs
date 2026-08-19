// Actually runs a diascript formula against synthetic bar data through the
// real client-side evaluator, so a formula that is grammatically valid but
// numerically broken never reaches a user's chart.
//
// diascript-validate only parses and type-checks -- it never executes a
// formula, so it cannot see two classes of bug: a formula that crashes on
// real bars (e.g. a `ref()` offset pointing past the edge of the loaded
// history), or one whose condition can never be satisfied (e.g. comparing a
// bar against a window that already includes that same bar). Both are
// invisible to a parser; only running the formula against real numbers
// catches them.
import { execSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";

const outputName = process.argv[2] || "result";

async function resolveEngine() {
  // diascript is installed globally (`npm install -g diascript`), the same
  // way diascript-validate itself is resolved via `shutil.which` on the
  // Python side -- so this has to find the global root at runtime rather
  // than assume a fixed path, since that differs between Windows dev and
  // the Linux container this also has to run in. A raw path join, not
  // require.resolve("diascript/dist/engine/engine.js") -- the package's own
  // package.json "exports" map does not list that deep subpath, so package
  // resolution refuses it even though the file is right there on disk.
  const globalRoot = execSync("npm root -g", { encoding: "utf8" }).trim();
  const enginePath = path.join(globalRoot, "diascript", "dist", "engine", "engine.js");
  return import(pathToFileURL(enginePath).href);
}

// Five deliberately different shapes. "flat" exists specifically to catch a
// division by a zero range (e.g. a Fisher Transform's highest()-lowest()) --
// the other four would never hit that denominator.
const SCENARIOS = ["up", "down", "choppy", "volatile", "flat"];

function makeBars(shape, n = 120) {
  return Array.from({ length: n }, (_, i) => {
    let base;
    if (shape === "up") base = 100 + i * 0.4;
    else if (shape === "down") base = 100 + (n - i) * 0.4;
    else if (shape === "choppy") base = 100 + Math.sin(i / 2.3) * 6;
    else if (shape === "volatile") base = 100 + Math.sin(i / 1.1) * 15 + (i % 5 === 0 ? 8 : 0);
    else base = 100; // flat -- zero range, on purpose
    const wiggle = shape === "flat" ? 0 : Math.sin(i / 3) * 1.5;
    return {
      time: 1700000000 + i * 900,
      open: base - 0.3 + wiggle * 0.2,
      high: shape === "flat" ? base : base + Math.abs(wiggle) + 1.1,
      low: shape === "flat" ? base : base - Math.abs(wiggle) - 1.2,
      close: base + wiggle * 0.3,
      volume: 10000 + (i % 7) * 500,
    };
  });
}

function report(valid, message) {
  const payload = valid ? { valid: true } : { valid: false, error: { message } };
  console.log(JSON.stringify(payload));
}

async function main() {
  let source = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) source += chunk;

  const { evaluate } = await resolveEngine();

  let outputType = null;
  let everTrue = false;

  for (const shape of SCENARIOS) {
    const bars = makeBars(shape);
    let result;
    try {
      result = await evaluate(source, bars, {});
    } catch (e) {
      report(false, `crashes when evaluated against real data (${shape}-market scenario): ${e.message}`);
      return;
    }

    const out = result.outputs[outputName];
    if (out) outputType = out.type;

    // fill() carries no series of its own -- outputs.ts wraps it as just
    // {between: [nameA, nameB], color}, never touching ctx.self, so
    // result.values[outputName] is stale/empty. The two series that matter
    // live under their OWN names in result.values instead.
    const seriesToCheck = outputType === "fill" && out
      ? [result.values[out.between[0]], result.values[out.between[1]]]
      : [result.values[outputName]];

    if (outputType === "marker" || outputType === "background") {
      const series = seriesToCheck[0];
      if (Array.isArray(series) && series.some(Boolean)) everTrue = true;
    } else {
      // line / histogram / band / fill: flag a non-finite value past a
      // generous warm-up window -- a division by zero (e.g. highest()-
      // lowest() both equal, the flat scenario) shows up here.
      for (const series of seriesToCheck) {
        if (!Array.isArray(series)) continue;
        // The LAST 40 bars, not the first 40 past some fixed offset -- a
        // fixed front-offset breaks for any period longer than it (e.g. a
        // 50-length ema() is still legitimately NaN at bar 40). The tail is
        // always past warmup for any period shorter than bars.length - 40,
        // and it's the freshest data anyway -- exactly what a trader looks at.
        const tail = series.slice(-40);
        const bad = tail.find((v) => typeof v === "number" && !Number.isFinite(v));
        if (bad !== undefined) {
          report(
            false,
            `produces a non-finite value (NaN/Infinity) in the ${shape}-market scenario -- likely a ` +
              "division by zero (e.g. highest() and lowest() equal on a flat run). Guard the " +
              "denominator, e.g. max(denominator, 0.0001).",
          );
          return;
        }
      }
    }
  }

  if ((outputType === "marker" || outputType === "background") && !everTrue) {
    report(
      false,
      "this condition is never true across 5 test scenarios (uptrend, downtrend, choppy, volatile, " +
        "flat) -- check the logic. A common cause: comparing the current bar against a window " +
        "(highest/lowest/sum/etc. over `length`) that already includes the current bar itself, which " +
        "can never be exceeded -- shift the window back with ref(..., 1) before comparing.",
    );
    return;
  }

  report(true);
}

main().catch((e) => report(false, `dynamic check crashed: ${e.message}`));
