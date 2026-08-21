import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const RUN = path.join(path.dirname(fileURLToPath(import.meta.url)), "run_pine.mjs");
const BARS = Array.from({ length: 30 }, (_, i) => ({
  open: 100 + i, high: 101 + i, low: 99 + i, close: 100.5 + i, volume: 1000, openTime: 1767000900000 + i * 60000,
}));

function run(input) {
  return JSON.parse(execFileSync("node", [RUN], { input: JSON.stringify(input), encoding: "utf8" }));
}

test("a known-good indicator script returns plot data as real {time, value} points, one per bar, real timestamps", () => {
  const result = run({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, true);
  const points = result.plots["SMA5"];
  assert.equal(points.length, BARS.length);
  // ta.sma(close, 5)'s first 4 points are genuinely NaN (not enough bars
  // yet -- real warmup behavior, not a defect); JSON.stringify turns NaN
  // into null on the wire, so checking a settled point (well past warmup)
  // is the honest assertion here, not points[0].
  assert.equal(typeof points[10].value, "number");
  assert.ok(Number.isFinite(points[10].value));
  // PineTS's own per-point time, not assumed to positionally match the
  // input bars -- it must equal the bar's openTime (confirmed: PineTS
  // silently drops time entirely if a bar is missing openTime, which is
  // exactly the bug this test exists to catch -- see the caller contract
  // note in worker.mjs).
  assert.equal(points[0].time, BARS[0].openTime);
  assert.equal(points[points.length - 1].time, BARS[BARS.length - 1].openTime);
});

test("a warmup-period NaN plot value survives the JSON round-trip as null, not a crash", () => {
  const result = run({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, true);
  // typeof null === "object" in JS -- this is exactly the trap the render
  // layer (pine-render.ts) has to guard against, not silently pass a null
  // "value" into a charting library expecting a number.
  assert.equal(result.plots["SMA5"][0].value, null);
});

test("a script that never returns is killed by the timeout, not left hanging", () => {
  // Pine uses indentation for blocks (Python-style), not {} -- a genuine
  // infinite loop, not a syntax error, is what actually exercises the
  // worker-terminate path below.
  const source = `//@version=5\nindicator("t")\nvar x = 0\nwhile true\n    x := x + 1`;
  const result = run({ source, bars: BARS, mode: "indicator", timeoutMs: 500 });
  assert.equal(result.ok, false);
  assert.match(result.error, /exceeded/);
});

test("malformed Pine returns a structured error, not a crash", () => {
  const result = run({ source: `this is not pine script @#$%`, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, false);
  assert.ok(result.error);
});

test("plots never leak PineTS's internal __xxx__ bookkeeping keys, only real plot() titles", () => {
  // Confirmed against the real installed package: ctx.plots always carries
  // __labels__/__lines__/__boxes__/__linefills__/__polylines__/__tables__
  // regardless of whether the script used any of those -- a plain
  // plot()-only script would otherwise leak six junk series to every caller.
  const result = run({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, true);
  assert.deepEqual(Object.keys(result.plots), ["SMA5"]);
});
