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

test("a known-good indicator script returns plot data", () => {
  const result = run({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, true);
  assert.ok(Array.isArray(result.plots["SMA5"]));
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
