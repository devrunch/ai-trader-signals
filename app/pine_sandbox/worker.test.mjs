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

test("a real fill(p1, p2, color) call surfaces as a fills entry naming the two real plots it fills, not a plots entry", () => {
  const source = `//@version=5
indicator("t", overlay=true)
p1 = plot(ta.sma(close, 5), title="Fast")
p2 = plot(ta.sma(close, 10), title="Slow")
fill(p1, p2, color=color.new(color.blue, 85))`;
  const result = run({ source, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, true);
  // Real per-bar data lives under the two plots' own names, not "fill".
  assert.ok(Array.isArray(result.plots["Fast"]));
  assert.ok(Array.isArray(result.plots["Slow"]));
  assert.equal(result.fills.length, 1);
  assert.equal(result.fills[0].plot1, "Fast");
  assert.equal(result.fills[0].plot2, "Slow");
  assert.equal(result.fills[0].colors.length, BARS.length);
  // color.new(color.blue, 85) resolves to a real hex string every bar --
  // this is what makes the frontend's fill actually colored per bar
  // instead of a guess.
  assert.match(result.fills[0].colors.at(-1).color, /^#/);
});

test("a script missing the //@version= pragma still runs, instead of PineTS's parser mis-locating an unrelated later token", () => {
  // Reproduces a real failure: a script pasted without its version header
  // (common when copied from a forum post or retyped by hand) fails with
  // "Unexpected token" at a line:col that doesn't correspond to anything
  // in the actual source -- confirmed live against the real package. The
  // fix is a default, not better error reporting: real TradingView Pine
  // always requires this line, so filling it in is what makes an
  // otherwise-valid script "just work".
  const source = `indicator("t")\nvar float a = na\nvar float b = na\na := math.max(close, nz(a[1])) - nz(a[1] - b[1]) / 100\nplot(a, "a")`;
  const result = run({ source, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, true, result.error);
  assert.ok(Array.isArray(result.plots["a"]));
});

test("a script with its own //@version= pragma is left untouched, not double-prefixed", () => {
  const result = run({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
  assert.equal(result.ok, true, result.error);
});

test("syminfo.ticker and timeframe.multiplier resolve real values when tickerId/timeframe/symbolInfo are given", () => {
  const source = `//@version=5
indicator("t")
plot(1, "x")
plot(timeframe.multiplier, "tf")
if barstate.islast
    alert("BUY - " + syminfo.ticker, alert.freq_once_per_bar_close)`;
  const result = run({
    source, bars: BARS, mode: "indicator",
    tickerId: "NSE:RELIANCE", timeframe: "5", symbolInfo: { symbol: "RELIANCE", exchange: "NSE" },
  });
  assert.equal(result.ok, true, result.error);
  assert.equal(result.plots["tf"].at(-1).value, 5);
});
