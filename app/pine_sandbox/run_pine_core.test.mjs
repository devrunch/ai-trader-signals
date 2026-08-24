import { test } from "node:test";
import assert from "node:assert/strict";
import { runPine, RECYCLE_AFTER_RUNS, __test_hasWarmWorker } from "./run_pine_core.mjs";

const BARS = Array.from({ length: 30 }, (_, i) => ({
  open: 100 + i, high: 101 + i, low: 99 + i, close: 100.5 + i, volume: 1000, openTime: 1767000900000 + i * 60000,
}));
const GOOD = { source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" };
// A single `while true` no longer hangs -- PineTS's own loop guard (500,000
// iterations per loop) throws a catchable error well inside timeoutMs, which
// resolves normally via onMessage (worker not corrupted, correctly not
// recycled). A loop NESTED inside another resets its own counter to zero on
// every re-entry, so each of the outer loop's 500,000 iterations gets a
// fresh 500,000-iteration budget -- genuinely longer than any timeoutMs
// worth testing with, which is what actually exercises the timeout path.
const HUNG = { source: `//@version=5\nindicator("t")\nvar x = 0\nfor i = 0 to 499999\n    for j = 0 to 499999\n        x := x + 1`, bars: BARS, mode: "indicator", timeoutMs: 300 };

test("a second run reuses the already-warm worker from the first, not a fresh spawn", async () => {
  const r1 = await runPine(GOOD);
  assert.equal(r1.ok, true);
  assert.equal(__test_hasWarmWorker(), true);

  const r2 = await runPine(GOOD);
  assert.equal(r2.ok, true);
  assert.equal(__test_hasWarmWorker(), true);
});

test("the worker recycles once RECYCLE_AFTER_RUNS is crossed, and the run right after still works", async () => {
  // RECYCLE_AFTER_RUNS + 1 calls guarantees crossing the boundary at least
  // once no matter how much of it earlier tests already used up on the
  // shared module-level worker (this file's tests all share one module
  // instance, same as production sharing one process).
  for (let i = 0; i < RECYCLE_AFTER_RUNS + 1; i++) {
    const r = await runPine(GOOD);
    assert.equal(r.ok, true, r.error);
  }
  assert.equal(__test_hasWarmWorker(), true);
});

test("a timed-out script recycles the worker, and the next run still succeeds", async () => {
  const r1 = await runPine(HUNG);
  assert.equal(r1.ok, false);
  assert.match(r1.error, /exceeded/);
  assert.equal(__test_hasWarmWorker(), false, "a timed-out run must not leave a terminated worker marked reusable");

  const r2 = await runPine(GOOD);
  assert.equal(r2.ok, true, r2.error);
  assert.equal(__test_hasWarmWorker(), true);
});
