import { test } from "node:test";
import assert from "node:assert/strict";
import { BarsProvider } from "./bars-provider.mjs";

test("getMarketData infers each bar's closeTime from real neighbor spacing, not a hardcoded 1-minute assumption", async () => {
  // 5-minute bars -- a hardcoded 60s closeTime (the previous behavior)
  // would be wrong here by a factor of 5, corrupting any script reading
  // time-dependent built-ins on a non-1m chart.
  const bars = [
    { openTime: 1767000000000, open: 100, high: 101, low: 99, close: 100.5, volume: 1000 },
    { openTime: 1767000300000, open: 100.5, high: 101.5, low: 99.5, close: 101, volume: 1100 },
    { openTime: 1767000600000, open: 101, high: 102, low: 100, close: 101.5, volume: 1200 },
  ];
  const provider = new BarsProvider(bars, { symbol: "RELIANCE", exchange: "NSE" });
  const klines = await provider.getMarketData();

  assert.equal(klines[0].closeTime, bars[1].openTime - 1);
  assert.equal(klines[1].closeTime, bars[2].openTime - 1);
  // Last bar has no next neighbor -- reuses the previous gap (5 minutes),
  // not a hardcoded 1-minute default.
  assert.equal(klines[2].closeTime, bars[2].openTime + (bars[2].openTime - bars[1].openTime) - 1);
});

test("getMarketData falls back to a 1-minute closeTime for a single-bar run (no neighbor to measure)", async () => {
  const bars = [{ openTime: 1767000000000, open: 100, high: 101, low: 99, close: 100.5, volume: 1000 }];
  const provider = new BarsProvider(bars, { symbol: "RELIANCE", exchange: "NSE" });
  const klines = await provider.getMarketData();
  assert.equal(klines[0].closeTime, bars[0].openTime + 60_000 - 1);
});

test("getSymbolInfo carries the real symbol/exchange through to ticker/prefix", async () => {
  const provider = new BarsProvider([], { symbol: "RELIANCE", exchange: "NSE" });
  const info = await provider.getSymbolInfo("NSE:RELIANCE");
  assert.equal(info.ticker, "RELIANCE");
  assert.equal(info.tickerid, "NSE:RELIANCE");
  assert.equal(info.prefix, "NSE");
});
