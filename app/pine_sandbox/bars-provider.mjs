/**
 * Wraps our own already-fetched bars as a real PineTS IProvider instead of
 * handing PineTS a raw array. Confirmed against the real package
 * (github.com/LuxAlgo/PineTS) that a raw-array `source` never populates
 * `context.pine.syminfo` at all -- that only happens inside PineTS's own
 * loadMarketData() when `source` has a getSymbolInfo() method to call.
 * Verified this produces IDENTICAL indicator output to the raw-array path
 * (same values, same bar count) for a plain script -- this is purely
 * additive, not a behavior change for scripts that don't touch syminfo/
 * timeframe.
 *
 * `getMarketData()` ignores its own tickerId/timeframe/limit/date-range
 * args and always returns the full bars this sandbox was given -- there is
 * no live fetch to make, the caller already loaded exactly the range it
 * wants tested.
 *
 * ISymbolInfo has ~35 fields (github.com/LuxAlgo/PineTS,
 * dist/types/marketData/IProvider.d.ts) covering fundamentals data
 * (industry, shareholders, analyst recommendations, ...) this app has no
 * source for -- those stay zeroed/empty rather than faked. The fields that
 * matter for how real scripts actually use syminfo (ticker, tickerid,
 * exchange-ish prefix, currency, timezone, mintick) are real.
 */
export class BarsProvider {
  constructor(bars, symbolInfo) {
    this.bars = bars;
    this.symbolInfo = symbolInfo;
  }

  async getMarketData() {
    return this.bars.map((b) => ({
      openTime: b.openTime,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      volume: b.volume ?? 0,
      // Real session-close time isn't known here (no calendar data in this
      // sandbox) -- next-bar-open minus 1ms is the same approximation
      // PineTS's own 24/7-market normalizeCloseTime() uses, close enough
      // for what scripts actually read closeTime for.
      closeTime: b.openTime + 60_000 - 1,
      quoteAssetVolume: 0,
      numberOfTrades: 0,
      takerBuyBaseAssetVolume: 0,
      takerBuyQuoteAssetVolume: 0,
      ignore: 0,
    }));
  }

  async getSymbolInfo(tickerId) {
    return {
      current_contract: "",
      description: this.symbolInfo?.name ?? tickerId,
      isin: "",
      main_tickerid: tickerId,
      prefix: this.symbolInfo?.exchange ?? "",
      root: "",
      ticker: this.symbolInfo?.symbol ?? tickerId,
      tickerid: tickerId,
      type: "stock",
      basecurrency: "INR",
      country: "IN",
      currency: "INR",
      timezone: "Asia/Kolkata",
      employees: 0,
      industry: "",
      sector: "",
      shareholders: 0,
      shares_outstanding_float: 0,
      shares_outstanding_total: 0,
      expiration_date: 0,
      session: "0915-1530",
      volumetype: "base",
      mincontract: 0,
      minmove: 1,
      mintick: 0.05,
      pointvalue: 1,
      pricescale: 100,
      recommendations_buy: 0,
      recommendations_buy_strong: 0,
      recommendations_date: 0,
      recommendations_hold: 0,
      recommendations_sell: 0,
      recommendations_sell_strong: 0,
      recommendations_total: 0,
      target_price_average: 0,
      target_price_date: 0,
      target_price_estimates: 0,
      target_price_high: 0,
      target_price_low: 0,
      target_price_median: 0,
    };
  }

  configure() {}
}
