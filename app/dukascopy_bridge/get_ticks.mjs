import { getHistoricRates } from "dukascopy-node";

/** One-shot: reads {instrument, fromMs, toMs, includePrice?} from stdin,
 *  writes JSON to stdout. Timestamps only by default -- the original,
 *  still-used caller (deriv_provider.py's tick-COUNT volume) already has
 *  real OHLC from its own REST candle fetch, and Dukascopy's own
 *  askVolume/bidVolume field is not reliably variable for every instrument
 *  (confirmed live for XAUUSD: constant across a 20s sample), so counting
 *  raw ticks per candle -- the same convention this app already uses for
 *  Deriv -- is the only honest number to build on there.
 *
 *  `includePrice: true` (Volume Footprint/TPO's own caller, fetch_ticks in
 *  dukascopy_bridge.py) instead returns {t, p} per tick -- p is the MID of
 *  Dukascopy's own bid/ask pair, since a tick here is a quote update, not a
 *  single-sided trade print. */
const input = JSON.parse(await new Promise((resolve) => {
  let data = "";
  process.stdin.on("data", (chunk) => { data += chunk; });
  process.stdin.on("end", () => resolve(data));
}));

const { instrument, fromMs, toMs, includePrice, bucketStartsSec } = input;
const data = await getHistoricRates({
  instrument,
  dates: { from: new Date(fromMs), to: new Date(toMs) },
  timeframe: "tick",
  format: "json",
  volumes: false,
  // The library fetches one file per hour and, by default, pauses a full
  // second between batches of 10 -- which made a 48-hour window cost ~5s,
  // the largest single component of a forex chart load. These are static
  // files on a CDN, so a wider batch and a short pause is not abuse.
  batchSize: 20,
  pauseBetweenBatchesMs: 100,
});

// Counting here rather than shipping the ticks: a 48-hour window is
// millions of timestamps, and writing them as JSON for the caller to bucket
// cost more than fetching them did. The counts are one number per candle.
if (bucketStartsSec) {
  const counts = new Array(bucketStartsSec.length).fill(0);
  for (const tick of data) {
    const seconds = Math.floor(tick.timestamp / 1000);
    // Binary search: the bucket starts are sorted, and a linear scan per
    // tick over millions of ticks is not.
    let lo = 0, hi = bucketStartsSec.length - 1, idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (bucketStartsSec[mid] <= seconds) { idx = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    if (idx >= 0) counts[idx] += 1;
  }
  process.stdout.write(JSON.stringify(counts));
} else {
  process.stdout.write(JSON.stringify(
    includePrice
      ? data.map((tick) => ({ t: tick.timestamp, p: (tick.bidPrice + tick.askPrice) / 2 }))
      : data.map((tick) => tick.timestamp),
  ));
}
