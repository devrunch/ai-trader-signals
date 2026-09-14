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
 *  single-sided trade print.
 *
 *  `timeframe: "m1"` (the event desk, measuring what price did after a
 *  release) returns OHLC bars instead. The same hour as ticks is millions of
 *  rows and as minute bars is sixty, and nothing about a reaction study needs
 *  tick resolution. */
const input = JSON.parse(await new Promise((resolve) => {
  let data = "";
  process.stdin.on("data", (chunk) => { data += chunk; });
  process.stdin.on("end", () => resolve(data));
}));

const { instrument, fromMs, toMs, includePrice, timeframe } = input;
const data = await getHistoricRates({
  instrument,
  dates: { from: new Date(fromMs), to: new Date(toMs) },
  timeframe: timeframe || "tick",
  format: "json",
  volumes: false,
  // The library fetches one file per hour and, by default, pauses a full
  // second between batches of 10 -- which made a 48-hour window cost ~5s,
  // the largest single component of a forex chart load. These are static
  // files on a CDN, so a wider batch and a short pause is not abuse.
  batchSize: 20,
  pauseBetweenBatchesMs: 100,
});

if (timeframe && timeframe !== "tick") {
  // Bars come back as [timestamp, open, high, low, close, volume].
  process.stdout.write(JSON.stringify(
    data.map((b) => (Array.isArray(b)
      ? { t: b[0], o: b[1], h: b[2], l: b[3], c: b[4] }
      : { t: b.timestamp, o: b.open, h: b.high, l: b.low, c: b.close })),
  ));
} else process.stdout.write(JSON.stringify(
  includePrice
    ? data.map((tick) => ({ t: tick.timestamp, p: (tick.bidPrice + tick.askPrice) / 2 }))
    : data.map((tick) => tick.timestamp),
));
