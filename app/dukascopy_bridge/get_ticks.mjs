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

const { instrument, fromMs, toMs, includePrice } = input;
const data = await getHistoricRates({
  instrument,
  dates: { from: new Date(fromMs), to: new Date(toMs) },
  timeframe: "tick",
  format: "json",
  volumes: false,
});

process.stdout.write(JSON.stringify(
  includePrice
    ? data.map((tick) => ({ t: tick.timestamp, p: (tick.bidPrice + tick.askPrice) / 2 }))
    : data.map((tick) => tick.timestamp),
));
