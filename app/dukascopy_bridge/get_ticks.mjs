import { getHistoricRates } from "dukascopy-node";

/** One-shot: reads {instrument, fromMs, toMs} from stdin, writes a JSON
 *  array of raw tick epoch-milliseconds to stdout. Timestamps only, not
 *  full tick objects (price/volume) -- the caller (deriv_provider.py)
 *  already has real OHLC from its own REST candle fetch, and Dukascopy's
 *  own askVolume/bidVolume field is not reliably variable for every
 *  instrument (confirmed live for XAUUSD: constant across a 20s sample),
 *  so counting raw ticks per candle -- the same convention this app
 *  already uses for Deriv -- is the only honest number to build on. */
const input = JSON.parse(await new Promise((resolve) => {
  let data = "";
  process.stdin.on("data", (chunk) => { data += chunk; });
  process.stdin.on("end", () => resolve(data));
}));

const { instrument, fromMs, toMs } = input;
const data = await getHistoricRates({
  instrument,
  dates: { from: new Date(fromMs), to: new Date(toMs) },
  timeframe: "tick",
  format: "json",
  volumes: false,
});

process.stdout.write(JSON.stringify(data.map((tick) => tick.timestamp)));
