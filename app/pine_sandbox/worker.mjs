import { parentPort } from "node:worker_threads";
import { PineTS } from "pinets";

parentPort.on("message", async ({ source, bars, mode }) => {
  try {
    const pine = new PineTS(bars);
    const ctx = await pine.run(source);
    if (mode === "strategy") {
      parentPort.postMessage({
        ok: true,
        plots: null,
        strategy: {
          opentrades: ctx.strategy?.opentrades ?? [],
          closedtrades: ctx.strategy?.closedtrades ?? [],
          pending_orders: ctx.strategy?.pending_orders ?? [],
        },
        error: null,
      });
    } else {
      const plots = {};
      for (const [name, plot] of Object.entries(ctx.plots ?? {})) {
        // PineTS's ctx.plots always carries its own internal bookkeeping
        // entries (labels/lines/boxes/linefills/polylines/tables) alongside
        // real plot() output, regardless of whether the script used any of
        // those drawing primitives -- confirmed against the real package,
        // not documented. Never real plot() titles (a script author can't
        // name a plot "__tables__" -- Pine plot() titles are arbitrary
        // strings but these are reserved by the runtime itself), so any
        // caller iterating ctx.plots without filtering leaks six junk
        // series into whatever it does with the result.
        if (name.startsWith("__") && name.endsWith("__")) continue;
        // Each point is {title, time, value, options} -- confirmed against
        // the real package, not documented. `time` and `value` are the only
        // fields worth forwarding; title duplicates the plot's own name key,
        // options is a per-point render hint no caller here uses.
        plots[name] = plot.data.map((p) => ({ time: p.time, value: p.value }));
      }
      parentPort.postMessage({ ok: true, plots, strategy: null, error: null });
    }
  } catch (err) {
    parentPort.postMessage({ ok: false, plots: null, strategy: null, error: String(err?.message ?? err) });
  }
});
