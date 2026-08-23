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
      const fills = [];
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
        // A real fill(p1, p2, color) call: confirmed against the real
        // package source (FillHelper in src/namespaces/Plots.ts) that its
        // plot object never carries a real per-bar value, only which two
        // OTHER plots it fills between (plot1/plot2, both keys elsewhere in
        // this same map) and a per-bar resolved color. Gradient fills
        // (fill(p1,p2,top_value,bottom_value,top_color,bottom_color)) carry
        // options.gradient and use a different, unsupported color model --
        // skipped rather than rendered wrong.
        if (plot.options?.style === "fill" && !plot.options?.gradient) {
          fills.push({
            name,
            plot1: plot.plot1,
            plot2: plot.plot2,
            colors: plot.data.map((p) => ({ time: p.time, color: p.options?.color ?? null })),
          });
          continue;
        }
        // Each point is {title, time, value, options} -- confirmed against
        // the real package, not documented. `time` and `value` are the only
        // fields worth forwarding; title duplicates the plot's own name key,
        // options is a per-point render hint no caller here uses.
        plots[name] = plot.data.map((p) => ({ time: p.time, value: p.value }));
      }
      parentPort.postMessage({ ok: true, plots, fills, strategy: null, error: null });
    }
  } catch (err) {
    parentPort.postMessage({ ok: false, plots: null, strategy: null, error: String(err?.message ?? err) });
  }
});
