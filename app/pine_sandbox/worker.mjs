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
      for (const [name, plot] of Object.entries(ctx.plots ?? {})) plots[name] = plot.data;
      parentPort.postMessage({ ok: true, plots, strategy: null, error: null });
    }
  } catch (err) {
    parentPort.postMessage({ ok: false, plots: null, strategy: null, error: String(err?.message ?? err) });
  }
});
