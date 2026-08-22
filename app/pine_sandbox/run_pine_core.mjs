import { Worker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import path from "node:path";

const WORKER_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), "worker.mjs");

/** Runs one Pine script in its own worker_thread -- resourceLimits cap its
 *  memory, and a timeout+terminate() bounds a script that never returns.
 *  Shared by run_pine.mjs (one-shot CLI) and run_pine_server.mjs (the
 *  persistent server) so a single execution's isolation guarantees are
 *  defined once, not duplicated between the two entry points. */
export async function runPine({ source, bars, mode, timeoutMs = 5000 }) {
  return new Promise((resolve) => {
    const worker = new Worker(WORKER_PATH, {
      resourceLimits: { maxOldGenerationSizeMb: 256, maxYoungGenerationSizeMb: 64 },
    });
    let settled = false;
    const finish = (result) => { if (settled) return; settled = true; worker.terminate(); resolve(result); };

    const timer = setTimeout(() => finish({ ok: false, plots: null, strategy: null, error: `Pine execution exceeded ${timeoutMs}ms` }), timeoutMs);

    worker.once("message", (result) => { clearTimeout(timer); finish(result); });
    worker.once("error", (err) => { clearTimeout(timer); finish({ ok: false, plots: null, strategy: null, error: String(err?.message ?? err) }); });
    worker.postMessage({ source, bars, mode });
  });
}
