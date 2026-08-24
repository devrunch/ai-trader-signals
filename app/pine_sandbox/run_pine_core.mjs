import { Worker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import path from "node:path";

const WORKER_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), "worker.mjs");

/** A fresh worker_thread pays PineTS's full module load (its own
 *  transpiler + every ta/math/matrix/strategy namespace) on every single
 *  spawn -- confirmed the actual cost driver on the production box (2GB
 *  RAM, routinely under 100MB free): reusing one worker across many runs
 *  means only the FIRST run after a (re)spawn pays that cost, every run
 *  after just posts a new script into an already-warm isolate. Recycled
 *  periodically anyway -- unlike a fresh-per-run worker, a reused one
 *  isn't a hard isolation boundary between scripts (a script that
 *  corrupts a global survives into the next one until recycle), and this
 *  app runs user-authored/AI-generated Pine, not just trusted presets. */
export const RECYCLE_AFTER_RUNS = 50;

let worker = null;
let runsSinceSpawn = 0;

function spawnWorker() {
  const w = new Worker(WORKER_PATH, {
    resourceLimits: { maxOldGenerationSizeMb: 256, maxYoungGenerationSizeMb: 64 },
  });
  worker = w;
  runsSinceSpawn = 0;
  // A live worker_thread otherwise keeps its host process's event loop
  // alive on its own -- fine for run_pine_server.mjs (its stdin/readline
  // listener already keeps that process alive independently), but
  // run_pine.mjs's one-shot CLI and this module's own tests need to exit
  // naturally once their real work is done, not hang on a warm worker
  // sitting idle waiting to be reused.
  w.unref();
  // Node's EventEmitter throws if an "error" event fires with zero
  // listeners attached -- fatal for the whole outer process if it happened
  // while this worker was idle between runs (no per-run listener attached
  // yet). This permanent listener is only a backstop for that idle case;
  // an in-flight run's own listener (see runPine below) is what actually
  // resolves that run's promise.
  w.on("error", () => { if (worker === w) worker = null; });
  w.once("exit", () => { if (worker === w) worker = null; });
}

function killWorker() {
  if (!worker) return;
  const w = worker;
  worker = null;
  w.terminate();
}

/** Runs one Pine script against the persistent worker (spawning or
 *  recycling it first if needed), timing it out and recycling on that
 *  timeout so the next call never posts into a worker that's mid-`terminate()`.
 *  Shared by run_pine.mjs (one-shot CLI) and run_pine_server.mjs (the
 *  persistent server) so a single execution's isolation guarantees are
 *  defined once, not duplicated between the two entry points. */
export async function runPine({ source, bars, mode, timeoutMs = 5000, tickerId, timeframe, symbolInfo, inputOverrides }) {
  if (!worker || runsSinceSpawn >= RECYCLE_AFTER_RUNS) {
    killWorker();
    spawnWorker();
  }
  const current = worker;
  runsSinceSpawn += 1;

  return new Promise((resolve) => {
    let settled = false;
    let timer;

    const cleanup = () => {
      clearTimeout(timer);
      current.off("message", onMessage);
      current.off("error", onError);
    };
    const finish = (result, recycle = false) => {
      if (settled) return;
      settled = true;
      if (recycle && worker === current) killWorker();
      resolve(result);
    };
    const onMessage = (result) => { cleanup(); finish(result); };
    const onError = (err) => { cleanup(); finish({ ok: false, plots: null, strategy: null, error: String(err?.message ?? err) }, true); };

    timer = setTimeout(() => {
      cleanup();
      finish({ ok: false, plots: null, strategy: null, error: `Pine execution exceeded ${timeoutMs}ms` }, true);
    }, timeoutMs);

    current.on("message", onMessage);
    current.on("error", onError);
    current.postMessage({ source, bars, mode, tickerId, timeframe, symbolInfo, inputOverrides });
  });
}

/** Test-only: whether a run right now would reuse an existing worker or
 *  have to spawn one -- lets a test assert the recycle boundary without
 *  reaching into module-private state directly. */
export function __test_hasWarmWorker() {
  return worker !== null && runsSinceSpawn < RECYCLE_AFTER_RUNS;
}
