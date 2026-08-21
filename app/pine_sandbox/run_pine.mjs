import { Worker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import path from "node:path";

const WORKER_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), "worker.mjs");

async function runPine({ source, bars, mode, timeoutMs = 5000 }) {
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

const input = JSON.parse(await new Promise((resolve) => {
  let data = "";
  process.stdin.on("data", (chunk) => { data += chunk; });
  process.stdin.on("end", () => resolve(data));
}));
const result = await runPine(input);
process.stdout.write(JSON.stringify(result));
