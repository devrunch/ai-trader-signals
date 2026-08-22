import { createInterface } from "node:readline";
import { runPine } from "./run_pine_core.mjs";

/**
 * A persistent replacement for spawning run_pine.mjs fresh per request.
 * Every Pine run used to pay full Node process startup (V8 init + pinets'
 * module load) on top of the actual computation -- cheap on an idle box,
 * but that repeated page-faulting-in-a-fresh-binary cost is exactly what
 * raced under the production box's memory pressure (confirmed live: a
 * burst of several indicators attaching at once, each spawning its own
 * Node process, occasionally lost the race against swap and got its stdin
 * closed before Python finished writing to it).
 *
 * This process stays alive and reads one JSON request per line from
 * stdin, writing one JSON response per line to stdout -- the caller
 * (sandbox.py) is responsible for never having more than one request in
 * flight at a time, so no request-id correlation is needed here. Each
 * request still runs in its own fresh worker_thread via runPine() --
 * per-run memory caps and timeout/crash isolation are unchanged, only the
 * OUTER process is now long-lived.
 */
const rl = createInterface({ input: process.stdin, terminal: false });

// A promise chain, not a naive per-line async handler: readline's "line"
// event fires synchronously as lines arrive, and without this a second
// line arriving before the first request finishes would start a second
// worker_thread concurrently -- defense in depth even though the Python
// side already serializes requests.
let chain = Promise.resolve();

rl.on("line", (line) => {
  chain = chain.then(async () => {
    let result;
    try {
      const input = JSON.parse(line);
      result = await runPine(input);
    } catch (err) {
      result = { ok: false, plots: null, strategy: null, error: String(err?.message ?? err) };
    }
    process.stdout.write(JSON.stringify(result) + "\n");
  });
});

// Stdin closing (the parent process exiting, or an explicit kill) is the
// normal shutdown path -- exit cleanly rather than sit on an event loop
// with nothing left to read.
rl.on("close", () => process.exit(0));
