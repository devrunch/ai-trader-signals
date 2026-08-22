import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SERVER = path.join(path.dirname(fileURLToPath(import.meta.url)), "run_pine_server.mjs");
const BARS = Array.from({ length: 30 }, (_, i) => ({
  open: 100 + i, high: 101 + i, low: 99 + i, close: 100.5 + i, volume: 1000, openTime: 1767000900000 + i * 60000,
}));

function startServer() {
  const proc = spawn("node", [SERVER]);
  const rl = createInterface({ input: proc.stdout, terminal: false });
  const pending = [];
  rl.on("line", (line) => {
    const resolve = pending.shift();
    resolve(JSON.parse(line));
  });
  return {
    proc,
    send(input) {
      return new Promise((resolve) => {
        pending.push(resolve);
        proc.stdin.write(JSON.stringify(input) + "\n");
      });
    },
    sendRaw(line) {
      return new Promise((resolve) => {
        pending.push(resolve);
        proc.stdin.write(line + "\n");
      });
    },
    close() { proc.kill(); },
  };
}

test("serves multiple requests over one persistent process, in order", async () => {
  const server = startServer();
  try {
    const r1 = await server.send({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
    assert.equal(r1.ok, true);
    assert.deepEqual(Object.keys(r1.plots), ["SMA5"]);

    const r2 = await server.send({ source: `//@version=5\nindicator("t")\nplot(ta.ema(close, 5), "EMA5")`, bars: BARS, mode: "indicator" });
    assert.equal(r2.ok, true);
    assert.deepEqual(Object.keys(r2.plots), ["EMA5"]);
  } finally {
    server.close();
  }
});

test("a hung script times out without killing the server -- the next request still works", async () => {
  const server = startServer();
  try {
    // Pine uses indentation for blocks, not {} -- a genuine infinite loop.
    const hung = `//@version=5\nindicator("t")\nvar x = 0\nwhile true\n    x := x + 1`;
    const r1 = await server.send({ source: hung, bars: BARS, mode: "indicator", timeoutMs: 300 });
    assert.equal(r1.ok, false);
    assert.match(r1.error, /exceeded/);

    const r2 = await server.send({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
    assert.equal(r2.ok, true);
  } finally {
    server.close();
  }
});

test("a malformed input line returns a structured error, not a crash", async () => {
  const server = startServer();
  try {
    const r1 = await server.sendRaw("not valid json {{{");
    assert.equal(r1.ok, false);
    assert.ok(r1.error);

    // The server itself must still be alive and correct afterward.
    const r2 = await server.send({ source: `//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")`, bars: BARS, mode: "indicator" });
    assert.equal(r2.ok, true);
  } finally {
    server.close();
  }
});
