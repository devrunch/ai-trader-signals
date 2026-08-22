import { runPine } from "./run_pine_core.mjs";

const input = JSON.parse(await new Promise((resolve) => {
  let data = "";
  process.stdin.on("data", (chunk) => { data += chunk; });
  process.stdin.on("end", () => resolve(data));
}));
const result = await runPine(input);
process.stdout.write(JSON.stringify(result));
