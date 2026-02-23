/**
 * Build rustdoc for a small batch of crates with math-heavy docs.
 *
 * Why:
 * - We want end-to-end validation that math renders in rustdoc output, not just "parses".
 * - We keep this build list intentionally small (curated), then scale out once the harness is stable.
 *
 * Output:
 * - Writes `rustdoc-math-targets.json` next to this file for Playwright tests to consume.
 */

import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// This file lives at: <dev>/subsume/docs/build-rustdoc-math.mjs
// So the dev root is exactly two parents up.
const devRoot = path.resolve(__dirname, "..", "..");

function cargoBin() {
  // `npm run` often runs without your interactive shell PATH.
  // Prefer an explicit override, then the standard rustup install location.
  return (
    process.env.CARGO ??
    path.join(process.env.HOME ?? "", ".cargo", "bin", "cargo")
  );
}

function runCapture(cmd, args, cwd) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd, env: process.env });
    let out = "";
    let err = "";
    p.stdout?.on("data", (d) => (out += String(d)));
    p.stderr?.on("data", (d) => (err += String(d)));
    p.on("error", reject);
    p.on("exit", (code) => {
      if (code === 0) resolve({ out, err });
      else reject(new Error(`Command failed (${code}): ${cmd} ${args.join(" ")}\n${err}`));
    });
  });
}

async function cargoTargetDir(cwd) {
  const { out } = await runCapture(
    cargoBin(),
    ["metadata", "--format-version", "1", "--no-deps"],
    cwd,
  );
  const meta = JSON.parse(out);
  return meta.target_directory;
}

/** Curated targets: repo path + cargo invocation + *package* name */
const targets = [
  {
    id: "wass",
    cwd: path.join(devRoot, "wass"),
    cargo: ["cargo", ["doc", "--no-deps"]],
    package: "wass",
  },
  {
    id: "cerno-retrieve",
    cwd: path.join(devRoot, "cerno"),
    cargo: ["cargo", ["doc", "--no-deps", "-p", "cerno-retrieve"]],
    package: "cerno-retrieve",
  },
  {
    id: "subsume-core",
    cwd: path.join(devRoot, "subsume"),
    cargo: ["cargo", ["doc", "--no-deps", "-p", "subsume-core"]],
    package: "subsume-core",
  },
];

function run(cmd, args, cwd) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, {
      cwd,
      stdio: "inherit",
      env: process.env,
    });
    p.on("error", reject);
    p.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Command failed (${code}): ${cmd} ${args.join(" ")}`));
    });
  });
}

console.log("Building rustdoc for math E2E targets:");
for (const t of targets) console.log(`- ${t.id}: ${t.cwd}`);

for (const t of targets) {
  const [cmd0, args] = t.cargo;
  const cmd = cmd0 === "cargo" ? cargoBin() : cmd0;
  console.log(`\n==> ${t.id}: ${cmd} ${args.join(" ")} (cwd=${t.cwd})`);
  await run(cmd, args, t.cwd);
}

// Normalize targets into concrete rustdoc paths.
const resolved = [];
for (const t of targets) {
  const targetDir = await cargoTargetDir(t.cwd);
  const docRoot = path.join(targetDir, "doc");
  const crate = t.package.replace(/-/g, "_");
  resolved.push({ ...t, targetDir, docRoot, crate });
}

const outPath = path.join(__dirname, "rustdoc-math-targets.json");
await writeFile(outPath, JSON.stringify({ targets: resolved }, null, 2) + "\n", "utf8");
console.log(`\nWrote ${outPath}`);

