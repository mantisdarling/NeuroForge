import { readFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const patterns = [
  /-----BEGIN (?:RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY BLOCK-----/,
  /\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b/,
  /\bglpat-[A-Za-z0-9_-]{20,}\b/,
  /\bnpm_[A-Za-z0-9]{36}\b/,
  /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/,
  /\bAKIA[0-9A-Z]{16}\b/,
  /\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b/,
  /(?:api[_-]?key|secret|token|password)\s*[:=]\s*["'][^"']{12,}["']/i,
  /authorization\s*[:=]\s*["']?bearer\s+[A-Za-z0-9._-]{12,}/i,
];

const { stdout } = await execFileAsync("git", ["ls-files", "-z"]);
const files = stdout.split("\0").filter(Boolean);
const findings = [];
for (const file of files) {
  const buffer = await readFile(file);
  if (buffer.includes(0)) continue;
  const content = buffer.toString("utf8");
  if (patterns.some((pattern) => pattern.test(content))) findings.push(file);
}

if (findings.length) {
  console.error(`Potential credential markers found: ${findings.join(", ")}`);
  process.exit(1);
}
console.log(`Secret-marker scan passed across ${files.length} files.`);
