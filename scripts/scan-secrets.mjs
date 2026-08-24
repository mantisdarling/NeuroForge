import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const roots = ["client/src", "tests", "scripts", ".github", "vercel.json", "package.json"];
const patterns = [
  /-----BEGIN (?:RSA|EC|OPENSSH) PRIVATE KEY-----/,
  /\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b/,
  /\bAKIA[0-9A-Z]{16}\b/,
  /(?:api[_-]?key|secret|token)\s*[:=]\s*["'][^"']{12,}["']/i,
];

async function collect(path) {
  try {
    const entries = await readdir(path, { withFileTypes: true });
    return (await Promise.all(entries.map((entry) => collect(join(path, entry.name))))).flat();
  } catch {
    return [path];
  }
}

const files = (await Promise.all(roots.map(collect))).flat();
const findings = [];
for (const file of files) {
  const content = await readFile(file, "utf8");
  if (patterns.some((pattern) => pattern.test(content))) findings.push(file);
}

if (findings.length) {
  console.error(`Potential credential markers found: ${findings.join(", ")}`);
  process.exit(1);
}
console.log(`Secret-marker scan passed across ${files.length} files.`);

