import { readFile } from "node:fs/promises";

const requiredHeaders = new Map([
  ["x-content-type-options", "nosniff"],
  ["x-frame-options", "DENY"],
  ["referrer-policy", "strict-origin-when-cross-origin"],
  ["cross-origin-opener-policy", "same-origin"],
  ["cross-origin-resource-policy", "same-origin"],
  ["origin-agent-cluster", "?1"],
  ["x-permitted-cross-domain-policies", "none"],
]);
const requiredCspDirectives = [
  "default-src 'self'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
  "frame-src 'none'",
  "object-src 'none'",
  "script-src 'self'",
  "script-src-elem 'self'",
  "connect-src 'self'",
  "worker-src 'none'",
];

const config = JSON.parse(await readFile("vercel.json", "utf8"));
const globalHeaders = config.headers?.find((rule) => rule.source === "/(.*)")?.headers ?? [];
const headerMap = new Map(globalHeaders.map((header) => [header.key.toLowerCase(), header.value]));
const failures = [];

for (const [key, value] of requiredHeaders) {
  if (headerMap.get(key) !== value) failures.push(`missing or weak ${key} header`);
}

const csp = headerMap.get("content-security-policy") ?? "";
for (const directive of requiredCspDirectives) {
  if (!csp.includes(directive)) failures.push(`CSP is missing ${directive}`);
}

const workflow = await readFile(".github/workflows/ci.yml", "utf8");
const actionRefs = [...workflow.matchAll(/^\s*- uses:\s+[^@\s]+@([^\s#]+)$/gm)].map((match) => match[1]);
if (!actionRefs.length || actionRefs.some((ref) => !/^[a-f0-9]{40}$/.test(ref))) failures.push("CI actions must be pinned to full commit SHAs");

if (failures.length) {
  console.error(`Security configuration check failed:\n${failures.join("\n")}`);
  process.exit(1);
}

console.log("Security configuration check passed.");
