import { gzipSync } from "node:zlib";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const assetsDirectory = new URL("../dist/assets/", import.meta.url);
const limits = { ".js": 160 * 1024, ".css": 24 * 1024 };
const assets = readdirSync(assetsDirectory).filter((file) => statSync(join(assetsDirectory.pathname, file)).isFile());
const failures = [];

for (const asset of assets) {
  const extension = asset.match(/\.(js|css)$/)?.[0];
  if (!extension || !(extension in limits)) continue;
  const bytes = gzipSync(readFileSync(join(assetsDirectory.pathname, asset))).byteLength;
  if (bytes > limits[extension]) failures.push(`${asset}: ${(bytes / 1024).toFixed(1)} KiB gzip exceeds ${(limits[extension] / 1024).toFixed(0)} KiB`);
}

if (failures.length) {
  console.error(`Delivery budget failed:\n${failures.join("\n")}`);
  process.exit(1);
}

console.log(`Delivery budget passed across ${assets.length} emitted assets.`);
