// The similarity table ships as data/sims/part-*.bin (split so it can be pushed over slow links).
import { readFile, readdir, writeFile, mkdir } from "node:fs/promises";

export async function readSims(dataDir) {
  const dir = new URL("sims/", dataDir);
  const parts = (await readdir(dir)).filter((f) => f.endsWith(".bin")).sort((a, b) => parseInt(a.slice(5)) - parseInt(b.slice(5)));
  const bufs = await Promise.all(parts.map((f) => readFile(new URL(f, dir))));
  const all = Buffer.concat(bufs);
  return new Float32Array(all.buffer, all.byteOffset, all.length / 4);
}

export async function writeSims(dataDir, f32, parts = 4) {
  const dir = new URL("sims/", dataDir);
  await mkdir(dir, { recursive: true });
  const b = Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength);
  const step = Math.ceil(b.length / 4 / parts) * 4;
  for (let i = 0; i < parts; i++) await writeFile(new URL(`part-${i}.bin`, dir), b.subarray(i * step, (i + 1) * step));
}
