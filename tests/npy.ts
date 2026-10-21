// Minimal .npy reader for golden fixtures written by tools/make_fixtures.py.
import { readFileSync } from "node:fs";

export interface Npy {
  shape: number[];
  data:
    | Float32Array<ArrayBuffer>
    | Float64Array<ArrayBuffer>
    | Int32Array<ArrayBuffer>
    | Uint8Array<ArrayBuffer>
    | BigInt64Array<ArrayBuffer>;
}

export function readNpy(path: string): Npy {
  const buf = readFileSync(path);
  if (buf[0] !== 0x93 || buf.toString("latin1", 1, 6) !== "NUMPY") throw new Error(`not an npy file: ${path}`);
  const major = buf[6];
  const headerLen = major >= 2 ? buf.readUInt32LE(8) : buf.readUInt16LE(8);
  const start = major >= 2 ? 12 : 10;
  const header = buf.toString("latin1", start, start + headerLen);
  const descr = /'descr':\s*'([^']+)'/.exec(header)![1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error("fortran order unsupported");
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)![1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const off = start + headerLen;
  const bytes = buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + buf.length);
  const ctor = { "<f4": Float32Array, "<f8": Float64Array, "<i4": Int32Array, "|b1": Uint8Array, "|u1": Uint8Array, "<i8": BigInt64Array }[descr];
  if (!ctor) throw new Error(`unsupported dtype ${descr} in ${path}`);
  return { shape, data: new ctor(bytes) as Npy["data"] };
}
