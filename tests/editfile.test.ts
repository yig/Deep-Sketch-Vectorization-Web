import { expect, test } from "vitest";
import { decodeEdits, encodeEdits } from "../src/editfile";

test("edit files round-trip and are validated", () => {
  const usm = new Uint8Array(40 * 30);
  usm.fill(1, 100, 180);
  usm.fill(2, 500, 520);
  const state = { marks: [{ type: 3 as const, x: 10.5, y: 4.25, cell: 101 }, { type: 1 as const, x: 2, y: 3, cell: null }], usm };
  const text = encodeEdits(state, { key: "abc", name: "x.png", width: 20, height: 15 }, "full", 512, { w: 40, h: 30 });
  expect(text.length).toBeLessThan(400);
  const { file, state: back } = decodeEdits(text);
  expect(file.model).toBe("full");
  expect(file.image.key).toBe("abc");
  expect(back.marks).toEqual(state.marks);
  expect([...back.usm]).toEqual([...usm]);
  expect(() => decodeEdits('{"format":"other"}')).toThrow(/not a Deep Sketch/);
  const bad = JSON.parse(text);
  bad.usm[1] += 1;
  expect(() => decodeEdits(JSON.stringify(bad))).toThrow();
});
