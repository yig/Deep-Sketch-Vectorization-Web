// Interactive refinement semantics on fixture network outputs: the edit state (marks + USM) fully
// determines the result, erasing a region undoes its surgery, and a keypoint can make a region refinable.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { readNpy } from "./npy";
import { finish, prepare, vectorizeFromNetworks, type Prepared } from "../src/pipeline/vectorize";
import { RegionStatus, snapMark, type Mark } from "../src/pipeline/topology";

const ROOT = join(__dirname, "fixtures");
const fixtures = existsSync(ROOT) ? readdirSync(ROOT).filter((d) => existsSync(join(ROOT, d, "meta.json"))) : [];

function load(name: string): Prepared {
  const dir = join(ROOT, name);
  const meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
  const f = (n: string) => readNpy(join(dir, n + ".npy")).data as Float32Array;
  return prepare({ H: meta.h, W: meta.w, aux: f("aux"), edge: f("edge_logits"), offset: f("offset") });
}

/** Cells of each region label. */
function regionCells(labels: Int32Array): Map<number, number[]> {
  const m = new Map<number, number[]>();
  labels.forEach((l, p) => {
    if (!l) return;
    if (!m.has(l)) m.set(l, []);
    m.get(l)!.push(p);
  });
  return m;
}

// The light-model fixtures have regions the automatic pass could not refine.
const name = fixtures.find((f) => f.startsWith("Art_freeform_GL_01")) ?? fixtures[0];

describe.skipIf(fixtures.length === 0)("interactive refinement", () => {
  const prep = load(name);
  const auto = finish(prep, prep.autoMarks, prep.autoUsm);
  const cells = regionCells(auto.regionLabels);
  const withStatus = (s: number) => [...cells.keys()].filter((l) => auto.regionStatus[l] === s);

  test("the automatic pass is surgery on the automatic marks and USM", () => {
    const meta = JSON.parse(readFileSync(join(ROOT, name, "meta.json"), "utf8"));
    const f = (n: string) => readNpy(join(ROOT, name, n + ".npy")).data as Float32Array;
    const v = vectorizeFromNetworks({ H: meta.h, W: meta.w, aux: f("aux"), edge: f("edge_logits"), offset: f("offset") });
    expect(v.segments).toEqual(auto.segments);
    expect(withStatus(RegionStatus.Applied).length).toBeGreaterThan(0);
  });

  test("erasing an applied region removes its surgery", () => {
    const l = withStatus(RegionStatus.Applied)[0];
    const usm = prep.autoUsm.slice();
    for (const p of cells.get(l)!) usm[p] = 0;
    const res = finish(prep, prep.autoMarks, usm);
    expect(withStatus(RegionStatus.Applied).length - 1).toBe(
      [...regionCells(res.regionLabels).keys()].filter((k) => res.regionStatus[k] === RegionStatus.Applied).length,
    );
    expect(res.segments).not.toEqual(auto.segments);
    // Undo by restoring the USM: identical to the automatic result.
    expect(finish(prep, prep.autoMarks, prep.autoUsm).segments).toEqual(auto.segments);
  });

  test("a keypoint inside a region that needs one makes it refinable", () => {
    const candidates = withStatus(RegionStatus.NotApplied);
    expect(candidates.length).toBeGreaterThan(0);
    let fixed = 0;
    for (const l of candidates.slice(0, 10)) {
      const rc = cells.get(l)!;
      const { pt, valence } = prep.base;
      const v = rc.find((p) => valence[p] > 0) ?? rc[0];
      for (const type of [3, 2, 1] as const) {
        const mark: Mark = { type, cell: v, x: pt[2 * v], y: pt[2 * v + 1] };
        const res = finish(prep, [...prep.autoMarks.filter((m) => !rc.includes(m.cell)), mark], prep.autoUsm);
        if (res.regionStatus[res.regionLabels[rc[0]]] === RegionStatus.Applied) {
          fixed++;
          break;
        }
      }
    }
    expect(fixed).toBeGreaterThan(0);
  });

  test("painting a new region over a line rebuilds it", () => {
    // Paint a disc around a valence-2 vertex away from any USM region. With no keypoint inside, a
    // region two polylines enter is rebuilt by joining them.
    const { h, w } = prep;
    const { valence } = prep.base;
    let target = -1;
    for (let p = 0; p < h * w && target < 0; p++) {
      const i = Math.floor(p / w);
      const j = p % w;
      if (valence[p] !== 2 || i < 8 || j < 8 || i > h - 9 || j > w - 9) continue;
      let clear = true;
      for (let di = -6; di <= 6 && clear; di++) for (let dj = -6; dj <= 6; dj++) if (prep.autoUsm[(i + di) * w + j + dj]) clear = false;
      if (clear) target = p;
    }
    expect(target).toBeGreaterThanOrEqual(0);
    const i0 = Math.floor(target / w);
    const j0 = target % w;
    const usm = prep.autoUsm.slice();
    for (let di = -2; di <= 2; di++) for (let dj = -2; dj <= 2; dj++) usm[(i0 + di) * w + j0 + dj] = 1;
    const res = finish(prep, prep.autoMarks, usm);
    expect(res.regionStatus[res.regionLabels[target]]).toBe(RegionStatus.Applied);
  });

  test("painting next to a detail-only region does not pull it into the surgery", () => {
    const { h, w } = prep;
    const usm = prep.autoUsm.slice();
    // A detail-only cell with a free 4-neighbor to paint.
    let detail = -1;
    let paint = -1;
    for (let p = 0; p < h * w && paint < 0; p++) {
      if (usm[p] !== 2) continue;
      for (const q of [p - 1, p + 1, p - w, p + w])
        if (q >= 0 && q < h * w && usm[q] === 0) {
          detail = p;
          paint = q;
          break;
        }
    }
    expect(paint).toBeGreaterThanOrEqual(0);
    usm[paint] = 1;
    const res = finish(prep, prep.autoMarks, usm);
    expect(res.regionLabels[paint]).not.toBe(res.regionLabels[detail]);
    expect(res.regionStatus[res.regionLabels[detail]]).toBe(RegionStatus.Inactive);
    expect(usm[detail]).toBe(2);
  });

  test("every region is explained", () => {
    for (const [l, rc] of cells) {
      const info = auto.regionInfo[l];
      expect(info.cells).toBe(rc.length);
      expect(info.explanation.length).toBeGreaterThan(0);
      const kinds = [0, 0, 0];
      for (const m of prep.autoMarks) if (auto.regionLabels[m.cell] === l) kinds[m.type - 1]++;
      expect(info.keypoints).toEqual(kinds);
      if (auto.regionStatus[l] === RegionStatus.NotApplied) expect(info.explanation).toMatch(/keep only one|add an end point|no non-crossing|no vertices/);
      if (auto.regionStatus[l] === RegionStatus.Applied) expect(info.entering).toBeGreaterThan(0);
    }
  });

  test("snapMark snaps to a nearby vertex", () => {
    const { pt, valence } = prep.base;
    const v = valence.findIndex((x) => x > 0);
    const m = snapMark(prep.base, 3, pt[2 * v] + 0.1, pt[2 * v + 1] - 0.1);
    expect(m.cell).toBe(v);
    expect([m.x, m.y]).toEqual([pt[2 * v], pt[2 * v + 1]]);
  });

  test("snapMark keeps the position when no vertex is near", () => {
    const m = snapMark(prep.base, 1, 0.3, 0.3, 0.01);
    expect([m.x, m.y]).toEqual([0.3, 0.3]);
    expect(m.cell).toBe(0);
  });
});
