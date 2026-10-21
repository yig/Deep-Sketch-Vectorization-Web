// Interactive refinement: zoom / pan, keypoint editing, USM painting, undo / redo.
//
// The editor owns the edit state (keypoints + USM). Every committed change bumps `version` and calls
// `onChange`; the worker re-runs the fast finishing stage on the whole state (no incremental
// bookkeeping), and `applyResult` takes back the worker's snapped keypoint positions and statuses.

import { CellStatus, type EditState, type RegionInfo, type RunResult } from "./protocol";

export type Tool = "pan" | "keypoint" | "brush" | "eraser" | "lasso";
export type MarkType = 1 | 2 | 3;

export interface EditorElements {
  stage: HTMLElement;
  canvas: HTMLElement;
  usm: HTMLCanvasElement;
  keypoints: SVGSVGElement;
  lasso: SVGSVGElement;
}

const HISTORY_LIMIT = 60;
const MARK_COLORS: Record<MarkType, string> = { 1: "var(--end)", 2: "var(--sharp)", 3: "var(--junc)" };
const STATUS_RGBA: Record<number, [number, number, number, number]> = {
  [CellStatus.Detail]: [147, 51, 234, 70],
  [CellStatus.Invalid]: [107, 114, 128, 150],
  [CellStatus.NotApplied]: [220, 38, 38, 130],
  [CellStatus.Applied]: [234, 88, 12, 130],
};
const PENDING_RGBA: [number, number, number, number] = [37, 99, 235, 110];

const e_isOverToolbar = (x: number, y: number) => !!document.elementFromPoint(x, y)?.closest(".toolbar");

const cloneState = (s: EditState): EditState => ({ marks: s.marks.map((m) => ({ ...m })), usm: s.usm.slice() });

export class Editor {
  tool: Tool = "pan";
  markType: MarkType = 1;
  /** Brush radius in source-image pixels. */
  brushRadius = 3;
  showKeypoints = false;
  showUsm = false;
  /** Incremented on every committed edit. */
  version = 0;
  onChange: () => void = () => {};
  onStatus: (text: string) => void = () => {};
  /** The region under the cursor changed (null: none). */
  onHover: (info: { info: RegionInfo; status: number } | null) => void = () => {};

  private state: EditState | null = null;
  private auto: EditState | null = null;
  private prepKey = "";
  private edited = false;
  private undoStack: EditState[] = [];
  private redoStack: EditState[] = [];
  private result: RunResult | null = null;
  private srcW = 1;
  private srcH = 1;
  private selected = -1;
  // View: canvas top-left in stage coordinates and displayed width.
  private view = { x: 0, y: 0, w: 0, fitted: false };
  private spaceHeld = false;
  /** Region label under the cursor (from the last result), 0 if none. */
  private hoverLabel = 0;
  private lastMouse: { x: number; y: number } | null = null;

  constructor(private els: EditorElements) {
    this.bindPointer();
    this.bindKeys();
    els.stage.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    new ResizeObserver(() => this.layout()).observe(els.stage);
  }

  // ---------- state ----------

  /** Edit state to send with the next run, or null when nothing was edited. */
  editRequest(): { prepKey: string; state: EditState } | null {
    return this.state && this.edited ? { prepKey: this.prepKey, state: cloneState(this.state) } : null;
  }

  get canUndo() {
    return this.undoStack.length > 0;
  }
  get canRedo() {
    return this.redoStack.length > 0;
  }
  get isEdited() {
    return this.edited;
  }

  setSource(w: number, h: number) {
    this.srcW = w;
    this.srcH = h;
    this.state = this.auto = this.result = null;
    this.undoStack = [];
    this.redoStack = [];
    this.edited = false;
    this.selected = -1;
    this.view.fitted = false;
    this.layout();
    this.render();
  }

  /** Take a result from the worker. `version` is the editor version the request was made at. */
  applyResult(result: RunResult, version: number) {
    this.result = result;
    if (!result.edited) {
      const hadEdits = this.edited && this.prepKey !== result.prepKey;
      this.auto = { marks: result.marks.map((m) => ({ ...m })), usm: result.usm.slice() };
      if (!this.edited || this.prepKey !== result.prepKey) {
        this.state = cloneState(this.auto);
        this.undoStack = [];
        this.redoStack = [];
        this.edited = false;
        this.selected = -1;
      }
      this.prepKey = result.prepKey;
      if (hadEdits) this.onStatus("Edits cleared: the network outputs changed.");
    } else if (version === this.version && this.state) {
      // Adopt the worker's snapped keypoint positions (same order as sent).
      this.state.marks = result.marks.map((m) => ({ ...m }));
    }
    // Labels and explanations changed: refresh the hover.
    this.hoverLabel = -1;
    if (this.lastMouse) this.updateHover(this.lastMouse.x, this.lastMouse.y);
    else this.setHover(0);
    this.render();
  }

  private commit(next: EditState) {
    if (!this.state) return;
    this.undoStack.push(this.state);
    if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
    this.redoStack = [];
    this.state = next;
    this.edited = true;
    this.version++;
    this.render();
    this.onChange();
  }

  undo() {
    if (!this.state || !this.undoStack.length) return;
    this.redoStack.push(this.state);
    this.state = this.undoStack.pop()!;
    this.afterHistory();
  }

  redo() {
    if (!this.state || !this.redoStack.length) return;
    this.undoStack.push(this.state);
    this.state = this.redoStack.pop()!;
    this.afterHistory();
  }

  private afterHistory() {
    this.edited = true;
    this.selected = -1;
    this.version++;
    this.render();
    this.onChange();
  }

  /** Back to the automatic result (undoable). */
  resetToAuto() {
    if (this.auto && this.state) this.commit(cloneState(this.auto));
  }

  setMarkType(t: MarkType) {
    this.markType = t;
    if (this.selected >= 0 && this.state && this.state.marks[this.selected].type !== t) {
      const next = cloneState(this.state);
      next.marks[this.selected].type = t;
      this.commit(next);
    }
  }

  deleteSelected() {
    if (this.selected < 0 || !this.state) return;
    const next = cloneState(this.state);
    next.marks.splice(this.selected, 1);
    this.selected = -1;
    this.commit(next);
  }

  // ---------- view ----------

  private get grid() {
    const r = this.result;
    return r ? { w: r.gridW, h: r.gridH, cell: r.scale / 2 } : null;
  }

  layout() {
    const { stage, canvas } = this.els;
    const sw = stage.clientWidth;
    const sh = stage.clientHeight;
    if (!this.view.fitted) {
      const pad = 24;
      const s = Math.min((sw - 2 * pad) / this.srcW, (sh - 2 * pad) / this.srcH, 4);
      this.view.w = Math.max(16, this.srcW * s);
      this.view.x = (sw - this.view.w) / 2;
      this.view.y = (sh - (this.view.w * this.srcH) / this.srcW) / 2;
      this.view.fitted = sw > 0;
    }
    canvas.style.left = `${this.view.x}px`;
    canvas.style.top = `${this.view.y}px`;
    canvas.style.width = `${this.view.w}px`;
    canvas.style.height = `${(this.view.w * this.srcH) / this.srcW}px`;
    this.renderMarks();
  }

  fit() {
    this.view.fitted = false;
    this.layout();
  }

  zoomBy(f: number, cx?: number, cy?: number) {
    const sw = this.els.stage.clientWidth;
    const sh = this.els.stage.clientHeight;
    cx ??= sw / 2;
    cy ??= sh / 2;
    const w = Math.min(Math.max(this.view.w * f, 32), this.srcW * 64);
    const k = w / this.view.w;
    this.view.x = cx - (cx - this.view.x) * k;
    this.view.y = cy - (cy - this.view.y) * k;
    this.view.w = w;
    this.layout();
  }

  private onWheel(e: WheelEvent) {
    if (this.els.canvas.hidden) return;
    e.preventDefault();
    const r = this.els.stage.getBoundingClientRect();
    if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) >= Math.abs(e.deltaX)) {
      this.zoomBy(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
    } else {
      this.view.x -= e.deltaX;
      this.layout();
    }
  }

  /** Client coordinates to source-image pixels. */
  private toSource(clientX: number, clientY: number) {
    const r = this.els.canvas.getBoundingClientRect();
    return { x: ((clientX - r.left) / r.width) * this.srcW, y: ((clientY - r.top) / r.height) * this.srcH };
  }

  /** Source pixels per screen pixel. */
  private get srcPerScreen() {
    return this.srcW / this.view.w;
  }

  // ---------- pointer interaction ----------

  private bindPointer() {
    const stage = this.els.stage;
    let mode: "none" | "pan" | "drag" | "paint" | "lasso" = "none";
    let path: { x: number; y: number }[] = [];
    let last = { x: 0, y: 0 };
    let moved = false;
    let draft: EditState | null = null;

    stage.addEventListener("pointerdown", (e) => {
      if (this.els.canvas.hidden || !this.state || e.button === 2) return;
      if ((e.target as Element).closest(".toolbar, .edit-hint")) return;
      e.preventDefault(); // no native image drag or text selection
      stage.setPointerCapture(e.pointerId);
      last = { x: e.clientX, y: e.clientY };
      moved = false;
      if (this.tool === "pan" || e.button === 1 || this.spaceHeld) {
        mode = "pan";
        stage.classList.add("panning");
        return;
      }
      const p = this.toSource(e.clientX, e.clientY);
      if (this.tool === "keypoint") {
        const hit = this.hitMark(p.x, p.y);
        draft = cloneState(this.state);
        if (hit >= 0) {
          this.selected = hit;
          if (e.altKey) {
            draft.marks.splice(hit, 1);
            this.selected = -1;
            this.commit(draft);
            draft = null;
            mode = "none";
            return;
          }
        } else {
          draft.marks.push({ type: this.markType, x: p.x, y: p.y, cell: null });
          this.selected = draft.marks.length - 1;
          moved = true;
        }
        mode = "drag";
        this.renderMarks(draft);
      } else if (this.tool === "lasso") {
        mode = "lasso";
        path = [p];
        this.renderLasso(path, e.altKey);
      } else {
        draft = cloneState(this.state);
        mode = "paint";
        this.paint(draft, p.x, p.y, p.x, p.y);
      }
    });

    stage.addEventListener("pointermove", (e) => {
      if (mode === "none") {
        if (e.pointerType === "mouse") this.updateHover(e.clientX, e.clientY);
        return;
      }
      if (mode === "pan") {
        this.view.x += e.clientX - last.x;
        this.view.y += e.clientY - last.y;
        last = { x: e.clientX, y: e.clientY };
        this.layout();
        return;
      }
      const a = this.toSource(last.x, last.y);
      const b = this.toSource(e.clientX, e.clientY);
      if (Math.hypot(e.clientX - last.x, e.clientY - last.y) > 0) moved = true;
      if (mode === "drag" && draft && this.selected >= 0) {
        const m = draft.marks[this.selected];
        m.x += b.x - a.x;
        m.y += b.y - a.y;
        m.cell = null;
        this.renderMarks(draft);
      } else if (mode === "paint" && draft) {
        this.paint(draft, a.x, a.y, b.x, b.y);
      } else if (mode === "lasso") {
        path.push(b);
        this.renderLasso(path, e.altKey);
      }
      last = { x: e.clientX, y: e.clientY };
    });

    const end = (e: PointerEvent) => {
      if (mode === "lasso") {
        this.renderLasso([], false);
        if (path.length >= 3 && this.state) {
          const next = cloneState(this.state);
          if (this.fillPolygon(next, path, e.altKey ? 0 : 1)) this.commit(next);
        }
        path = [];
      } else if (mode === "pan") stage.classList.remove("panning");
      else if ((mode === "drag" || mode === "paint") && draft && moved) this.commit(draft);
      else this.render();
      mode = "none";
      draft = null;
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointerleave", () => {
      this.lastMouse = null;
      this.setHover(0);
    });
    stage.addEventListener("pointercancel", () => {
      // Abandon the gesture.
      if (mode === "lasso") this.renderLasso([], false);
      stage.classList.remove("panning");
      mode = "none";
      draft = null;
      path = [];
      this.render();
    });

    stage.addEventListener("dblclick", (e) => {
      if (this.tool !== "keypoint" || !this.state || (e.target as Element).closest(".toolbar")) return;
      const p = this.toSource(e.clientX, e.clientY);
      const hit = this.hitMark(p.x, p.y);
      if (hit < 0) return;
      const next = cloneState(this.state);
      const m = next.marks[hit];
      m.type = ((m.type % 3) + 1) as MarkType;
      this.selected = hit;
      this.commit(next);
    });
  }

  private updateHover(clientX: number, clientY: number) {
    this.lastMouse = { x: clientX, y: clientY };
    const r = this.result;
    const g = this.grid;
    if (!r || !g || this.tool === "pan" || (e_isOverToolbar(clientX, clientY))) return this.setHover(0);
    const p = this.toSource(clientX, clientY);
    const i = Math.floor(p.y / g.cell);
    const j = Math.floor(p.x / g.cell);
    this.setHover(i >= 0 && j >= 0 && i < g.h && j < g.w ? r.regionLabels[i * g.w + j] : 0);
  }

  private setHover(label: number) {
    if (label === this.hoverLabel) return;
    this.hoverLabel = label;
    const r = this.result;
    this.onHover(label && r ? { info: r.regionInfo[label], status: r.cellStatus[r.regionLabels.indexOf(label)] } : null);
    this.render();
  }

  private hitMark(x: number, y: number): number {
    if (!this.state) return -1;
    const r = 9 * this.srcPerScreen;
    let best = -1;
    let bd = r;
    this.state.marks.forEach((m, i) => {
      const d = Math.hypot(m.x - x, m.y - y);
      if (d <= bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  }

  /** Paint (brush) or erase a capsule from a to b into draft.usm and redraw the overlay. */
  private paint(draft: EditState, ax: number, ay: number, bx: number, by: number) {
    const g = this.grid;
    if (!g) return;
    const value = this.tool === "eraser" ? 0 : 1;
    const r = this.brushRadius / g.cell;
    const [ai, aj, bi, bj] = [ay / g.cell, ax / g.cell, by / g.cell, bx / g.cell];
    const i0 = Math.max(0, Math.floor(Math.min(ai, bi) - r));
    const i1 = Math.min(g.h - 1, Math.ceil(Math.max(ai, bi) + r));
    const j0 = Math.max(0, Math.floor(Math.min(aj, bj) - r));
    const j1 = Math.min(g.w - 1, Math.ceil(Math.max(aj, bj) + r));
    const di = bi - ai;
    const dj = bj - aj;
    const len2 = di * di + dj * dj;
    for (let i = i0; i <= i1; i++)
      for (let j = j0; j <= j1; j++) {
        const ci = i + 0.5;
        const cj = j + 0.5;
        const t = len2 > 0 ? Math.min(1, Math.max(0, ((ci - ai) * di + (cj - aj) * dj) / len2)) : 0;
        if (Math.hypot(ci - ai - t * di, cj - aj - t * dj) <= r) draft.usm[i * g.w + j] = value;
      }
    this.renderUsm(draft);
  }

  /** Set the cells whose centers lie inside the polygon (even-odd rule). Returns whether any changed. */
  private fillPolygon(draft: EditState, poly: { x: number; y: number }[], value: number): boolean {
    const g = this.grid;
    if (!g) return false;
    const ys = poly.map((q) => q.y / g.cell);
    const xs = poly.map((q) => q.x / g.cell);
    const i0 = Math.max(0, Math.floor(Math.min(...ys)));
    const i1 = Math.min(g.h - 1, Math.ceil(Math.max(...ys)));
    let changed = false;
    for (let i = i0; i <= i1; i++) {
      const cy = i + 0.5;
      // Crossings of the scanline with each polygon edge.
      const cross: number[] = [];
      for (let k = 0; k < xs.length; k++) {
        const k2 = (k + 1) % xs.length;
        const [ya, yb] = [ys[k], ys[k2]];
        if (ya <= cy !== yb <= cy) cross.push(xs[k] + ((cy - ya) / (yb - ya)) * (xs[k2] - xs[k]));
      }
      cross.sort((a, b) => a - b);
      for (let c = 0; c + 1 < cross.length; c += 2) {
        const j0 = Math.max(0, Math.ceil(cross[c] - 0.5));
        const j1 = Math.min(g.w - 1, Math.floor(cross[c + 1] - 0.5));
        for (let j = j0; j <= j1; j++) {
          const p = i * g.w + j;
          if (draft.usm[p] !== value) {
            draft.usm[p] = value;
            changed = true;
          }
        }
      }
    }
    return changed;
  }

  private renderLasso(path: { x: number; y: number }[], erase: boolean) {
    const svg = this.els.lasso;
    svg.setAttribute("viewBox", `0 0 ${this.srcW} ${this.srcH}`);
    svg.setAttribute("preserveAspectRatio", "none");
    if (path.length < 2) {
      svg.innerHTML = "";
      return;
    }
    const k = this.srcPerScreen;
    const pts = path.map((q) => `${q.x},${q.y}`).join(" ");
    const color = erase ? "#6b7280" : "#2563eb";
    svg.innerHTML = `<polygon points="${pts}" fill="${color}" fill-opacity="0.15" stroke="${color}" stroke-width="${1.5 * k}" stroke-dasharray="${4 * k} ${3 * k}"/>`;
  }

  private bindKeys() {
    window.addEventListener("keydown", (e) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, select, textarea")) return;
      if (e.key === " ") {
        this.spaceHeld = true;
        e.preventDefault();
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.key === " ") this.spaceHeld = false;
    });
  }

  // ---------- rendering ----------

  render() {
    this.renderUsm();
    this.renderMarks();
  }

  private renderUsm(state: EditState | null = this.state) {
    const cv = this.els.usm;
    const r = this.result;
    const show = this.showUsm || this.tool !== "pan";
    cv.style.display = show && r && state ? "" : "none";
    if (!r || !state || !show) return;
    if (cv.width !== r.gridW || cv.height !== r.gridH) {
      cv.width = r.gridW;
      cv.height = r.gridH;
    }
    const ctx = cv.getContext("2d")!;
    const img = ctx.createImageData(r.gridW, r.gridH);
    const sameGrid = r.usm.length === state.usm.length;
    const hover = this.hoverLabel;
    for (let p = 0; p < state.usm.length; p++) {
      const v = state.usm[p];
      if (!v) continue;
      const settled = sameGrid && r.usm[p] === v;
      const rgba = settled ? STATUS_RGBA[r.cellStatus[p]] ?? PENDING_RGBA : PENDING_RGBA;
      img.data.set(rgba, 4 * p);
      if (settled && hover && r.regionLabels[p] === hover) img.data[4 * p + 3] = Math.min(255, rgba[3] * 2);
    }
    ctx.putImageData(img, 0, 0);
    cv.style.width = `${((r.prepared.W * r.scale) / this.srcW) * 100}%`;
    cv.style.height = `${((r.prepared.H * r.scale) / this.srcH) * 100}%`;
  }

  private renderMarks(state: EditState | null = this.state) {
    const svg = this.els.keypoints;
    const show = this.showKeypoints || this.tool !== "pan";
    svg.style.display = show && state ? "" : "none";
    if (!state || !show) return;
    svg.setAttribute("viewBox", `0 0 ${this.srcW} ${this.srcH}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const k = this.srcPerScreen;
    const r = 4.5 * k;
    const res = this.result;
    const hover = this.hoverLabel;
    svg.innerHTML = state.marks
      .map((m, i) => {
        const sel = i === this.selected && this.tool === "keypoint";
        const inside = hover > 0 && res !== null && m.cell !== null && res.regionLabels[m.cell] === hover;
        return (
          (inside ? `<circle cx="${m.x}" cy="${m.y}" r="${r + 4 * k}" fill="none" stroke="#facc15" stroke-width="${2.5 * k}"/>` : "") +
          `<circle cx="${m.x}" cy="${m.y}" r="${r}" fill="${MARK_COLORS[m.type]}" fill-opacity="0.9" stroke="#fff" stroke-width="${1.2 * k}"/>` +
          (sel ? `<circle cx="${m.x}" cy="${m.y}" r="${r + 3 * k}" fill="none" stroke="var(--ink)" stroke-width="${1.5 * k}"/>` : "")
        );
      })
      .join("");
  }
}
