// Anonymous usage counts via GoatCounter (no cookies, no image data: only a path and screen size).
//
// GoatCounter's count.js is cross-origin, which the page's COEP (require-corp) blocks, so hits go
// straight to its /count endpoint with sendBeacon, as count.js itself does. A beacon reads no
// response, so cross-origin isolation does not affect it.

/**
 * GoatCounter site codes (https://<code>.goatcounter.com); every hit goes to each. The first site has
 * sessions on (unique visitors), the second has them off (every hit). Empty disables counting.
 */
const GOATCOUNTER_CODES: string[] = [
  "cragl-deep-sketch-vectorization-web",
  "cragl-deep-sketch-vectorization-web-vectorizations",
];

const enabled =
  GOATCOUNTER_CODES.length > 0 &&
  import.meta.env.PROD &&
  !["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);

function count(path: string, title: string, event: boolean) {
  if (!enabled) return;
  const q = new URLSearchParams({
    p: path,
    t: title,
    s: `${screen.width},${screen.height},${window.devicePixelRatio || 1}`,
    rnd: Math.random().toString(36).slice(2),
  });
  if (event) q.set("e", "true");
  else if (document.referrer) q.set("r", document.referrer);
  for (const code of GOATCOUNTER_CODES) {
    try {
      navigator.sendBeacon(`https://${code}.goatcounter.com/count?${q}`);
    } catch {
      // Counting is best-effort.
    }
  }
}

/** Counts a visit, unless coi-serviceworker is about to reload the page (first visit). */
export function countPageview() {
  const sw = navigator.serviceWorker;
  if (!crossOriginIsolated && sw && !sw.controller) return;
  count(location.pathname, document.title, false);
}

/** Counts one vectorized sketch. */
export function countVectorized() {
  count("vectorize", "Sketch vectorized", true);
}
