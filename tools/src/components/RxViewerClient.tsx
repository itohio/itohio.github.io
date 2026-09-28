"use client";

/**
 * RX Blind-Spot Viewer.
 * The viewer is served from a frozen build snapshot under /legacy (its React source is not
 * part of this tree). It is embedded full-viewport; its own "back" navigation is forwarded to
 * the parent hash router by a small script inside the snapshot.
 */
export default function RxViewerClient() {
  return (
    <iframe
      src="/legacy/?tool=rxmap"
      title="RX Blind-Spot Viewer"
      style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh", border: 0, background: "#0d1117" }}
    />
  );
}
