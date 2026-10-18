/* Shared design tokens + API client. Kept free of React components so
 * react-refresh (HMR) can treat component files separately. */

export const C = {
  bg: "#FFFFFF",
  surfaceAlt: "#F4F6F9",
  surfaceAlt2: "#EEF1F6",
  border: "#E2E6EC",
  ink: "#132133",
  inkMuted: "#5B6572",
  inkDim: "#8A93A2",
  navy: "#163A5F",
  navySoft: "#1F4C79",
  amber: "#C57C21",
  amberSoft: "#FBEDD5",
  teal: "#1F7A63",
  tealSoft: "#E4F2EC",
  critical: "#C4463F",
  criticalSoft: "#F9E4E2",
  caution: "#946B0C",
  cautionSoft: "#FBEDD5",
  console: "#0E1620",
  consoleBorder: "#1F2C3A",
};

/** Base URL of the FastAPI backend.
 * In production (Render, Hugging Face, or unified server), API is on the same origin (empty string "").
 * In standalone Vite dev server (npm run dev on port 5173), it defaults to localhost:8000.
 */
export const API_BASE =
  import.meta.env.VITE_API_BASE !== undefined
    ? import.meta.env.VITE_API_BASE
    : import.meta.env.DEV
      ? "http://127.0.0.1:8000"
      : "";

export async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    const d = detail.detail;
    throw new Error(
      typeof d === "string"
        ? d
        : Array.isArray(d)
          ? d.map((e) => e.msg).join("; ")
          : `${res.status} ${res.statusText}`
    );
  }
  return res.json();
}
