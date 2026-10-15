/* Audible alert engine for the critical fatigue cascade.
 *
 * Uses the Web Audio API to synthesize an escalating two-tone siren (no audio
 * asset needed) and the SpeechSynthesis API for a spoken warning. Also fires
 * a desktop Notification when the user has granted permission.
 */

let ctx = null;
let sirenTimer = null;
let chimeTimer = null;

function audioCtx() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
  }
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

function beep(freq, dur, when = 0, type = "sine", gain = 0.16) {
  const ac = audioCtx();
  if (!ac) return;
  const osc = ac.createOscillator();
  const g = ac.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, ac.currentTime + when);
  g.gain.exponentialRampToValueAtTime(gain, ac.currentTime + when + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + when + dur);
  osc.connect(g).connect(ac.destination);
  osc.start(ac.currentTime + when);
  osc.stop(ac.currentTime + when + dur + 0.05);
}

export function unlockAudio() {
  audioCtx();
}

/** Soft two-note chime for the warning tier. */
export function playWarningChime() {
  beep(660, 0.18, 0, "sine", 0.12);
  beep(880, 0.22, 0.2, "sine", 0.12);
}

/** Escalating two-tone siren, repeated while a critical cascade is active. */
export function startCriticalSiren() {
  if (sirenTimer) return;
  const cycle = () => {
    beep(760, 0.28, 0, "square", 0.14);
    beep(560, 0.28, 0.32, "square", 0.14);
    beep(760, 0.28, 0.64, "square", 0.14);
    beep(560, 0.28, 0.96, "square", 0.14);
  };
  cycle();
  sirenTimer = setInterval(cycle, 1500);
}

export function stopCriticalSiren() {
  if (sirenTimer) {
    clearInterval(sirenTimer);
    sirenTimer = null;
  }
}

/** Spoken warning via SpeechSynthesis (if available). */
export function speak(text) {
  try {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    u.pitch = 1.0;
    u.volume = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  } catch {
    /* speech not available */
  }
}

export function notifyDesktop(title, body) {
  try {
    if (!("Notification" in window)) return;
    if (Notification.permission === "granted") {
      new Notification(title, { body });
    } else if (Notification.permission !== "denied") {
      Notification.requestPermission().then((p) => {
        if (p === "granted") new Notification(title, { body });
      });
    }
  } catch {
    /* notifications not available */
  }
}

export function stopAllAudio() {
  stopCriticalSiren();
  if (chimeTimer) {
    clearInterval(chimeTimer);
    chimeTimer = null;
  }
  try {
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  } catch {
    /* noop */
  }
}
