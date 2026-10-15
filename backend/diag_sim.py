"""Drive a simulation session and print the drowsiness trajectory."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=10))

def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=10))

# reset any session
try:
    post("/api/monitor/stop", {})
except Exception:
    pass
time.sleep(0.5)

print(post("/api/monitor/start", {
    "driver_label": "SimTest", "threshold": 0.5,
    "use_camera": False, "use_simulation": True}))

print(f"\n{'t':>5} {'frames':>7} {'p_drowsy':>9} {'p_closed':>9} {'closed_s':>9} {'perclos':>8} {'lstm':>7} {'tier':>9}")
print("-" * 74)
t0 = time.time()
last_frames = -1
while time.time() - t0 < 30:
    s = get("/api/monitor/state")
    frames = len(s.get("attention_hist", []))
    if frames != last_frames:
        last_frames = frames
        t = s.get("temporal", {})
        print(f"{time.time()-t0:5.1f} {frames:7d} {s.get('p_drowsy',0):9.3f} {s.get('p_closed',0):9.3f} "
              f"{s.get('eye_closed_seconds',0):9.2f} {s.get('perclos',0):8.3f} "
              f"{(t.get('p_drowsy_lstm') if t.get('p_drowsy_lstm') is not None else 0):7.3f} "
              f"{s.get('tier','?'):>9}")
    time.sleep(0.25)

post("/api/monitor/stop", {})
print("\nevents:")
for e in get("/api/monitor/state").get("events", []):
    print("  ", e.get("tier"), "-", e.get("text"))
