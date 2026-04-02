import cv2
import mediapipe as mp
import serial
import time
import numpy as np

# ─────────────────────────────────────────
#  CONFIGURE THIS
# ─────────────────────────────────────────
PORT      = "COM4"       # Windows: "COM4" | Linux/Mac: "/dev/ttyACM0"
BAUD_RATE = 9600
# ─────────────────────────────────────────

# ── Smoothing buffer (reduces jitter) ────
SMOOTH_FACTOR = 5
angle_buffer  = {"base": [], "vertical": []}

def smooth_angle(motor, new_angle):
    """Average last N angles to smooth servo movement."""
    angle_buffer[motor].append(new_angle)
    if len(angle_buffer[motor]) > SMOOTH_FACTOR:
        angle_buffer[motor].pop(0)
    return int(np.mean(angle_buffer[motor]))

def map_value(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another."""
    return int((value - in_min) * (out_max - out_min) /
               (in_max - in_min) + out_min)

# ── MediaPipe Setup ───────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode       = False,
    max_num_faces           = 1,
    refine_landmarks        = True,
    min_detection_confidence= 0.5,
    min_tracking_confidence = 0.5
)
NOSE_TIP = 1

# ── Serial Setup ──────────────────────────
try:
    arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {PORT}")
except serial.SerialException as e:
    print(f"Connection Error: {e}")
    exit()

# ── Camera Setup ─────────────────────────
cap = cv2.VideoCapture(0)

prev_base     = -1
prev_vertical = -1

print("=" * 45)
print("   Face Tracking Servo Control")
print("=" * 45)
print("  Nose X → Base    Servo (pin 9 )")
print("  Nose Y → Vertical Servo (pin 11)")
print("  Press Q to quit")
print("=" * 45)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip + convert
    frame    = cv2.flip(frame, 1)
    h, w, _  = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results  = face_mesh.process(rgb)

    # Draw center crosshair
    cv2.line(frame, (w // 2, 0),     (w // 2, h),     (50, 50, 50), 1)
    cv2.line(frame, (0, h // 2),     (w, h // 2),     (50, 50, 50), 1)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # ── Get nose tip coords ───────────────
            nose   = face_landmarks.landmark[NOSE_TIP]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            nose_z = round(nose.z, 4)

            # ── Map pixel coords → servo angles ───
            # nose_x: 0 to w  →  base:     0 to 180
            # nose_y: 0 to h  →  vertical: 180 to 0  (inverted Y axis)
            raw_base     = map_value(nose_x, 0, w, 0,   180)
            raw_vertical = map_value(nose_y, 0, h, 180, 0)    # inverted

            # ── Smooth angles ─────────────────────
            base_angle     = smooth_angle("base",     raw_base)
            vertical_angle = smooth_angle("vertical", raw_vertical)

            # ── Send to Arduino only if changed ───
            if base_angle != prev_base or vertical_angle != prev_vertical:
                command = f"{base_angle},{vertical_angle}\n"
                arduino.write(command.encode())
                prev_base     = base_angle
                prev_vertical = vertical_angle

                # Read Arduino response
                while arduino.in_waiting > 0:
                    response = arduino.readline().decode().strip()
                    print(f"  Arduino → {response}")

            # ── Draw nose dot ─────────────────────
            cv2.circle(frame, (nose_x, nose_y), 6, (0, 255, 100), -1)
            cv2.circle(frame, (nose_x, nose_y), 8, (255, 255, 255),  1)

            # ── Draw line from center to nose ─────
            cv2.line(frame,
                     (w // 2, h // 2),
                     (nose_x, nose_y),
                     (0, 200, 255), 1)

            # ── Display info on frame ─────────────
            cv2.putText(frame,
                        f"Nose  : ({nose_x}, {nose_y})  z={nose_z}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 100), 2)

            cv2.putText(frame,
                        f"Base  : {base_angle} deg",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2)

            cv2.putText(frame,
                        f"Vertical: {vertical_angle} deg",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2)

            # ── Console output ────────────────────
            print(f"  Nose → x:{nose_x} y:{nose_y} | "
                  f"Base:{base_angle}° Vertical:{vertical_angle}°")

    else:
        cv2.putText(frame,
                    "No Face Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("Face Tracking Servo Control (Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────
cap.release()
cv2.destroyAllWindows()
arduino.close()
print("Closed.")
# ```

# ---

# **How it works:**
# ```
# Camera Feed
#     │
#     ▼
# MediaPipe FaceMesh
#     │
#     ▼
# Nose Tip (x, y) in pixels
#     │
#     ▼
# map_value()
#   nose_x (0 → width)   →  Base Angle     (0°  → 180°)
#   nose_y (0 → height)  →  Vertical Angle (180° →   0°)  ← inverted
#     │
#     ▼
# smooth_angle()  ← removes jitter
#     │
#     ▼
# Serial → Arduino → Servo moves