import cv2
import os
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import time
import sys

# ✅ Windows: bring OpenCV window to foreground so ESC key is captured
if sys.platform == "win32":
    import ctypes
    def _force_window_focus(win_name):
        hwnd = ctypes.windll.user32.FindWindowW(None, win_name)
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
else:
    def _force_window_focus(win_name):
        pass  # Not needed on Linux/macOS

# ========== CONFIG ==========
SERVICE_ACCOUNT_FILE = "credentials.json"
SHEET_ID = "12UIwAY-gibNz__4TJqpfZB92MvRy1vBNHKcQoJ-lZLw"
WORKSHEET_NAME = "Attendance"
DATASET_PATH = "dataset"
MODEL_PATH = "face_model.yml"
LABEL_MAP_PATH = "label_map.txt"

CONFIDENCE_THRESHOLD = 80
LOG_COOLDOWN_SECONDS = 10
CAPTURE_COUNT = 60

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ========== GOOGLE SHEETS ==========
def connect_sheet():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    try:
        worksheet = sheet.worksheet(WORKSHEET_NAME)
        print(f"✅ Found worksheet: '{WORKSHEET_NAME}'")
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=WORKSHEET_NAME, rows=5000, cols=5)
        print(f"✅ Created new worksheet: '{WORKSHEET_NAME}'")

    existing_data = worksheet.get_all_values()
    expected_headers = ["Name", "Status", "Confidence", "Date", "Time"]
    if len(existing_data) == 0 or existing_data[0] != expected_headers:
        worksheet.clear()
        worksheet.append_row(expected_headers)
    return worksheet


def log_to_sheet(worksheet, name, status, confidence):
    now = datetime.now()
    row = [name, status, f"{float(confidence):.2f}", now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]
    worksheet.append_row(row, value_input_option="USER_ENTERED")
    print(f"📋 Logged → {name} | {status} | Conf: {confidence:.2f} | {row[3]} {row[4]}")


# ========== LABEL MAP ==========
def save_label_map(label_map):
    with open(LABEL_MAP_PATH, "w") as f:
        for lid, name in label_map.items():
            f.write(f"{lid}:{name}\n")


def load_label_map():
    label_map = {}
    if not os.path.exists(LABEL_MAP_PATH):
        return label_map
    with open(LABEL_MAP_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                lid, name = line.split(":", 1)
                label_map[int(lid)] = name
    return label_map


# ========== STEP 1: ENROLL ==========
def enroll_person(name):
    person_path = os.path.join(DATASET_PATH, name)
    os.makedirs(person_path, exist_ok=True)

    existing = [f for f in os.listdir(person_path) if f.endswith(".jpg")]
    start_count = len(existing)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera. Check it is connected and not in use.")
        return

    count = 0
    print(f"\n📸 Enrolling '{name}' — Look at the camera. Capturing {CAPTURE_COUNT} images...")
    print("   Move your head slightly (left/right/tilt) for better coverage.")
    print("   Press ESC to cancel early.\n")

    cv2.namedWindow("Enrollment", cv2.WINDOW_NORMAL)
    cv2.imshow("Enrollment", np.zeros((100, 400), dtype=np.uint8))
    cv2.waitKey(1)
    _force_window_focus("Enrollment")

    while count < CAPTURE_COUNT:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            count += 1
            img_path = os.path.join(person_path, f"{start_count + count}.jpg")
            cv2.imwrite(img_path, face_resized)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {count}/{CAPTURE_COUNT}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Enrolling: {name} — ESC to cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("   ⚠️  Enrollment cancelled early.")
            break
        if key == ord('q'):
            print("   ⚠️  Enrollment cancelled (q pressed).")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Enrollment done! {count} images saved for '{name}'.")


# ========== STEP 2: TRAIN ==========
def train_model():
    print("\n🧠 Training model on dataset...")

    faces_data, labels, label_map = [], [], {}
    label_id = 0

    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset folder not found.")
        return None, {}

    persons = sorted([
        p for p in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, p))
    ])

    if not persons:
        print("❌ No persons found in dataset.")
        return None, {}

    for person in persons:
        p_path = os.path.join(DATASET_PATH, person)
        label_map[label_id] = person
        person_count = 0

        for img_name in os.listdir(p_path):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(p_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (200, 200))
            img_eq = cv2.equalizeHist(img_resized)
            faces_data.append(img_eq)
            labels.append(label_id)
            person_count += 1

        print(f"   → {person}: {person_count} images  (label {label_id})")
        label_id += 1

    if not faces_data:
        print("❌ No valid images found.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces_data, np.array(labels))
    recognizer.save(MODEL_PATH)
    save_label_map(label_map)

    print(f"✅ Training complete! {len(faces_data)} images | {len(label_map)} person(s).")
    print(f"   Persons: {list(label_map.values())}\n")
    return recognizer, label_map


# ========== STEP 3: RECOGNITION + ATTENDANCE ==========
def run_attendance(recognizer, label_map, worksheet):
    if recognizer is None:
        print("❌ No model loaded. Train first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera. Check it is connected and not in use by another app.")
        return

    print("🎯 Starting Face Recognition... Press ESC or Q to quit.\n")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}  (scores below this = recognised)\n")

    cv2.namedWindow("Face Attendance", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Attendance", np.zeros((100, 400), dtype=np.uint8))
    cv2.waitKey(1)
    _force_window_focus("Face Attendance")

    last_logged = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame grab failed — camera disconnected?")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces_detected = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces_detected:
            face_roi = cv2.resize(gray_eq[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                detected_name = label_map.get(label, "Unknown")
                status = "Present"
                color = (0, 255, 0)     # Green
            else:
                detected_name = "Unknown"
                status = "Unknown"
                color = (0, 0, 255)     # Red

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, detected_name, (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            now = time.time()
            # ✅ Only log recognised (Present) persons — Unknowns are skipped
            if worksheet and status == "Present" and (now - last_logged.get(detected_name, 0)) > LOG_COOLDOWN_SECONDS:
                try:
                    log_to_sheet(worksheet, detected_name, status, confidence)
                    last_logged[detected_name] = now
                except Exception as e:
                    print(f"⚠️  Sheet logging failed: {e}")

        enrolled = ", ".join(label_map.values()) if label_map else "None"
        cv2.putText(frame, f"Enrolled: {enrolled}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "ESC / Q = Quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Face Attendance", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("\n🛑 Quit key pressed — stopping.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Attendance session ended.")


# ========== MAIN MENU ==========
def main():
    print("=" * 50)
    print("     FACE ATTENDANCE SYSTEM — MULTI-PERSON")
    print("=" * 50)

    worksheet = None
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"⚠️  '{SERVICE_ACCOUNT_FILE}' not found — running WITHOUT sheet logging.\n")
    else:
        try:
            print("🔗 Connecting to Google Sheets...")
            worksheet = connect_sheet()
            print("✅ Google Sheets ready!\n")
        except Exception as e:
            print(f"❌ Sheet connection failed: {e}\n")

    if worksheet is None:
        print("⚠️  Continuing WITHOUT Google Sheets logging.\n")

    recognizer = None
    label_map = load_label_map()
    if os.path.exists(MODEL_PATH) and label_map:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        print(f"✅ Loaded existing model. Enrolled: {list(label_map.values())}\n")
    else:
        print("⚠️  No trained model found. Please enroll (1) then train (2) first.\n")

    while True:
        print("\n--- MENU ---")
        print("1. Enroll a new person")
        print("2. Train / retrain model")
        print("3. Start attendance (recognition)")
        print("4. List enrolled persons")
        print("5. Exit")
        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            name = input("Enter person's name: ").strip()
            if name:
                enroll_person(name)
                print("💡 Remember to retrain (option 2) before recognising.")
            else:
                print("⚠️  Name cannot be empty.")

        elif choice == "2":
            recognizer, label_map = train_model()

        elif choice == "3":
            if recognizer is None:
                print("⚠️  No trained model found. Enroll and train first.")
            else:
                run_attendance(recognizer, label_map, worksheet)

        elif choice == "4":
            if label_map:
                print(f"\n👥 Enrolled persons ({len(label_map)}):")
                for lid, name in label_map.items():
                    p = os.path.join(DATASET_PATH, name)
                    imgs = len(os.listdir(p)) if os.path.exists(p) else 0
                    print(f"   [{lid}] {name} — {imgs} training images")
            else:
                print("⚠️  No persons enrolled yet.")

        elif choice == "5":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice.")


if __name__ == "__main__":
    main()