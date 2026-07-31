import json
import os
import urllib.request
from collections import deque
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template
from flask_sock import Sock
from simple_websocket import ConnectionClosed


# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Model
MODEL_URL = "https://huggingface.co/nandaputric/drowsiness-model/resolve/main/model.onnx"
MODEL_PATH = "/tmp/model.onnx"
MODEL_MIN_BYTES = 1_000_000  # ambang batas "file terlalu kecil buat jadi model asli"

# Inference / deteksi wajah
INPUT_SIZE = (224, 224)
DROWSY_THRESHOLD = 0.38
DROWSY_FRAMES = 6
CONF_MIN = 0.85            # dinaikin dari 0.75, masih ada false-positive di background
MIN_FACE_SIZE = 80
DETECT_MAX_WIDTH = 480      # deteksi wajah di frame yang di-downscale ke lebar ini biar lebih cepat
                            # (bbox MediaPipe relatif 0-1, jadi tetap valid buat crop di frame asli)

SMOOTH_WINDOW = 2           # rata-rata probabilitas 2 frame terakhir buat anti-flicker
FACE_MIN_STREAK = 2         # wajah harus valid N frame berturut-turut dulu sebelum dipercaya
                            # (anti false-positive dari deteksi yang cuma "kedip" 1 frame doang)
                            # NB: harus >= 2, soalnya face_streak di-increment DULU baru dicek
                            # (< FACE_MIN_STREAK), jadi kalau nilainya 1 kondisinya gak akan
                            # pernah kepenuhin dan blok logic-nya jadi dead code.



# MODEL DOWNLOAD & CACHE
def _download_model_to(path: str) -> None:
    print(f"[INFO] Downloading model dari: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, path)


def _is_valid_model_file(path: str) -> bool:
    """Cek apakah file di `path` adalah binary ONNX yang valid (bukan
    HTML error page atau Git-LFS pointer, dan bukan file kosong/kekecilan)."""
    if not os.path.exists(path):
        return False

    size = os.path.getsize(path)
    if size < MODEL_MIN_BYTES:
        print(f"[WARN] Model file cuma {size} bytes (< {MODEL_MIN_BYTES}), kemungkinan corrupt.")
        return False

    # ONNX itu protobuf biner, bukan text -> cek signature HTML/LFS pointer
    try:
        with open(path, "rb") as f:
            head = f.read(200)
    except OSError:
        return False

    if head.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")) or \
            head.startswith(b"version https://git-lfs"):
        print("[WARN] Model file isinya HTML/LFS-pointer, bukan binary ONNX.")
        return False

    return True


def ensure_model_downloaded() -> None:
    """Pastikan model ada & valid di MODEL_PATH; download ulang kalau perlu."""
    if os.path.exists(MODEL_PATH):
        if _is_valid_model_file(MODEL_PATH):
            print(f"[INFO] Model udah ada di cache: {MODEL_PATH}")
            return
        print(f"[WARN] Cache di {MODEL_PATH} corrupt, hapus & download ulang.")
        try:
            os.remove(MODEL_PATH)
        except OSError:
            pass

    tmp_path = MODEL_PATH + ".part"
    try:
        _download_model_to(tmp_path)
        if not _is_valid_model_file(tmp_path):
            raise RuntimeError(
                "Hasil download gagal validasi (file terlalu kecil atau "
                "isinya bukan binary ONNX -- kemungkinan HF error page, "
                "rate limit, atau koneksi keputus di tengah jalan)."
            )
        os.replace(tmp_path, MODEL_PATH)  # atomic rename
        print(f"[INFO] Model berhasil didownload ke: {MODEL_PATH} "
              f"({os.path.getsize(MODEL_PATH) / (1024 * 1024):.1f} MB)")
    except Exception as e:
        print(f"[ERROR] Gagal download model: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


# ONNX RUNTIME SESSION
def _build_ort_session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def load_session_with_retry() -> ort.InferenceSession:
    """Load ONNX session. Kalau gagal (misal model cache korup), hapus
    cache dan coba download + load ulang sekali."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = _build_ort_session_options()
    try:
        return ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=providers)
    except Exception as e:
        print(f"[WARN] Load model gagal ({e}), hapus cache & retry download sekali.")
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except OSError:
                pass
        ensure_model_downloaded()
        return ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=providers)


def run_inference(face_input: np.ndarray) -> float:
    out = session.run([_output_name], {_input_name: face_input})[0]
    return float(out[0][0])


# --- Inisialisasi model (dijalankan sekali saat modul di-import) ---
ensure_model_downloaded()

print(f"[INFO] Loading model dari: {MODEL_PATH}")
# Providers list -> otomatis pakai CUDA kalau ada, fallback ke CPU (CUDA
# emang gak available di Vercel, warning fallback-nya normal).
session = load_session_with_retry()
_input_name = session.get_inputs()[0].name
_output_name = session.get_outputs()[0].name
print(f"[INFO] Model berhasil dimuat. Provider aktif: {session.get_providers()[0]}")
print(f"[INFO] Input: {_input_name} {session.get_inputs()[0].shape}  "
      f"Output: {_output_name} {session.get_outputs()[0].shape}")

# Warm-up model saat server start biar delay inisialisasi ONNX ga kerasa
# pas user nyalain kamera pertama kali.
_dummy_input = np.zeros((1, *INPUT_SIZE, 3), dtype=np.float32)
run_inference(_dummy_input)


# FACE DETECTION (MediaPipe)
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=1,  # full-range, lebih akurat drpd short-range (0)
    min_detection_confidence=CONF_MIN,
)


def crop_face_mediapipe(frame: np.ndarray, detection, padding: float = 0.2):
    """Crop wajah dari `frame` berdasarkan bbox relatif MediaPipe, dengan
    padding tambahan di tiap sisi. Return (face_crop, (x1, y1, x2, y2))
    atau (None, None) kalau bbox-nya invalid."""
    h, w = frame.shape[:2]
    bbox = detection.location_data.relative_bounding_box

    x1 = int((bbox.xmin - padding * bbox.width) * w)
    y1 = int((bbox.ymin - padding * bbox.height) * h)
    x2 = int((bbox.xmin + bbox.width + padding * bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height + padding * bbox.height) * h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    face_crop = frame[y1:y2, x1:x2]
    return face_crop, (x1, y1, x2, y2)


def preprocess_face(face_rgb: np.ndarray, target_size=INPUT_SIZE) -> np.ndarray:
    """Resize + normalisasi crop wajah jadi input model.
    face_rgb udah RGB (crop diambil dari frame_rgb di process_frame),
    jadi gak perlu cvtColor lagi di sini."""
    face_resized = cv2.resize(face_rgb, target_size)
    face_norm = face_resized.astype(np.float32) / 255.0
    return np.expand_dims(face_norm, axis=0)


def decode_image_bytes(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decode frame yang dikirim client sebagai binary WebSocket message
    (JPEG Blob), bukan lagi lewat multipart/form-data."""
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


# SESSION STATE & FRAME PROCESSING
class SessionState:
    """State per-koneksi WebSocket: hitungan drowsy dan window untuk
    smoothing probabilitas."""

    def __init__(self):
        self.drowsy_count = 0
        self.prob_window = deque(maxlen=SMOOTH_WINDOW)
        self.face_streak = 0

    def reset(self):
        self.drowsy_count = 0
        self.prob_window.clear()
        self.face_streak = 0


def _no_face_response(state: SessionState, response: dict) -> dict:
    """Update state & response untuk kasus "wajah tidak valid/tidak ada"
    (dipakai di beberapa early-return supaya gak duplikat logic)."""
    state.drowsy_count = max(state.drowsy_count - 2, 0)
    response["drowsy_count"] = state.drowsy_count
    return response


def process_frame(frame: np.ndarray, state: SessionState) -> dict:
    """Jalankan 1 frame lewat pipeline: deteksi wajah -> crop -> model ->
    update state drowsiness. Return dict siap di-JSON-kan ke client."""
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Downscale dulu buat deteksi biar lebih cepat; bbox tetap relatif (0-1)
    # jadi crop di frame asli (resolusi penuh) tetap akurat.
    if w > DETECT_MAX_WIDTH:
        scale = DETECT_MAX_WIDTH / w
        detect_input = cv2.resize(
            frame_rgb, (DETECT_MAX_WIDTH, int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detect_input = frame_rgb

    results = detector.process(detect_input)

    response = {
        "face_detected": False,
        "face_too_small": False,
        "label": None,
        "prob": None,
        "bbox": None,  # [x1, y1, x2, y2] dalam pixel, relatif ke frame yg dikirim
        "drowsy_count": state.drowsy_count,
        "drowsy_frames": DROWSY_FRAMES,
        "alarm": False,
    }

    if not results.detections:
        state.prob_window.clear()
        state.face_streak = 0
        return _no_face_response(state, response)

    detection = results.detections[0]
    raw_bbox = detection.location_data.relative_bounding_box
    face_px_w = int(raw_bbox.width * w)
    face_px_h = int(raw_bbox.height * h)

    response["face_detected"] = True

    if face_px_w < MIN_FACE_SIZE or face_px_h < MIN_FACE_SIZE:
        # wajah kejauhan/kekecilan -> sinyal ga reliable, perlakukan sama kayak "ga ada wajah"
        response["face_too_small"] = True
        state.prob_window.clear()
        state.face_streak = 0
        _no_face_response(state, response)
        x1 = int(raw_bbox.xmin * w)
        y1 = int(raw_bbox.ymin * h)
        x2 = int((raw_bbox.xmin + raw_bbox.width) * w)
        y2 = int((raw_bbox.ymin + raw_bbox.height) * h)
        response["bbox"] = [x1, y1, x2, y2]
        return response

    face_crop, bbox = crop_face_mediapipe(frame_rgb, detection)
    if face_crop is None or face_crop.size == 0:
        state.face_streak = 0
        return _no_face_response(state, response)

    state.face_streak += 1
    if state.face_streak < FACE_MIN_STREAK:
        # wajah baru kedeteksi, belum cukup streak -> jangan buru2 percaya,
        # tunggu frame berikutnya dulu
        response["bbox"] = list(bbox)
        return _no_face_response(state, response)

    face_input = preprocess_face(face_crop)
    raw_prob = run_inference(face_input)

    # raw_prob mendekati 0 = drowsy, mendekati 1 = nondrowsy
    prob_drowsy = 1.0 - raw_prob
    state.prob_window.append(prob_drowsy)
    smoothed_prob_drowsy = float(np.mean(state.prob_window))

    if smoothed_prob_drowsy > (1.0 - DROWSY_THRESHOLD):
        label = "DROWSY"
        prob = smoothed_prob_drowsy
        state.drowsy_count = min(state.drowsy_count + 1, DROWSY_FRAMES)
    else:
        label = "NON-DROWSY"
        prob = 1.0 - smoothed_prob_drowsy
        state.drowsy_count = max(state.drowsy_count - 2, 0)

    response.update({
        "label": label,
        "prob": round(prob, 4),
        "bbox": list(bbox),
        "drowsy_count": state.drowsy_count,
        "alarm": state.drowsy_count >= DROWSY_FRAMES,
    })
    return response


# FLASK APP & ROUTES
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
sock = Sock(app)


@app.route("/")
def index():
    return render_template("index.html")


@sock.route("/ws")
def ws_predict(ws):
    state = SessionState()
    try:
        while True:
            data = ws.receive()
            if data is None:
                break  # koneksi ditutup client

            if isinstance(data, (bytes, bytearray)):
                frame = decode_image_bytes(data)
                if frame is None:
                    ws.send(json.dumps({"error": "gagal decode gambar"}))
                    continue
                response = process_frame(frame, state)
                ws.send(json.dumps(response))

            elif isinstance(data, str):
                if data == "reset":
                    state.reset()
                    ws.send(json.dumps({"status": "reset", "drowsy_count": 0}))
                else:
                    ws.send(json.dumps({"error": f"pesan teks tidak dikenal: {data!r}"}))
    except ConnectionClosed:
        pass