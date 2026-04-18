# config.py
import cv2

# ===== MODELS =====
MODEL_DET_PATH = "yolov8n.pt"
MODEL_POSE_PATH = "yolov8n-pose.pt"

CAM_INDEX = 0
CAMERA_BACKEND = "AUTO"

# ===== YOLO DETECT (person + bottle) =====
IMGSZ = 384
PERSON_ID = 0
BOTTLE_ID = 39

CONF_PERSON = 0.50
CONF_BOTTLE = 0.35

YOLO_EVERY_N = 3
MIN_AREA = 12000
YOLO_DEVICE = "cpu"

# ===== YOLO POSE =====
POSE_EVERY_N = 4
POSE_DEVICE = "cpu"
POSE_CONF = 0.15
POSE_IMGSZ = 512

# ===== Fall =====
FALL_CONFIRM_SEC = 1.20
FALL_COOLDOWN_SEC = 8.0
FALL_TRANSITION_SEC = 2.5
FALL_MIN_BOX_AR = 1.45
FALL_MIN_LOW_CENTER_RATIO = 0.52
FALL_MAX_TORSO_SLOPE = 0.28

# ===== FACE =====
NGUONG_SIM = 0.45
NHAN_DIEN_MOI = 3.0
FACE_CTX_ID = -1
FACE_DET_SIZE = (128, 128)

# ===== Tracking =====
MISS_MAX = 25

# ===== CAMERA =====
MIRROR = True
ROTATE_MODE = None

# ===== CSV + embedding =====
CSV_PATH = "nhan_su.csv"
EMB_DIR = "embeddings"
SNAP_DIR = "snapshots"
FIELDNAMES = ["person_id", "ho_ten", "ma_nv", "bo_phan", "ngay_sinh", "emb_file"]

# ===== Bottle holding =====
HOLD_DIST_RATIO = 0.40
HOLD_COOLDOWN_SEC = 2.0

# ===== Crowd warning =====
CROWD_WARN_N = 4
