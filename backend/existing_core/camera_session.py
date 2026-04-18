import time
from datetime import datetime
import numpy as np
import cv2

from . import config
from . import utils_cv
from . import face_recog
from . import events
from . import pose_fall
from .sort_tracker import Sort

DISPLAY_SCALE = 1.0
MAX_POSE_PERSONS = 4
DEFAULT_BOX_COLOR = (0, 255, 0)    # green
ALARM_BOX_COLOR = (0, 0, 255)      # red


def _open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    for backend in backends:
        try:
            cap = cv2.VideoCapture(config.CAM_INDEX, backend) if backend != 0 else cv2.VideoCapture(config.CAM_INDEX)
            if cap is not None and cap.isOpened():
                return cap
        except Exception:
            pass
    return None


def _find_nearest_bottle_for_person(px1, py1, px2, py2, bottles):
    if not bottles:
        return None

    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    pw = (px2 - px1) + 1e-6

    best = None
    best_dist = 1e18

    for bx1, by1, bx2, by2, score in bottles:
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2
        dist = ((bcx - pcx) ** 2 + (bcy - pcy) ** 2) ** 0.5

        if dist < best_dist:
            best_dist = dist
            best = (bx1, by1, bx2, by2, score, dist)

    if best is None:
        return None

    if best[5] <= config.HOLD_DIST_RATIO * pw:
        return best

    return None


def _run_pose_on_roi(pose_model, frame, x1, y1, x2, y2):
    roi = utils_cv.cat_roi_an_toan(frame, x1, y1, x2, y2, pad=10)
    if roi is None or roi.size == 0:
        return None

    res = pose_model.predict(
        roi,
        imgsz=config.POSE_IMGSZ,
        conf=config.POSE_CONF,
        device=config.POSE_DEVICE,
        verbose=False
    )[0]

    if res.keypoints is None or len(res.keypoints) == 0:
        return None

    kps = res.keypoints.data[0].cpu().numpy()
    if kps is None or getattr(kps, "ndim", 0) != 2 or kps.shape[0] < 17 or kps.shape[1] < 3:
        return None

    kps[:, 0] += (x1 - 10)
    kps[:, 1] += (y1 - 10)
    return kps


def _apply_external_command(
    item,
    paused,
    show_help,
    yolo_every_n,
    nguong_sim,
    mirror,
    rotate_mode,
    frame=None
):
    if not item:
        return None, paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode

    cmd = (item.get("command") or "").lower()

    if cmd == "pause":
        paused = True
    elif cmd == "resume":
        paused = False
    elif cmd == "help":
        show_help = not show_help
    elif cmd == "mirror":
        mirror = not mirror
    elif cmd == "rotate":
        if rotate_mode is None:
            rotate_mode = cv2.ROTATE_90_CLOCKWISE
        elif rotate_mode == cv2.ROTATE_90_CLOCKWISE:
            rotate_mode = cv2.ROTATE_180
        elif rotate_mode == cv2.ROTATE_180:
            rotate_mode = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            rotate_mode = None
    elif cmd == "sim_inc":
        nguong_sim = min(0.95, nguong_sim + 0.02)
    elif cmd == "sim_dec":
        nguong_sim = max(0.10, nguong_sim - 0.02)
    elif cmd == "set_yolo":
        payload = item.get("payload") or {}
        yolo_every_n = max(1, min(3, int(payload.get("yolo_every_n", yolo_every_n))))
    elif cmd == "snapshot" and frame is not None:
        try:
            path = utils_cv.save_snapshot(frame, prefix="manual")
            print("[SNAP] Saved:", path)
        except Exception as ex:
            print("[SNAP] Error:", ex)
    elif cmd in ("register", "register_start"):
        return "REGISTER", paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode
    elif cmd == "edit":
        return "EDIT", paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode
    elif cmd == "delete":
        return "DELETE", paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode
    elif cmd == "reload":
        return "RELOAD", paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode
    elif cmd == "stop":
        return "EXIT", paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode

    return None, paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode


def run_camera_session(
    det_model,
    pose_model,
    face_app,
    ds_nhan_su,
    yolo_every_n,
    nguong_sim,
    nhan_dien_moi,
    mirror,
    rotate_mode,
    logger=None,
    web_mode=False,
    command_fetcher=None,
    state_getter=None,
    frame_callback=None,
):
    cap = _open_camera()
    if cap is None:
        print(f"Không mở được webcam. CAM_INDEX={config.CAM_INDEX}")
        return ("EXIT", (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[BBOX_FEATURES_RESTORED] active")

    tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.35)
    fstate = pose_fall.FallState()

    last_recog = {}
    tid_to_person = {}
    tid_to_showid = {}
    free_showids = []
    next_showid = 1
    miss_count = {}
    tid_posture = {}
    tid_is_fall = {}

    frame_id = 0
    dets_cache = np.empty((0, 5), dtype=np.float32)
    bottles_cache = []

    paused = False
    show_help = False

    t0 = time.time()
    frames = 0
    fps = 0.0
    last_frame = None
    last_alarm_time = {}
    alarm_gap = 8.0
    last_event_type = None

    while True:
        if state_getter is not None:
            try:
                s = state_getter() or {}
                mirror = bool(s.get("mirror", mirror))
                yolo_every_n = int(s.get("yolo_every_n", yolo_every_n))
                nguong_sim = float(s.get("sim_threshold", nguong_sim))
                paused = bool(s.get("paused", paused))
            except Exception:
                pass

        if command_fetcher is not None:
            while True:
                item = command_fetcher()
                if not item:
                    break

                action, paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode = _apply_external_command(
                    item, paused, show_help, yolo_every_n, nguong_sim, mirror, rotate_mode, frame=last_frame
                )
                if action is not None:
                    cap.release()
                    if not web_mode:
                        cv2.destroyAllWindows()
                    return (action, (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))

        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            if rotate_mode is not None:
                frame = cv2.rotate(frame, rotate_mode)
            if mirror:
                frame = cv2.flip(frame, 1)

            last_frame = frame.copy()
            H, W = frame.shape[:2]
            frame_id += 1
            now = time.time()
            last_event_type = None

            if frame_id % max(1, yolo_every_n) == 0:
                res = det_model.predict(
                    frame,
                    imgsz=min(int(getattr(config, "IMGSZ", 384)), 384),
                    conf=min(float(getattr(config, "CONF_PERSON", 0.5)), float(getattr(config, "CONF_BOTTLE", 0.35))),
                    device=getattr(config, "YOLO_DEVICE", "cpu"),
                    verbose=False
                )[0]

                dets_person = []
                dets_bottle = []

                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    score = float(box.conf[0])

                    if cls_id == getattr(config, "PERSON_ID", 0):
                        area = (x2 - x1) * (y2 - y1)
                        if area >= getattr(config, "MIN_AREA", 12000) and score >= getattr(config, "CONF_PERSON", 0.5):
                            dets_person.append([x1, y1, x2, y2, score])

                    elif cls_id == getattr(config, "BOTTLE_ID", 39):
                        if score >= getattr(config, "CONF_BOTTLE", 0.35):
                            dets_bottle.append([x1, y1, x2, y2, score])

                dets_cache = np.array(dets_person, dtype=np.float32) if dets_person else np.empty((0, 5), dtype=np.float32)
                bottles_cache = dets_bottle

            tracks = tracker.update(dets_cache)
            cur_tids = set(int(t[4]) for t in tracks) if len(tracks) else set()

            for tid in list(tid_to_showid.keys()):
                if tid not in cur_tids:
                    miss_count[tid] = miss_count.get(tid, 0) + 1
                    if miss_count[tid] >= getattr(config, "MISS_MAX", 25):
                        sid = tid_to_showid.pop(tid, None)
                        if sid is not None:
                            free_showids.append(sid)
                            free_showids.sort()
                        miss_count.pop(tid, None)
                        last_recog.pop(tid, None)
                        tid_to_person.pop(tid, None)
                        tid_posture.pop(tid, None)
                        tid_is_fall.pop(tid, None)
                        fstate.tid_last_pose.pop(tid, None)
                        fstate.tid_suspect_start.pop(tid, None)
                        fstate.tid_last_posture.pop(tid, None)
                        fstate.tid_last_upright_time.pop(tid, None)
                else:
                    miss_count[tid] = 0

            for tid in cur_tids:
                if tid not in tid_to_showid:
                    if free_showids:
                        tid_to_showid[tid] = free_showids.pop(0)
                    else:
                        tid_to_showid[tid] = next_showid
                        next_showid += 1

            tid_holding = events.detect_bottle_holding(tracks, bottles_cache)

            for x1, y1, x2, y2, tid in tracks:
                x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)

                if (tid not in last_recog) or (now - last_recog[tid] > nhan_dien_moi):
                    person = None
                    roi = utils_cv.cat_roi_an_toan(frame, x1, y1, x2, y2, pad=10)

                    if roi is not None and roi.size != 0 and face_app is not None:
                        try:
                            faces = face_app.get(roi)
                            f = utils_cv.pick_face_largest(faces)
                            if f is not None and getattr(f, "normed_embedding", None) is not None:
                                emb = f.normed_embedding.astype(np.float32)
                                person, _sim = face_recog.so_khop(emb, ds_nhan_su, nguong_sim)
                        except Exception:
                            person = None

                    tid_to_person[tid] = person
                    last_recog[tid] = now

            if pose_model is not None and len(tracks) and frame_id % max(1, getattr(config, "POSE_EVERY_N", 4)) == 0:
                tracks_sorted = sorted(tracks, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]), reverse=True)

                for idx, (x1, y1, x2, y2, tid) in enumerate(tracks_sorted):
                    x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)

                    if idx < MAX_POSE_PERSONS:
                        kps = _run_pose_on_roi(pose_model, frame, x1, y1, x2, y2)
                        if kps is not None:
                            fstate.tid_last_pose[tid] = kps
                    else:
                        kps = fstate.tid_last_pose.get(tid)

                    tid_posture[tid] = pose_fall.classify_posture(
                        kps,
                        person_bbox=(x1, y1, x2, y2),
                        frame_h=H
                    )
                    tid_is_fall[tid] = pose_fall.update_fall_by_pose(
                        fstate,
                        tid,
                        now,
                        kps,
                        posture=tid_posture[tid],
                        person_bbox=(x1, y1, x2, y2),
                        frame_h=H,
                    )

            people_n = len(tracks)
            frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frames / elapsed
                t0 = time.time()
                frames = 0

            realtime_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            for x1, y1, x2, y2, tid in tracks:
                x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
                cam_id = tid_to_showid.get(tid, tid)

                person = tid_to_person.get(tid)
                name = "Unknown" if person is None else person.get("ho_ten", "Unknown")

                person_id_show = "--"
                person_id_val = None
                if person is not None and person.get("person_id") is not None:
                    try:
                        person_id_val = int(person["person_id"])
                        person_id_show = f"{person_id_val:02d}"
                    except Exception:
                        person_id_val = None

                posture = tid_posture.get(tid, "UNKNOWN")
                is_fall = bool(tid_is_fall.get(tid, False))
                is_lying = posture == "NAM"
                holding = bool(tid_holding.get(tid, False))
                posture_show = "TE NGA" if is_fall else posture

                is_alarm = is_fall or is_lying or holding
                box_color = ALARM_BOX_COLOR if is_alarm else DEFAULT_BOX_COLOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                line1 = f"ID:{person_id_show} | {name}"
                utils_cv.put_text_bg(
                    frame,
                    line1,
                    (x1, max(22, y1 - 10)),
                    font_scale=0.55,
                    color=(255, 255, 255),
                    bg=(0, 0, 0),
                    alpha=0.55
                )

                line2_parts = []
                if posture_show and posture_show != "UNKNOWN":
                    line2_parts.append(f"POSE:{posture_show}")
                if holding:
                    line2_parts.append("Bottle:YES")

                if line2_parts:
                    utils_cv.put_text_bg(
                        frame,
                        " | ".join(line2_parts),
                        (x1, min(H - 8, y2 + 22)),
                        font_scale=0.55,
                        color=(255, 255, 255),
                        bg=(0, 90, 0) if not is_alarm else (0, 0, 255),
                        alpha=0.55
                    )

                if holding:
                    bottle_box = _find_nearest_bottle_for_person(x1, y1, x2, y2, bottles_cache)
                    if bottle_box is not None:
                        bx1, by1, bx2, by2, bscore, _dist = bottle_box
                        bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)

                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                        utils_cv.put_text_bg(
                            frame,
                            f"Chai nuoc {bscore:.2f}",
                            (bx1, max(22, by1 - 8)),
                            font_scale=0.50,
                            color=(255, 255, 255),
                            bg=(255, 0, 0),
                            alpha=0.55
                        )

                alarm_type = None
                if is_fall:
                    alarm_type = "FALL"
                elif is_lying:
                    alarm_type = "LYING"
                elif holding:
                    alarm_type = "BOTTLE"

                if alarm_type is not None:
                    last_t = last_alarm_time.get((tid, alarm_type), 0.0)
                    if now - last_t >= alarm_gap:
                        if logger is not None:
                            try:
                                logger.log_event(
                                    event_type=alarm_type,
                                    cam_id=cam_id,
                                    person_id=person_id_val,
                                    person_name=name,
                                    extra={
                                        "people_count": people_n,
                                        "posture": posture_show,
                                        "holding": holding,
                                        "realtime_str": realtime_str,
                                    },
                                )
                            except Exception as ex:
                                print("[LOGGER] log_event error:", repr(ex))

                        print(f"[EVENT] {alarm_type} | cam_id={cam_id} | time={realtime_str}")
                        last_alarm_time[(tid, alarm_type)] = now
                        last_event_type = alarm_type

            if frame_callback is not None:
                try:
                    frame_callback(
                        frame,
                        {
                            "fps": round(float(fps), 1),
                            "people_count": int(people_n),
                            "last_event": last_event_type,
                            "preview_ready": True,
                            "note": f"tracked={people_n}, bottles={len(bottles_cache)}",
                        },
                    )
                except Exception as ex:
                    print(f"[PREVIEW] frame_callback error: {repr(ex)}")

            if not web_mode:
                frame_show = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("NHAN DIEN", frame_show)

        if web_mode:
            time.sleep(0.001)
            continue

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('p'), ord('P')):
            paused = not paused
        if key in (ord('h'), ord('H')):
            show_help = not show_help
        if key in (ord('r'), ord('R')):
            cap.release()
            cv2.destroyAllWindows()
            return ("REGISTER", (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))
        if key == ord('+') or key == ord('='):
            nguong_sim = min(0.95, nguong_sim + 0.02)
        if key == ord('-') or key == ord('_'):
            nguong_sim = max(0.10, nguong_sim - 0.02)
        if key == ord('1'):
            yolo_every_n = 1
        if key == ord('2'):
            yolo_every_n = 2
        if key == ord('3'):
            yolo_every_n = 3
        if key in (ord('m'), ord('M')):
            mirror = not mirror
        if key in (ord('t'), ord('T')):
            if rotate_mode is None:
                rotate_mode = cv2.ROTATE_90_CLOCKWISE
            elif rotate_mode == cv2.ROTATE_90_CLOCKWISE:
                rotate_mode = cv2.ROTATE_180
            elif rotate_mode == cv2.ROTATE_180:
                rotate_mode = cv2.ROTATE_90_COUNTERCLOCKWISE
            else:
                rotate_mode = None
        if key in (ord('s'), ord('S')):
            try:
                path = utils_cv.save_snapshot(frame, prefix="manual")
                print("[SNAP] Saved:", path)
            except Exception as ex:
                print("[SNAP] Error:", ex)
        if key in (ord('e'), ord('E')):
            cap.release()
            cv2.destroyAllWindows()
            return ("EDIT", (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))
        if key in (ord('x'), ord('X')):
            cap.release()
            cv2.destroyAllWindows()
            return ("DELETE", (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))
        if key in (ord('l'), ord('L')):
            cap.release()
            cv2.destroyAllWindows()
            return ("RELOAD", (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return ("EXIT", (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))

    cap.release()
    if not web_mode:
        cv2.destroyAllWindows()
    return ("EXIT", (yolo_every_n, nguong_sim, nhan_dien_moi, mirror, rotate_mode))