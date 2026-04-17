from flask import Blueprint, jsonify, request
from existing_core import csv_db
from services import camera_service
import traceback

personnel_bp = Blueprint("personnel_bp", __name__)

@personnel_bp.route("/api/personnel/list", methods=["GET"])
def get_personnel_list():
    try:
        ds = csv_db.tai_tat_ca_csv()
        # Remove 'embed' from response since it's un-serializable and large
        persons = []
        for p in ds:
            p_dict = {
                "person_id": p.get("person_id"),
                "ho_ten": p.get("ho_ten"),
                "ma_nv": p.get("ma_nv"),
                "bo_phan": p.get("bo_phan"),
                "ngay_sinh": p.get("ngay_sinh"),
                "emb_file": p.get("emb_file")
            }
            persons.append(p_dict)
        return jsonify({"success": True, "persons": persons})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@personnel_bp.route("/api/personnel/register/start", methods=["POST"])
def register_start():
    # Tell camera service to switch to capture mode
    return jsonify(camera_service.enqueue_command("register_start"))

@personnel_bp.route("/api/personnel/register/finish", methods=["POST"])
def register_finish():
    try:
        data = request.form
        if request.is_json:
            data = request.json

        ho_ten = data.get("ho_ten", "").strip()
        ma_nv = data.get("ma_nv", "").strip()
        bo_phan = data.get("bo_phan", "").strip()
        ngay_sinh = data.get("ngay_sinh", "").strip()

        if not ho_ten:
            return jsonify({"success": False, "error": "Họ tên không được để trống"}), 400

        # Check if we have the embedding sitting in the camera state
        emb = camera_service.state.temp_embedding

        if emb is None:
            return jsonify({"success": False, "error": "Chưa có embedding từ camera stream"}), 400

        # Save to DB
        new_id = csv_db.them_nhan_su_csv(
            person_id=None,
            ho_ten=ho_ten,
            ma_nv=ma_nv,
            bo_phan=bo_phan,
            ngay_sinh=ngay_sinh,
            embed=emb
        )

        if new_id is None:
            return jsonify({"success": False, "error": "Lỗi lưu file hoặc cập nhật Mongo"}), 500

        # Clear the temp_embedding and go back to normal mode
        camera_service.state.temp_embedding = None
        if camera_service.state.mode in ("register_form", "register_form_ready"):
            camera_service.state.mode = "running"
        camera_service.emit_status()

        # Reload so vision core has the new person
        camera_service.enqueue_command("reload")

        return jsonify({"success": True, "message": f"Đăng ký thành công nhân sự ID {new_id}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@personnel_bp.route("/api/personnel/edit", methods=["PUT"])
def edit_personnel():
    try:
        data = request.json or {}
        person_id = data.get("person_id")
        if not person_id:
            return jsonify({"success": False, "error": "Missing person_id"}), 400

        ok = csv_db.sua_thong_tin_csv_data(int(person_id), data)
        if ok:
            camera_service.enqueue_command("reload")  # reload model so new data used
            return jsonify({"success": True, "message": "Cập nhật thành công"})
        else:
            return jsonify({"success": False, "error": "Lỗi cập nhật hoặc person_id không tồn tại"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@personnel_bp.route("/api/personnel/delete", methods=["DELETE"])
def delete_personnel():
    try:
        data = request.json or {}
        person_id = data.get("person_id")
        if not person_id:
            return jsonify({"success": False, "error": "Missing person_id"}), 400

        ok = csv_db.xoa_person_va_reindex(int(person_id))
        if ok:
            camera_service.enqueue_command("reload")
            return jsonify({"success": True, "message": "Xóa thành công"})
        else:
            return jsonify({"success": False, "error": "Lỗi xóa hoặc person_id không tồn tại"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
