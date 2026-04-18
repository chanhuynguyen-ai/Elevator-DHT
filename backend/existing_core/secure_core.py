from __future__ import annotations

from datetime import datetime
from typing import Any

from config import settings


def get_mongo_uri() -> str:
    return settings.mongo_uri


def get_db_name() -> str:
    return settings.database_name


def get_personnels_collection_name() -> str:
    return settings.personnels_collection


def get_events_collection_name() -> str:
    return settings.events_collection


def _normalize_datetime_string(value: Any) -> tuple[str, str, str, str]:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            try:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = datetime.now()

    ts = dt.isoformat()
    return ts, dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), dt.strftime("%A").upper()


def build_person_doc(
    mongo_id: int,
    person_id: int,
    ho_ten: str,
    ma_nv: str,
    bo_phan: str,
    ngay_sinh: str,
    emb_file: str,
) -> dict:
    return {
        "_id": int(mongo_id),
        "person_id": int(person_id),
        "ho_ten": ho_ten or "",
        "ma_nv": ma_nv or "",
        "bo_phan": bo_phan or "",
        "ngay_sinh": ngay_sinh or "",
        "emb_file": emb_file or "",
    }


def build_event_doc(
    mongo_id: int,
    event_type: str,
    timestamp: Any,
    date: str,
    time: str,
    weekday: str,
    cam_id: str,
    person_id: Any,
    person_name: str,
    extra: Any,
) -> dict:
    ts, d, t, wd = _normalize_datetime_string(timestamp)
    return {
        "_id": int(mongo_id),
        "event_type": (event_type or "").upper(),
        "timestamp": ts,
        "date": date or d,
        "time": time or t,
        "weekday": weekday or wd,
        "cam_id": str(cam_id or ""),
        "person_id": None if person_id in (None, "", "null") else int(person_id),
        "person_name": person_name or "Unknown",
        "extra": extra if isinstance(extra, dict) else extra,
    }
