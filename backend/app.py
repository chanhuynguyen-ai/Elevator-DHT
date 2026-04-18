from __future__ import annotations

import os
import sys
from typing import IO


def _ensure_valid_std_handles() -> None:
    """Avoid WinError 6 on Windows when a dependency uses subprocess with invalid std handles."""
    if os.name != "nt":
        return

    keepalive: list[IO] = []
    for name, mode in (("stdin", "r"), ("stdout", "a"), ("stderr", "a")):
        current = getattr(sys, name, None)
        current_dunder = getattr(sys, f"__{name}__", None)

        def _is_valid(stream) -> bool:
            try:
                return stream is not None and stream.fileno() >= 0
            except Exception:
                return False

        if not _is_valid(current):
            fallback = open(os.devnull, mode, encoding="utf-8", buffering=1)
            keepalive.append(fallback)
            setattr(sys, name, fallback)
            current = fallback

        if not _is_valid(current_dunder):
            setattr(sys, f"__{name}__", current)

    setattr(sys, "_smartelevator_std_keepalive", keepalive)


_ensure_valid_std_handles()

from flask import Flask
from flask_cors import CORS

from config import settings
from routes import register_blueprints
from services.socket_service import init_socketio
from services.log_service import install_std_redirects, setup_logging
from services import chat_service, mongo_service, camera_service

logger = setup_logging(settings.log_level)
# install_std_redirects()

app = Flask(__name__)
app.config["SECRET_KEY"] = settings.secret_key

CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=True,
)

socketio = init_socketio(app)
register_blueprints(app)

try:
    mongo_service.connect()
except Exception as ex:
    logger.warning(f"Mongo init failed: {ex}")

if settings.chatbot_enabled:
    try:
        chat_service.init_db()
    except Exception as ex:
        logger.warning(f"Chat service DB init failed: {ex}")


@socketio.on("connect")
def handle_socket_connect():
    logger.info("Socket client connected")


@socketio.on("disconnect")
def handle_socket_disconnect():
    logger.info("Socket client disconnected")


if __name__ == "__main__":
    logger.info("SmartElevator backend is running...")
    socketio.run(
        app,
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
        allow_unsafe_werkzeug=True,
    )
