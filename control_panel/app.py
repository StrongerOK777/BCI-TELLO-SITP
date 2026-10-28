#!/usr/bin/env python3
"""Local web control panel for all three devices (car / drone / arm).

    python control_panel/app.py            # 模拟设备 + 模拟脑环，打开 http://127.0.0.1:8765
    python control_panel/app.py --real     # 允许在页面上（二次确认后）连接真实硬件

The panel is a thin Flask layer over ``bin.hub.ControlHub``; the brain-control
logic is the same ``bin.control.BrainControlSession`` every ``brain_control.py``
uses. The server only listens on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402

from bin.hub import ControlHub  # noqa: E402
from control_panel.devices import device_options  # noqa: E402

def create_app(hub: ControlHub) -> Flask:
    app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False

    def body() -> Dict[str, Any]:
        return request.get_json(silent=True) or {}

    def guarded(fn: Callable[[], Any]):
        try:
            return jsonify(fn())
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except (KeyError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc).strip("'")}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:  # hardware / serial errors -> visible message, server keeps running
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.get("/")
    def index():
        return send_from_directory(HERE / "static", "index.html")

    @app.get("/api/state")
    def state():
        return jsonify(hub.state())

    @app.post("/api/connect")
    def connect():
        data = body()
        real = bool(data.get("real_hardware"))
        if real and not data.get("confirm"):
            return jsonify({"error": "连接真实硬件需要确认（confirm=true）"}), 400
        return guarded(lambda: hub.connect(
            str(data.get("device", "")), real_hardware=real, real_headset=bool(data.get("real_headset"))
        ))

    @app.post("/api/disconnect")
    def disconnect():
        return guarded(hub.disconnect)

    @app.post("/api/brain/start")
    def brain_start():
        return guarded(hub.start_brain)

    @app.post("/api/brain/stop")
    def brain_stop():
        return guarded(hub.stop_brain)

    @app.post("/api/command")
    def command():
        return guarded(lambda: hub.command(str(body().get("intent", ""))))

    @app.post("/api/emergency")
    def emergency():
        return guarded(hub.emergency)

    @app.post("/api/config")
    def config():
        return guarded(lambda: hub.update_config(**body()))

    @app.post("/api/sim-headset")
    def sim_headset():
        data = body()
        return guarded(lambda: hub.set_sim_headset(
            attention=data.get("attention"), meditation=data.get("meditation"),
            poor_signal=data.get("poor_signal"), blinks=int(data.get("blinks", 0)),
        ))

    @app.get("/api/stream")
    def stream():
        q = hub.subscribe()
        once = request.args.get("once") == "1"  # tests: return the first event and close

        def events():
            try:
                while True:
                    try:
                        event, payload = q.get(timeout=15)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if once:
                        return
            finally:
                hub.unsubscribe(q)

        return Response(events(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="脑控统一控制面板（本地 Web）")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--real", action="store_true", help="允许在页面上连接真实硬件（仍需二次确认）")
    parser.add_argument("--model-path", default=None, help="model 转弯模式使用的模型，默认 src/models/FinalModel.pth")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    hub = ControlHub(device_options(args.model_path), allow_real=args.real)
    app = create_app(hub)
    print("=" * 60)
    print(f"脑控控制面板：http://127.0.0.1:{args.port}")
    print("真实硬件：" + ("允许（页面上需二次确认）" if args.real else "未允许（只用模拟设备）"))
    print("=" * 60)
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)
    finally:
        hub.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
