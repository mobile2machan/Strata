#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strata インテリジェント・モデル切り替えルーター (Proxy API)
OpenCode や各クライアントからのリクエストに応じて、IQ3_S と Swift 1.5 を動的に切り替えます。
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# モデル構成の定義
MODELS = {
    "swift-1.5-iq2_xs-dual-gpu": {
        "aliases": ["swift", "swift-1.5", "swift-iq2_xs", "swift-1.5-iq2_xs"],
        "name": "Swift 1.5 IQ2_XS (Dual GPU / 200+ t/s)",
        "config": ROOT / "strata-swift-dual-gpu.json",
        "env": {
            "STRATA_SPLIT_OWN": "1",
            "STRATA_GR_DOWN_MAX4": "1",
            "STRATA_PF_FUSED": "1"
        }
    },
    "qwen3.8-flash-next-iq3_s-dual-gpu": {
        "aliases": ["iq3_s", "qwen", "qwen3.8-flash-next", "qwen3.8-flash-next-iq3_s"],
        "name": "Qwen 3.8 Flash Next IQ3_S (Dual GPU)",
        "config": ROOT / "strata-iq3_s-dual-gpu.json",
        "env": {
            "STRATA_SPLIT_OWN": "1",
            "STRATA_GR_DOWN_MAX4": "1",
            "STRATA_PF_FUSED": "1"
        }
    }
}

DEFAULT_MODEL = "swift-1.5-iq2_xs-dual-gpu"
ROUTER_HOST = "0.0.0.0"
ROUTER_PORT = 8080
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8081

class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_model = None
        self.backend_proc = None
        self.status = "stopped"  # "stopped", "starting", "ready"
        self.start_time = None

    def resolve_model(self, model_name: str) -> str:
        if not model_name:
            return DEFAULT_MODEL
        target = model_name.strip().lower()
        for key, conf in MODELS.items():
            if target == key.lower():
                return key
            for alias in conf["aliases"]:
                if target == alias.lower():
                    return key
        # 部分一致のフォールバック
        if "swift" in target:
            return "swift-1.5-iq2_xs-dual-gpu"
        if "iq3" in target or "qwen" in target:
            return "qwen3.8-flash-next-iq3_s-dual-gpu"
        return DEFAULT_MODEL

    def stop_backend(self):
        if self.backend_proc:
            pid = self.backend_proc.pid
            print(f"[ルーター] 現在のモデルプロセス (PID: {pid}) を停止中...", flush=True)
            self.status = "stopping"
            try:
                if os.name == "nt":
                    # Windows で子プロセス（strata.exe 等）も含めて強制終了
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                   capture_output=True, check=False)
                else:
                    self.backend_proc.terminate()
            except Exception as e:
                print(f"[ルーター] プロセス停止中にエラー: {e}", flush=True)
            self.backend_proc = None
            time.sleep(1.5)  # VRAM の解放待機
            self.current_model = None
            self.status = "stopped"
            print("[ルーター] プロセス停止完了。VRAM を解放しました。", flush=True)

    def start_backend(self, model_key: str):
        if model_key not in MODELS:
            raise ValueError(f"未知のモデル: {model_key}")
        
        self.stop_backend()

        conf = MODELS[model_key]
        cfg_path = conf["config"]
        if not cfg_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {cfg_path}")

        print(f"[ルーター] 新しいモデルを起動します: {conf['name']}", flush=True)
        self.status = "starting"
        self.current_model = model_key

        env = os.environ.copy()
        for k, v in conf["env"].items():
            env[k] = v

        cmd = [
            PYTHON,
            str(ROOT / "serve" / "server.py"),
            "--engine", "strata",
            "--config", str(cfg_path),
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--fit-max-tokens"
        ]

        self.backend_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        self.start_time = time.time()
        print(f"[ルーター] プロセス起動開始 (PID: {self.backend_proc.pid})。モデルロード中...", flush=True)

        # 起動待機 (ポーリング)
        timeout = 90
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.backend_proc.poll() is not None:
                self.status = "stopped"
                self.current_model = None
                raise RuntimeError(f"Strata バックエンドが異常終了しました (終了コード: {self.backend_proc.returncode})")
            
            try:
                url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/v1/models"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1.0) as res:
                    if res.status == 200:
                        self.status = "ready"
                        load_time = time.time() - self.start_time
                        print(f"[ルーター] ★ モデルロード完了 ({load_time:.1f} 秒): {conf['name']}", flush=True)
                        return
            except Exception:
                time.sleep(1.0)

        self.stop_backend()
        raise TimeoutError("モデルの起動がタイムアウトしました。")

    def ensure_model(self, requested_model_name: str):
        target = self.resolve_model(requested_model_name)
        with self.lock:
            if self.current_model == target and self.status == "ready":
                return target
            print(f"[ルーター] モデル切り替え要求を検知: 現行={self.current_model} -> 要求={target}", flush=True)
            self.start_backend(target)
            return target

manager = ModelManager()

class RouterHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        # 1. モデル一覧
        if self.path == "/v1/models" or self.path == "/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            data = {
                "object": "list",
                "data": [
                    {
                        "id": k,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "strata",
                        "active": (k == manager.current_model and manager.status == "ready"),
                        "name": conf["name"]
                    }
                    for k, conf in MODELS.items()
                ]
            }
            self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
            return

        # 2. ステータス確認
        if self.path == "/v1/models/status" or self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            cur = manager.current_model
            data = {
                "current_model": cur,
                "current_name": MODELS[cur]["name"] if cur in MODELS else None,
                "status": manager.status,
                "available_models": list(MODELS.keys())
            }
            self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
            return

        # 3. 明示的な切り替え (GETクエリ: /switch?model=swift)
        if self.path.startswith("/switch?") or self.path.startswith("/v1/models/switch?"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            target = query.get("model", [None])[0]
            if target:
                try:
                    resolved = manager.ensure_model(target)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "ok": True,
                        "switched_to": resolved,
                        "name": MODELS[resolved]["name"]
                    }, ensure_ascii=False).encode("utf-8"))
                    return
                except Exception as e:
                    self.send_response(500)
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                    return

        # それ以外はバックエンドへ透過転送
        self.proxy_request("GET")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # 1. 明示的な切り替え (POST: /v1/models/switch)
        if self.path in ("/v1/models/switch", "/switch"):
            target = None
            try:
                payload = json.loads(body.decode("utf-8"))
                target = payload.get("model")
            except Exception:
                pass
            if target:
                try:
                    resolved = manager.ensure_model(target)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "ok": True,
                        "switched_to": resolved,
                        "name": MODELS[resolved]["name"]
                    }, ensure_ascii=False).encode("utf-8"))
                    return
                except Exception as e:
                    self.send_response(500)
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                    return

        # 2. チャット補完 (POST: /v1/chat/completions)
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            try:
                payload = json.loads(body.decode("utf-8"))
                requested_model = payload.get("model", "")
                if requested_model:
                    manager.ensure_model(requested_model)
            except Exception as e:
                print(f"[ルーター] モデル判定/切り替え中にエラー: {e}", flush=True)

        # バックエンドへ透過転送
        self.proxy_request("POST", body)

    def proxy_request(self, method: str, body: bytes = None):
        target_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}{self.path}"
        req_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        
        req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.send_cors_headers()
                self.end_headers()

                # ストリーミング対応でチャンク読み出し & 送信
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(k, v)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(503)
            self.send_cors_headers()
            self.end_headers()
            err_msg = json.dumps({"error": {"message": f"Strata バックエンド接続エラー: {e}"}}, ensure_ascii=False)
            self.wfile.write(err_msg.encode("utf-8"))

def main():
    print("=" * 65)
    print(" Strata インテリジェント・モデル切り替えルーター (Proxy API)")
    print(f" ルーター待受アドレス: http://{ROUTER_HOST}:{ROUTER_PORT}")
    print(f" バックエンド内部接続: http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(" 登録モデル:")
    for k, v in MODELS.items():
        print(f"   - {k} ({v['name']})")
    print("=" * 65)

    # 初回デフォルトモデルを起動
    try:
        manager.ensure_model(DEFAULT_MODEL)
    except Exception as e:
        print(f"[ルーター] 初期モデル起動中に警告: {e}", flush=True)

    server = ThreadingHTTPServer((ROUTER_HOST, ROUTER_PORT), RouterHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ルーター] 停止中...", flush=True)
        manager.stop_backend()
        server.server_close()
        print("[ルーター] 終了しました。", flush=True)

if __name__ == "__main__":
    main()
