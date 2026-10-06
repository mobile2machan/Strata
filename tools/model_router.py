#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strata インテリジェント・モデル切り替えルーター (Proxy API)
OpenCode や各種 API クライアントからの要求に応じて、IQ3_S と Swift 1.5 を動的・安全に切り替えます。
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# winjob のインポート (Windows における子プロセス孤児化防止)
sys.path.insert(0, str(ROOT / "serve"))
try:
    import winjob
except ImportError:
    winjob = None

# モデル構成の定義 (厳格な完全一致・エイリアス定義)
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


def is_port_available(host: str, port: int) -> bool:
    """ポートが現在バインド可能（空き）か検査する"""
    hosts_to_check = [host]
    if host == "0.0.0.0":
        hosts_to_check.append("127.0.0.1")
    for h in hosts_to_check:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.bind((h, port))
            except OSError:
                return False
    return True


def cleanup_strata_engine_processes(exclude_pids: set[int] | None = None):
    """自前の子以外の strata.exe / strata-vision.exe を終了して VRAM を全解放する"""
    if os.name != "nt":
        return
    exclude = exclude_pids or set()
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq strata*", "/FO", "CSV", "/NH"],
            text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            parts = line.strip().split('","')
            if len(parts) >= 2:
                name = parts[0].strip('"')
                pid_str = parts[1].strip('"')
                if pid_str.isdigit() and ("strata.exe" in name.lower() or "strata-vision.exe" in name.lower()):
                    pid = int(pid_str)
                    if pid not in exclude:
                        print(f"[ルーター] 競合するエンジンプロセス {name} (PID: {pid}) を検出。VRAM 解放のため終了します...", flush=True)
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
    except Exception:
        pass


class ModelManager:
    def __init__(self, backend_host: str, backend_port: int, api_key: str = ""):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.api_key = api_key
        self.lock = threading.Lock()
        self.current_model: str | None = None
        self.backend_proc: subprocess.Popen | None = None
        self.status = "stopped"  # "stopped", "switching", "ready"
        self.target_model: str | None = None
        self.switch_start_time: float = 0.0
        self.last_error: str | None = None

    def resolve_model(self, model_name: str) -> str | None:
        """モデル名またはエイリアスの完全一致のみを判定。部分一致は一切行わない。"""
        if not model_name:
            return None
        target = model_name.strip().lower()
        for key, conf in MODELS.items():
            if target == key.lower():
                return key
            for alias in conf["aliases"]:
                if target == alias.lower():
                    return key
        return None

    def stop_backend(self):
        """バックエンドプロセスを停止し、残存エンジンを強制一掃してポートおよび VRAM を完全解放する"""
        if self.backend_proc:
            pid = self.backend_proc.pid
            print(f"[ルーター] バックエンドプロセス (PID: {pid}) を停止中...", flush=True)
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                   capture_output=True, check=False)
                else:
                    self.backend_proc.terminate()
            except Exception as e:
                print(f"[ルーター] プロセス停止中にエラー: {e}", flush=True)
            self.backend_proc = None

        # 孤立した strata.exe / strata-vision.exe を完全に一掃し、VRAM を 100% 解放
        cleanup_strata_engine_processes()

        # ポート解放の待機検査
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if is_port_available(self.backend_host, self.backend_port):
                break
            time.sleep(0.5)

        if not is_port_available(self.backend_host, self.backend_port):
            print(f"[ルーター] 警告: ポート {self.backend_host}:{self.backend_port} がまだ解放されていません。", flush=True)
        else:
            print(f"[ルーター] ポート {self.backend_host}:{self.backend_port} の解放を確認しました。", flush=True)

        self.current_model = None
        self.status = "stopped"

    def _switch_worker(self, model_key: str):
        """別スレッドでモデル切り替えを実行するルーチン"""
        try:
            self._do_switch(model_key)
        except Exception as e:
            print(f"[ルーター] モデル切り替えに失敗しました: {e}", flush=True)
            with self.lock:
                self.status = "stopped"
                self.target_model = None
                self.last_error = str(e)

    def _do_switch(self, model_key: str):
        # 1. 既存バックエンドおよび競合エンジンの完全停止 (VRAM 解放)
        self.stop_backend()

        # 2. ポート空き確認
        if not is_port_available(self.backend_host, self.backend_port):
            raise RuntimeError(
                f"バックエンドポート {self.backend_host}:{self.backend_port} は既に使用されています。"
                f"他の Strata サーバーが稼働中か確認してください。"
            )

        conf = MODELS[model_key]
        cfg_path = conf["config"]
        if not cfg_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {cfg_path}")

        print(f"[ルーター] 新しいモデルを起動します: {conf['name']}", flush=True)

        env = os.environ.copy()
        for k, v in conf["env"].items():
            env[k] = v

        cmd = [
            PYTHON,
            str(ROOT / "serve" / "server.py"),
            "--engine", "strata",
            "--config", str(cfg_path),
            "--host", self.backend_host,
            "--port", str(self.backend_port),
            "--fit-max-tokens"
        ]
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        self.backend_proc = proc

        # Windows Job Object に登録 (ルーター終了時に道連れ終了)
        if winjob and os.name == "nt":
            if winjob.contain(proc):
                print(f"[ルーター] プロセス PID {proc.pid} を Job Object に登録しました。", flush=True)

        print(f"[ルーター] プロセス起動開始 (PID: {proc.pid})。モデルロード待機中...", flush=True)

        # ログファイルのパス取得
        log_path = None
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            if "log" in cfg_data:
                log_path = Path(cfg_data["log"])
        except Exception:
            pass

        last_progress_time = time.time()
        last_log_size = log_path.stat().st_size if log_path and log_path.exists() else 0

        # 全体最大上限 (20分 = 1200秒)
        overall_deadline = time.time() + 1200.0

        while time.time() < overall_deadline:
            # 1. 子プロセスが異常終了していないか検査
            if proc.poll() is not None:
                ret = proc.returncode
                self.backend_proc = None
                with self.lock:
                    self.status = "stopped"
                    self.target_model = None
                raise RuntimeError(f"Strata バックエンドが起動処理中に異常終了しました (終了コード: {ret})")

            # 2. 進捗検知: ログファイルサイズの増加（エンジンが still starting 等を出力中）
            if log_path and log_path.exists():
                try:
                    sz = log_path.stat().st_size
                    if sz != last_log_size:
                        last_log_size = sz
                        last_progress_time = time.time()
                except OSError:
                    pass

            # 3. ポートの HTTP レスポンス検査 (/v1/models)
            try:
                url = f"http://{self.backend_host}:{self.backend_port}/v1/models"
                req = urllib.request.Request(url)
                if self.api_key:
                    req.add_header("Authorization", f"Bearer {self.api_key}")
                with urllib.request.urlopen(req, timeout=1.0) as res:
                    if res.status == 200:
                        body = json.loads(res.read().decode("utf-8"))
                        models_reported = [m.get("id") for m in body.get("data", [])]
                        cfg_model_name = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("model_name", "")
                        if model_key in models_reported or cfg_model_name in models_reported:
                            with self.lock:
                                self.current_model = model_key
                                self.status = "ready"
                                self.target_model = None
                            load_time = time.time() - self.switch_start_time
                            print(f"[ルーター] ★ モデルロード完了 ({load_time:.1f} 秒): {conf['name']}", flush=True)
                            return
                        else:
                            last_progress_time = time.time()
            except Exception:
                pass

            # 4. 180秒間ログの進捗が完全に途絶えた場合のみハングアップと判定
            if time.time() - last_progress_time > 180.0:
                print(f"[ルーター] 警告: モデルロードが 180 秒間進捗停止（ハングアップ）しました。", flush=True)
                self.stop_backend()
                with self.lock:
                    self.target_model = None
                raise TimeoutError("モデルロードが180秒間進捗停止したためタイムアウトしました。")

            time.sleep(1.0)

        # 20分タイムアウト
        self.stop_backend()
        with self.lock:
            self.target_model = None
        raise TimeoutError("モデルロードが全体最大時間 (20分) を超過しました。")

    def ensure_model(self, requested_model_name: str) -> tuple[bool, str]:
        """
        要求モデルの状態を確認。
        - 稼働中 (ready): (True, target) を即座に返す
        - 切替中 (switching): (False, target) を即座に返す (HTTP 503 用)
        - 別のモデルまたは停止中: 非同期バックグラウンド切替を開始し、(False, target) を即座に返す (HTTP 503 用)
        """
        target = self.resolve_model(requested_model_name)
        if not target:
            raise ValueError(f"未知のモデル名です: {requested_model_name}")

        with self.lock:
            # 1. すでに要求モデルが ready
            if self.current_model == target and self.status == "ready":
                return True, target

            # 2. 現在切り替え中
            if self.status == "switching":
                return False, self.target_model or target

            # 3. 新規に切り替えを開始 (非同期スレッド)
            self.status = "switching"
            self.target_model = target
            self.switch_start_time = time.time()
            self.last_error = None
            t = threading.Thread(target=self._switch_worker, args=(target,), daemon=True)
            t.start()
            return False, target


class RouterHandler(BaseHTTPRequestHandler):
    manager: ModelManager
    api_key: str

    def check_auth(self) -> bool:
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            key = auth[len("Bearer "):].strip()
            if key == self.api_key:
                return True
        if self.headers.get("x-api-key", "").strip() == self.api_key:
            return True
        return False

    def is_trusted_origin(self, origin: str) -> bool:
        if not origin:
            return True
        try:
            parsed = urlparse(origin)
            host = parsed.hostname or ""
            if host in ("localhost", "127.0.0.1"):
                return True
            if host.startswith("192.168.") or host.startswith("10.") or (host.startswith("172.") and 16 <= int(host.split(".")[1]) <= 31):
                return True
        except Exception:
            pass
        return False

    def send_cors_headers(self):
        origin = self.headers.get("Origin", "")
        if self.is_trusted_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin if origin else "*")
        else:
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if not self.check_auth():
            self.send_response(401)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Unauthorized: API key required"}}')
            return

        # 1. モデル一覧 (/v1/models)
        if self.path in ("/v1/models", "/models"):
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
                        "active": (k == self.manager.current_model and self.manager.status == "ready"),
                        "name": conf["name"],
                        "aliases": conf["aliases"]
                    }
                    for k, conf in MODELS.items()
                ]
            }
            self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
            return

        # 2. ステータス確認 (/status, /v1/models/status)
        if self.path in ("/status", "/v1/models/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            cur = self.manager.current_model
            elapsed = round(time.time() - self.manager.switch_start_time, 1) if self.manager.status == "switching" else None
            data = {
                "current_model": cur,
                "current_name": MODELS[cur]["name"] if cur in MODELS else None,
                "status": self.manager.status,
                "target_model": self.manager.target_model,
                "switching_elapsed_seconds": elapsed,
                "last_error": self.manager.last_error,
                "available_models": list(MODELS.keys())
            }
            self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
            return

        # 3. 明示的な切り替え (/switch?model=...)
        if self.path.startswith("/switch?") or self.path.startswith("/v1/models/switch?"):
            query = parse_qs(urlparse(self.path).query)
            target = query.get("model", [None])[0]
            if not target:
                self.send_response(400)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "Missing model query parameter"}}')
                return

            resolved = self.manager.resolve_model(target)
            if not resolved:
                self.send_response(400)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {
                        "message": f"Unknown model: '{target}'. Available: {list(MODELS.keys())}"
                    }
                }, ensure_ascii=False).encode("utf-8"))
                return

            try:
                ready, target_key = self.manager.ensure_model(target)
                elapsed = round(time.time() - self.manager.switch_start_time, 1)
                if not ready:
                    self.send_response(202)
                    self.send_header("Retry-After", "30")
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "ok": True,
                        "status": "switching",
                        "target_model": target_key,
                        "name": MODELS[target_key]["name"],
                        "message": f"Model switching to '{target_key}' in progress ({elapsed}s elapsed). Please wait ~1-3 minutes.",
                        "elapsed_seconds": elapsed
                    }, ensure_ascii=False).encode("utf-8"))
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "ok": True,
                    "status": "ready",
                    "current_model": target_key,
                    "name": MODELS[target_key]["name"]
                }, ensure_ascii=False).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": str(e)}}, ensure_ascii=False).encode("utf-8"))
                return

        # それ以外はバックエンドへ透過転送
        self.proxy_request("GET")

    def do_POST(self):
        if not self.check_auth():
            self.send_response(401)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Unauthorized: API key required"}}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # 1. 明示的な切り替え (POST: /switch, /v1/models/switch)
        if self.path in ("/switch", "/v1/models/switch"):
            target = None
            try:
                payload = json.loads(body.decode("utf-8"))
                target = payload.get("model")
            except Exception:
                pass
            if not target:
                self.send_response(400)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "Missing model field in JSON body"}}')
                return

            resolved = self.manager.resolve_model(target)
            if not resolved:
                self.send_response(400)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {
                        "message": f"Unknown model: '{target}'. Available: {list(MODELS.keys())}"
                    }
                }, ensure_ascii=False).encode("utf-8"))
                return

            try:
                ready, target_key = self.manager.ensure_model(target)
                elapsed = round(time.time() - self.manager.switch_start_time, 1)
                if not ready:
                    self.send_response(202)
                    self.send_header("Retry-After", "30")
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "ok": True,
                        "status": "switching",
                        "target_model": target_key,
                        "name": MODELS[target_key]["name"],
                        "message": f"Model switching to '{target_key}' in progress ({elapsed}s elapsed). Please wait ~1-3 minutes.",
                        "elapsed_seconds": elapsed
                    }, ensure_ascii=False).encode("utf-8"))
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "ok": True,
                    "status": "ready",
                    "current_model": target_key,
                    "name": MODELS[target_key]["name"]
                }, ensure_ascii=False).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": str(e)}}, ensure_ascii=False).encode("utf-8"))
                return

        # 2. チャット補完 (/v1/chat/completions)
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            try:
                payload = json.loads(body.decode("utf-8"))
                requested_model = payload.get("model", "")
                if requested_model:
                    resolved = self.manager.resolve_model(requested_model)
                    if not resolved:
                        self.send_response(400)
                        self.send_cors_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "error": {
                                "message": f"Unknown model: '{requested_model}'. Available models: {list(MODELS.keys())}"
                            }
                        }, ensure_ascii=False).encode("utf-8"))
                        return

                    ready, target_key = self.manager.ensure_model(requested_model)
                    if not ready:
                        elapsed = round(time.time() - self.manager.switch_start_time, 1)
                        self.send_response(503)
                        self.send_header("Retry-After", "30")
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_cors_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "error": {
                                "message": f"Model is currently loading: '{target_key}' ({elapsed}s elapsed, models take 1-3 minutes to load). Please retry in 30 seconds.",
                                "type": "model_switching",
                                "target_model": target_key,
                                "elapsed_seconds": elapsed
                            }
                        }, ensure_ascii=False).encode("utf-8"))
                        return
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"[ルーター] モデル判定/切替中にエラー: {e}", flush=True)

        # バックエンドへ透過転送
        self.proxy_request("POST", body)

    def proxy_request(self, method: str, body: bytes = None):
        target_url = f"http://{self.manager.backend_host}:{self.manager.backend_port}{self.path}"
        req_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        req_headers["Host"] = f"{self.manager.backend_host}:{self.manager.backend_port}"

        if self.api_key:
            req_headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.send_cors_headers()
                self.end_headers()

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
    parser = argparse.ArgumentParser(description="Strata Model Switcher Router")
    parser.add_argument("--host", default="0.0.0.0", help="ルーター待受ホスト (デフォルト: 0.0.0.0 - LANアクセス許可)")
    parser.add_argument("--port", type=int, default=8080, help="ルーター待受ポート (デフォルト: 8080)")
    parser.add_argument("--backend-host", default="127.0.0.1", help="Strata バックエンドホスト (デフォルト: 127.0.0.1)")
    parser.add_argument("--backend-port", type=int, default=8091, help="Strata バックエンドポート (デフォルト: 8091)")
    parser.add_argument("--api-key", default="", help="クライアント認証およびバックエンドに渡す API キー")
    parser.add_argument("--default-model", default=DEFAULT_MODEL, help="起動時のデフォルトモデル")
    args = parser.parse_args()

    # 1. ルーターポートの空き確認
    if not is_port_available(args.host, args.port):
        print(f"[エラー] ルーター待受ポート {args.host}:{args.port} は既に使用されています。既存のサーバーを終了するか別のポートを指定してください。", file=sys.stderr)
        sys.exit(1)

    # 2. 起動前に既存の孤児・競合 Strata エンジンプロセスを一掃 (VRAM 解放)
    cleanup_strata_engine_processes()

    manager = ModelManager(args.backend_host, args.backend_port, api_key=args.api_key)
    atexit.register(manager.stop_backend)

    class BoundHandler(RouterHandler):
        pass
    BoundHandler.manager = manager
    BoundHandler.api_key = args.api_key

    print("=" * 70)
    print(" Strata インテリジェント・モデル切り替えルーター (Proxy API)")
    print(f" ルーター待受アドレス: http://{args.host}:{args.port}")
    print(f" バックエンド内部接続: http://{args.backend_host}:{args.backend_port}")
    if args.api_key:
        print(" API キー認証: 有効")
    else:
        print(" API キー認証: なし (LAN / 127.0.0.1 信頼オリジン)")
    print(" 登録モデル (完全一致のみ有効):")
    for k, v in MODELS.items():
        print(f"   - {k} ({v['name']}) [aliases: {', '.join(v['aliases'])}]")
    print("=" * 70)

    # 初回デフォルトモデルをバックグラウンドスレッドで起動
    try:
        manager.ensure_model(args.default_model)
    except Exception as e:
        print(f"[ルーター] 初期モデル起動中に警告: {e}", file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), BoundHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ルーター] 終了要求を受信。停止中...", flush=True)
    finally:
        manager.stop_backend()
        server.server_close()
        print("[ルーター] 正常に終了しました。", flush=True)


if __name__ == "__main__":
    main()
