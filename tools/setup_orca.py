import os
import sys
import time
import urllib.request
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "orca-iq3_xxs"
PACK_DIR = ROOT / "packs" / "orca-iq3_xxs"
HF_REPO = "https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/resolve/main/"
SHARDS = [
    "Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf"
]

def download_file(url, dst: Path, token=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    done_marker = dst.with_name(dst.name + ".done")
    if dst.exists() and done_marker.exists():
        print(f"  [ok] {dst.name} already downloaded")
        return True

    headers = {"User-Agent": "strata-orca-setup"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"\n[エラー] Hugging Faceの認証が必要です (HTTP 401 Unauthorized)。")
            print("このモデルはアクセス承認制(Gated Repo)のため、Hugging FaceのAccess Tokenが必要です。")
            return False
        raise

    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    print(f"\n--- Downloading {dst.name} ({total / 1e9:.2f} GB) ---")
    headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as r, open(part, "ab" if have else "wb") as f:
        last = 0
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                pct = (100 * have / total) if total else 0
                print(f"\r  {dst.name}: {have / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)

    print()
    part.replace(dst)
    done_marker.write_text("ok", encoding="utf-8")
    print(f"  [ok] {dst.name} finished.")
    return True

def main():
    print("=" * 60)
    print(" Strata - OrcaRouter Uncensored IQ3_XXS セットアップ")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for shard in SHARDS:
        target = MODELS_DIR / shard
        if not (target.exists() and target.with_name(target.name + ".done").exists()):
            url = HF_REPO + shard
            success = False
            while not success:
                success = download_file(url, target, token)
                if not success:
                    print("\nHugging Faceで https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF を開き、")
                    print("「Access repository」で利用規約を承諾後、Settings -> Access Tokens で取得したTokenを入力してください。")
                    token = input("\nHugging Face Access Token (hf_...): ").strip()
                    if not token:
                        print("キャンセルされました。手動でモデルを以下のフォルダに配置することも可能です:")
                        print(f"  {MODELS_DIR}")
                        sys.exit(1)

    print("\n=== モデルのStrata互換パック生成 (--compat-bf16) ===")
    llama_dir = ROOT / "third_party" / "llama.cpp"
    env = dict(os.environ, STRATA_GGUF_PY=str(llama_dir / "gguf-py"))

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "iq_pack.py"),
        "--gguf", str(MODELS_DIR / SHARDS[0]),
        "--out", str(PACK_DIR),
        "--compat-bf16"
    ]
    print("実行中: " + " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print("[エラー] パック生成に失敗しました。")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" [成功] セットアップが完了いたしました！")
    print(" 「run-orca-iq3_xxs.bat」をダブルクリックして起動できます。")
    print("=" * 60)

if __name__ == "__main__":
    main()
