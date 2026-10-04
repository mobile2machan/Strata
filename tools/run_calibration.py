import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import calibrate as CAL

def main():
    target_json = ROOT / "strata-iq3_s-dual-gpu.json"
    if len(sys.argv) > 1:
        target_json = Path(sys.argv[1])

    print("=" * 65)
    print(f"  Strata Dual-GPU 自動キャリブレーション")
    print(f"  対象設定: {target_json.name}")
    print("=" * 65)
    print("実機で推論テストを行い、最適なPCIe転送比率・投機閾値・CPUスレッド数を測定します。")
    print("測定には約3〜5分かかります。そのままお待ちください...\n")

    cfg = json.loads(target_json.read_text(encoding="utf-8-sig"))
    res = CAL.run(cfg, say=print)

    settings = res.get("settings", {})
    report = res.get("report", {})

    print("\n" + "=" * 65)
    print("  [測定完了] キャリブレーション結果")
    print("=" * 65)
    if report.get("tok_s"):
        print(f"  測定生成速度: {report['tok_s']} tok/s")
    print(f"  所要時間: {report.get('seconds', 0)} 秒")

    if settings:
        print("\n  最適化されたパラメータ:")
        for k, v in settings.items():
            print(f"    {k}: {v}")

        # 設定ファイルへ反映
        cfg["args"] = CAL.apply(cfg["args"], settings)
        target_json.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        print(f"\n  [成功] 最適設定を {target_json.name} に自動保存いたしました！")
    else:
        print("\n  現在の設定（デフォルト値）がすでに最適値であることが確認されました。")
        print("  設定ファイルの変更は不要です。")
    print("=" * 65)

if __name__ == "__main__":
    main()
