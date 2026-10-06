# Strata モデル切り替えルーター (Model Router)

複数モデル（Swift 1.5 と Qwen 3.8 Flash Next IQ3_S など）を、OpenCode や各種 API クライアントからのリクエストに応じて動的・安全に自動切り替え（スワップ）するプロキシサーバーです。

---

## 1. 起動方法

専用バッチファイル、または Python スクリプトから起動します。

```bat
run-router.bat
```

または：

```bat
"D:\GenerativeAI\Strata\.venv\Scripts\python.exe" tools/model_router.py --host 0.0.0.0 --port 8080 --backend-host 127.0.0.1 --backend-port 8081
```

* **待受ポート**: `0.0.0.0:8080`（ローカル `127.0.0.1:8080`、LAN `192.168.1.254:8080`）
* **内部バックエンド**: `127.0.0.1:8081`（ルーター配下で Strata サーバーが稼働）
* **デフォルトモデル**: Swift 1.5 IQ2_XS

---

## 2. API 仕様

* **自動切り替え**:
  `/v1/chat/completions` 等に渡す `model` パラメータに応じて、バックエンドが自動でスワップされます。
* **手動切り替え API**:
  * Swift 1.5 へ: `GET/POST http://192.168.1.254:8080/switch?model=swift`
  * Qwen IQ3_S へ: `GET/POST http://192.168.1.254:8080/switch?model=iq3_s`
* **ステータス確認**:
  `GET http://192.168.1.254:8080/v1/models/status`

---

## 3. 実測データ (Dual GPU)

### 計測環境
* **OS**: Windows 11 (WDDM)
* **GPU**: NVIDIA GeForce RTX 3090 (24GB) + NVIDIA RTX PRO 4000 Blackwell (24GB)（合計 48GB VRAM）
* **システム RAM**: 128GB
* **コミット**: `1a47634`

### 実測結果

| 段階 | 測定結果 |
| :--- | :--- |
| **Qwen 切り替え要求** | 503 応答: 0.01 秒 (`Retry-After: 30`) |
| **Swift 1.5 → Qwen IQ3_S 状態遷移** | `switching` → **105.2 秒** で `ready` 復帰 (`last_error=None`) |
| **Qwen IQ3_S 推論速度** | 200 OK、**57.7 tok/s** |
| **Swift 切り替え要求** | 503 応答: 0.02 秒 (`Retry-After: 30`) |
| **Qwen IQ3_S → Swift 1.5 状態遷移** | `switching` → **90.1 秒** で `ready` 復帰 (`last_error=None`) |
| **Swift 1.5 推論速度** | 200 OK、**98.4 tok/s** |
| **往復切り替え合計** | **197.7 秒**（タイムアウトなし、自己終了なし、孤児プロセス 0件） |

---

## 4. 運用上の制約事項

* **1マシン1系統限定**:
  ルーターは、モデル切り替え時に前のモデルの VRAM（24.0GB + 24.5GB）およびシステム RAM（40〜55GB）を確実に全解放するため、内部の `cleanup_competing_processes()` によって同一マシン上の競合する Strata エンジンおよび他の `model_router.py` プロセスを一括クリーンアップします。
  そのため、同一マシン上で複数系統のルーターを同時に起動することはできません（別のルーター系統を起動してスイッチを実行すると、互いのプロセスが終了対象となります）。1台のマシンにつき 1系統のみ稼働させてください。
