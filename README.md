# go2_isaac_lab

## 概要

本パッケージは、`unitree_rl_lab`のコードを引き継ぎつつ、拡張性を高めるためにフォルダ構成やコード等を見直したものである。

## テスト環境

### GPU

- NVIDIA GeForce RTX 5070 Ti
- Driver Version: 580.95.05
- CUDA Version: 12.9.86

### ソフトウェア

- Ubuntu22.04
- Isaac Sim 5.1.0(uv install)
- Python 3.11

## 環境構築手順

```bash
pip install -e source/go2_isaac_lab/
```

## 実行手順

- 利用可能なタスク一覧を表示する

  ```bash
  python scripts/list_envs.py
  ```

- タスクを実行する

  ```bash
  python scripts/rsl_rl/train.py --task <TASK_NAME>
  ```

- 途中まで学習したポリシーで学習を再開する

   ```bash
   python scripts/rsl_rl/train.py --task <TASK_NAME> --resume --load_run <FOLDER_NAME> --checkpoint <PT_FILE>
   ```

   以下に例を示す
   ```bash
   python scripts/rsl_rl/train.py \
     --task Go2-Isaac-Lab-Velocity-Parkour-Stationary-v0 \
     --resume \
     --load_run 2025-11-17_23-12-46 \
     --checkpoint model_15000.pt
   ```

## 開発者用

- `pre-commit`の導入

  ```bash
  pip install pre-commit
  pre-commit install
  ```

## メモ

### 転倒復帰の最新のポリシー

- `/home/yuma/go2_isaac_lab/logs/rsl_rl/go2_parkour/2025-11-19_08-33-56`

### しゃがみ歩行の最新のポリシー

- `/home/yuma/go2_isaac_lab/logs/rsl_rl/go2_parkour/2025-11-28_22-45-36`
