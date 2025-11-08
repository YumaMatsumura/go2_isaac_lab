# go2_isaac_lab

## 概要

本パッケージは、`unitree_rl_lab`のコードを引き継ぎつつ、拡張性を高めるためにフォルダ構成やコード等を見直したものである。

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

## 開発者用

- `pre-commit`の導入

  ```bash
  pip install pre-commit
  pre-commit install
  ```
