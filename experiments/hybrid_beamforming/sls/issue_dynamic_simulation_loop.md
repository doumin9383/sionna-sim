# Issue: Dynamic Simulation Loop & Convergence Strategy

## 背景
現在のSLSシミュレーションは「Snapshot (Drop-based)」であり、静的な配置においてRank/MCS/干渉が収束するまでループを回している（`max_la_iterations`, `max_rank_selection_iterations`）。
これは静特性評価としては正しいが、将来的に「Dynamic (Slot-based)」な時系列シミュレーションを行う場合、1スロット内で収束するまでループするのは計算量的に過剰であり、また現実のUE/BSの処理能力（1スロット内でのフィードバック回数制限）とも乖離する可能性がある。

## 課題
Dynamicシミュレーションにおいては、以下の挙動がより適切である可能性がある。
1.  **反復回数の制限**: `max_rank_selection_iterations = 1` とし、1スロットにつき1回だけランク・MCSを更新する。
2.  **状態の継続性**: 前のスロットのRank/MCS/干渉状態を次のスロットの初期値として引き継ぎ、時間をかけて環境変化に追従させる（実際のLink Adaptationに近い挙動）。

## アクションアイテム
*   [ ] Dynamicシミュレーションモード実装時に、ループ回数を制御できるオプションを追加検討する。
*   [ ] 「収束」か「追従」かを選択できるアーキテクチャになっているか再評価する。
