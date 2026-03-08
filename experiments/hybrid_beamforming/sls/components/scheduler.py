import tensorflow as tf
import numpy as np


class PFScheduler:
    def __init__(self, config, num_ut, num_rb):
        self.config = config
        self.num_ut = num_ut
        self.num_rb = num_rb

        # 移動平均エクスウィンドー T_avg_u
        self.avg_throughput = tf.zeros([num_ut], dtype=tf.float32)

        # PFメトリックに加算する微小定数 (0割防止)
        self.epsilon = (
            self.config.pf_epsilon if hasattr(self.config, "pf_epsilon") else 1e-3
        )

        # IIRフィルタの重み
        self.beta = self.config.pf_beta if hasattr(self.config, "pf_beta") else 0.98

    def schedule(self, pre_allocation_results, channel_quality_metric=None):
        """
        [Phase 2: Resource Mapping (Algorithm 2)]
        事前計算結果（要求RB数、期待スループット）を用いてPFメトリックを計算し、
        波形に応じたリソースマッピング（RBの割り振り）を実行する。

        Args:
            pre_allocation_results (dict):
                "n_req": [Batch, UT]
                "max_thp": [Batch, UT] 期待スループット
                "mcs_opt": [Batch, UT]
                "rank_opt": [Batch, UT]
            channel_quality_metric (tf.Tensor, optional): [Batch, UT, RB] の各RBの通信品質（SINR等）

        Returns:
            dict:
                "allocation_mask": [Batch, UT, RB] (bool) 実際に割り当てられたRBのマスク
                "scheduled_rbs": [Batch, UT] (int32) 実際に割り当てられたRB数
                "scheduled_thp": [Batch, UT] (float32) スケジューリング結果に基づく推定スループット
        """
        B = tf.shape(pre_allocation_results["max_thp"])[0]
        N_UT = self.num_ut
        N_RB = self.num_rb

        n_req = pre_allocation_results["n_req"]
        max_thp = pre_allocation_results["max_thp"]
        max_thp_f = tf.cast(max_thp, tf.float32)

        # 1. PFメトリックの計算
        avg_thp_expanded = tf.broadcast_to(self.avg_throughput[None, :], [B, N_UT])
        pf_metric = max_thp_f / (avg_thp_expanded + self.epsilon)

        # -1のスループット（割当不可）を持つUEは除外
        valid_mask = max_thp_f > 0
        pf_metric = tf.where(valid_mask, pf_metric, -1.0)

        allocation_mask = np.zeros((B, N_UT, N_RB), dtype=bool)
        scheduled_rbs = np.zeros((B, N_UT), dtype=np.int32)

        for b in range(B):
            # 優先度順にUEをソート (降順)
            pf_b = pf_metric[b].numpy()
            n_req_b = n_req[b].numpy()
            cqi_b = (
                channel_quality_metric[b].numpy()
                if channel_quality_metric is not None
                else np.zeros((N_UT, N_RB))
            )

            # None-zero/valid ones
            ue_indices = np.where(pf_b > 0)[0]
            sorted_ues = ue_indices[np.argsort(-pf_b[ue_indices])]

            free_rbs = np.ones(N_RB, dtype=bool)

            for u in sorted_ues:
                req_rb = n_req_b[u]
                if req_rb <= 0 or np.sum(free_rbs) < req_rb:
                    continue

                selected_rbs = []

                if self.config.waveform == "CP-OFDM":
                    # [CP-OFDM: 非連続・Greedy選択]
                    # 空きRBの中でCQIが高い上位 req_rb 個を取得
                    free_indices = np.where(free_rbs)[0]
                    free_cqi = cqi_b[u, free_indices]

                    # Sort by CQI descending
                    top_k_indices = np.argsort(-free_cqi)[:req_rb]
                    selected_rbs = free_indices[top_k_indices]

                else:
                    # [DFT-s-OFDM: 連続・RBGベース選択]
                    # 最小RBGサイズ(min of S_FDRA)を満たすチャンク単位での連続割当
                    min_rbg = (
                        min(self.config.s_fdra_options)
                        if hasattr(self.config, "s_fdra_options")
                        else 4
                    )

                    # If target is non-multiple of chunk, round down or up. Usually it is already a size from s_fdra.
                    needed_chunks = req_rb // min_rbg
                    chunks_total = N_RB // min_rbg

                    if needed_chunks == 0:
                        continue

                    # チャンク単位の空き状況配列
                    free_chunks = np.array(
                        [
                            np.all(free_rbs[c * min_rbg : (c + 1) * min_rbg])
                            for c in range(chunks_total)
                        ]
                    )

                    # 連続チャンクの探索
                    contiguous_candidates = []
                    for c in range(chunks_total - needed_chunks + 1):
                        if np.all(free_chunks[c : c + needed_chunks]):
                            contiguous_candidates.append(c)

                    if len(contiguous_candidates) > 0:
                        # 複数候補がある場合、該当領域のCQI平均が最大のものを選択
                        best_c = -1
                        best_cqi = -np.inf
                        for c in contiguous_candidates:
                            start_rb = c * min_rbg
                            end_rb = start_rb + (needed_chunks * min_rbg)
                            avg_cqi = np.mean(cqi_b[u, start_rb:end_rb])
                            if avg_cqi > best_cqi:
                                best_cqi = avg_cqi
                                best_c = c

                        start_rb = best_c * min_rbg
                        actual_rbs_to_allocate = needed_chunks * min_rbg
                        selected_rbs = np.arange(
                            start_rb, start_rb + actual_rbs_to_allocate
                        )
                    else:
                        # 連続ブロックが確保できない場合はスキップ
                        continue

                if len(selected_rbs) > 0:
                    allocation_mask[b, u, selected_rbs] = True
                    free_rbs[selected_rbs] = False
                    scheduled_rbs[b, u] = len(selected_rbs)

        # TF Tensorに変換
        allocation_mask_tf = tf.convert_to_tensor(allocation_mask, dtype=tf.bool)
        scheduled_rbs_tf = tf.convert_to_tensor(scheduled_rbs, dtype=tf.int32)

        # 今回の割当に基づく推定スループット計算
        req_rb_f = tf.maximum(tf.cast(n_req, tf.float32), 1.0)
        sch_rb_f = tf.cast(scheduled_rbs_tf, tf.float32)
        scheduled_thp = max_thp_f * (sch_rb_f / req_rb_f)

        # 2. 平均スループットの更新 (IIR filter)
        batch_avg_thp = tf.reduce_mean(scheduled_thp, axis=0)  # [N_UT]
        self.avg_throughput = (
            self.beta * self.avg_throughput + (1.0 - self.beta) * batch_avg_thp
        )

        return {
            "allocation_mask": allocation_mask_tf,
            "scheduled_rbs": scheduled_rbs_tf,
            "scheduled_thp": scheduled_thp,
        }
