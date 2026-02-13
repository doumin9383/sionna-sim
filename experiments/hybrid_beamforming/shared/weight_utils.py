import tensorflow as tf


def compute_digital_weights(h, granularity, rbg_size_sc=None, weight_type="svd"):
    """
    粒度に応じたデジタル重み（プリコーダ/コンバイナ）を計算する。

    Args:
        h (tf.Tensor): チャネル行列 [..., RxP, TxP]
        granularity (str): 'wideband', 'subband', 'narrowband', 'carrierwise'
        rbg_size_sc (int): Subband時のサブキャリア数
        weight_type (str): 'svd' (将来的に 'lmmse' 等を追加可能)

    Returns:
        s, u, v: SVD結果
    """
    # 1. 粒度に応じた平均化処理
    # h の形状想定: [Batch, UT, Freq, RxP, TxP] or similar
    # 最後の2軸以外が空間・周波数

    if granularity == "wideband":
        # 全有効サブキャリア（軸-3）で平均
        h_coarse = tf.reduce_mean(h, axis=-3, keepdims=True)
    elif granularity == "subband":
        if rbg_size_sc is None:
            raise ValueError("rbg_size_sc is required for subband granularity")
        # RBG単位で平均
        # [..., F, RxP, TxP] -> [..., Num_RBG, RBG_Size, RxP, TxP]
        shape = tf.shape(h)
        n_f = shape[-3]
        num_rbg = (n_f + rbg_size_sc - 1) // rbg_size_sc

        # パディングが必要な場合（端数）
        pad_size = num_rbg * rbg_size_sc - n_f
        paddings = [[0, 0]] * (h.shape.rank - 3) + [[0, pad_size], [0, 0], [0, 0]]
        h_padded = tf.pad(h, paddings)

        h_reshaped = tf.reshape(
            h_padded,
            tf.concat([shape[:-3], [num_rbg, rbg_size_sc], shape[-2:]], axis=0),
        )
        h_coarse = tf.reduce_mean(h_reshaped, axis=-3)
    elif granularity == "narrowband":
        # 1 RB = 12 SC 単位で平均
        rb_size = 12
        shape = tf.shape(h)
        n_f = shape[-3]
        num_rb = (n_f + rb_size - 1) // rb_size

        pad_size = num_rb * rb_size - n_f
        paddings = [[0, 0]] * (h.shape.rank - 3) + [[0, pad_size], [0, 0], [0, 0]]
        h_padded = tf.pad(h, paddings)

        h_reshaped = tf.reshape(
            h_padded, tf.concat([shape[:-3], [num_rb, rb_size], shape[-2:]], axis=0)
        )
        h_coarse = tf.reduce_mean(h_reshaped, axis=-3)
    else:  # carrierwise or default
        h_coarse = h

    # 2. 重み計算
    if weight_type == "svd":
        # h_coarse: [..., Coarse_Freq, RxP, TxP]
        # Transpose for SVD: tf.linalg.svd expects [..., M, N]
        s, u, v = tf.linalg.svd(h_coarse)
        return s, u, v
    else:
        raise NotImplementedError(f"Weight type {weight_type} not implemented")


def expand_weights(w, target_fft_size, granularity, rbg_size_sc=None):
    """
    計算された重みをフル有効サブキャリアサイズに展開する。

    Args:
        w (tf.Tensor): 重み行列 [..., Num_Blocks, P, R]
        target_fft_size (int): 展開後の有効サブキャリア数
        granularity (str): 'wideband', 'subband', 'narrowband', 'carrierwise'
        rbg_size_sc (int): Subband時のサブキャリア数

    Returns:
        w_expanded: 展開後の重み [..., target_fft_size, P, R]
    """
    if granularity == "wideband":
        # w: [..., 1, P, R]
        multiples = [1] * (w.shape.rank - 3) + [target_fft_size, 1, 1]
        return tf.tile(w, multiples)

    elif granularity == "subband":
        # w: [..., Num_RBG, P, R]
        w_rep = tf.repeat(w, repeats=rbg_size_sc, axis=-3)
        return w_rep[..., :target_fft_size, :, :]

    elif granularity == "narrowband":
        # w: [..., Num_RB, P, R]
        w_rep = tf.repeat(w, repeats=12, axis=-3)
        return w_rep[..., :target_fft_size, :, :]

    else:  # carrierwise
        return w
