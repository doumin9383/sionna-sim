#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import tensorflow as tf


def expand_precoder(w_coarse, total_subcarriers, granularity_type, rbg_size_sc=None):
    """
    Expands a coarse precoder (Wideband/Subband) to the full number of subcarriers.

    Args:
        w_coarse (tf.Tensor): Coarse precoder of shape [..., num_blocks, num_tx_ant, num_layers].
        total_subcarriers (int): Target number of subcarriers.
        granularity_type (str): "Wideband", "Subband", or "Narrowband".
        rbg_size_sc (int, optional): Number of subcarriers per block (for Subband).

    Returns:
        w_full (tf.Tensor): Expanded precoder [..., total_subcarriers, num_tx_ant, num_layers].
    """
    if granularity_type == "Narrowband":
        # w_coarse is already full size
        return w_coarse

    if granularity_type == "Wideband":
        # w_coarse has shape [..., 1, num_tx, num_layers]
        # Tile it to cover all subcarriers
        # Dimension of blocks is -3
        multiples = [1] * len(w_coarse.shape)
        multiples[-3] = total_subcarriers
        return tf.tile(w_coarse, multiples)

    if granularity_type == "Subband":
        if rbg_size_sc is None:
            raise ValueError("rbg_size_sc must be provided for Subband granularity.")

        # w_coarse has shape [..., num_blocks, num_tx, num_layers]

        # We need to repeat each block rbg_size_sc times.
        # tf.repeat repeats elements along an axis.
        # Input: [..., num_blocks, num_tx, num_layers]
        # Output: [..., num_blocks * rbg_size_sc, num_tx, num_layers]

        w_expanded = tf.repeat(w_coarse, repeats=rbg_size_sc, axis=-3)

        # Crop to total_subcarriers if the last block exceeds boundary
        # The w_expanded shape might be larger than total_subcarriers if num_blocks * rbg_size_sc > total_subcarriers
        return w_expanded[..., :total_subcarriers, :, :]

    raise ValueError(f"Unknown granularity type: {granularity_type}")
