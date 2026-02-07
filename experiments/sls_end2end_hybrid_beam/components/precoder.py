#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import tensorflow as tf
from sionna.utils import insert_dims


def svd_precoder(h, return_g=False):
    """
    Computes SVD-based precoding matrix for MIMO channels.

    Args:
        h (tf.Tensor): Channel matrix of shape [..., num_rx_ant, num_tx_ant].
        return_g (bool): If True, returns the singular values (channel gains) as well.

    Returns:
        w (tf.Tensor): Precoding matrix of shape [..., num_tx_ant, num_streams].
                       Currently assumes full rank transmission (num_streams = min(num_rx, num_tx)).
        g (tf.Tensor): Singular values (channel gains), returned if return_g=True.
    """

    # h shape: [..., num_rx, num_tx]
    # SVD: h = u * s * v_h (v_h is Hermitian transpose of v)
    # Tensorflow svd returns s (singular values), u, v (adjoint=True gives v_h, adjoint=False gives v)

    s, u, v = tf.linalg.svd(h)

    # v shape: [..., num_tx, num_tx]
    # The columns of v are the right singular vectors, which are the optimal precoding vectors.
    w = v

    if return_g:
        return w, s
    else:
        return w


def zero_forcing_precoder(h):
    """
    Computes Zero-Forcing (ZF) precoding matrix.
    Pseudo-inverse of H.
    """
    # h shape: [..., num_rx, num_tx]
    # pinv(h) = (h^H * h)^-1 * h^H (if num_rx >= num_tx)
    # or h^H * (h * h^H)^-1 (if num_rx < num_tx)

    w = tf.linalg.pinv(h)

    # Verify shape: pinv should be [..., num_tx, num_rx]
    # But usually ZF precoder is column-normalized or power constrained.
    # For now, just returning pseudo-inverse which orthogonalizes the channel.
    # Note: This W is applied as x = W * s.
    # If H * W = I, then received signal y = H * W * s + n = s + n.

    return w
