#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.channel import (
    ChannelModel,
    GenerateOFDMChannel,
    GenerateTimeChannel,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
)

# Note: time_lag_discrete_time_channel might be needed for time channel,
# or we use cir_to_time_channel if available (Sionna >0.14 usually has it or similar).
# Checking imports from previous context: Step 82 showed 'GenerateTimeChannel', 'TimeChannel'.
# 'cir_to_ofdm_channel' exists. 'cir_to_time_channel' is not explicitly listed in Step 82 but might be part of utils or channel.
# Step 82 listed: ..., 'GenerateTimeChannel', ...
# We will trust standard Sionna APIs or import basic operations.

from sionna.phy.utils import flatten_last_dims, split_dim
from sionna.phy.channel.tr38901 import UMi, UMa, RMa


class RBGChannelModel(ChannelModel):
    """
    Abstract Base Class / Wrapper for Physical Channel Models (UMa, UMi, RMa).
    Ensures consistent path generation logic.

    In this revision, it acts as a wrapper around the standard 3GPP/Sionna models
    to provide a unified interface for RBG-based systems.
    """

    def __init__(
        self,
        scenario,
        carrier_frequency,
        ut_array,
        bs_array,
        direction,
        enable_pathloss=True,
        enable_shadow_fading=True,
        o2i_model="low",
        average_street_width=20.0,
        average_building_height=5.0,
        precision=None,
    ):
        super().__init__(
            precision=precision,
        )

        # Internal Channel Model instantiation
        common_params = {
            "carrier_frequency": carrier_frequency,
            "ut_array": ut_array,
            "bs_array": bs_array,
            "direction": direction,
            "enable_pathloss": enable_pathloss,
            "enable_shadow_fading": enable_shadow_fading,
            "precision": precision,
        }

        if scenario == "umi":
            self._model = UMi(o2i_model=o2i_model, **common_params)
        elif scenario == "uma":
            self._model = UMa(o2i_model=o2i_model, **common_params)
        elif scenario == "rma":
            self._model = RMa(
                average_street_width=average_street_width,
                average_building_height=average_building_height,
                **common_params,
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        pass

    def set_topology(self, *args, **kwargs):
        """Pass topology down to the internal model"""
        self._model.set_topology(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Delegate the call to the internal model.
        Returns: (a, tau)
        """
        return self._model(*args, **kwargs)

    @property
    def model(self):
        """Access the underlying internal model"""
        return self._model


class ChunkedTimeChannel(GenerateTimeChannel):
    """
    Generates Time Domain Channel (CIR or Waveform) supporting FDRA masks.
    Inherits from GenerateTimeChannel.
    """

    def __init__(
        self,
        channel_model,
        bandwidth,
        num_time_samples,
        l_min,
        l_max,
        normalize_channel=False,
        precision=None,
    ):
        super().__init__(
            channel_model=channel_model,
            bandwidth=bandwidth,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            normalize_channel=normalize_channel,
            precision=precision,
        )
        self._channel_model = channel_model

    def get_cir(self, batch_size=None):
        """
        Explicitly get CIR (Discrete Time Channel Impulse Response) from the model.
        Returns:
            h_time: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_path, num_time_samples]
            (The discrete time channel taps)
        """
        # 1. Generate paths (a, tau)
        # Use _num_time_steps and _bandwidth (acting as sampling freq)
        # However, for cir_to_time_channel, we need to ensure we have enough time steps
        # to cover the delay spread + filter length.
        # But here _channel_model usually just needs (num_time_steps, sampling_frequency)
        # to generate time-variant paths if mobility is involved.
        # If static, it might ignore num_time_steps.

        # We use the total number of time steps required for the output
        total_time_steps = self._num_time_steps + self._l_tot - 1
        a, tau = self._channel_model(total_time_steps, self._bandwidth)

        # 2. Convert to Discrete Time Channel (h_time)
        h_time = cir_to_time_channel(
            self._bandwidth,
            a,
            tau,
            l_min=self._l_min,
            l_max=self._l_max,
            normalize=self._normalize_channel,
        )

        return h_time

    def call(self, batch_size=None):
        # Default behavior of GenerateTimeChannel
        return super().call(batch_size)


class ChunkedOFDMChannel(GenerateOFDMChannel):
    """
    Generates Frequency Domain Channel with optimizations for RBG-based processing (SLS).
    Inherits from GenerateOFDMChannel.
    """

    def __init__(
        self,
        channel_model,
        resource_grid,
        normalize_channel=True,
        precision=None,
        use_rbg_granularity=False,
        rbg_size=1,
    ):
        super().__init__(
            channel_model=channel_model,
            resource_grid=resource_grid,
            normalize_channel=normalize_channel,
            precision=precision,
        )
        self._channel_model = channel_model
        self._resource_grid = resource_grid
        self.use_rbg_granularity = use_rbg_granularity
        self.rbg_size = rbg_size

        # Pre-compute all frequencies and RBG centers
        self._all_frequencies = subcarrier_frequencies(
            resource_grid.fft_size, resource_grid.subcarrier_spacing
        )
        # TODO: Define RBG centers if we want strict sampling only at centers

    def __call__(self, batch_size=None):
        if self.use_rbg_granularity:
            return self.get_rbg_channel(batch_size, self.rbg_size)
        else:
            return super().__call__(batch_size)

    def get_paths(self, batch_size=None):
        """
        Expose path generation for consistency verification.
        """
        # GenerateOFDMChannel stores necessary parameters in:
        # self._resource_grid.num_ofdm_symbols
        # sampling_frequency = 1 / self._resource_grid.ofdm_symbol_duration

        num_ofdm_symbols = self._resource_grid.num_ofdm_symbols
        sampling_frequency = 1.0 / self._resource_grid.ofdm_symbol_duration

        a, tau = self._channel_model(num_ofdm_symbols, sampling_frequency)
        return a, tau

    def get_rbg_channel(self, batch_size, rbg_size, active_rbgs=None):
        """
        Get channel sampled at RBG granularity.
        """
        a, tau = self.get_paths(batch_size)

        # Calculate RBG center frequencies
        # RBG size is in number of subcarriers
        # We assume RBGs are contiguous blocks

        # Create RBG indices (centers)
        # Shape: [num_rbgs]
        # Example: rbg_size=16. Indices: 8, 24, 40...
        num_subcarriers = self._resource_grid.fft_size
        num_rbgs = tf.maximum(num_subcarriers // rbg_size, 1)

        # Calculate indices of RBG centers
        # If num_rbgs is 1 because of fallback, we take the center of the available band
        if num_subcarriers < rbg_size:
            rbg_indices = tf.constant([num_subcarriers // 2], dtype=tf.int32)
        else:
            rbg_indices = tf.range(num_rbgs) * rbg_size + (rbg_size // 2)

        # Gather frequencies at these indices
        rbg_freqs = tf.gather(self._all_frequencies, rbg_indices)

        # If active_rbgs is provided (mask), we might want to filter further,
        # but usually we want the whole grid sampled at RBG resolution for scheduling.
        # If active_rbgs corresponds to specific RBG indices required:
        if active_rbgs is not None:
            rbg_freqs = tf.gather(rbg_freqs, active_rbgs)

        # Generate Channel freq response only at these frequencies
        h_rbg = cir_to_ofdm_channel(
            rbg_freqs, a, tau, normalize=self._normalize_channel
        )
        return h_rbg


class HybridOFDMChannel(ChunkedOFDMChannel):
    """
    Adds Analog Beamforming capabilities to the ChunkedOFDMChannel.
    """

    def __init__(
        self,
        channel_model,
        resource_grid,
        tx_array,
        rx_array,
        num_tx_ports,
        num_rx_ports,
        normalize_channel=True,
        precision=None,
        use_rbg_granularity=False,
        rbg_size=1,
    ):
        super().__init__(
            channel_model=channel_model,
            resource_grid=resource_grid,
            normalize_channel=normalize_channel,
            precision=precision,
            use_rbg_granularity=use_rbg_granularity,
            rbg_size=rbg_size,
        )

        self.tx_array = tx_array
        self.rx_array = rx_array
        self.num_tx_ports = num_tx_ports
        self.num_rx_ports = num_rx_ports

        # Initialize Default Weights
        self._init_default_weights()

    def _init_default_weights(self):
        # Default: Identity mapping (first N elements)
        self.w_rf = tf.eye(
            self.tx_array.num_ant, num_columns=self.num_tx_ports, dtype=tf.complex64
        )
        self.a_rf = tf.eye(
            self.rx_array.num_ant, num_columns=self.num_rx_ports, dtype=tf.complex64
        )

    def set_analog_weights(self, w_rf, a_rf):
        self.w_rf = w_rf
        self.a_rf = a_rf

    def _apply_weights(self, h_elem, w_rf, a_rf):
        """
        Applies weights using Einsum.
        h_elem: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_sc]
        w_rf  : [num_tx_ant, num_tx_ports] or broadcastable
        a_rf  : [num_rx_ant, num_rx_ports] or broadcastable
        """
        # TX Beamforming
        # h_elem indices: b (batch), r (rx), u (rx_ant), t (tx), v (tx_ant), s (sym), c (sc)
        # w_rf indices:   v (tx_ant), p (tx_port)
        # Target:         b, r, u, t, p, s, c

        # Determine equation based on w_rf rank
        if len(w_rf.shape) == 2:  # [v, p] - static weights for all
            eq_tx = "brutvsc,vp->brutpsc"
        elif len(w_rf.shape) == 3:  # [t, v, p] or [b, v, p] - depend on context
            # Assuming [num_tx, num_tx_ant, num_tx_ports] if mismatch batch
            if w_rf.shape[0] == h_elem.shape[3]:  # matches num_tx
                eq_tx = "brutvsc,tvp->brutpsc"
            else:  # assumes [batch, v, p]
                eq_tx = "brutvsc,bvp->brutpsc"
        else:  # [b, t, v, p]
            eq_tx = "brutvsc,btvp->brutpsc"

        h_tx_bf = tf.einsum(eq_tx, h_elem, w_rf)

        # RX Beamforming
        # h_tx_bf indices: b, r, u, t, p, s, c
        # a_rf indices:    u (rx_ant), q (rx_port)
        # Target:          b, r, q, t, p, s, c

        if len(a_rf.shape) == 2:  # [u, q]
            eq_rx = "brutpsc,uq->brqtpsc"
        elif len(a_rf.shape) == 3:  # [r, u, q] or [b, u, q]
            if a_rf.shape[0] == h_elem.shape[1]:  # matches num_rx
                eq_rx = "brutpsc,ruq->brqtpsc"
            else:  # assumes [batch, u, q]
                eq_rx = "brutpsc,buq->brqtpsc"
        else:  # [b, r, u, q]
            eq_rx = "brutpsc,bruq->brqtpsc"

        h_port = tf.einsum(eq_rx, h_tx_bf, tf.math.conj(a_rf))
        return h_port

    def __call__(self, batch_size=None):
        """
        Return the port-domain channel.
        """
        if self.use_rbg_granularity:
            # get_rbg_channel already applies weights
            return self.get_rbg_channel(batch_size, self.rbg_size)

        # 1. Get physical channel (Element domain) - calling grandparent directly to avoid ChunkedOFDMChannel logic
        # wrapping back to get_rbg_channel if we called super().__call__
        h_elem = GenerateOFDMChannel.__call__(self, batch_size)

        # 2. Apply Analog Beamforming
        h_port = self._apply_weights(h_elem, self.w_rf, self.a_rf)

        return h_port

    def get_rbg_channel(self, batch_size, rbg_size, active_rbgs=None):
        """
        Get port-domain channel sampled at RBG granularity.
        """
        # 1. Get physical RBG channel (Element domain)
        h_elem = super().get_rbg_channel(batch_size, rbg_size, active_rbgs)

        # 2. Apply Analog Beamforming
        h_port = self._apply_weights(h_elem, self.w_rf, self.a_rf)

        return h_port


class GeneratHybridBeamformingTimeChannel(ChunkedTimeChannel):
    """
    Adds Analog Beamforming capabilities to the ChunkedTimeChannel.
    Generates time-domain channel impulse response (CIR) after applying analog beamforming weights.
    """

    def __init__(
        self,
        channel_model,
        bandwidth,
        num_time_samples,
        tx_array,
        rx_array,
        num_tx_ports,
        num_rx_ports,
        l_min,
        l_max,
        normalize_channel=False,
        precision=None,
    ):
        super().__init__(
            channel_model=channel_model,
            bandwidth=bandwidth,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            normalize_channel=normalize_channel,
            precision=precision,
        )

        self.tx_array = tx_array
        self.rx_array = rx_array
        self.num_tx_ports = num_tx_ports
        self.num_rx_ports = num_rx_ports

        # Initialize Default Weights
        self._init_default_weights()

    def _init_default_weights(self):
        # Default: Identity mapping (first N elements)
        self.w_rf = tf.eye(
            self.tx_array.num_ant, num_columns=self.num_tx_ports, dtype=tf.complex64
        )
        self.a_rf = tf.eye(
            self.rx_array.num_ant, num_columns=self.num_rx_ports, dtype=tf.complex64
        )

    def set_analog_weights(self, w_rf, a_rf):
        self.w_rf = w_rf
        self.a_rf = a_rf

    def _apply_weights(self, h_elem, w_rf, a_rf):
        """
        Applies weights using Einsum to Time Domain Channel (CIR).
        h_elem: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_path, num_time_samples]
        w_rf  : [num_tx_ant, num_tx_ports] or broadcastable
        a_rf  : [num_rx_ant, num_rx_ports] or broadcastable

        Returns:
        h_port: [batch, num_rx, num_rx_ports, num_tx, num_tx_ports, num_path, num_time_samples]
        """
        # TX Beamforming
        # h_elem indices: b (batch), r (rx), u (rx_ant), t (tx), v (tx_ant), p (path), s (sample)
        # w_rf indices:   v (tx_ant), m (tx_port)
        # Target:         b, r, u, t, m, p, s

        # Determine equation based on w_rf rank
        if len(w_rf.shape) == 2:  # [v, m] - static weights for all
            eq_tx = "brutvps,vm->brutmps"
        elif len(w_rf.shape) == 3:  # [t, v, m] or [b, v, m]
            if w_rf.shape[0] == h_elem.shape[3]:  # matches num_tx
                eq_tx = "brutvps,tvm->brutmps"
            else:  # assumes [batch, v, m]
                eq_tx = "brutvps,bvm->brutmps"
        else:  # [b, t, v, m]
            eq_tx = "brutvps,btvm->brutmps"

        h_tx_bf = tf.einsum(eq_tx, h_elem, w_rf)

        # RX Beamforming
        # h_tx_bf indices: b, r, u, t, m, p, s
        # a_rf indices:    u (rx_ant), n (rx_port)
        # Target:          b, r, n, t, m, p, s

        if len(a_rf.shape) == 2:  # [u, n]
            eq_rx = "brutmps,un->brntmps"
        elif len(a_rf.shape) == 3:  # [r, u, n] or [b, u, n]
            if a_rf.shape[0] == h_elem.shape[1]:  # matches num_rx
                eq_rx = "brutmps,run->brntmps"
            else:  # assumes [batch, u, n]
                eq_rx = "brutmps,bun->brntmps"
        else:  # [b, r, u, n]
            eq_rx = "brutmps,brun->brntmps"

        h_port = tf.einsum(eq_rx, h_tx_bf, tf.math.conj(a_rf))
        return h_port

    def get_cir(self, batch_size=None):
        """
        Return the port-domain CIR.
        """
        # 1. Get physical channel CIR (Element domain) - calling parent
        # Parent (ChunkedTimeChannel) returns [b, r, u, t, v, p, s]
        h_elem = super().get_cir(batch_size)

        # 2. Apply Analog Beamforming
        h_port = self._apply_weights(h_elem, self.w_rf, self.a_rf)

        return h_port
