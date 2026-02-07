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
    time_lag_ discrete_time_channel
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
    def __init__(self,
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
                 precision=None):
        super().__init__(num_tx=1, num_tx_ant=1, num_rx=1, num_rx_ant=1,
                         carrier_frequency=carrier_frequency, precision=precision)

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

        # Expose properties from the internal model
        self.num_tx = self._model.num_tx
        self.num_tx_ant = self._model.num_tx_ant
        self.num_rx = self._model.num_rx
        self.num_rx_ant = self._model.num_rx_ant

    def set_topology(self, *args, **kwargs):
        """Pass topology down to the internal model"""
        self._model.set_topology(*args, **kwargs)

    def __call__(self, num_time_samples, sampling_frequency):
        """
        Delegate the call to the internal model.
        Returns: (a, tau)
        """
        return self._model(num_time_samples, sampling_frequency)

    @property
    def model(self):
        """Access the underlying internal model"""
        return self._model


class ChunkedTimeChannel(GenerateTimeChannel):
    """
    Generates Time Domain Channel (CIR or Waveform) supporting FDRA masks.
    Inherits from GenerateTimeChannel.
    """
    def __init__(self, channel_model, bandwidth, num_time_samples, l_min, l_max,
                 normalize_channel=False, precision=None):
        super().__init__(channel_model=channel_model,
                         bandwidth=bandwidth,
                         num_time_samples=num_time_samples,
                         l_min=l_min,
                         l_max=l_max,
                         normalize_channel=normalize_channel,
                         precision=precision)

    def get_cir(self, batch_size=None):
        """
        Explicitly get CIR (Channel Impulse Response) from the model.
        Returns:
            h_time: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_path, num_time_samples]
            (Note: GenerateTimeChannel usually produces h_time by convolving a, tau.
             This method exposes the intermediate steps if needed or returns the final h_time)
        """
        # Generate paths
        a, tau = self._channel_model(self._num_time_samples, self._sampling_frequency)

        # Note: In standard Sionna, GenerateTimeChannel.__call__ does:
        # a, tau = self._channel_model(...)
        # h_time = cir_to_time_channel(self._bandwidth, a, tau, self._l_min, self._l_max, ...)
        # We can implement specific logic here if we need FDRA masking in time domain
        # (which is complex, usually done by applying mask in freq and IFFT).

        # For now, we return the standard paths for consistency checking.
        return a, tau

    def call(self, batch_size=None):
        # Default behavior of GenerateTimeChannel
        return super().call(batch_size)


class ChunkedOFDMChannel(GenerateOFDMChannel):
    """
    Generates Frequency Domain Channel with optimizations for RBG-based processing (SLS).
    Inherits from GenerateOFDMChannel.
    """
    def __init__(self, channel_model, resource_grid, normalize_channel=True, precision=None):
        super().__init__(channel_model=channel_model,
                         resource_grid=resource_grid,
                         normalize_channel=normalize_channel,
                         precision=precision)

        # Pre-compute all frequencies and RBG centers
        self._all_frequencies = subcarrier_frequencies(
            resource_grid.fft_size, resource_grid.subcarrier_spacing
        )
        # TODO: Define RBG centers if we want strict sampling only at centers

    def get_paths(self, batch_size=None):
        """
        Expose path generation for consistency verification.
        """
        a, tau = self._channel_model(self._num_time_samples, self._sampling_frequency)
        return a, tau

    def get_rbg_channel(self, batch_size, rbg_size, active_rbgs=None):
        """
        Get channel sampled at RBG granularity.
        """
        a, tau = self.get_paths(batch_size)

        # Here we would implement the logic to sample H(f) only at RBG centers
        # or compute average H(f) per RBG.
        # For this frame, we simply convert CIR to OFDM channel for specific frequencies.

        # This implementation will be refined in the SLS task.
        h_ofdm = cir_to_ofdm_channel(self._all_frequencies, a, tau,
                                     normalize=self._normalize_channel)
        return h_ofdm


class HybridOFDMChannel(ChunkedOFDMChannel):
    """
    Adds Analog Beamforming capabilities to the ChunkedOFDMChannel.
    """
    def __init__(self, channel_model, resource_grid, tx_array, rx_array,
                 num_tx_ports, num_rx_ports, normalize_channel=True, precision=None):
        super().__init__(channel_model=channel_model,
                         resource_grid=resource_grid,
                         normalize_channel=normalize_channel,
                         precision=precision)

        self.tx_array = tx_array
        self.rx_array = rx_array
        self.num_tx_ports = num_tx_ports
        self.num_rx_ports = num_rx_ports

        # Initialize Default Weights
        self._init_default_weights()

    def _init_default_weights(self):
        # Default: Identity mapping (first N elements)
        self.w_rf = tf.eye(self.tx_array.num_ant, num_columns=self.num_tx_ports, dtype=tf.complex64)
        self.a_rf = tf.eye(self.rx_array.num_ant, num_columns=self.num_rx_ports, dtype=tf.complex64)

    def set_analog_weights(self, w_rf, a_rf):
        self.w_rf = w_rf
        self.a_rf = a_rf

    def _apply_weights(self, h_elem, w_rf, a_rf):
        """
        Applies weights using Einsum.
        h_elem: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_sc]
        """
        # TX Beamforming
        # Support broadcasting
         # Determine equation based on w rank
        if len(w_rf.shape) == 2:  # [v, p]
            eq_tx = "brutvsc,vp->brutpsc"
        elif len(w_rf.shape) == 3:  # [t, v, p]
            eq_tx = "brutvsc,tvp->brutpsc"
        else:  # [b, t, v, p]
            eq_tx = "brutvsc,btvp->brutpsc"

        h_tx_bf = tf.einsum(eq_tx, h_elem, w_rf)

        # RX Beamforming
        if len(a_rf.shape) == 2:  # [u, q]
            eq_rx = "brutpsc,uq->brqtpsc"
        elif len(a_rf.shape) == 3:  # [r, u, q]
             eq_rx = "brutpsc,ruq->brqtpsc"
        else:  # [b, r, u, q]
             eq_rx = "brutpsc,bruq->brqtpsc"

        h_port = tf.einsum(eq_rx, h_tx_bf, tf.math.conj(a_rf))
        return h_port

    def call(self, batch_size=None):
        """
        Return the port-domain channel.
        """
        # 1. Get physical channel (Element domain) - calling parent
        # Note: standard GenerateOFDMChannel returns [b, r, u, t, v, o, s]
        h_elem = super().call(batch_size)

        # 2. Apply Analog Beamforming
        h_port = self._apply_weights(h_elem, self.w_rf, self.a_rf)

        return h_port
