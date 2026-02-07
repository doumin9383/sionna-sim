#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import tensorflow as tf
from sionna.phy.nr import PUSCHConfig, PUSCHReceiver, CarrierConfig
from .pusch_transmitter_wrapper import HybridPUSCHTransmitter
from .channel_models import GeneratHybridBeamformingTimeChannel

# from sionna.channel import ApplyTimeChannel # Check if available, or usetf.nn.convolution
from sionna.phy.channel import ApplyTimeChannel


class PUSCHCommunicationModel(tf.keras.Model):
    """
    Model for PUSCH Link Level Simulation.
    Integrates Configuration, Transmitter, Channel, and Receiver.
    """

    def __init__(
        self,
        carrier_frequency=3.5e9,
        subcarrier_spacing=30e3,
        num_tx_ant=4,
        num_rx_ant=4,
        num_layers=1,
        num_rb=50,
        domain="time",  # "time" or "freq"
        enable_transform_precoding=False,
        mcs_index=16,
        precoding_granularity=None,  # None, "Wideband", or int (in RBs)
        papr_oversampling_factor=4,
    ):
        super().__init__()

        # 1. Configuration
        carrier_config = CarrierConfig()
        carrier_config.subcarrier_spacing = subcarrier_spacing / 1e3
        carrier_config.carrier_frequency = carrier_frequency
        carrier_config.n_size_grid = num_rb  # Set bandwidth by RB count

        self.pusch_config = PUSCHConfig(carrier=carrier_config)
        self.pusch_config.output_domain = domain

        # Adjust config based on inputs
        # In Sionna NR, num_antenna_ports must match num_layers for PUSCHConfig
        # when using it to map layers to ports.
        # We will handle precoding (port to antenna) in our wrapper if needed.
        self.pusch_config.num_antenna_ports = num_layers
        self.pusch_config.num_layers = num_layers

        # Modulation / MCS
        self.pusch_config.mcs_index = mcs_index

        # Transform Precoding (DFT-s-OFDM)
        self.pusch_config.transform_precoding = False
        self.manual_transform_precoding = enable_transform_precoding

        # 2. Transmitter
        self.transmitter = HybridPUSCHTransmitter(
            self.pusch_config,
            enable_transform_precoding=self.manual_transform_precoding,
            output_domain=domain,
            num_tx_ant=num_tx_ant,
            precoding_granularity=precoding_granularity,
        )

        # 4. Receiver
        self.receiver = PUSCHReceiver(self.transmitter)

        self.papr_oversampling_factor = papr_oversampling_factor

    def call(self, batch_size=1):
        """
        Runs the full chain: Tx -> Channel -> Rx
        Returns: b, b_hat
        """
        # Tx
        x = self.transmitter(batch_size)

        # Channel (Bypass for now or Identity)
        # y = self.channel(x)
        y = x  # Identity channel for now

        # Rx
        b_hat = self.receiver(y)

        # Retrieve original bits (from transmitter's encoder? PUSCHTransmitter doesn't expose easily?)
        # Standard PUSCHReceiver returns LLRs.
        # We need transmitted bits for BER.
        # PUSCHTransmitter typically regenerates bits every call if not fixed?
        # Actually Sionna transmitters usually generate new random bits.
        # Unless we pass them? PUSCHTransmitter call() takes 'bits' argument?
        # Let's check signature. PUSCHTransmitter(pusch_config).call(batch_size, bits=None)

        return b_hat

    def generate_signal(self, batch_size=1):
        """
        Generates only the transmitted signal x.
        """
        return self.transmitter(batch_size)

    def compute_papr(self, x):
        """
        Computes PAPR of time-domain signal x.
        Applies oversampling to get accurate Peak.
        """
        # x shape: [batch, tx_ant, time_samples]

        if self.papr_oversampling_factor > 1:
            # Oversampling via FFT -> Zero Pad -> IFFT

            # 1. FFT
            x_f = tf.signal.fft(tf.cast(x, tf.complex64))

            # 2. Zero Padding
            # Split into positive/negative freq (fftshift style) or just insert in middle?
            # Standard FFT layout: DC, Pos Freqs, Neg Freqs.
            # We insert zeros in the "middle" (high frequencies).

            num_samples = x.shape[-1]
            num_zeros = num_samples * (self.papr_oversampling_factor - 1)

            # Axis is -1
            # Split at N/2
            n_div_2 = num_samples // 2

            x_f_left = x_f[..., :n_div_2]
            x_f_right = x_f[..., n_div_2:]

            zeros = tf.zeros(list(x.shape[:-1]) + [num_zeros], dtype=x_f.dtype)

            x_f_padded = tf.concat([x_f_left, zeros, x_f_right], axis=-1)

            # 3. IFFT
            x_oversampled = tf.signal.ifft(x_f_padded)

            # Scale adjustment? IFFT(Pad(FFT(x))) preserves energy/power?
            # Parseval: Energy is sum(|x|^2).
            # With padding, energy might change.
            # We care about Ratio (Peak/Avg), so global scaling cancels out?
            # Max power / Mean power.
            # If we scale signal by K, both Peak and Mean scale by K^2. PAPR invariant.
            # So scaling doesn't matter for PAPR.

            signal_for_papr = x_oversampled
        else:
            signal_for_papr = x

        # Compute PAPR
        # Power
        power = tf.abs(signal_for_papr) ** 2

        # Peak Power per antenna/batch
        peak_power = tf.reduce_max(power, axis=-1)

        # Average Power per antenna/batch
        mean_power = tf.reduce_mean(power, axis=-1)

        # PAPR Linear
        papr_linear = peak_power / mean_power

        # PAPR dB
        papr_db = 10.0 * tf.math.log(papr_linear) / tf.math.log(10.0)

        return papr_db
