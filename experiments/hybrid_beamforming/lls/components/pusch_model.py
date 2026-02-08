#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import tensorflow as tf
from sionna.phy.nr import PUSCHConfig, PUSCHReceiver, CarrierConfig
from .pusch_transmitter_wrapper import HybridPUSCHTransmitter
from experiments.hybrid_beamforming.shared.channel_models import (
    GeneratHybridBeamformingTimeChannel,
)

# from sionna.channel import ApplyTimeChannel # Check if available, or usetf.nn.convolution
from sionna.phy.channel import ApplyTimeChannel


class PUSCHCommunicationModel(tf.keras.Model):
    """
    Model for PUSCH Link Level Simulation.
    Integrates Configuration, Transmitter, Channel, and Receiver.
    """

    def __init__(
        self,
        config,
        num_layers,
        enable_transform_precoding,
        precoding_granularity,
        domain="time",
    ):
        super().__init__()

        # 1. Configuration
        self.pusch_config = PUSCHConfig(
            carrier=config.carrier_config, tb_config=config.tb_config
        )
        self.pusch_config.output_domain = domain

        # Adjust config based on inputs
        # In Sionna NR, num_antenna_ports must match num_layers for PUSCHConfig
        # when using it to map layers to ports.
        # We will handle precoding (port to antenna) in our wrapper if needed.
        self.pusch_config.num_antenna_ports = config.num_ut_ant
        self.pusch_config.num_layers = num_layers
        self.pusch_config.precoding = "codebook"
        self.pusch_config.tpmi = 1

        # Transform Precoding (DFT-s-OFDM)
        self.pusch_config.transform_precoding = False
        self.manual_transform_precoding = enable_transform_precoding

        # 2. Transmitter
        self.transmitter = HybridPUSCHTransmitter(
            self.pusch_config,
            enable_transform_precoding=self.manual_transform_precoding,
            output_domain=domain,
            num_tx_ant=config.num_ut_ant,
            precoding_granularity=precoding_granularity,
            rbg_size_rb=config.rbg_size_rb,
        )

        # 4. Receiver
        self.receiver = PUSCHReceiver(self.transmitter)

        self.papr_oversampling_factor = config.papr_oversampling_factor

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

    def compute_papr(self, x, exclude_dmrs=False):
        """
        Computes PAPR of time-domain signal x.
        Applies oversampling to get accurate Peak.

        Args:
            x: Time domain signal [batch, tx_ant, time_samples]
            exclude_dmrs: If True, excludes time samples corresponding to DMRS symbols.
        """
        # x shape: [batch, tx_ant, time_samples]

        # 1. Apply Oversampling (Global FFT interpolation)
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
            # As noted before, scaling cancels out for PAPR.

            signal_for_papr = x_oversampled

            # Update scaling factor for indices
            scale_factor = self.papr_oversampling_factor
        else:
            signal_for_papr = x
            scale_factor = 1

        # 2. Exclude DMRS symbols if requested
        if exclude_dmrs:
            # Get DMRS indices
            dmrs_indices = self.pusch_config.dmrs_symbol_indices

            # Get Symbol Structure
            rg = self.transmitter.resource_grid
            fft_size = rg.fft_size
            cp_len = rg.cyclic_prefix_length
            num_symbols = rg.num_ofdm_symbols

            # Handle cp_len (scalar or list)
            # In Sionna ResourceGrid, it's typically int, but we handle list just in case
            # if we can access checking it, but ResourceGrid forces int.
            # However, let's assume it could be iterable if some custom config.
            if isinstance(cp_len, int):
                cp_lens = [cp_len] * num_symbols
            else:
                # Assume iterable
                cp_lens = list(cp_len)

            filtered_samples = []

            current_idx = 0

            # Iterate through symbols in the time domain
            # We need to slice "signal_for_papr" which is scaled by 'scale_factor'
            # The indices in original 'x' are:
            # Symbol 0: [0 : fft + cp[0]]
            # Symbol 1: [end_0 : end_0 + fft + cp[1]]
            # etc.
            # In 'signal_for_papr', these are multiplied by 'scale_factor'.

            # We slice along the last axis (time)
            # signal_for_papr shape: [batch, tx, time]

            for i in range(num_symbols):
                sym_len_orig = fft_size + cp_lens[i]
                sym_len_over = sym_len_orig * scale_factor

                start_idx = current_idx
                end_idx = current_idx + sym_len_over

                # Verify we don't exceed bounds (due to standard rounding?)
                # integer math should be exact here if strict integer oversampling.

                if i not in dmrs_indices:
                    # Keep this symbol (Data)
                    slice_data = signal_for_papr[..., start_idx:end_idx]
                    filtered_samples.append(slice_data)

                current_idx = end_idx

            if not filtered_samples:
                # Fallback if everything is DMRS (unlikely)
                print("Warning: All symbols excluded as DMRS? Using full signal.")
            else:
                # Concatenate along time axis
                signal_for_papr = tf.concat(filtered_samples, axis=-1)

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
