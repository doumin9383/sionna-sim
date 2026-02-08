#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import tensorflow as tf
from sionna.phy.nr import PUSCHTransmitter
from sionna.phy.ofdm import OFDMModulator


class HybridPUSCHTransmitter(PUSCHTransmitter):
    """
    Wraps standard PUSCHTransmitter to provide manual control over Transform Precoding (DFT-s-OFDM).

    This class inherits from PUSCHTransmitter and overrides the `call` method to:
    1. Utilize the internal components (_tb_encoder, _mapper, _layer_mapper) derived from PUSCHConfig.
    2. Inject a manual DFT spreading step if `enable_transform_precoding` is True.
    3. Map the (spread) symbols to the Resource Grid.
    4. Perform OFDM Modulation.

    This allows bypassing Sionna's internal Transform Precoding limitations while maintaining
    compatibility with PUSCHConfig and other Sionna components.
    """

    def __init__(
        self,
        pusch_config,
        enable_transform_precoding=False,
        output_domain="time",
        num_tx_ant=4,
        precoding_granularity=None,
        **kwargs,
    ):
        super().__init__(pusch_config, output_domain=output_domain, **kwargs)
        self.enable_transform_precoding = enable_transform_precoding
        self.num_tx_ant = max(
            num_tx_ant, self._num_layers
        )  # Ensure enough antennas for layers
        self.precoding_granularity = precoding_granularity

    def call(self, batch_size=1):
        """
        Executes the PUSCH transmission chain.
        """

        # 1. Generate Bits and Encode (Bits -> Coded Bits)
        b = self._binary_source([batch_size, self._num_tx, self._tb_size])
        c = self._tb_encoder(b)

        # 2. Modulation (Coded Bits -> Symbols)
        x_symbols = self._mapper(c)

        # 3. Layer Mapping (Symbols -> Layers)
        x_layers = self._layer_mapper(x_symbols)

        # 4. Transform Precoding (DFT Spreading) - OPTIONAL
        if self.enable_transform_precoding:
            x_layers = self._apply_dft_spreading(x_layers)

        # 5. Resource Grid Mapping
        # Inputs: x_layers (symbols) [batch, layers, symbols]
        # Outputs: Resource Grid [batch, layers, num_ofdm_symbols, subcarriers]
        x_rg = self._resource_grid_mapper(x_layers)

        # 5.5 Precoding (Mapping layers/ports to antennas)
        # If num_tx_ant > layers or granularity is specified, we apply manual precoding
        x_rg_precoded = self._apply_precoding(x_rg)

        # 6. OFDM Modulation
        # New modulator for the correct number of tx antennas
        # PUSCHTransmitter's modulator is bound to its resource_grid_mapper output ports.
        # We might need to use the OFDMModulator directly or adjust.
        # However, _ofdm_modulator in Sionna is just an OFDMModulator(self.resource_grid).
        # It takes [batch, ports, symbols, subcarriers].
        # If we change ports from layers to num_tx_ant, we need it to handle that.

        # In Sionna, OFDMModulator just does IFFT on the last dimension and CP.
        # It doesn't care about the number of 'ports' as long as it's the second-to-last dim?
        # Actually it's [batch, tx_ant, num_symbols, num_samples].
        x_time = self._ofdm_modulator(x_rg_precoded)

        return x_time

    def _apply_precoding(self, x_rg):
        """
        Applies frequency-selective precoding to the resource grid.
        x_rg shape: [batch, num_layers, num_ofdm_symbols, num_subcarriers]
        """
        if len(x_rg.shape) == 5 and x_rg.shape[1] == 1:
            x_rg = tf.squeeze(x_rg, axis=1)

        batch_size = tf.shape(x_rg)[0]
        num_layers = self._num_layers
        num_symbols = tf.shape(x_rg)[2]
        num_subcarriers = tf.shape(x_rg)[3]

        # Determine granularity in subcarriers
        if self.precoding_granularity == "Narrowband":
            g_sc = 1
        elif (
            self.precoding_granularity == "Wideband"
            or self.precoding_granularity is None
        ):
            g_sc = num_subcarriers
        else:
            g_sc = int(self.precoding_granularity) * 12  # Assume RB=12 subcarriers

        num_blocks = (num_subcarriers + g_sc - 1) // g_sc

        # Generate precoding matrices: [batch, num_blocks, num_tx_ant, num_layers]
        W_blocks = self._generate_precoder_matrices(batch_size, num_blocks)

        # Repeat/Tile W_blocks to cover all subcarriers
        # We need to handle the last block potentially being smaller
        W_list = []
        for i in range(num_blocks):
            block_size = g_sc
            if (i + 1) * g_sc > num_subcarriers:
                block_size = num_subcarriers - i * g_sc

            # W_blocks[:, i, :, :] is [batch, tx, layers]
            # Tile to [batch, block_size, tx, layers]
            W_tile = tf.tile(
                tf.expand_dims(W_blocks[:, i, :, :], 1), [1, block_size, 1, 1]
            )
            W_list.append(W_tile)

        W_sc = tf.concat(W_list, axis=1)  # [batch, num_subcarriers, tx, layers]

        # x_rg is [batch, layers, symbols, subcarriers]
        # Transpose to [batch, symbols, subcarriers, layers]
        x_rg_t = tf.transpose(x_rg, [0, 2, 3, 1])

        # Multiply: [batch, symbols, subcarriers, tx]
        # x_rg_t indices: b (batch), s (symbols), f (subcarriers), k (layers)
        # W_sc indices: b (batch), f (subcarriers), j (tx_ant), k (layers)
        # Result indices: b, s, f, j
        # Contraction over k (layers).
        x_precoded = tf.einsum("bsfk, bfjk -> bsfj", x_rg_t, W_sc)

        # Transpose back to [batch, num_tx_ant, num_symbols, num_subcarriers]
        y = tf.transpose(x_precoded, [0, 3, 1, 2])

        return y

    def _generate_precoder_matrices(self, batch_size, num_blocks):
        """
        Generates random unitary precoding matrices.
        """
        shape = [batch_size, num_blocks, self.num_tx_ant, self.num_tx_ant]
        # Random complex
        real = tf.random.normal(shape)
        imag = tf.random.normal(shape)
        z = tf.complex(real, imag)

        # QR decomposition to get unitary
        q, _ = tf.linalg.qr(z)

        # Take first num_layers columns
        W = q[..., : self._num_layers]
        return W

    def _apply_dft_spreading(self, x):
        """
        Applies DFT spreading to the input symbols.
        Transform precoding in NR is applied per OFDM symbol.
        """
        # M_sc is the number of subcarriers per OFDM symbol allocated for this PUSCH
        # num_subcarriers is total length. We need the data-carrying subcarriers.
        M_sc = (
            self.resource_grid.num_effective_subcarriers
            // self.resource_grid.num_ofdm_symbols
        )

        shape = x.shape
        total_len = shape[-1]

        # Reshape to [batch, layers, num_ofdm_symbols_with_data, M_sc]
        x_reshaped = tf.reshape(x, list(shape[:-1]) + [-1, M_sc])

        # DFT along the last dimension (subcarriers)
        x_dft = tf.signal.fft(tf.cast(x_reshaped, tf.complex64))

        # Power Normalization (1/sqrt(M_sc))
        dft_size_cast = tf.cast(M_sc, x_dft.dtype)
        x_dft = x_dft / tf.sqrt(dft_size_cast)

        # Flatten back
        y = tf.reshape(x_dft, shape)
        return y
