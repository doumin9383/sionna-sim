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
        dtype=tf.complex64,
        **kwargs,
    ):
        """
        Args:
            pusch_config (PUSCHConfig): The PUSCH configuration.
            enable_transform_precoding (bool): If True, applies DFT spreading (DFTS-OFDM).
                                               Make sure pusch_config.transform_precoding is False (CP-OFDM)
                                               to avoid double processing or MCS table mismatches.
            dtype (tf.DType): Complex dtype for processing.
        """
        super().__init__(pusch_config, dtype=dtype, **kwargs)
        self.enable_transform_precoding = enable_transform_precoding

        # Instantiate OFDM Modulator as it's not exposed as a public attribute in PUSCHTransmitter
        # We rely on self.resource_grid which is initialized by PUSCHTransmitter
        # OFDMModulator takes cyclic_prefix_length as argument, not resource_grid
        self.ofdm_modulator = OFDMModulator(
            self.resource_grid.cyclic_prefix_length, dtype=dtype
        )

    def call(self, batch_size=1, bits=None):
        """
        Executes the PUSCH transmission chain.
        """

        # 1. TB Encoding (Bits -> Coded Bits)
        # _tb_encoder returns coded bits.
        if bits is None:
            # Generate random bits if not provided
            # TB Size is determined by config
            # We access tb_size from the PUSCHConfig object which is stored as self._pusch_config
            tb_size = self._pusch_config.tb_size
            bits = self._binary_source([batch_size, tb_size])

        b = self._tb_encoder(bits)

        # 2. Modulation (Coded Bits -> Symbols)
        c = self._mapper(b)

        # 3. Layer Mapping (Symbols -> Layers)
        # Output shape: [batch_size, num_layers, num_symbols_per_layer]
        x_layers = self._layer_mapper(c)

        # 4. Transform Precoding (DFT Spreading) - OPTIONAL
        if self.enable_transform_precoding:
            x_layers = self._apply_dft_spreading(x_layers)

        # 5. Resource Grid Mapping
        # Inputs: x_layers (symbols)
        # Outputs: Resource Grid [batch, tx, num_ofdm_symbols, fft_size]
        # This inserts DMRS as well.

        # Check if x_layers needs reshaping to Rank 4 for ResourceGridMapper (Sionna v0.14+ quirks)
        # We need the flattened update vector to be [N, 1] to match Output [..., 1].
        # flatten_last_dims(inputs, 3) followed by transpose requires inputs to be 4D [B, 1, 1, N]
        # to produce [B, N] -> [N, B] where B=1.
        # So we reshape to [batch, 1, 1, -1] assuming single stream/batch=1 flow.
        x_layers_reshaped = tf.reshape(x_layers, [batch_size, 1, 1, -1])

        x_rg = self._resource_grid_mapper(x_layers_reshaped)

        # 6. OFDM Modulation
        x_time = self.ofdm_modulator(x_rg)

        return x_time

    def _apply_dft_spreading(self, x):
        """
        Applies DFT spreading to the input symbols.
        Assumes the input x [batch, layers, total_symbols] is structured such that
        it can be divided into chunks of size M_sc (allocated subcarriers).
        """
        M_sc = self.resource_grid.num_effective_subcarriers

        shape = x.shape
        total_len = shape[-1]

        if total_len % M_sc != 0:
            pass

        # Reshape to [..., Num_OFDM_Symbols, M_sc]
        x_reshaped = tf.reshape(x, list(shape[:-1]) + [-1, M_sc])

        # DFT along the last dimension (subcarriers)
        x_dft = tf.signal.fft(tf.cast(x_reshaped, tf.complex64))

        # Power Normalization (1/sqrt(M_sc))
        dft_size_cast = tf.cast(M_sc, x_dft.dtype)
        x_dft = x_dft / tf.sqrt(dft_size_cast)

        # Flatten back to [batch, layers, total_symbols]
        y = tf.reshape(x_dft, shape)
        return y
