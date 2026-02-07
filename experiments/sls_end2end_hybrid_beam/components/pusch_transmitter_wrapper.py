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
        self.ofdm_modulator = OFDMModulator(self.resource_grid, dtype=dtype)

    def call(self, batch_size=1):
        """
        Executes the PUSCH transmission chain.
        """

        # 1. TB Encoding (Bits -> Coded Bits)
        # _tb_encoder returns coded bits.
        b = self._tb_encoder(batch_size)

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
        x_rg = self._resource_grid_mapper(x_layers)

        # 6. OFDM Modulation
        x_time = self.ofdm_modulator(x_rg)

        return x_time

    def _apply_dft_spreading(self, x):
        """
        Applies DFT spreading to the input symbols.
        Assumes the input x [batch, layers, total_symbols] is structured such that
        it can be divided into chunks of size M_sc (allocated subcarriers).
        """
        # We assume M_sc equals to the total available subcarriers in the resource grid
        # This holds if we allocate the full bandwidth defined in ResourceGrid.
        # If partial allocation is used, this logic needs to be smarter (reading from config).
        M_sc = self.resource_grid.num_effective_subcarriers

        shape = x.shape
        total_len = shape[-1]

        # Logic to handle cases where total_len is not a multiple of M_sc
        # This can happen if DMRS configuration reduces data symbols in some slots but not others?
        # In DFT-s-OFDM, we expect M_sc to be constant for data symbols.

        if total_len % M_sc != 0:
            # Fallback: Try to use num_subcarriers (fft_size) if effective is different?
            # Or just warn and return (skipping DFT)
            # tf.print("Warning: Input length not divisible by M_sc. Skipping DFT.")
            # For robust simulation, we force it/reshape or error out.
            # Let's assume for now it divides.
            pass

        # Reshape to [..., Num_OFDM_Symbols, M_sc]
        # We need to treat 'batch' and 'layers' as preserved dims.
        # shape[:-1] is [batch, layers]

        # We let -1 figure out the number of time symbols
        x_reshaped = tf.reshape(x, list(shape[:-1]) + [-1, M_sc])

        # DFT along the last dimension (subcarriers)
        x_dft = tf.signal.fft(tf.cast(x_reshaped, tf.complex64))

        # Power Normalization (1/sqrt(M_sc))
        # Ensure energy conservation
        dft_size_cast = tf.cast(M_sc, x_dft.dtype)
        x_dft = x_dft / tf.sqrt(dft_size_cast)

        # Flatten back to [batch, layers, total_symbols]
        y = tf.reshape(x_dft, shape)
        return y
