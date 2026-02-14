import tensorflow as tf
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import ResourceGrid

# Setup dummy environment
rg = ResourceGrid(num_ofdm_symbols=14, fft_size=64, subcarrier_spacing=15e3)


def dummy_channel_model(num_time_samples, sampling_frequency):
    return tf.zeros([1, 1, 1, 1, 14]), tf.zeros([1, 1, 1, 1])


gc = GenerateOFDMChannel(dummy_channel_model, rg)
print("Attributes of GenerateOFDMChannel:")
for attr in dir(gc):
    if not attr.startswith("__"):
        print(f"  {attr}")
