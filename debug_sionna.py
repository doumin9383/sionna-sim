import sionna
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel.tr38901 import UMa, PanelArray


def inspect_channel():
    # print(f"Sionna version: {sionna.__version__}")

    # Setup dummy
    rg = ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=72,
        subcarrier_spacing=30e3,
        num_tx=1,
        num_streams_per_tx=1,
    )
    bs = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        antenna_pattern="omni",
        carrier_frequency=3.5e9,
    )
    ut = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        antenna_pattern="omni",
        carrier_frequency=3.5e9,
    )
    model = UMa(
        carrier_frequency=3.5e9,
        o2i_model="low",
        ut_array=ut,
        bs_array=bs,
        direction="downlink",
    )

    goc = GenerateOFDMChannel(model, rg)

    print("GenerateOFDMChannel attributes:")
    print(dir(goc))

    if hasattr(goc, "_channel_model"):
        print("Has _channel_model")
    else:
        print("No _channel_model")

    if hasattr(goc, "channel_model"):
        print("Has channel_model")

    # Check sampling params
    if hasattr(goc, "_num_time_samples"):
        print("Has _num_time_samples")
    else:
        print("No _num_time_samples")


if __name__ == "__main__":
    inspect_channel()
