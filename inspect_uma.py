import tensorflow as tf
from sionna.phy.channel.tr38901 import UMa, PanelArray


def inspect():
    carrier_frequency = 3.5e9
    bs_array = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=4,
        num_cols_per_panel=4,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency,
    )
    ut_array = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )
    uma = UMa(
        carrier_frequency=carrier_frequency,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
    )

    ut_loc = tf.random.uniform([1, 1, 3])
    bs_loc = tf.random.uniform([1, 1, 3])
    uma.set_topology(ut_loc, bs_loc)

    print("--- Attributes containing 'loc' ---")
    for attr in dir(uma):
        if "loc" in attr:
            print(attr)


if __name__ == "__main__":
    inspect()
