import tensorflow as tf
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

from sionna.phy.channel.tr38901 import UMi, PanelArray


def inspect_umi():
    print("Inspecting UMi internals...")

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

    model = UMi(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=True,
        enable_shadow_fading=True,
    )

    print("Model instantiated.")
    print(f"Type: {type(model)}")
    print(f"Dir(model): {dir(model)}")

    # Check for angle-related attributes
    attributes = dir(model)
    potential_angle_attrs = [
        attr
        for attr in attributes
        if "angle" in attr
        or "deg" in attr
        or "doa" in attr
        or "dod" in attr
        or "_params" in attr
        or "cir" in attr
    ]
    print(f"Potential angle attributes: {potential_angle_attrs}")

    # Inspect the __call__ signature
    import inspect

    print(f"Signature of __call__: {inspect.signature(model.__call__)}")

    # Dummy topology
    batch_size = 2
    num_ut = 2
    num_bs = 1
    ut_loc = tf.zeros([batch_size, num_ut, 3])
    bs_loc = tf.zeros([batch_size, num_bs, 3])
    ut_orient = tf.zeros([batch_size, num_ut, 3])
    bs_orient = tf.zeros([batch_size, num_bs, 3])
    ut_vel = tf.zeros([batch_size, num_ut, 3])
    in_state = tf.zeros([batch_size, num_ut], dtype=tf.bool)

    model.set_topology(ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state)

    print(
        "Topology set. attempting call without arguments first (if default exists)..."
    )
    try:
        ret = model()
        print("Call model() success.")
    except Exception as e:
        print(f"Call model() failed: {e}")
        try:
            # Try with arguments from GenerateOFDMChannel usage
            ret = model(1, 15e3)  # num_time_steps, sampling_frequency
            print("Call model(1, 15e3) success.")
        except Exception as e2:
            print(f"Call model(1, 15e3) failed: {e2}")

            try:
                # Check if it is num_samples instead of num_time_steps?
                ret = model(1)
                print("Call model(1) success.")
            except Exception as e3:
                print(f"Call model(1) failed: {e3}")

    # After calling (or attempting), check if internal state populated
    if hasattr(model, "_cir_sampler"):
        print("Inspecting _cir_sampler...")
        print(dir(model._cir_sampler))

        # Assuming _cir_sampler has something like 'params' or 'rays'


if __name__ == "__main__":
    inspect_umi()
