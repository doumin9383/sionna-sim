import tensorflow as tf
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

from sionna.phy.channel.tr38901 import UMi, PanelArray


class SpyUMi(UMi):
    def __call__(self, *args, **kwargs):
        print(f"SpyUMi called with args: {args}, kwargs: {kwargs}")

        # Call super, but we want to intercept what happens inside.
        # However, if the logic is monolithic, we might miss it.
        # Let's try to check attributes before and after.

        ret = super().__call__(*args, **kwargs)

        print("Super call finished.")
        # Check if any new attributes appeared
        # print(f"Dir after call: {dir(self)}")

        # Check specific internal attributes often used in Sionna
        # _cir_sampler might have been called.
        if hasattr(self, "_cir_sampler"):
            print("Checking _cir_sampler attributes...")
            # print(dir(self._cir_sampler))

            # In some versions, rays (angles) are in _cir_sampler.rays?
            # Or _cir_sampler.topology?
            pass

        return ret

    # Try to intercept potential step methods if they are defined in UMi or inherited
    # These names are guessed from previous dir() output on UMi
    def _step_11_field_matrix(self, *args, **kwargs):
        print("_step_11_field_matrix called!")
        # This method likely calculates the field (gains).
        # It probably takes angles as input?
        # Let's print args shapes/values
        # for i, arg in enumerate(args):
        #    print(f"Arg {i}: {type(arg)}")
        return super()._step_11_field_matrix(*args, **kwargs)


def inspect_spy():
    print("Inspecting SpyUMi...")

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

    model = SpyUMi(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=True,
        enable_shadow_fading=True,
    )

    # Dummy topology
    batch_size = 1
    num_ut = 1
    num_bs = 1
    ut_loc = tf.zeros([batch_size, num_ut, 3])
    bs_loc = tf.zeros([batch_size, num_bs, 3])
    ut_orient = tf.zeros([batch_size, num_ut, 3])
    bs_orient = tf.zeros([batch_size, num_bs, 3])
    ut_vel = tf.zeros([batch_size, num_ut, 3])
    in_state = tf.zeros([batch_size, num_ut], dtype=tf.bool)

    model.set_topology(ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state)

    print("Calling model...")
    a, tau = model(1, 15e3)
    print("Model called.")


if __name__ == "__main__":
    inspect_spy()
