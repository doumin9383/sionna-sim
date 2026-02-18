import tensorflow as tf
import sionna.sys
import inspect


def probe_dl_power_control():
    print("TensorFlow Version:", tf.__version__)

    # Check if downlink_fair_power_control exists in sionna.sys
    if hasattr(sionna.sys, "downlink_fair_power_control"):
        func = sionna.sys.downlink_fair_power_control
        print("\n[Found downlink_fair_power_control]")
        try:
            sig = inspect.signature(func)
            print(f"Signature: {sig}")
            print("\nDocstring:")
            print(func.__doc__)
        except Exception as e:
            print(f"Error inspecting signature: {e}")

    else:
        print("\n[downlink_fair_power_control NOT found in sionna.sys]")


if __name__ == "__main__":
    try:
        probe_dl_power_control()
    except Exception as e:
        print(f"Error: {e}")
