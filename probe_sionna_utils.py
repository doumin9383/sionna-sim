import tensorflow as tf
import sionna.sys.utils
import inspect


def probe_sys_utils():
    print("TensorFlow Version:", tf.__version__)

    potential_funcs = ["spread_across_subcarriers", "get_pathloss"]

    for func_name in potential_funcs:
        if hasattr(sionna.sys.utils, func_name):
            func = getattr(sionna.sys.utils, func_name)
            print(f"\n[Found {func_name}]")
            try:
                sig = inspect.signature(func)
                print(f"Signature: {sig}")
                print("\nDocstring:")
                print(func.__doc__)

                # Retrieve source code if possible
                try:
                    source = inspect.getsource(func)
                    print("\nSource Code:")
                    print(source)
                except Exception as e:
                    print(f"Could not get source: {e}")

            except Exception as e:
                print(f"Error inspecting {func_name}: {e}")
        else:
            print(f"\n[{func_name} NOT found in sionna.sys.utils]")


if __name__ == "__main__":
    try:
        probe_sys_utils()
    except Exception as e:
        print(f"Error: {e}")
