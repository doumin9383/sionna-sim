import inspect
from sionna.phy.channel.tr38901 import UMa


def check():
    print(f"UMa.__call__ signature: {inspect.signature(UMa.__call__)}")


if __name__ == "__main__":
    check()
