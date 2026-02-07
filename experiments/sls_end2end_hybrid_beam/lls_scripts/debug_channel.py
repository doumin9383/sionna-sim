import sys
import os

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

try:
    from experiments.sls_end2end_hybrid_beam.components import channel_models

    print("Import successful!")
    print(dir(channel_models))
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
