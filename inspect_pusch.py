import tensorflow as tf
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter

try:
    pusch_config = PUSCHConfig()
    transmitter = PUSCHTransmitter(pusch_config)
    print("PUSCHTransmitter attributes:")
    print(dir(transmitter))
    print("\nChecking for specific sub-components:")
    components = ["encoder", "mapper", "resource_grid_mapper", "ofdm_modulator"]
    for comp in components:
        if hasattr(transmitter, comp):
            print(f"Has {comp}")
            # Try to see if it's a callable or object
            attr = getattr(transmitter, comp)
            print(f"  Type: {type(attr)}")
        else:
            print(f"Does NOT have {comp}")

except Exception as e:
    print(f"Error: {e}")
