import sys
import os
import inspect
from sionna.phy.ofdm import OFDMModulator

print("OFDMModulator init signature:")
print(inspect.signature(OFDMModulator.__init__))
print("OFDMModulator docstring:")
print(OFDMModulator.__init__.__doc__)
