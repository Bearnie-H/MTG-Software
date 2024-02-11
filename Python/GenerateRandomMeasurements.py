#!/usr/bin/env python3

import time
import random
import sys

def RandomReading() -> str:

    LayerIndex_Bits: int = 3
    DeviceIndex_Bits: int = 3
    MagneticField_Bits: int = 12
    TemperatureField_Bits: int = 12
    Timestamp_Bits: int = 32

    LayerIndex: int = random.randint(0, (2**LayerIndex_Bits)-1)
    DeviceIndex: int = random.randint(0, (2**DeviceIndex_Bits)-1)
    I2C_Address: int = LayerIndex * 2**4 + DeviceIndex

    FieldX: int = random.randint(-(2**(MagneticField_Bits-1)), (2**(MagneticField_Bits-1))-1)
    FieldY: int = random.randint(-(2**(MagneticField_Bits-1)), (2**(MagneticField_Bits-1))-1)
    FieldZ: int = random.randint(-(2**(MagneticField_Bits-1)), (2**(MagneticField_Bits-1))-1)
    Temperature: int = random.randint(0, (2**TemperatureField_Bits)-1)
    Timestamp: int = time.time_ns() % (2**Timestamp_Bits)

    return f"{I2C_Address:3d},{LayerIndex},{DeviceIndex},{FieldX:5d},{FieldY:5d},{FieldZ:5d},{Temperature:4d},{Timestamp}"

if ( len(sys.argv) > 1 ):
    for i in range(int(sys.argv[1])):
        print(RandomReading())
else:
    print(RandomReading())
