#!/usr/bin/env python3

import time
import random
import sys

def RandomReading() -> str:
    LayerIndex: int = random.randint(0, 7)
    DeviceIndex: int = random.randint(0, 7)
    I2C_Address: int = LayerIndex * 2**4 + DeviceIndex

    FieldX: int = random.randint(-2048, 2047)
    FieldY: int = random.randint(-2048, 2047)
    FieldZ: int = random.randint(-2048, 2047)
    Temperature: int = random.randint(0, 4095)
    Timestamp: int = time.time_ns() % 2**32

    return f"{I2C_Address},{LayerIndex},{DeviceIndex},{FieldX},{FieldY},{FieldZ},{Temperature},{Timestamp}"

if ( len(sys.argv) > 1 ):
    Count: int = int(sys.argv[1])
    for i in range(Count):
        print(RandomReading())

else:
    print(RandomReading())
