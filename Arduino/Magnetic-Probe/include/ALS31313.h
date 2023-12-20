/*
    ALS31313.h

    Author: Joseph Sadden
    Date:   12th December, 2023

    This library...
*/

/*
    MIT License

    Copyright (c) 2023 Joseph Sadden

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef ALS31313_H
#define ALS31313_H

#ifdef __cplusplus
extern "C" {
#endif

/* +++ Begin Top-level Header Includes +++ */
#include <stdbool.h>
#include <stdint.h>
/* --- End Top-level Header Includes --- */

/* +++ Begin Local Header Includes +++ */
/* --- End Local Header Includes --- */
#include "common.h"
#include "I2CBus.h"

/* +++ Begin Library Macro Definitions +++ */

#define DEFAULT_ADDRESS 0b0000000   // ...
#define MINIMUM_ADDRESS 0b0000000
#define MAXIMUM_ADDRESS 0b1111111

#define SENSOR_READING_BYTE_LENGTH (1+sizeof(MagneticSensorReading_t))

/* --- End Library Macro Definitions --- */

/* +++ Begin Library Typedefs +++ */
typedef struct MagneticSensorReading_t {

    uint8_t I2CAddress;

    uint8_t LayerIndex: 4;
    uint8_t DeviceIndex: 4;

    uint16_t FieldX;
    uint16_t FieldY;
    uint16_t FieldZ;

    uint16_t Temperature;

    uint32_t Timestamp;

    size_t AsBytes(uint8_t* Buffer);
    void SetMeasurementTime(uint32_t Timestamp);

} MagneticSensorReading_t;

class ALS31313_t {

    uint8_t LayerIndex: 4;
    uint8_t DeviceIndex: 4;

    uint8_t I2CAddress;

    public:
        ALS31313_t(uint8_t LayerIndex, uint8_t DeviceIndex);

        bool AssertAddress(uint8_t I2CAddress, I2CBus_t& Bus);
        bool InitializeSettings(I2CBus_t& Bus);
        bool ReadMeasurements(MagneticSensorReading_t& CurrentReading, I2CBus_t& Bus) const;

    private:
        uint8_t ValidateI2CAddress(uint8_t Expected, I2CBus_t& Bus);
        bool EnableWriteAccess(I2CBus_t& Bus) const;
        bool SetI2CAddress(uint8_t NewAddress, I2CBus_t& Bus);
        uint32_t ReadRegister(uint8_t RegisterAddress) const;
        bool WriteRegister(uint8_t RegisterAddress, uint32_t Data) const;
};
/* --- End Library Typedefs --- */

/* +++ Begin Library Function Definitions +++ */
/* --- End Library Function Definitions --- */

/* +++ Begin Library Global Declarations +++ */
extern uint8_t SensorReading_Bytes[SENSOR_READING_BYTE_LENGTH];
/* --- End Library Global Declarations --- */

#ifdef __cplusplus
}
#endif


#endif
