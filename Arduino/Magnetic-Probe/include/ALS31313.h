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

#define SENSOR_BIT_DEPTH 12

#define SENSOR_READING_BYTE_LENGTH (2 + sizeof(MagneticSensorReading_t))

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

} MagneticSensorReading_t;

typedef enum I2C_ADC_Status_t: uint32_t {
    I2C_ADC_Enable  = 0b0,
    I2C_ADC_Disable = 0b1,
} I2C_ADC_Status_t;

typedef enum Bandwidth_Select_Mode_t: uint32_t {
    Bandwidth_Select_LowSpeed_WithFilter         = 0b000,
    Bandwidth_Select_MediumSpeed_WithFilter      = 0b001,
    Bandwidth_Select_HighSpeed_WithFilter        = 0b010,
    Bandwidth_Select_LowSpeed_WithoutFilter      = 0b100,
    Bandwidth_Select_MediumSpeed_WithoutFilter   = 0b101,
    Bandwidth_Select_HighSpeed_WithoutFilter     = 0b110,
} Bandwidth_Select_Mode_t;

typedef enum Hall_Effect_Mode_t: uint32_t {
    Hall_Effect_Mode_SingleEnded     = 0b00,
    Hall_Effect_Mode_Differential    = 0b01,
    Hall_Effect_Mode_CommonMode      = 0b10,
    Hall_Effect_Mode_AlternatingMode = 0b11,
} Hall_Effect_Mode_t;

typedef enum I2C_CRC_Mode_t: uint32_t {
    I2C_CRC_Disabled = 0b0,
    I2C_CRC_Enabled  = 0b1,
} I2C_CRC_Mode_t;

typedef enum I2C_Voltage_Threshold_t: uint32_t {
    I2C_Voltage_Threshold_High_3V_Mode = 0b0,
    I2C_Voltage_Threshold_Low_1V8_Mode = 0b1,
} I2C_Voltage_Threshold_t;

typedef enum Axis_Enable_Status_t: uint32_t {
    Axis_Enable____ = 0b000,
    Axis_Enable_X__ = 0b001,
    Axis_Enable__Y_ = 0b010,
    Axis_Enable_XY_ = 0b011,
    Axis_Enable___Z = 0b100,
    Axis_Enable_X_Z = 0b101,
    Axis_Enable__YZ = 0b110,
    Axis_Enable_XYZ = 0b111,
} Axis_Enable_Status_t;

typedef enum I2C_Loop_Mode_t: uint32_t {
    I2C_Loop_Mode_Single   = 0b00,
    I2C_Loop_Mode_FastLoop = 0b01,
    I2C_Loop_Mode_FullLoop = 0b10,
} I2C_Loop_Mode_t;

typedef enum Sensor_Power_Mode_t: uint32_t {
    Sensor_Power_Mode_Standard = 0b00,
    Sensor_Power_Mode_Sleep    = 0b01,
    Sensor_Power_Mode_LowPower = 0b10,
} Sensor_Power_Mode_t;

typedef enum SensorError_t {
    Success,
    I2CAddressWriteFailure_DataTooLong,
    I2CAddressWriteFailure_AddressNACK,
    I2CAddressWriteFailure_DataNACK,
    I2CAddressWriteFailure_Unknown,
    I2CAddressWriteFailure_Timeout,
    RegisterContentsMismatch,
    DeviceNotFound,
    WriteAccessNotEnabled,
    I2CADCStatusChangeFailure,
    I2CAddressChangeFailure,
    BandwidthSelectChangeFailure,
    HallEffectModeChangeFailure,
    I2CCRCModeChangeFailure,
    I2CThresholdChangeFailure,
    MagneticFieldAxesEnableFailure,
    I2CLoopModeChangeFailure,
    PowerModeChangeFailure,
    DataNotReady,
} SensorError_t;

class ALS31313_t {

    private:
        uint8_t LayerIndex: 4;
        uint8_t DeviceIndex: 4;

    public:
        uint8_t I2CAddress;
        ALS31313_t(uint8_t LayerIndex, uint8_t DeviceIndex, bool I2C_Initialized = false);

        uint8_t I2CAddressFromIndices();

        SensorError_t EnableWriteAccess(I2CBus_t& Bus);
        SensorError_t SetI2CADCStatus(I2CBus_t& Bus, I2C_ADC_Status_t Mode);
        SensorError_t SetBandwidthSelectMode(I2CBus_t& Bus, Bandwidth_Select_Mode_t Mode);
        SensorError_t SetHallEffectMode(I2CBus_t& Bus, Hall_Effect_Mode_t Mode);
        SensorError_t SetI2CCRCMode(I2CBus_t& Bus, I2C_CRC_Mode_t Mode);
        SensorError_t SetI2CAddress(I2CBus_t& Bus, uint8_t NewAddress);
        SensorError_t SetI2CThreshold(I2CBus_t& Bus, I2C_Voltage_Threshold_t Threshold);
        SensorError_t SetMagneticFieldChannels(I2CBus_t& Bus, Axis_Enable_Status_t Axes);
        SensorError_t SetI2CLoopMode(I2CBus_t& Bus, I2C_Loop_Mode_t Mode);
        SensorError_t SetPowerMode(I2CBus_t& Bus, Sensor_Power_Mode_t Mode);

        uint8_t GetI2CAddress(I2CBus_t& Bus);

        SensorError_t DefaultInitialize(I2CBus_t& Bus);
        SensorError_t ReadMeasurements(I2CBus_t& Bus, MagneticSensorReading_t& CurrentMeasurement);

    // private:
        SensorError_t WriteRegisterAddress(I2CBus_t& Bus, uint8_t RegisterAddress);
        SensorError_t ReadRegister(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t* Out);
        SensorError_t WriteRegister(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t Data);
        SensorError_t UpdateRegister(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t Data, uint32_t Mask, bool Force = false);
        SensorError_t ValidateRegisterContents(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t ExpectedData);
};
/* --- End Library Typedefs --- */

/* +++ Begin Library Function Definitions +++ */
void Log_CheckAddress(uint8_t Address);
void Log_DeviceCurrentI2CStatus(uint8_t ExpectedAddress, uint8_t CurrentAddress);
void Log_EnableWriteAccess(uint8_t I2CAddress);
void Log_SetI2CADCStatus(uint8_t I2CAddress, I2C_ADC_Status_t Mode);
void Log_SetBandwidthSelectMode(uint8_t I2CAddress, Bandwidth_Select_Mode_t Mode);
void Log_SetHallEffectMode(uint8_t I2CAddress, Hall_Effect_Mode_t Mode);
void Log_SetI2CCRCMode(uint8_t I2CAddress, I2C_CRC_Mode_t Mode);
void Log_SetI2CAddress(uint8_t I2CAddress, uint8_t NewAddress);
void Log_SetI2CThreshold(uint8_t I2CAddress, I2C_Voltage_Threshold_t Threshold);
void Log_SetMagneticFieldChannels(uint8_t I2CAddress, Axis_Enable_Status_t Axes);
void Log_SetI2CLoopMode(uint8_t I2CAddress, I2C_Loop_Mode_t Mode);
void Log_SetPowerMode(uint8_t I2CAddress, Sensor_Power_Mode_t Mode);
void Log_NoDeviceToInitialize(uint8_t I2CAddress);
void Log_ReadSensorMeasurements(uint8_t I2CAddress);
void Log_WriteRegister(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t Data);
void Log_RegisterContents(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t RegisterContents, bool Newline = true);
void Log_RegisterContentMismatch(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t Expected, uint32_t Actual);
void Log_SensorError(SensorError_t Error, bool Newline = true);
/* --- End Library Function Definitions --- */

/* +++ Begin Library Global Declarations +++ */
extern uint8_t SensorReading_Bytes[SENSOR_READING_BYTE_LENGTH];
/* --- End Library Global Declarations --- */

#ifdef __cplusplus
}
#endif

#endif
