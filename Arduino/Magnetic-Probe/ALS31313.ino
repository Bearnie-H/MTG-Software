/*
    ALS31313.ino

    Author(s):  Joseph Sadden
    Date:       1st December, 2023

    This file contains and provides the Hardware Abstraction Layer (HAL)
    for the ALS31313 Hall-Effect sensor. This sensor forms the foundation of the
    operation of this sensor array. This sensor is accessed via the I2C
    protocol, and provides some additional configuration options beyond
    simply reporting the measurement values from the internal sensing elements.
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

/* +++ Begin Top-level Header Includes +++ */
#include <stdbool.h>
#include <stdint.h>
/* --- End Top-level Header Includes --- */

/* +++ Begin Custom Header Includes +++ */
#include "include/ALS31313.h"
/* --- End Custom Header Includes --- */


/* +++ Begin Private Macro Definitions +++ */
/* --- End Private Macro Definitions --- */

/* +++ Begin Private Typedefs +++ */
/* --- End Private Typedefs --- */

/* +++ Begin Private Constant Definitions +++ */
uint8_t SensorReading_Bytes[SENSOR_READING_BYTE_LENGTH] = { 0x00 };
/* --- End Private Constant Definitions --- */

/* +++ Begin Private Function Declarations +++ */
template<typename N> N SignExtendValue(N Value, N Bits);
/* --- End Private Function Declarations --- */

/* +++ Begin Private Function Definitions +++ */
/* --- End Private Function Definitions --- */

/* +++ Begin Struct/Class Method Definitions +++ */
size_t MagneticSensorReading_t::AsBytes(uint8_t* Buffer) {

    // Clear the buffer, asserting 0x00 in all bytes
    memset(Buffer, 0x00, SENSOR_READING_BYTE_LENGTH);

    // Write a start-byte to indicate a measurement value over the serial line, rather than a log message to be written out for the user...
    Buffer[0] = 0x00;

    // Write the I2C address of the device...
    Buffer[1] = this->I2CAddress;

    // Write the device and layer indices associated with the measurement...
    Buffer[2] = this->LayerIndex;
    Buffer[3] = this->DeviceIndex;

    // Write out the X, Y, Z and Temperature values...
    memcpy(&(Buffer[4]), (uint8_t*)&(this->FieldX), 2);
    memcpy(&(Buffer[6]), (uint8_t*)&(this->FieldY), 2);
    memcpy(&(Buffer[8]), (uint8_t*)&(this->FieldZ), 2);
    memcpy(&(Buffer[10]), (uint8_t*)&(this->Temperature), 2);

    // Write out the timestamp value...
    memcpy(&(Buffer[12]), (uint8_t*)&(this->Timestamp), 4);

    // Send some garbage, known, stop bytes
    Buffer[16] = 0xAA;
    Buffer[17] = 0x55;

    return SENSOR_READING_BYTE_LENGTH;
}

ALS31313_t::ALS31313_t(uint8_t LayerIndex, uint8_t DeviceIndex, bool I2C_Initialized) {

    this->LayerIndex = LayerIndex;
    this->DeviceIndex = DeviceIndex;
    if ( I2C_Initialized ) {
        this->I2CAddress = this->I2CAddressFromIndices();
    } else {
        this->I2CAddress = NO_VALID_ADDRESS;
    }

    return;
}

uint8_t ALS31313_t::I2CAddressFromIndices() {
    return (( this->LayerIndex << 4 ) | this->DeviceIndex) + RESERVED_I2C_ADDRESS_END;
}

SensorError_t ALS31313_t::EnableWriteAccess(I2CBus_t& Bus) {

    static const uint8_t WriteAccessRegisterAddress = 0x35;
    static const uint32_t WriteAccessRegisterData = 0x2C413534;

    Log_EnableWriteAccess(this->I2CAddress);

    return this->WriteRegister(Bus, WriteAccessRegisterAddress, WriteAccessRegisterData);
}

SensorError_t ALS31313_t::SetI2CADCStatus(I2CBus_t& Bus, I2C_ADC_Status_t Mode) {

    static const uint8_t I2CADCRegisterAddress = 0x02;
    static const uint32_t I2CADCDataMask       = 0x00FDFFE0;
    static const uint32_t I2CADCBitOffset      = 17;

    Log_SetI2CADCStatus(this->I2CAddress, Mode);

    return this->UpdateRegister(Bus, I2CADCRegisterAddress, Mode << I2CADCBitOffset, I2CADCDataMask);
}

SensorError_t ALS31313_t::SetBandwidthSelectMode(I2CBus_t& Bus, Bandwidth_Select_Mode_t Mode) {

    static const uint8_t BandwidthSelectRegisterAddress = 0x02;
    static const uint32_t BandwidthSelectDataMask       = 0x001FFFE0;
    static const uint32_t BandwidthSelectBitOffset      = 21;

    Log_SetBandwidthSelectMode(this->I2CAddress, Mode);

    return this->UpdateRegister(Bus, BandwidthSelectRegisterAddress, Mode << BandwidthSelectBitOffset, BandwidthSelectDataMask);
}

SensorError_t ALS31313_t::SetHallEffectMode(I2CBus_t& Bus, Hall_Effect_Mode_t Mode) {

    static const uint8_t HallEffectRegisterAddress = 0x02;
    static const uint32_t HallEffectDataMask       = 0x00E7FFE0;
    static const uint32_t HallEffectBitOffset      = 19;

    Log_SetHallEffectMode(this->I2CAddress, Mode);

    return this->UpdateRegister(Bus, HallEffectRegisterAddress, Mode << HallEffectBitOffset, HallEffectDataMask);;
}

SensorError_t ALS31313_t::SetI2CCRCMode(I2CBus_t& Bus, I2C_CRC_Mode_t Mode) {

    static const uint8_t I2CCRCModeRegisterAddress = 0x02;
    static const uint32_t I2CCRCModeDataMask       = 0x00FBFFE0;
    static const uint32_t I2CCRCModeBitOffset      = 18;

    Log_SetI2CCRCMode(this->I2CAddress, Mode);

    return this->UpdateRegister(Bus, I2CCRCModeRegisterAddress, Mode << I2CCRCModeBitOffset, I2CCRCModeDataMask);
}

SensorError_t ALS31313_t::SetI2CAddress(I2CBus_t& Bus, uint8_t NewAddress) {

    static const uint8_t I2CAddressRegisterAddress = 0x02;
    static const uint32_t I2CAddressDataMask       = 0xFFFE03E0;
    static const uint32_t I2CAddressBitOffset      = 10;

    NewAddress &= 0x7F;

    Log_SetI2CAddress(this->I2CAddress, NewAddress);

    SensorError_t Status = this->UpdateRegister(Bus, I2CAddressRegisterAddress, ((uint32_t)NewAddress) << I2CAddressBitOffset, I2CAddressDataMask, true);
    if ( Status != Success ) {
        return Status;
    }

    if ( this->I2CAddress != NewAddress ) {
        Status = I2CAddressPowerCycleRequired;
    }

    this->I2CAddress = NewAddress;
    return Status;
}

SensorError_t ALS31313_t::SetI2CThreshold(I2CBus_t& Bus, I2C_Voltage_Threshold_t Threshold) {

    static const uint8_t I2CThresholdRegisterAddress = 0x02;
    static const uint32_t I2CThresholdDataMask       = 0x00FFFDE0;
    static const uint32_t I2CThresholdBitOffset      = 9;

    Log_SetI2CThreshold(this->I2CAddress, Threshold);

    return this->UpdateRegister(Bus, I2CThresholdRegisterAddress, Threshold << I2CThresholdBitOffset, I2CThresholdDataMask);
}

SensorError_t ALS31313_t::SetMagneticFieldChannels(I2CBus_t& Bus, Axis_Enable_Status_t Axes) {

    static const uint8_t MagneticFieldChannelsRegisterAddress = 0x02;
    static const uint32_t MagneticFieldChannelsDataMask       = 0x00FFFE00;
    static const uint32_t MagneticFieldChannelsBitOffset      = 6;

    Log_SetMagneticFieldChannels(this->I2CAddress, Axes);

    return this->UpdateRegister(Bus, MagneticFieldChannelsRegisterAddress, Axes << MagneticFieldChannelsBitOffset, MagneticFieldChannelsDataMask);
}

SensorError_t ALS31313_t::SetI2CLoopMode(I2CBus_t& Bus, I2C_Loop_Mode_t Mode) {

    static const uint8_t I2CLoopModeRegisterAddress = 0x27;
    static const uint32_t I2CLoopModeDataMask       = 0x00000073;
    static const uint32_t I2CLoopModeBitOffset      = 2;

    Log_SetI2CLoopMode(this->I2CAddress, Mode);

    return this->UpdateRegister(Bus, I2CLoopModeRegisterAddress, Mode << I2CLoopModeBitOffset, I2CLoopModeDataMask);
}

SensorError_t ALS31313_t::SetPowerMode(I2CBus_t& Bus, Sensor_Power_Mode_t Mode) {

    static const uint8_t PowerModeRegisterAddress = 0x27;
    static const uint32_t PowerModeDataMask       = 0x0000007C;
    static const uint32_t PowerModeBitOffset      = 0;

    Log_SetPowerMode(this->I2CAddress, Mode);

    return this->UpdateRegister(Bus, PowerModeRegisterAddress, Mode << PowerModeBitOffset, PowerModeDataMask);
}

uint8_t ALS31313_t::GetI2CAddress(I2CBus_t& Bus) {

    uint8_t ExpectedAddress = this->I2CAddressFromIndices();

    // First, check if there's even anything attached to the bus to query...
    Log_CheckBusEmpty();
    if ( Bus.Empty() ) {
        return NO_VALID_ADDRESS;
    }

    // Check if the device is on the bus with the expected address...
    Log_CheckAddress(ExpectedAddress);
    if ( Bus.AddressExists(ExpectedAddress) ) {
        this->I2CAddress = ExpectedAddress;
        return ExpectedAddress;
    }

    // If not, check if it's on the bus with the default address...
    if ( DEFAULT_ADDRESS != ExpectedAddress ){
        Log_CheckAddress(DEFAULT_ADDRESS);
        if ( Bus.AddressExists(DEFAULT_ADDRESS) ) {
            this->I2CAddress = DEFAULT_ADDRESS;
            return DEFAULT_ADDRESS;
        }
    }

    // If neither, iterate over the possible address space, searching for ANY address corresponding to a device.
    for ( uint8_t TestAddress = MINIMUM_ADDRESS; TestAddress <= MAXIMUM_ADDRESS; TestAddress++ ) {

        // Skip over the addresses we've already tried.
        if (( TestAddress == DEFAULT_ADDRESS ) || ( TestAddress == ExpectedAddress )) {
            continue;
        }

        // Check each next address...
        if ( Bus.AddressExists(TestAddress) ) {
            this->I2CAddress = TestAddress;
            return TestAddress;
        }
    }

    this->I2CAddress = NO_VALID_ADDRESS;
    return NO_VALID_ADDRESS;
}

SensorError_t ALS31313_t::DefaultInitialize(I2CBus_t& Bus) {

    // See if the sensor is already initialized with the correct I2C Address
    uint8_t ExpectedAddress = this->I2CAddressFromIndices();
    this->I2CAddress = this->GetI2CAddress(Bus);
    Log_DeviceCurrentI2CStatus(ExpectedAddress, this->I2CAddress);
    if ( this->I2CAddress == NO_VALID_ADDRESS ) {
        Log_NoDeviceToInitialize(ExpectedAddress);
        return DeviceNotFound;
    }

    // If the device exists, we always need to enable write access to set
    // the power mode and I2C looping style.
    if ( this->EnableWriteAccess(Bus) != Success ) {
        Log_OperationFailed();
        return WriteAccessNotEnabled;
    }

    if ( this->SetI2CADCStatus(Bus, I2C_ADC_Enable) != Success ) {
        Log_OperationFailed();
        return I2CADCStatusChangeFailure;
    }

    // Set the bandwidth select mode.
    if ( this->SetBandwidthSelectMode(Bus, Bandwidth_Select_LowSpeed_WithFilter) != Success ) {
        Log_OperationFailed();
        return BandwidthSelectChangeFailure;
    }

    // Set the Hall-Effect Sensing mode.
    if ( this->SetHallEffectMode(Bus, Hall_Effect_Mode_SingleEnded) != Success ) {
        Log_OperationFailed();
        return HallEffectModeChangeFailure;
    }

    // Set the I2C CRC.
    if ( this->SetI2CCRCMode(Bus, I2C_CRC_Disabled) != Success ) {
        Log_OperationFailed();
        return I2CCRCModeChangeFailure;
    }

    // Set the I2C voltage threshold.
    if ( this->SetI2CThreshold(Bus, I2C_Voltage_Threshold_High_3V_Mode) != Success ) {
        Log_OperationFailed();
        return I2CThresholdChangeFailure;
    }

    // Enable the magnetic field measurement channels.
    if ( this->SetMagneticFieldChannels(Bus, Axis_Enable_XYZ) != Success ) {
        Log_OperationFailed();
        return MagneticFieldAxesEnableFailure;
    }

    // And always set the looping mode and power mode.
    // Set the correct I2C looping mode.
    if ( this->SetI2CLoopMode(Bus, I2C_Loop_Mode_Single) != Success ) {
        Log_OperationFailed();
        return I2CLoopModeChangeFailure;
    }

    // Set the correct device power mode.
    if ( this->SetPowerMode(Bus, Sensor_Power_Mode_Standard) != Success ) {
        Log_OperationFailed();
        return PowerModeChangeFailure;
    }

    // Write the new I2C address to the device.
    SensorError_t Status = this->SetI2CAddress(Bus, ExpectedAddress);
    if ( Status != Success) {
        Log_OperationFailed();
        return Status;
    }

    return Success;
}

SensorError_t ALS31313_t::ReadMeasurements(I2CBus_t& Bus, MagneticSensorReading_t& CurrentMeasurement) {

    Log_ReadSensorMeasurements(this->I2CAddress);

    static const uint8_t MeasurementMSBRegisterAddress = 0x28;
    static const uint8_t MeasurementLSBRegisterAddress = 0x29;

    static const uint32_t NewDataFlagDataMask    = 0x00000080;
    static const uint32_t FieldXMSBDataMask      = 0xFF000000;
    static const uint32_t FieldYMSBDataMask      = 0x00FF0000;
    static const uint32_t FieldZMSBDataMask      = 0x0000FF00;
    static const uint32_t TemperatureMSBDataMask = 0x0000003F;
    static const uint32_t FieldXLSBDataMask      = 0x000F0000;
    static const uint32_t FieldYLSBDataMask      = 0x0000F000;
    static const uint32_t FieldZLSBDataMask      = 0x00000F00;
    static const uint32_t TemperatureLSBDataMask = 0x0000003F;

    static const uint32_t FieldXMSBBitOffset      = 24;
    static const uint32_t FieldYMSBBitOffset      = 16;
    static const uint32_t FieldZMSBBitOffset      = 8;
    static const uint32_t TemperatureMSBBitOffset = 0;
    static const uint32_t FieldXLSBBitOffset      = 16;
    static const uint32_t FieldYLSBBitOffset      = 12;
    static const uint32_t FieldZLSBBitOffset      = 8;
    static const uint32_t TemperatureLSBBitOffset = 0;

    uint32_t MSBs = 0x00000000, LSBs = 0x00000000;

    SensorError_t Status = this->ReadRegister(Bus, MeasurementMSBRegisterAddress, &MSBs);
    if ( Status != Success ) {
        return Status;
    }

    Status = this->ReadRegister(Bus, MeasurementLSBRegisterAddress, &LSBs);
    if ( Status != Success ) {
        return Status;
    }

    if (( MSBs & NewDataFlagDataMask ) == 0 ) {
        return DataNotReady;
    }

    CurrentMeasurement.I2CAddress  = this->I2CAddress;
    CurrentMeasurement.DeviceIndex = this->DeviceIndex;
    CurrentMeasurement.LayerIndex  = this->LayerIndex;

    CurrentMeasurement.FieldX       = (((MSBs & FieldXMSBDataMask)      >> FieldXMSBBitOffset)      << 4) | ((LSBs & FieldXLSBDataMask)      >> FieldXLSBBitOffset     );
    CurrentMeasurement.FieldY       = (((MSBs & FieldYMSBDataMask)      >> FieldYMSBBitOffset)      << 4) | ((LSBs & FieldYLSBDataMask)      >> FieldYLSBBitOffset     );
    CurrentMeasurement.FieldZ       = (((MSBs & FieldZMSBDataMask)      >> FieldZMSBBitOffset)      << 4) | ((LSBs & FieldZLSBDataMask)      >> FieldZLSBBitOffset     );
    CurrentMeasurement.Temperature  = (((MSBs & TemperatureMSBDataMask) >> TemperatureMSBBitOffset) << 6) | ((LSBs & TemperatureLSBDataMask) >> TemperatureLSBBitOffset);

    CurrentMeasurement.FieldX      = SignExtendValue(CurrentMeasurement.FieldX, SENSOR_BIT_DEPTH);
    CurrentMeasurement.FieldY      = SignExtendValue(CurrentMeasurement.FieldY, SENSOR_BIT_DEPTH);
    CurrentMeasurement.FieldZ      = SignExtendValue(CurrentMeasurement.FieldZ, SENSOR_BIT_DEPTH);

    CurrentMeasurement.Timestamp   = (uint32_t)micros();

    return Success;
}

SensorError_t ALS31313_t::WriteRegisterAddress(I2CBus_t& Bus, uint8_t RegisterAddress, bool EndTransmission) {

    // Begin an I2C transaction with the I2C address of this device.
    Bus.Wire.beginTransmission(this->I2CAddress);

    // Write the register address we wish to interact with...
    if ( Bus.Wire.write(RegisterAddress) != 1 ) {
        return I2CAddressWriteFailure_Unknown;
    }

    if ( EndTransmission ) {
        // Wait for the response from this sending byte...
        switch ( Bus.Wire.endTransmission(false) ) {
            case 0:
                return Success;
            case 1:
                return I2CAddressWriteFailure_DataTooLong;
            case 2:
                return I2CAddressWriteFailure_AddressNACK;
            case 3:
                return I2CAddressWriteFailure_DataNACK;
            case 5:
                return I2CAddressWriteFailure_Timeout;
            default:
                return I2CAddressWriteFailure_Unknown;
        }
    }

    return Success;
}

SensorError_t ALS31313_t::ReadRegister(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t* Out) {

    SensorError_t Status = this->WriteRegisterAddress(Bus, RegisterAddress, true);
    if ( Status != Success ) {
        return Status;
    }

    // Request the 4 byte contents of the register, sending a STOP once all bytes have been read.
    Bus.Wire.requestFrom((uint8_t)this->I2CAddress, (uint8_t)4, (uint8_t)true);
    for ( int32_t i = 3; i >= 0; i-- ) {
        *Out |= ((uint32_t)Wire.read() << (i * __CHAR_BIT__));
    }

    Log_RegisterContents(this->I2CAddress, RegisterAddress, *Out);

    return Status;
}

SensorError_t ALS31313_t::WriteRegister(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t Data) {

    Log_WriteRegister(this->I2CAddress, RegisterAddress, Data);

    SensorError_t Status = this->WriteRegisterAddress(Bus, RegisterAddress, false);
    if ( Status != Success ) {
        return Status;
    }

    // Write the data...
    for ( int32_t i = 3; i >= 0; i-- ) {
        if ( Bus.Wire.write((uint8_t)(Data >> (i * __CHAR_BIT__))) != 1 ) {
            return I2CAddressWriteFailure_Unknown;
        }
    }

    switch ( Bus.Wire.endTransmission(true) ) {
        case 0:
            return Success;
            Log_RegisterContents(this->I2CAddress, RegisterAddress, Data);
        case 1:
            return I2CAddressWriteFailure_DataTooLong;
        case 2:
            return I2CAddressWriteFailure_AddressNACK;
        case 3:
            return I2CAddressWriteFailure_DataNACK;
        case 5:
            return I2CAddressWriteFailure_Timeout;
        default:
            return I2CAddressWriteFailure_Unknown;
    }
}

SensorError_t ALS31313_t::UpdateRegister(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t Data, uint32_t Mask, bool Force) {

    uint32_t RegisterContents = 0x00;

    SensorError_t Status = this->ReadRegister(Bus, RegisterAddress, &RegisterContents);
    if ( Status != Success ) {
        return Status;
    }

    uint32_t NewContents = (RegisterContents & Mask) | Data;
    if ( ! Force ) {
        if (( NewContents & ( ~Mask )) == ( RegisterContents & ( ~Mask ))) {
            return Success;
        }
    }

    return this->WriteRegister(Bus, RegisterAddress, NewContents);
}

SensorError_t ALS31313_t::ValidateRegisterContents(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t ExpectedData) {

    uint32_t RegisterContents = 0x00;
    SensorError_t Status = this->ReadRegister(Bus, RegisterAddress, &RegisterContents);
    if ( Status != Success ) {
        return Status;
    }

    if ( RegisterContents != ExpectedData ) {
        Log_RegisterContentMismatch(this->I2CAddress, RegisterAddress, ExpectedData, RegisterContents);
        return RegisterContentsMismatch;
    }

    return Success;
}

uint16_t SignExtendValue(uint16_t Value, uint16_t Bits) {

    uint16_t Sign = (1 << (Bits - 1)) & Value;
    uint16_t Mask = ((~0U) >> (Bits - 1)) << (Bits - 1);

    if ( Sign == 0 ) {
        return (Value & (~Mask));
    } else {
        return (Value | Mask);
    }
}

/* --- End Struct/Class Method Definitions --- */

#if defined (DEBUG)

void Log_CheckAddress(uint8_t Address) {

    Serial.print("Checking for device at ");
    if ( Address == DEFAULT_ADDRESS ) {
        Serial.print("default ");
    }
    Log_I2CAddress(Address);

    return;
}

void Log_DeviceCurrentI2CStatus(uint8_t ExpectedAddress, uint8_t CurrentAddress) {

    Serial.print("Device address is ");
    if ( CurrentAddress == NO_VALID_ADDRESS ) {
        Serial.print("None");
    } else {
        Serial.print(CurrentAddress);
    }
    Serial.print(" - expected ");
    Log_I2CAddress(ExpectedAddress);

    return;
}

void Log_EnableWriteAccess(uint8_t I2CAddress) {

    Serial.print("Enabling write access for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetI2CADCStatus(uint8_t I2CAddress, I2C_ADC_Status_t Mode) {

    Serial.print("I2C ADC ");
    switch ( Mode ) {
        case I2C_ADC_Enable:
            Serial.print("On");
            break;
        case I2C_ADC_Disable:
            Serial.print("Off");
            break;
        default:
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetBandwidthSelectMode(uint8_t I2CAddress, Bandwidth_Select_Mode_t Mode) {

    Serial.print("Bandwidth select mode ");
    switch ( Mode ) {
        case Bandwidth_Select_LowSpeed_WithFilter:
            Serial.print("Low-Filter");
            break;
        case Bandwidth_Select_MediumSpeed_WithFilter:
            Serial.print("Medium-Filter");
            break;
        case Bandwidth_Select_HighSpeed_WithFilter:
            Serial.print("High-Filter");
            break;
        case Bandwidth_Select_LowSpeed_WithoutFilter:
            Serial.print("Low");
            break;
        case Bandwidth_Select_MediumSpeed_WithoutFilter:
            Serial.print("Medium");
            break;
        case Bandwidth_Select_HighSpeed_WithoutFilter:
            Serial.print("High");
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetHallEffectMode(uint8_t I2CAddress, Hall_Effect_Mode_t Mode) {

    Serial.print("Sensing mode ");
    switch ( Mode ) {
        case Hall_Effect_Mode_SingleEnded:
            Serial.print("Single-Ended");
            break;
        case Hall_Effect_Mode_Differential:
            Serial.print("Differential");
            break;
        case Hall_Effect_Mode_CommonMode:
            Serial.print("Common-Mode");
            break;
        case Hall_Effect_Mode_AlternatingMode:
            Serial.print("Alternating");
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetI2CCRCMode(uint8_t I2CAddress, I2C_CRC_Mode_t Mode) {

    Serial.print("I2C CRC ");
    switch ( Mode ) {
        case I2C_CRC_Disabled:
            Serial.print("Off");
            break;
        case I2C_CRC_Enabled:
            Serial.print("On");
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetI2CAddress(uint8_t I2CAddress, uint8_t NewAddress) {

    if ( I2CAddress == NewAddress ) {
        return;
    }

    Serial.print("Changing I2C Address from ");
    Serial.print(I2CAddress);
    Serial.print(" to ");
    Serial.println(NewAddress);

    Serial.println("Note: New I2C Address will only take effect after a power cycle!");

    return;
}

void Log_SetI2CThreshold(uint8_t I2CAddress, I2C_Voltage_Threshold_t Threshold) {

    Serial.print("I2C threshold ");
    switch ( Threshold ) {
        case I2C_Voltage_Threshold_High_3V_Mode:
            Serial.print("3V");
            break;
        case I2C_Voltage_Threshold_Low_1V8_Mode:
            Serial.print("1.8V");
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetMagneticFieldChannels(uint8_t I2CAddress, Axis_Enable_Status_t Axes) {

    Serial.print("Enabling field axes ");
    switch ( Axes ) {
        case Axis_Enable____:
            Serial.print("None");
            break;
        case Axis_Enable_X__:
            Serial.print("X");
            break;
        case Axis_Enable__Y_:
            Serial.print("Y");
            break;
        case Axis_Enable_XY_:
            Serial.print("X,Y");
            break;
        case Axis_Enable___Z:
            Serial.print("Z");
            break;
        case Axis_Enable_X_Z:
            Serial.print("X,Z");
            break;
        case Axis_Enable__YZ:
            Serial.print("Y,Z");
            break;
        case Axis_Enable_XYZ:
            Serial.print("X,Y,Z");
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetI2CLoopMode(uint8_t I2CAddress, I2C_Loop_Mode_t Mode) {

    Serial.print("I2C Loop Mode ");
    switch ( Mode ) {
        case I2C_Loop_Mode_Single:
            Serial.print("Single");
            break;
        case I2C_Loop_Mode_FastLoop:
            Serial.print("Fast");
            break;
        case I2C_Loop_Mode_FullLoop:
            Serial.print("Full");
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_SetPowerMode(uint8_t I2CAddress, Sensor_Power_Mode_t Mode) {

    Serial.print("Power Mode ");
    switch ( Mode ) {
        case Sensor_Power_Mode_Standard:
            Serial.print("Normal");
            break;
        case Sensor_Power_Mode_Sleep:
            Serial.print("Sleep");
            break;
        case Sensor_Power_Mode_LowPower:
            Serial.print("Low");
            break;
    }
    Serial.print(" for ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_NoDeviceToInitialize(uint8_t I2CAddress) {

    Serial.print("Device not found with ");
    Log_I2CAddress(I2CAddress);

    return;
}

void Log_ReadSensorMeasurements(uint8_t I2CAddress) {

    Serial.print("Reading ");
    Log_DeviceAddress(I2CAddress);

    return;
}

void Log_WriteRegister(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t Data) {

    Serial.print("Writing (I2C, Reg, Data): ");
    Serial.print(I2CAddress);
    Serial.print(",0x");
    Serial.print(RegisterAddress, 16);
    Serial.print(",0x");
    Serial.println(Data, 16);

    return;
}

void Log_RegisterContents(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t RegisterContents, bool Newline) {

    Serial.print("Reading (I2C, Reg, Data): ");
    Serial.print(I2CAddress);
    Serial.print(",0x");
    Serial.print(RegisterAddress, 16);
    Serial.print(",0x");
    Serial.print(RegisterContents, 16);

    if ( Newline ) {
        Serial.println();
    }

    return;
}

void Log_RegisterContentMismatch(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t Expected, uint32_t Actual) {

    Log_RegisterContents(I2CAddress, RegisterAddress, Expected, false);
    Serial.print("Expected 0x");
    Serial.println(Expected, 16);

    return;
}

void Log_SensorError(SensorError_t Error, bool Newline) {

       switch ( Error ) {
        case Success:
            Serial.print("Success");
            break;
        case I2CAddressWriteFailure_DataTooLong:
            Serial.print("I2C Data Too Long");
            break;
        case I2CAddressWriteFailure_AddressNACK:
            Serial.print("I2C Addr NACK");
            break;
        case I2CAddressWriteFailure_DataNACK:
            Serial.print("I2C Data Nack");
            break;
        case I2CAddressWriteFailure_Unknown:
            Serial.print("Unknown I2C Error");
            break;
        case I2CAddressWriteFailure_Timeout:
            Serial.print("I2C Timeout");
            break;
        case RegisterContentsMismatch:
            Serial.print("Register Mismatch");
            break;
        case DeviceNotFound:
            Serial.print("No Device Found");
            break;
        case WriteAccessNotEnabled:
            Serial.print("Write Access Failed");
            break;
        case I2CADCStatusChangeFailure:
            Serial.print("I2C ADC Not Changed");
            break;
        case I2CAddressChangeFailure:
            Serial.print("I2C Address Not Changed");
            break;
        case I2CAddressPowerCycleRequired:
            Serial.print("New I2C Address Requires Power Cycle");
            break;
        case BandwidthSelectChangeFailure:
            Serial.print("Bandwidth Select Not Changed");
            break;
        case HallEffectModeChangeFailure:
            Serial.print("Sensing Mode Not Changed");
            break;
        case I2CCRCModeChangeFailure:
            Serial.print("I2C CRC Not Changed");
            break;
        case I2CThresholdChangeFailure:
            Serial.print("I2C Voltage Not Changed");
            break;
        case MagneticFieldAxesEnableFailure:
            Serial.print("Sensing Channels Not Changed");
            break;
        case I2CLoopModeChangeFailure:
            Serial.print("I2C Loop Mode Not Changed");
            break;
        case PowerModeChangeFailure:
            Serial.print("Power Mode Not Changed");
            break;
        case DataNotReady:
            Serial.print("Data Not Ready");
            break;
        default:
            break;
    }

    if ( Newline ) {
        Serial.println();
    }

    return;
}

#else

void Log_CheckAddress(uint8_t Address) {
    return;
}

void Log_DeviceCurrentI2CStatus(uint8_t ExpectedAddress, uint8_t CurrentAddress) {
    return;
}

void Log_EnableWriteAccess(uint8_t I2CAddress) {
    return;
}

void Log_SetI2CADCStatus(uint8_t I2CAddress, I2C_ADC_Status_t Mode) {
    return;
}

void Log_SetBandwidthSelectMode(uint8_t I2CAddress, Bandwidth_Select_Mode_t Mode) {
    return;
}

void Log_SetHallEffectMode(uint8_t I2CAddress, Hall_Effect_Mode_t Mode) {
    return;
}

void Log_SetI2CCRCMode(uint8_t I2CAddress, I2C_CRC_Mode_t Mode) {
    return;
}

void Log_SetI2CAddress(uint8_t I2CAddress, uint8_t NewAddress) {
    return;
}

void Log_SetI2CThreshold(uint8_t I2CAddress, I2C_Voltage_Threshold_t Threshold) {
    return;
}

void Log_SetMagneticFieldChannels(uint8_t I2CAddress, Axis_Enable_Status_t Axes) {
    return;
}

void Log_SetI2CLoopMode(uint8_t I2CAddress, I2C_Loop_Mode_t Mode) {
    return;
}

void Log_SetPowerMode(uint8_t I2CAddress, Sensor_Power_Mode_t Mode) {
    return;
}

void Log_NoDeviceToInitialize(uint8_t I2CAddress) {
    return;
}

void Log_ReadSensorMeasurements(uint8_t I2CAddress) {
    return;
}

void Log_WriteRegister(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t Data) {
    return;
}

void Log_RegisterContents(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t RegisterContents, bool Newline) {
    return;
}

void Log_RegisterContentMismatch(uint8_t I2CAddress, uint8_t RegisterAddress, uint32_t Expected, uint32_t Actual) {
    return;
}

void Log_SensorError(SensorError_t Error, bool Newline) {
    return;
}

#endif
