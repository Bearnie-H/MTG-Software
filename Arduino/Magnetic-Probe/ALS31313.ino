/*
    ALS31313.ino

    Author(s):  Joseph Sadden
    Date:       1st December, 2023

    This file...
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

#define WRITE_ACCESS_ADDRESS        (uint8_t)0x35
#define WRITE_ACCESS_DATA           (uint32_t)0x2C413534

#define BW_SELECT_ADDRESS           (uint8_t)0x02
#define HALL_EFFECT_MODE_ADDRESS    (uint8_t)0x02
#define I2C_CRC_ENABLE_ADDRESS      (uint8_t)0x02
#define I2C_SLAVE_ADDRESS_ADDRESS   (uint8_t)0x02
#define I2C_THRESHOLD_ADDRESS       (uint8_t)0x02
#define CHANNEL_X_ENABLE_ADDRESS    (uint8_t)0x02
#define CHANNEL_Z_ENABLE_ADDRESS    (uint8_t)0x02
#define CHANNEL_Y_ENABLE_ADDRESS    (uint8_t)0x02

#define I2C_LOOP_MODE_ADDRESS       (uint8_t)0x27
#define DEVICE_SLEEP_MODE_ADDRESS   (uint8_t)0x27

#define SENSOR_MSBs_ADDRESS         (uint8_t)0x28
#define FIELD_X_MSB_ADDRESS         (uint8_t)0x28
#define FIELD_Y_MSB_ADDRESS         (uint8_t)0x28
#define FIELD_Z_MSB_ADDRESS         (uint8_t)0x28
#define NEW_DATA_ADDRESS            (uint8_t)0x28
#define TEMPERATURE_MSB_ADDRESS     (uint8_t)0x28

#define SENSOR_LSBs_ADDRESS         (uint8_t)0x29
#define FIELD_X_LSB_ADDRESS         (uint8_t)0x29
#define FIELD_Y_LSB_ADDRESS         (uint8_t)0x29
#define FIELD_Z_LSB_ADDRESS         (uint8_t)0x29
#define TEMPERATURE_LSB_ADDRESS     (uint8_t)0x29

#define BW_SELECT_BIT_OFFSET                21
#define HALL_EFFECT_MODE_BIT_OFFSET         19
#define I2C_CRC_ENABLE_BIT_OFFSET           18
#define I2C_SLAVE_ADDRESS_BIT_OFFSET        10
#define I2C_VOLTAGE_THRESHOLD_BIT_OFFSET    9
#define ENABLE_Z_CHANNEL_BIT_OFFSET         8
#define ENABLE_Y_CHANNEL_BIT_OFFSET         7
#define ENABLE_X_CHANNEL_BIT_OFFSET         6

#define I2C_LOOP_MODE_BIT_OFFSET    2
#define SLEEP_MODE_BIT_OFFSET       0

#define REGISTER_0x02_CLEAR_MASK    0xFF00FE0F
#define REGISTER_0x27_CLEAR_MASK    0xFFFFFF80

#define I2C_SLAVE_ADDRESS_MASK  ((uint32_t)0x000000FF << 10)

#define FIELD_X_MSB_MASK        (uint32_t)0xFF000000
#define FIELD_X_LSB_MASK        (uint32_t)0x000F0000
#define FIELD_X_MSB_RIGHT_SHIFT 20
#define FIELD_X_LSB_RIGHT_SHIFT 16

#define FIELD_Y_MSB_MASK        (uint32_t)0x00FF0000
#define FIELD_Y_LSB_MASK        (uint32_t)0x0000F000
#define FIELD_Y_MSB_RIGHT_SHIFT 12
#define FIELD_Y_LSB_RIGHT_SHIFT 12

#define FIELD_Z_MSB_MASK        (uint32_t)0x0000FF00
#define FIELD_Z_LSB_MASK        (uint32_t)0x00000F00
#define FIELD_Z_MSB_RIGHT_SHIFT 4
#define FIELD_Z_LSB_RIGHT_SHIFT 8

#define TEMPERATURE_MSB_MASK        (uint32_t)0x0000003f
#define TEMPERATURE_LSB_MASK        (uint32_t)0x0000003f
#define TEMPERATURE_MSB_LEFT_SHIFT  6
#define TEMPERATURE_LSB_RIGHT_SHIFT 0


#define NEW_DATA_MASK           (uint32_t)0x00000080

/* --- End Private Macro Definitions --- */

/* +++ Begin Private Typedefs +++ */

typedef enum Bandwidth_Select_Mode_t: uint32_t {
    Bandwidth_Select_LowSpeed_WithFilter         = ((uint32_t)0b000 << BW_SELECT_BIT_OFFSET),
    Bandwidth_Select_MediumSpeed_WithFilter      = ((uint32_t)0b001 << BW_SELECT_BIT_OFFSET),
    Bandwidth_Select_HighSpeed_WithFilter        = ((uint32_t)0b010 << BW_SELECT_BIT_OFFSET),
    Bandwidth_Select_LowSpeed_WithoutFilter      = ((uint32_t)0b100 << BW_SELECT_BIT_OFFSET),
    Bandwidth_Select_MediumSpeed_WithoutFilter   = ((uint32_t)0b101 << BW_SELECT_BIT_OFFSET),
    Bandwidth_Select_HighSpeed_WithoutFilter     = ((uint32_t)0b110 << BW_SELECT_BIT_OFFSET),
} Bandwidth_Select_Mode_t;

typedef enum Hall_Effect_Mode_t: uint32_t {
    Hall_Effect_Mode_SingleEnded     = ((uint32_t)0b00 << HALL_EFFECT_MODE_BIT_OFFSET),
    Hall_Effect_Mode_Differential    = ((uint32_t)0b01 << HALL_EFFECT_MODE_BIT_OFFSET),
    Hall_Effect_Mode_CommonMode      = ((uint32_t)0b10 << HALL_EFFECT_MODE_BIT_OFFSET),
    Hall_Effect_Mode_AlternatingMode = ((uint32_t)0b11 << HALL_EFFECT_MODE_BIT_OFFSET),
} Hall_Effect_Mode_t;

typedef enum I2C_CRC_Status_t: uint32_t {
    I2C_CRC_Disabled = ((uint32_t)0b0 << I2C_CRC_ENABLE_BIT_OFFSET),
    I2C_CRC_Enabled  = ((uint32_t)0b1 << I2C_CRC_ENABLE_BIT_OFFSET),
} I2C_CRC_Status_t;

typedef enum I2C_Voltage_Threshold_t: uint32_t {
    I2C_Voltage_Threshold_High_3V_Mode = ((uint32_t)0b0 << I2C_VOLTAGE_THRESHOLD_BIT_OFFSET),
    I2C_Voltage_Threshold_Low_1V8_Mode = ((uint32_t)0b1 << I2C_VOLTAGE_THRESHOLD_BIT_OFFSET),
} I2C_Voltage_Threshold_t;

typedef enum Axis_Enable_Status_t: uint32_t {
    Axis_Enable____ = ((int32_t)0b000 << ENABLE_X_CHANNEL_BIT_OFFSET),
    Axis_Enable_X__ = ((int32_t)0b001 << ENABLE_X_CHANNEL_BIT_OFFSET),
    Axis_Enable__Y_ = ((int32_t)0b010 << ENABLE_X_CHANNEL_BIT_OFFSET),
    Axis_Enable_XY_ = ((int32_t)0b011 << ENABLE_X_CHANNEL_BIT_OFFSET),
    Axis_Enable___Z = ((int32_t)0b100 << ENABLE_X_CHANNEL_BIT_OFFSET),
    Axis_Enable_X_Z = ((int32_t)0b101 << ENABLE_X_CHANNEL_BIT_OFFSET),
    Axis_Enable__YZ = ((int32_t)0b110 << ENABLE_X_CHANNEL_BIT_OFFSET),
    Axis_Enable_XYZ = ((int32_t)0b111 << ENABLE_X_CHANNEL_BIT_OFFSET),
} Axis_Enable_Status_t;

typedef enum I2C_Loop_Mode_t: uint32_t {
    I2C_Loop_Mode_Single   = ((int32_t)0b00 << I2C_LOOP_MODE_BIT_OFFSET),
    I2C_Loop_Mode_FastLoop = ((int32_t)0b01 << I2C_LOOP_MODE_BIT_OFFSET),
    I2C_Loop_Mode_FullLoop = ((int32_t)0b10 << I2C_LOOP_MODE_BIT_OFFSET),
} I2C_Loop_Mode_t;

typedef enum Sensor_Power_Mode_t: uint32_t {
    Sensor_Power_Mode_Standard = ((int32_t)0b00 << SLEEP_MODE_BIT_OFFSET),
    Sensor_Power_Mode_Sleep    = ((int32_t)0b01 << SLEEP_MODE_BIT_OFFSET),
    Sensor_Power_Mode_LowPower = ((int32_t)0b10 << SLEEP_MODE_BIT_OFFSET),
}  Sensor_Power_Mode_t;

/* --- End Private Typedefs --- */

/* +++ Begin Private Constant Definitions +++ */
uint8_t SensorReading_Bytes[SENSOR_READING_BYTE_LENGTH] = { 0x00 };
/* --- End Private Constant Definitions --- */

/* +++ Begin Private Function Declarations +++ */
/* --- End Private Function Declarations --- */

/* +++ Begin Private Function Definitions +++ */
/* --- End Private Function Definitions --- */

/* +++ Begin Struct/Class Method Definitions +++ */

void MagneticSensorReading_t::SetMeasurementTime(uint32_t Timestamp) {

    this->Timestamp = Timestamp;

    return;
}

size_t MagneticSensorReading_t::AsBytes(uint8_t* Buffer) {

    // Clear the buffer, asserting 0x00 in all bytes
    memset(Buffer, 0x00, SENSOR_READING_BYTE_LENGTH);

    // Write a start-byte to indicate a measurement value over the serial line, rather than a log message to be written out for the user...
    Buffer[0] = 0x00;

    // Write the I2C address of the device...
    Buffer[1] = this->I2CAddress;

    // Write the device and layer indices associated with the measurement...
    Buffer[2] = (( this->LayerIndex & 0xF0 ) >> 4);
    Buffer[3] = (this->DeviceIndex & 0x0F);

    // Write out the X, Y, Z and Temperature values...
    memcpy(&(Buffer[4]), (uint8_t*)&(this->FieldX), 2);
    memcpy(&(Buffer[6]), (uint8_t*)&(this->FieldY), 2);
    memcpy(&(Buffer[8]), (uint8_t*)&(this->FieldZ), 2);
    memcpy(&(Buffer[10]), (uint8_t*)&(this->Temperature), 2);

    // Write out the timestamp value...
    memcpy(&(Buffer[12]), (uint8_t*)&(this->Timestamp), 4);

    // Write out an EOL character to easily signal when the data message is finished.
    Buffer[16] = '\n';

    return SENSOR_READING_BYTE_LENGTH;
}

ALS31313_t::ALS31313_t(uint8_t LayerIndex, uint8_t DeviceIndex) {

    this->LayerIndex = LayerIndex;
    this->DeviceIndex = DeviceIndex;

    return;
}

uint8_t ALS31313_t::DetermineI2CAddress() {
    return ( this->LayerIndex << 4 ) | this->DeviceIndex;
}

bool ALS31313_t::AssertAddress(I2CBus_t& Bus, uint8_t I2CAddress) {

    uint8_t CurrentAddress = this->ValidateI2CAddress(Bus, I2CAddress);
    if ( CurrentAddress == NO_VALID_ADDRESS ) {
        return false;
    }

    if ( ! this->EnableWriteAccess(Bus) ) {
        return false;
    }

    if ( CurrentAddress == I2CAddress ) {
        return true;
    }

    return this->SetI2CAddress(Bus, I2CAddress);
}

bool ALS31313_t::InitializeSettings(I2CBus_t& Bus) {

    // Set Register 0x02 first. Read the current settings and only change the masked values;
    uint32_t Register0x02 = this->ReadRegister(Bus, BW_SELECT_ADDRESS) & REGISTER_0x02_CLEAR_MASK;

    // Then get Register 0x27 for the remaining few settings.
    uint32_t Register0x27 = this->ReadRegister(Bus, I2C_LOOP_MODE_ADDRESS) & REGISTER_0x27_CLEAR_MASK;

    // Set the bandwidth-select setting of the sensor...
    Register0x02 |= Bandwidth_Select_HighSpeed_WithFilter;

    // Set the Hall-Effect sensing mode of the sensor...
    Register0x02 |= Hall_Effect_Mode_SingleEnded;

    // Enable or disable the I2C CRC for data integrity checks...
    Register0x02 |= I2C_CRC_Disabled;

    // Set the I2C voltage threshold level to 3.3V operation...
    Register0x02 |= I2C_Voltage_Threshold_High_3V_Mode;

    // Enable the X, Y, and Z channels of the magnetic field sensing...
    Register0x02 |= Axis_Enable_XYZ;

    // Turn off all low-power mode options...
    Register0x27 |= Sensor_Power_Mode_Standard;

    // Turn off I2C loop mode...
    Register0x27 |= I2C_Loop_Mode_Single;

    return this->WriteRegister(Bus, BW_SELECT_ADDRESS, Register0x02) && this->WriteRegister(Bus, I2C_LOOP_MODE_ADDRESS, Register0x27);
}

bool ALS31313_t::ReadMeasurements(I2CBus_t& Bus, MagneticSensorReading_t& CurrentReading) const {

    // Perform two 4-byte reads of registers 0x28 and 0x29
    uint32_t MSBs = this->ReadRegister(Bus, SENSOR_MSBs_ADDRESS);
    uint32_t LSBs = this->ReadRegister(Bus, SENSOR_LSBs_ADDRESS);

    // Verify that the new_data flag is set for this reading, returning if not.
    bool NewData = (0 != (MSBs & NEW_DATA_MASK));
    if ( ! NewData ) {
        return false;
    }

    CurrentReading.DeviceIndex = this->DeviceIndex;
    CurrentReading.LayerIndex  = this->LayerIndex;
    CurrentReading.I2CAddress  = this->I2CAddress;

    // Extract out the MSBs and LSBs of each field axis and the temperature reading, stitching them together into the full 12-bit output values.
    CurrentReading.FieldX       = ((MSBs & FIELD_X_MSB_MASK) >> FIELD_X_MSB_RIGHT_SHIFT) | ((LSBs & FIELD_X_LSB_MASK) >> FIELD_X_LSB_RIGHT_SHIFT);
    CurrentReading.FieldY       = ((MSBs & FIELD_Y_MSB_MASK) >> FIELD_Y_MSB_RIGHT_SHIFT) | ((LSBs & FIELD_Y_LSB_MASK) >> FIELD_Y_LSB_RIGHT_SHIFT);
    CurrentReading.FieldZ       = ((MSBs & FIELD_Z_MSB_MASK) >> FIELD_Z_MSB_RIGHT_SHIFT) | ((LSBs & FIELD_Z_LSB_MASK) >> FIELD_Z_LSB_RIGHT_SHIFT);
    CurrentReading.Temperature  = ((MSBs & TEMPERATURE_MSB_MASK) << TEMPERATURE_MSB_LEFT_SHIFT) | ((LSBs & TEMPERATURE_LSB_MASK) >> TEMPERATURE_LSB_RIGHT_SHIFT);

    return true;
}

uint8_t ALS31313_t::ValidateI2CAddress(I2CBus_t& Bus, uint8_t Expected) {

    // Does the default address exist on the bus?
    if ( Bus.AddressExists(DEFAULT_ADDRESS) ) {
        this->I2CAddress = DEFAULT_ADDRESS;
        return this->I2CAddress;
    }

    // Is there already a device with the expected address on the bus?
    if ( Bus.AddressExists(Expected) ) {
        this->I2CAddress = Expected;
        return this->I2CAddress;
    }

    // Is there ANY address on the bus??
    for ( uint8_t Address = MINIMUM_ADDRESS; Address <= MAXIMUM_ADDRESS; Address++ ) {

        // Skip the addresses we've already checked
        if (( Address == DEFAULT_ADDRESS) || ( Address == Expected )) {
            continue;
        }

        if ( Bus.AddressExists(Address)) {
            this->I2CAddress = Address;
            return this->I2CAddress;
        }
    }

    // If, somehow, this fails, return an error code since this is obviously failing.
    return this->I2CAddress = NO_VALID_ADDRESS, NO_VALID_ADDRESS;
}

bool ALS31313_t::EnableWriteAccess(I2CBus_t& Bus) const {

    // Send the write enable code to the write-enable address.
    return this->WriteRegister(Bus, WRITE_ACCESS_ADDRESS, WRITE_ACCESS_DATA);
}

bool ALS31313_t::SetI2CAddress(I2CBus_t& Bus, uint8_t NewAddress) {

    // Read the current contents of the 32-bit register holding the I2C slave address of the sensor,
    // so that when writing the new address we don't change any bits other than the 8 address bits.
    uint32_t CurrentRegister = this->ReadRegister(Bus, I2C_SLAVE_ADDRESS_ADDRESS);

    // Clear the bits associated with the I2C address
    uint32_t UpdatedRegister = CurrentRegister & ~(I2C_SLAVE_ADDRESS_MASK);

    // Write in the new address to the existing register contents
    UpdatedRegister |= ((uint32_t)NewAddress << I2C_SLAVE_ADDRESS_BIT_OFFSET);

    // Write the new I2C address to the device, over I2C using the existing address.
    if ( ! this->WriteRegister(Bus, I2C_SLAVE_ADDRESS_ADDRESS, UpdatedRegister) ) {
        return false;
    }

    this->I2CAddress = NewAddress;
    return true;
}

uint32_t ALS31313_t::ReadRegister(I2CBus_t& Bus, uint8_t RegisterAddress) const {

    // Write the I2C address and wait for ACK/NACK
    Bus.beginTransmission(this->I2CAddress);

    // Write the Register address and wait for ACK/NACK
    Bus.write(RegisterAddress);

    // Request the 4 byte contents of the register, sending a STOP once all bytes have been read.
    Bus.requestFrom(this->I2CAddress, (uint8_t)sizeof(uint32_t), (uint8_t)true);
    uint32_t RegisterContents = 0;
    while ( Bus.available() ) {
        RegisterContents = ((RegisterContents << 8) | Bus.read());
    }

    return RegisterContents;
}

bool ALS31313_t::WriteRegister(I2CBus_t& Bus, uint8_t RegisterAddress, uint32_t Data) const {

    // Write the I2C address and wait for ACK/NACK
    Bus.beginTransmission(this->I2CAddress);

    // Write the 8-bit register address and wait for the ACK/NACK;
    Bus.write(RegisterAddress);

    // Write the 4-byte register contents, waiting for ACK/NACK between each byte;
    size_t nWritten = Bus.write(Data);
    if ( nWritten != 4 ) {
        // ...
    }

    // Send the STOP condition and release the bus.
    uint8_t Error = Bus.endTransmission(true);
    return (Error == 0);
}

/* --- End Struct/Class Method Definitions --- */
