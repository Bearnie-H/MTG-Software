/*
    JSMP001.ino

    Author(s):  Joseph Sadden
    Date:       15th December, 2023

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
#include "include/JSMP001.h"
/* --- End Custom Header Includes --- */

/* +++ Begin Private Macro Definitions +++ */
/* --- End Private Macro Definitions --- */

/* +++ Begin Private Typedefs +++ */
/* --- End Private Typedefs --- */

/* +++ Begin Private Constant Definitions +++ */
const SN74LV4051A_Multiplexer_Channel_t LayerSelectionTranslationTable[MAXIMUM_LAYERS] = {
    CHANNEL_2,
    CHANNEL_1,
    CHANNEL_0,
    CHANNEL_3,
    CHANNEL_4,
    CHANNEL_6,
    CHANNEL_7,
    CHANNEL_5,
};

const SN74LV4051A_Multiplexer_Channel_t DeviceSelectionTranslationTable[DEVICES_PER_LAYER] = {
    CHANNEL_1,
    CHANNEL_0,
    CHANNEL_3,
    CHANNEL_4,
    CHANNEL_2,
    CHANNEL_5,
    CHANNEL_7,
    CHANNEL_6,
};
/* --- End Private Constant Definitions --- */

/* +++ Begin Private Function Declarations +++ */
/* --- End Private Function Declarations --- */

/* +++ Begin Private Function Definitions +++ */
/* --- End Private Function Definitions --- */

/* +++ Begin Struct/Class Method Definitions +++ */
MagneticSensorArray_t::MagneticSensorArray_t(SN74LV4051A_Multiplexer_t &LayerSelect, SN74LV4051A_Multiplexer_t &DeviceSelect, I2CBus_t &I2CBus) {

    this->LayerSelect = LayerSelect;
    this->DeviceSelect = DeviceSelect;

    this->I2CBus = I2CBus;

    this->LayerIndex = 0;
    this->DeviceIndex = 0;
    this->Initialized = false;

    for (int i = 0; i < MAXIMUM_LAYERS; i++) {
        this->DeviceMasks[i] = (uint8_t)(~0U);
    }

    return;
}

SensorError_t MagneticSensorArray_t::InitializeHardware() {

    this->LayerIndex = 0;
    this->DeviceIndex = 0;

    this->LayerSelect.EnableDevice();
    this->DeviceSelect.EnableDevice();
    this->ResetAddressingMasks();

    ALS31313_t Sensor = this->CurrentSensor();
    for ( int i = 0; i < (MAXIMUM_LAYERS * DEVICES_PER_LAYER); i++ ) {

        this->LayerSelect.EnableOutputChannel(LayerSelectionTranslationTable[this->LayerIndex]);
        this->DeviceSelect.EnableOutputChannel(DeviceSelectionTranslationTable[this->DeviceIndex]);

        Log_InitializeSensor(this->LayerIndex, this->DeviceIndex);
        SensorError_t InitializationError = Sensor.DefaultInitialize(this->I2CBus);
        Log_SensorInitializationStatus(this->LayerIndex, this->DeviceIndex, InitializationError);

        if ( InitializationError != Success ) {
            // If an error occurs, mask this device to not be included in the iterator.
            Log_DisablingDevice(this->LayerIndex, this->DeviceIndex, InitializationError);
            this->DeviceMasks[this->LayerIndex] &= ~(0b1 << this->DeviceIndex);
        }

        Sensor = this->NextSensor();
    }

    // Check each layer, masking out any layers with 0 devices from the iterator.
    uint8_t AnyDevices = 0x00;
    for ( int i = 0; i < MAXIMUM_LAYERS; i++ ) {
        if ( 0 == this->DeviceMasks[i] ) {
            Log_DisablingLayer(i);
        }
        AnyDevices |= this->DeviceMasks[i];
    }

    if ( AnyDevices == 0 ) {
        return DeviceNotFound;
    }

    this->LayerIndex = MAXIMUM_LAYERS;
    this->DeviceIndex = DEVICES_PER_LAYER;
    this->Initialized = true;

    return Success;
}

ALS31313_t MagneticSensorArray_t::CurrentSensor() {
    return ALS31313_t(this->LayerIndex, this->DeviceIndex, this->Initialized);
}

SensorError_t MagneticSensorArray_t::ReadNextDevice(MagneticSensorReading_t &CurrentMeasurement) {
    ALS31313_t Device = this->NextSensor();
    this->LayerSelect.EnableOutputChannel(LayerSelectionTranslationTable[this->LayerIndex]);
    this->DeviceSelect.EnableOutputChannel(DeviceSelectionTranslationTable[this->DeviceIndex]);
    return Device.ReadMeasurements(this->I2CBus, CurrentMeasurement);
}

void MagneticSensorArray_t::ResetAddressingMasks(void) {

    for ( int i = 0; i < MAXIMUM_LAYERS; i++ ) {
        this->DeviceMasks[i] = (uint8_t)(~0U);
    }

    return;
}

ALS31313_t MagneticSensorArray_t::NextSensor() {

    uint8_t InitialLayerIndex = this->LayerIndex;
    uint8_t InitialDeviceIndex = this->DeviceIndex;
    uint16_t CurrentMask = 0x00;

    do {
        this->DeviceIndex++;

        if ( this->DeviceIndex >= DEVICES_PER_LAYER ) {
            this->DeviceIndex = 0;
            this->LayerIndex = (this->LayerIndex + 1) % MAXIMUM_LAYERS;
        }

        CurrentMask = this->DeviceMasks[this->LayerIndex];
        CurrentMask >>= (uint16_t)this->DeviceIndex;

        if (( CurrentMask & 0b1 ) == 0b1 ) {
            break;
        }
    } while ( !(( this->LayerIndex == InitialLayerIndex ) && ( this->DeviceIndex == InitialDeviceIndex )));

    Log_ReportNextSensor(this->LayerIndex, this->DeviceIndex);

    return ALS31313_t(this->LayerIndex, this->DeviceIndex, this->Initialized);
}

/* --- End Struct/Class Method Definitions --- */

#if defined(DEBUG)

void Log_ReportNextSensor(uint8_t LayerIndex, uint8_t DeviceIndex) {

    Serial.print("Next sensor: (L,D) = ");
    Serial.print(LayerIndex);
    Serial.print(",");
    Serial.println(DeviceIndex);


    return;
}

void Log_InitializeSensor(uint8_t LayerIndex, uint8_t DeviceIndex) {

    Serial.print("Default initializing ALS31313 to ");
    Log_I2CAddress(ALS31313_t(LayerIndex, DeviceIndex).I2CAddressFromIndices());

    return;
}

void Log_SensorInitializationStatus(uint8_t LayerIndex, uint8_t DeviceIndex, SensorError_t InitializationError) {

    Serial.print("Initialized ALS31313 at: (L,D) = ");
    Serial.print(LayerIndex);
    Serial.print(",");
    Serial.print(DeviceIndex);
    Serial.print(" with status: ");

    Log_SensorError(InitializationError);

    return;
}

void Log_DisablingDevice(uint8_t LayerIndex, uint8_t DeviceIndex, SensorError_t InitializationError) {

    Serial.print("Marking ALS31313 at: ");
    Serial.print(LayerIndex);
    Serial.print(",");
    Serial.print(DeviceIndex);
    Serial.print(" disabled - error: ");
    Log_SensorError(InitializationError);

    return;
}

void Log_DisablingLayer(uint8_t LayerIndex) {

    Serial.print("Disabling layer ");
    Serial.print(LayerIndex);
    Serial.println(", no devices initialized");

    return;
}

#else

void Log_ReportNextSensor(uint8_t LayerIndex, uint8_t DeviceIndex) {
    return;
}

void Log_InitializeSensor(uint8_t LayerIndex, uint8_t DeviceIndex) {
    return;
}

void Log_SensorInitializationStatus(uint8_t LayerIndex, uint8_t DeviceIndex, SensorError_t InitializationError) {
    return;
}

void Log_DisablingDevice(uint8_t LayerIndex, uint8_t DeviceIndex, SensorError_t InitializationError) {
    return;
}

void Log_DisablingLayer(uint8_t LayerIndex) {
    return;
}

#endif