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
    this->LayerMask = (uint16_t)((1 << MAXIMUM_LAYERS) - 1);

    this->DeviceIndex = 0;

    return;
}

bool MagneticSensorArray_t::InitializeI2CAddressing() {

    this->LayerSelect.EnableDevice();
    this->DeviceSelect.EnableDevice();

    do {
        ALS31313_t Sensor = this->NextSensor();

        this->LayerSelect.EnableOutputChannel(LayerSelectionTranslationTable[this->LayerIndex]);
        this->DeviceSelect.EnableOutputChannel(DeviceSelectionTranslationTable[this->DeviceIndex]);

        if ( ! Sensor.AssertAddress(this->I2CBus, Sensor.DetermineI2CAddress()) ) {
            this->LayerMask &= ~(1 << this->LayerIndex);
            this->NextLayer();
        }

        if ( ! Sensor.InitializeSettings(this->I2CBus) ) {
            return false;
        }

    } while ( this->DeviceIndex != 0 && this->LayerIndex != 0 );

    this->LayerIndex = 0;
    this->DeviceIndex = 0;

    this->LayerSelect.DisableDevice();
    this->DeviceSelect.DisableDevice();

    // Enable the shared I2C bus, rather than using the multiplexors
    this->I2CBus.Enable();

    return this->LayerMask != 0;
}

bool MagneticSensorArray_t::ReadNextDevice(MagneticSensorReading_t &CurrentMeasurement) {

    ALS31313_t Sensor = this->NextSensor();

    this->LayerSelect.EnableOutputChannel(LayerSelectionTranslationTable[this->LayerIndex]);
    this->DeviceSelect.EnableOutputChannel(DeviceSelectionTranslationTable[this->DeviceIndex]);

    return Sensor.ReadMeasurements(this->I2CBus, CurrentMeasurement);
}

ALS31313_t MagneticSensorArray_t::NextSensor() {

    this->DeviceIndex++;
    if ( this->DeviceIndex >= DEVICES_PER_LAYER ) {
        this->NextLayer();
    }

    return ALS31313_t(this->LayerIndex, this->DeviceIndex);
}

bool MagneticSensorArray_t::NextLayer() {

    this->DeviceIndex = 0;
    uint16_t LayerMask = (this->LayerMask >> this->LayerIndex);

    do {
        LayerMask >>= 1;
        this->LayerIndex++;

        if ( this->LayerIndex >= MAXIMUM_LAYERS ) {
            this->LayerIndex = 0;
            LayerMask = this->LayerMask;
        }
    } while ((LayerMask & 1) != 0);

    return true;
}

/* --- End Struct/Class Method Definitions --- */
