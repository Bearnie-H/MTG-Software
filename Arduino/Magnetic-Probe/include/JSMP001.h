/*
    JSMP001.h

    Author: Joseph Sadden
    Date:   15th December, 2023

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

#ifndef JSMP001_H
#define JSMP001_H

#ifdef __cplusplus
extern "C" {
#endif

/* +++ Begin Top-level Header Includes +++ */
#include <stdbool.h>
#include <stdint.h>

#include <Wire.h>
/* --- End Top-level Header Includes --- */

/* +++ Begin Local Header Includes +++ */
/* --- End Local Header Includes --- */
#include "common.h"
#include "ALS31313.h"
#include "SN74LV4051A.h"
#include "I2CBus.h"

/* +++ Begin Library Macro Definitions +++ */
#define DEVICES_PER_LAYER   8   // The number of ALS31313 sensors per daughterboard
#define MAXIMUM_LAYERS      8   // The maximum number of daughterboards supported by the motherboard.
/* --- End Library Macro Definitions --- */

/* +++ Begin Library Constant Definitions +++ */
extern const SN74LV4051A_Multiplexer_Channel_t LayerSelectionTranslationTable[MAXIMUM_LAYERS];
extern const SN74LV4051A_Multiplexer_Channel_t DeviceSelectionTranslationTable[DEVICES_PER_LAYER];
/* --- End Library Constant Definitions --- */

/* +++ Begin Library Typedefs +++ */
class MagneticSensorArray_t {

    SN74LV4051A_Multiplexer_t LayerSelect;
    SN74LV4051A_Multiplexer_t DeviceSelect;

    I2CBus_t I2CBus;

    uint8_t DeviceIndex;
    uint8_t LayerIndex;

    uint8_t DeviceMasks[MAXIMUM_LAYERS] = { (~0U) };

    bool Initialized;

    public:
        MagneticSensorArray_t(SN74LV4051A_Multiplexer_t &LayerSelect, SN74LV4051A_Multiplexer_t &DeviceSelect, I2CBus_t &I2CBusEnable);
        ~MagneticSensorArray_t() = default;

        SensorError_t InitializeHardware(void);
        ALS31313_t CurrentSensor(void);
        SensorError_t ReadNextDevice(MagneticSensorReading_t &CurrentMeasurement);
        void ResetAddressingMasks(void);

    private:
        ALS31313_t NextSensor(void);
};
/* --- End Library Typedefs --- */

/* +++ Begin Library Function Definitions +++ */
void Log_ReportNextSensor(uint8_t LayerIndex, uint8_t DeviceIndex);
void Log_InitializeSensor(uint8_t LayerIndex, uint8_t DeviceIndex);
void Log_SensorInitializationStatus(uint8_t LayerIndex, uint8_t DeviceIndex, SensorError_t InitializationError);
void Log_DisablingDevice(uint8_t LayerIndex, uint8_t DeviceIndex, SensorError_t InitializationError);
void Log_DisablingLayer(uint8_t LayerIndex);
/* --- End Library Function Definitions --- */

#ifdef __cplusplus
}
#endif

#endif
