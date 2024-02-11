/*
    common.ino

    Author(s):  Joseph Sadden
    Date:       1st February, 2024

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
#include "include/common.h"
/* --- End Custom Header Includes --- */

#if defined (DEBUG)

void Log_OperationFailed(void) {

    Serial.println("Error: Operation Failed");

    return;
}

void Log_DeviceAddress(uint8_t I2CAddress, bool Newline) {

    Serial.print("device ");
    Serial.print(I2CAddress);

    if ( Newline ) {
        Serial.println();
    }

    return;
}

void Log_I2CAddress(uint8_t I2CAddress, bool Newline) {

    Serial.print("address ");
    Serial.print(I2CAddress);

    if ( Newline ) {
        Serial.println();
    }

    return;
}

void Log_PinDigitalLevel(Pin_t Pin, bool State, bool Newline) {

    Serial.print("Pin ");
    Serial.print(Pin);
    Serial.print("=");
    Serial.print(State == 0 ? "LOW" : "HIGH");

    if ( Newline ) {
        Serial.println();
    }

    return;
}

#else

void Log_OperationFailed(void) {
    return;
}

void Log_DeviceAddress(uint8_t I2CAddress, bool Newline) {
    return;
}

void Log_I2CAddress(uint8_t I2CAddress, bool Newline) {
    return;
}

#endif
