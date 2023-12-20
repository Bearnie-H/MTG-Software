/*
    I2CBus.ino

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
#include "include/I2CBus.h"
/* --- End Custom Header Includes --- */

/* +++ Begin Private Macro Definitions +++ */
/* --- End Private Macro Definitions --- */

/* +++ Begin Private Typedefs +++ */
/* --- End Private Typedefs --- */

/* +++ Begin Private Constant Definitions +++ */
/* --- End Private Constant Definitions --- */

/* +++ Begin Private Function Declarations +++ */
/* --- End Private Function Declarations --- */

/* +++ Begin Private Function Definitions +++ */
/* --- End Private Function Definitions --- */

/* +++ Begin Struct/Class Method Definitions +++ */
/* --- End Struct/Class Method Definitions --- */

I2CBus_t::I2CBus_t() {

    this->DataPin = PIN_WIRE_SDA;
    this->ClockPin = PIN_WIRE_SCL;
    this->EnablePin = NOT_DISABLEABLE;
    this->Enabled = true;

    Wire.begin();

    return;
}

I2CBus_t::I2CBus_t(Pin_t Data, Pin_t Clock, Pin_t Enable) {

    this->DataPin = Data;
    this->DataPin = Clock;
    this->EnablePin = Enable;

    this->Disable();

    Wire.begin();

    return;
}

bool I2CBus_t::Enable() {

    if ( this->EnablePin != NOT_DISABLEABLE ) {
        digitalWrite(this->EnablePin, HIGH);
        if ( ! this->Enabled ) {
            Wire.begin();
        }
        this->Enabled = true;
    }

    return true;
}

bool I2CBus_t::Disable() {

    if ( this->EnablePin != NOT_DISABLEABLE ) {
        digitalWrite(this->EnablePin, LOW);
        if ( this->Enabled ) {
            Wire.end();
        }
        this->Enabled = false;
    }

    return true;
}

bool I2CBus_t::IsEnabled() const {
    return this->Enabled;
}

bool  I2CBus_t::AddressExists(uint8_t Address) {

    // Send a message to the given address, and check if we get an ACK or a NACK back.
    // ...
    Wire.beginTransmission(Address);
    uint8_t error = Wire.endTransmission(true);

    switch (error) {
        case 0:
            return true;
        case 2:
            return false;
        default:
            // Some unexpected error occurred.
            // ...
        return false;
    }
}