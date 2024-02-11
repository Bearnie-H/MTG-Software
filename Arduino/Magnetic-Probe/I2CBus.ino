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
#define I2C_WIRE_SPEED (uint32_t)100000  // Hz
#define I2C_WIRE_TIMEOUT (uint32_t)(((1.0 / I2C_WIRE_SPEED) * 1e6) * 500) // µs
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

    this->EnablePin = NOT_DISABLEABLE;
    this->Enabled = true;
    this->Wire = Wire;

    this->Wire.setClock(I2C_WIRE_SPEED);
    this->Wire.setWireTimeout(I2C_WIRE_TIMEOUT, true);
    this->Wire.begin();

    return;
}

I2CBus_t::I2CBus_t(Pin_t Enable) {

    this->EnablePin = Enable;
    this->Enabled = false;
    this->Wire = Wire;

    pinMode(this->EnablePin, OUTPUT);
    digitalWrite(this->EnablePin, LOW);

    this->Wire.setClock(I2C_WIRE_SPEED);
    this->Wire.setWireTimeout(I2C_WIRE_TIMEOUT, true);
    this->Wire.begin();

    return;
}

bool I2CBus_t::Enable() {

    if ( this->EnablePin != NOT_DISABLEABLE ) {
        Log_EnableI2CBus(this->EnablePin);
        pinMode(this->EnablePin, OUTPUT);
        digitalWrite(this->EnablePin, HIGH);
        if ( ! this->Enabled ) {
            this->Wire.begin();
        }
        this->Enabled = true;
    }

    return true;
}

bool I2CBus_t::Disable() {

    if ( this->EnablePin != NOT_DISABLEABLE ) {
        Log_DisableI2CBus(this->EnablePin);
        digitalWrite(this->EnablePin, LOW);
        pinMode(this->EnablePin, INPUT);
        if ( this->Enabled ) {
            this->Wire.end();
        }
        this->Enabled = false;
    }

    return true;
}

bool I2CBus_t::IsEnabled() {
    return this->Enabled;
}

bool I2CBus_t::Empty() {

    this->Wire.beginTransmission(EMPTY_BUS_TEST_ADDRESS);;
    uint8_t Error = this->Wire.endTransmission(true);

    // Only on a timeout can we be certain the bus is empty of responding devices.
    bool Empty = (Error == 5);
    Log_BusEmpty(Empty);

    return Empty;
}

bool I2CBus_t::AddressExists(uint8_t Address) {

    // Log_I2CBusAddressCheck(Address);

    // Send a message to the given address, and check if we get an ACK or a NACK back.
    // ...
    this->Wire.beginTransmission(Address);
    uint8_t error = this->Wire.endTransmission(true);

    switch (error) {
        case 0:
            Log_I2CBusAddressExists(Address, true);
            return true;
        case 2:
            // Log_I2CBusAddressExists(Address, false);
            return false;
        default:
            // Some unexpected error occurred.
            Log_I2CBusAddressCheckUnexpectedError(Address, error);
            // ...
        return false;
    }
}

#if defined(DEBUG)

void Log_EnableI2CBus(Pin_t EnablePin) {

    Serial.print("Enabling I2C bus: ");
    Serial.print(EnablePin);
    Serial.println(" (HIGH)");

    return;
}

void Log_CheckBusEmpty(void) {

    Serial.println("Checking if I2C Bus is empty...");
    return;
}

void Log_BusEmpty(bool IsEmpty) {

    Serial.print("I2C Bus is ");
    if ( ! IsEmpty ) {
        Serial.print("not ");
    }
    Serial.println("empty");

    return;
}

void Log_I2CBusAddressCheck(uint8_t Address) {

    Serial.print("Checking I2C Bus for ");
    Log_I2CAddress(Address);

    return;
}

void Log_I2CBusAddressExists(uint8_t Address, bool Exist) {

    Serial.print("Address ");
    Serial.print(Address);
    Serial.print(" does ");
    if ( ! Exist ) {
        Serial.print("not ");
    }
    Serial.println("exist on the bus");

    return;
}

void Log_I2CBusAddressCheckUnexpectedError(uint8_t Address, uint8_t Error) {

    Serial.print("Failed to check bus for ");
    Log_I2CAddress(Address, false);

    Serial.print(" - error: ");
    switch (Error) {
        case 1:
            Serial.println("Data too long");
            break;
        case 2:
            Serial.println("Addr NACK");
            break;
        case 3:
            Serial.println("Data NACK");
            break;
        case 4:
            Serial.println("Unknown");
            break;
        case 5:
            Serial.println("Timeout");
            break;
        default:
            Serial.println("None");
    }

    return;
}

void Log_DisableI2CBus(Pin_t EnablePin) {

    Serial.print("Disabling I2C bus: ");
    Serial.print(EnablePin);
    Serial.println(" (LOW)");

    return;
}

#else

void Log_EnableI2CBus(Pin_t EnablePin) {
    return;
}

void Log_CheckBusEmpty(void) {
    return;
}

void Log_BusEmpty(bool IsEmpty) {
    return;
}

void Log_I2CBusAddressCheck(uint8_t Address) {
    return;
}

void Log_I2CBusAddressCheckUnexpectedError(uint8_t Address, uint8_t Error) {
    return;
}

void Log_I2CBusAddressExists(uint8_t Address, bool Exist) {
    return;
}

void Log_DisableI2CBus(Pin_t EnablePin) {
    return;
}

#endif
