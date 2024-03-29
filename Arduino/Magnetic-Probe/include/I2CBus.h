/*
    I2CBus.h

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

#ifndef I2CBUS_H
#define I2CBUS_H

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

/* +++ Begin Library Macro Definitions +++ */
#define NOT_DISABLEABLE (uint8_t)-1

#define EMPTY_BUS_TEST_ADDRESS (uint8_t)42
#define NO_VALID_ADDRESS (uint8_t)-1
#define RESERVED_I2C_ADDRESS_END (uint8_t)8
/* --- End Library Macro Definitions --- */

/* +++ Begin Library Typedefs +++ */
class I2CBus_t {

    Pin_t EnablePin;
    bool Enabled;

    public:
        TwoWire Wire;

        I2CBus_t();
        I2CBus_t(Pin_t Enable);

        bool IsEnabled();
        bool Enable();
        bool Disable();

        bool AddressExists(uint8_t Address);
        bool Empty();
        // ...
};
/* --- End Library Typedefs --- */

/* +++ Begin Library Function Definitions +++ */
void Log_EnableI2CBus(Pin_t EnablePin);
void Log_CheckBusEmpty(void);
void Log_BusEmpty(bool IsEmpty);
void Log_I2CBusAddressCheck(uint8_t Address);
void Log_I2CBusAddressExists(uint8_t Address, bool Exist);
void Log_I2CBusAddressCheckUnexpectedError(uint8_t Address, uint8_t Error);
void Log_DisableI2CBus(Pin_t EnablePin);
/* --- End Library Function Definitions --- */

#ifdef __cplusplus
}
#endif

#endif
