/*
    SN74LV4051A.ino

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
#include "include/SN74LV4051A.h"
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
SN74LV4051A_Multiplexer_t::SN74LV4051A_Multiplexer_t(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, Pin_t Enable) {

    this->ADDR_0 = Addr0;
    this->ADDR_1 = Addr1;
    this->ADDR_2 = Addr2;

    this->Enable = Enable;

    pinMode(Addr0, OUTPUT);
    pinMode(Addr1, OUTPUT);
    pinMode(Addr2, OUTPUT);

    pinMode(Enable, OUTPUT);

    this->DisableDevice();

    return;
}

SN74LV4051A_Multiplexer_t::~SN74LV4051A_Multiplexer_t() {

    this->DisableDevice();

    return;
}

void SN74LV4051A_Multiplexer_t::DisableDevice() {

    // Log_DisableMux(this->ADDR_0, this->ADDR_1, this->ADDR_2, this->Enable);

    digitalWrite(this->Enable, HIGH);

    this->doEnableOutputChannel(CHANNEL_0);

    return;
}

void SN74LV4051A_Multiplexer_t::EnableDevice() {

    // Log_EnableMux(this->ADDR_0, this->ADDR_1, this->ADDR_2, this->Enable);

    // The enable signal for the SN74LV4051A is active-low, so pull the enable pin low.
    digitalWrite(this->Enable, LOW);

    return;
}

void SN74LV4051A_Multiplexer_t::EnableOutputChannel(SN74LV4051A_Multiplexer_Channel_t Channel) {

    this->DisableDevice();
    // Log_EnableMuxChannel(this->ADDR_0, this->ADDR_1, this->ADDR_2, Channel);
    this->doEnableOutputChannel(Channel);
    this->EnableDevice();

    return;
}

void SN74LV4051A_Multiplexer_t::doEnableOutputChannel(SN74LV4051A_Multiplexer_Channel_t Channel) {

    digitalWrite(this->ADDR_0, (Channel >> 0) & 0b001);
    digitalWrite(this->ADDR_1, (Channel >> 1) & 0b001);
    digitalWrite(this->ADDR_2, (Channel >> 2) & 0b001);

    return;
}

/* --- End Struct/Class Method Definitions --- */

#if defined(DEBUG)

void Log_EnableMux(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, Pin_t Enable) {

    Serial.print("Enabling Mux: ");
    Log_PinDigitalLevel(Addr0, digitalRead(Addr0), false);
    Serial.print(", ");
    Log_PinDigitalLevel(Addr1, digitalRead(Addr1), false);
    Serial.print(", ");
    Log_PinDigitalLevel(Addr2, digitalRead(Addr2), false);
    Serial.print(", (Enable): ");
    Log_PinDigitalLevel(Enable, LOW);

    return;
}

void Log_EnableMuxChannel(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, SN74LV4051A_Multiplexer_Channel_t Channel) {

    Serial.print("Enabling Channel ");
    Serial.print(Channel);
    Serial.print(" of Mux: ");
    Log_PinDigitalLevel(Addr0, Channel & 0b001, false);
    Serial.print(", ");
    Log_PinDigitalLevel(Addr1, Channel & 0b010, false);
    Serial.print(", ");
    Log_PinDigitalLevel(Addr2, Channel & 0b100);

    return;
}

void Log_DisableMux(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, Pin_t Enable) {

    Serial.print("Disabling Mux: ");
    Log_PinDigitalLevel(Addr0, digitalRead(Addr0), false);
    Serial.print(", ");
    Log_PinDigitalLevel(Addr1, digitalRead(Addr1), false);
    Serial.print(", ");
    Log_PinDigitalLevel(Addr2, digitalRead(Addr2), false);
    Serial.print(", (Enable): ");
    Log_PinDigitalLevel(Enable, HIGH);

    return;
}

#else

void Log_EnableMux(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, Pin_t Enable) {
    return;
}

void Log_EnableMuxChannel(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, SN74LV4051A_Multiplexer_Channel_t Channel) {
    return;
}

void Log_DisableMux(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, Pin_t Enable) {
     return;
}

#endif
