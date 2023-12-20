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

    return;
}

SN74LV4051A_Multiplexer_t::~SN74LV4051A_Multiplexer_t() {

    this->DisableDevice();

    return;
}

bool SN74LV4051A_Multiplexer_t::DisableDevice() const {

    digitalWrite(this->Enable, HIGH);

    digitalWrite(this->ADDR_0, LOW);
    digitalWrite(this->ADDR_1, LOW);
    digitalWrite(this->ADDR_2, LOW);

    return true;
}

bool SN74LV4051A_Multiplexer_t::EnableDevice() const {

    digitalWrite(this->ADDR_0, LOW);
    digitalWrite(this->ADDR_1, LOW);
    digitalWrite(this->ADDR_2, LOW);

    digitalWrite(this->Enable, LOW);

    return true;
}

bool SN74LV4051A_Multiplexer_t::EnableOutputChannel(SN74LV4051A_Multiplexer_Channel_t Channel) const {

    this->DisableDevice();

    // Should I just replace this with the boolean operators on the Channel_t type directly? Or just rely on the compiler to do this for me?
    switch (Channel) {
        case CHANNEL_0:
            digitalWrite(this->ADDR_0, LOW);
            digitalWrite(this->ADDR_1, LOW);
            digitalWrite(this->ADDR_2, LOW);
            break;
        case CHANNEL_1:
            digitalWrite(this->ADDR_0, HIGH);
            digitalWrite(this->ADDR_1, LOW);
            digitalWrite(this->ADDR_2, LOW);
            break;
        case CHANNEL_2:
            digitalWrite(this->ADDR_0, LOW);
            digitalWrite(this->ADDR_1, HIGH);
            digitalWrite(this->ADDR_2, LOW);
            break;
        case CHANNEL_3:
            digitalWrite(this->ADDR_0, HIGH);
            digitalWrite(this->ADDR_1, HIGH);
            digitalWrite(this->ADDR_2, LOW);
            break;
        case CHANNEL_4:
            digitalWrite(this->ADDR_0, LOW);
            digitalWrite(this->ADDR_1, LOW);
            digitalWrite(this->ADDR_2, HIGH);
            break;
        case CHANNEL_5:
            digitalWrite(this->ADDR_0, HIGH);
            digitalWrite(this->ADDR_1, LOW);
            digitalWrite(this->ADDR_2, HIGH);
            break;
        case CHANNEL_6:
            digitalWrite(this->ADDR_0, LOW);
            digitalWrite(this->ADDR_1, HIGH);
            digitalWrite(this->ADDR_2, HIGH);
            break;
        case CHANNEL_7:
            digitalWrite(this->ADDR_0, HIGH);
            digitalWrite(this->ADDR_1, HIGH);
            digitalWrite(this->ADDR_2, HIGH);
            break;
    }

    this->EnableDevice();

    return true;
}

/* --- End Struct/Class Method Definitions --- */
