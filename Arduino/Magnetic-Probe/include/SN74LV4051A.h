/*
    SN74LV4051A.h

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

#ifndef SN74LV4051A_H
#define SN74LV4051A_H

#ifdef __cplusplus
extern "C" {
#endif

/* +++ Begin Top-level Header Includes +++ */
#include <stdbool.h>
#include <stdint.h>
/* --- End Top-level Header Includes --- */

/* +++ Begin Local Header Includes +++ */
/* --- End Local Header Includes --- */
#include "common.h"

/* +++ Begin Library Macro Definitions +++ */
/* --- End Library Macro Definitions --- */

/* +++ Begin Library Typedefs +++ */

/*
    SN74LV4051A_Multiplexer_Output_t

    This enumeration...
*/
typedef enum SN74LV4051A_Multiplexer_Channel_t {
    CHANNEL_0 = 0,
    CHANNEL_1 = 1,
    CHANNEL_2 = 2,
    CHANNEL_3 = 3,
    CHANNEL_4 = 4,
    CHANNEL_5 = 5,
    CHANNEL_6 = 6,
    CHANNEL_7 = 7,
} SN74LV4051A_Multiplexer_Channel_t;

/*
    SN74LV4051A_Multiplexer_t

    This class...
*/
class SN74LV4051A_Multiplexer_t {

    /*
        ...
    */
    Pin_t ADDR_0;
    Pin_t ADDR_1;
    Pin_t ADDR_2;

    /*
    */
    Pin_t IO;

    /*
        ...
    */
    Pin_t Enable;

    public:
        /*
            Default Constructor

            This function...

            Return (SN74LV4051A_Multiplexer_t):
                ...
        */
        SN74LV4051A_Multiplexer_t() = default;

        /*
            Constructor

            This function...

            Addr0:
                ...
            Addr1:
                ...
            Addr2:
                ...
            Enable:
                ...

            Return (SN74LV4051A_Multiplexer_t):
                ...
        */
        SN74LV4051A_Multiplexer_t(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, Pin_t Enable);

        /*
            Destructor

            This function...

            Return (void):
                ...
        */
        ~SN74LV4051A_Multiplexer_t();

        /*
            EnableDevice

            This function...

            Return (void):
                ...
        */
        void EnableDevice();

        /*
            DisableDevice

            This function...

            Return (void):
                ...
        */
        void DisableDevice();

        /*
            EnableOutputChannel

            This function...

            Return (void):
                ...
        */
        void EnableOutputChannel(SN74LV4051A_Multiplexer_Channel_t Channel);

    private:

        void doEnableOutputChannel(SN74LV4051A_Multiplexer_Channel_t Channel);

};
/* --- End Library Typedefs --- */

/* +++ Begin Library Function Definitions +++ */
void Log_EnableMux(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2);
void Log_EnableMuxChannel(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2, SN74LV4051A_Multiplexer_Channel_t Channel);
void Log_DisableMux(Pin_t Addr0, Pin_t Addr1, Pin_t Addr2);
/* --- End Library Function Definitions --- */

#ifdef __cplusplus
}
#endif

#endif
