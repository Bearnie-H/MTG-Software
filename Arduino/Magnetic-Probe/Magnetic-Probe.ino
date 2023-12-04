/*
    Author(s):  Joseph Sadden
    Date:       1st December, 2023
    Purpose:    This sketch...
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
/* --- End Top-level Header Includes --- */

/* +++ Begin Custom Header Includes +++ */
/* --- End Custom Header Includes --- */

/*
    DEBUG

    This is the top-level macro which enables all debugging and instrumentation
    behaviour of the program. This will significantly increase the code size and
    decrease the execution speed as a result of the significant logging
    behaviour this typically implies. Comment out or otherwise undefine this
    macro for production or experimental trials where debugging and
    instrumentation can be turned off.
*/
#define DEBUG 1

#if defined(NDEBUG)
    #if defined(DEBUG)
        #undef DEBUG
    #endif
#endif

/* +++ Begin Program Macro Definitions +++ */
/* --- End Program Macro Definitions --- */

/* +++ Begin Program Typedefs +++ */
/* --- End Program Typedefs --- */

/* +++ Begin Function Prototype Forward Declarations +++ */
/* --- End Function Prototype Forward Declarations --- */

/* +++ Begin Interrupt Service Routine Function Forward Declarations +++ */
/* --- End Interrupt Service Routine Function Forward Declarations --- */

/* +++ Begin Template Function Prototype Forward Declarations +++ */
/* --- End Template Function Prototype Forward Declarations --- */

/* +++ Begin Debugging Function Prototype Forward Declarations +++ */
/* --- End Debugging Function Prototype Forward Declarations --- */

/* +++ Begin Program Global Constant Definitions +++ */
/* --- End Program Global Constant Definitions --- */

/* +++ Begin Pin Definitions +++ */
/* --- End Pin Definitions --- */

/* +++ Begin Hardware Register Declarations +++ */
/* --- End Hardware Register Declarations --- */

/* +++ Begin Program Global Variable Definitions +++ */
/* --- End Program Global Variable Definitions –-- */

/* +++ Begin Arduino Implementation Functions +++ */
/*
    This section contains the default functions provided by the Arduino
    implementation These functions must be present, with these defined function
    signatures, and provide the two interfaces for providing custom code for the
    controller to execute.

    In essence, the full structure of the Arduino program is roughly as follows:

    void main(void) {
        RunBootloader();

        setup();
        while ( true ) {
            loop();
        }
    }

    where the `RunBootloader()` is a placeholder for whatever system
    initialization the processor runs after a Reset before dropping control to
    the `setup()` and `loop()` functions.
*/
/*
    setup

    This function provides the starting point for the Arduino Sketch. This
    performs all of the non-static program and device initialization before
    passing control on to the `loop()` function as the primary control loop.

    Return (void):
        This function returns nothing to the caller, as all initialization it
        performs must be done through global variables.
*/

void setup(void) {

    // ...

    return;
}

/*
    loop

    This function provides the main control loop of the Arduino program. The
    behaviour provided in this function is called within a while(true) infinite
    loop provided by the implementation, and does not require explicit looping
    to operate indefinitely.

    Return (void):
        This function returns nothing to the caller.
*/
void loop(void) {

    // ...

    return;
}

/* --- End Arduino Implementation Functions --- */

/* +++ Begin Program Function Definitions +++ */
/* --- End Program Function Definitions --- */

/* +++ Begin Struct/Class Method Definitions +++ */
/* --- End Struct/Class Method Definitions --- */

/* +++ Begin Template Function Definitions +++ */
/* --- End Template Function Definitions --- */

/* +++ Begin Interrupt Service Routine Function Definitions +++ */
/* --- Begin Interrupt Service Routine Function Definitions --- */

/* +++ Begin Debugging Function Definitions +++ */
#if defined(DEBUG)
/*
    This section contains the function definitions for any functions related to
    debugging, logging, or otherwise instrumentation of the controller during
    operation. These functions MUST be optional for the successful functioning
    of the device, as ALL functionality provided in these functions will be
    removed when the top-level DEBUG symbol is undefined.

    A function related to debugging must return void, but may take an arbitrary
    number and type of arguments. These MUST NOT modify any global variables,
    but may read either global constants or variables if required.
*/

#else
/*
    Place empty function definition for the debug-only functions in this
    section. These should all be void functions with a function definition
    consisting of only a single return statement. These will be safely optimized
    out during compilation and will produce no runtime penalty for calling when
    running in non-debug mode.
*/

#endif
/* --- End Debugging Function Definitions --- */
