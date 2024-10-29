/*
    Author(s):  Joseph Sadden, Freya Hik
    Date:       1st February, 2023
    Purpose:    This program is the control software for the magnetic field
                    emitter created for the Mend-the-Gap collaboration. This
                    software provides the controller functionality for the
                    original 4-winding, 2-phase emitter, as well as the full
                    6-winding, 3-phase emitter.
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

/*
    avr/wdt.h

    This header includes the functionality to enable and work with the included
    hardware watchdog timer of the Arduino. This allows us to force reset the
    Arduino if and when some event occurs which prevents the main control loop
    from executing.
*/
#include <avr/wdt.h>

/* --- End Top-level Header Includes --- */

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

/*
    CORRECT_DELAY

    This top-level macro defines whether or not the `delay()` function is
    corrected should the pre-scaler for Timer0 be modified. If the `delay()`
    function is never used in the code, then this can either be commented out or
    explicitly undefined.

    If the `delay()` function IS used and this macro is not defined, the actual
    delay caused by the function will be incorrect by an amount corresponding to
    the relative difference in the Timer0 pre-scaler value expected by the
    `delay()` function.

    If this is defined, this also defines the macro "DELAY_MS" as a corrected
    replacement to the implementation-provided `delay()` function which ensures
    a delay of the requested duration, regardless of the Timer0 pre-scaler
    value.
*/
#define CORRECT_DELAY
#if defined(NCORRECT_DELAY)
    #undef CORRECT_DELAY
#endif

/* +++ Begin Program Macro Definitions +++ */
/*
    This section contains the macro definitions for the function-like macros
    used in this program, or those macros which cannot be replaced with typesafe
    constexpr expressions.
*/

/*
    The total number of electromagnetic windings in the stator. This can either be
    4 for the 2-phase design, or 6 for the 3-phase design.

    This is a pre-processor macro rather than a constexpr global to allow
    validation of this value at compile-time.
*/
#define WINDING_COUNT 4

/*
    Validate the WINDING_COUNT macro to be either 4 or 6. Raising a compiler
    error if the value is invalid.
*/
#if (4 == WINDING_COUNT)
    /*
        TWO_PHASE

        If defined, this macro indicates that the controller is running in
        two-phase mode, with four windings spaced π/2 radians apart.
    */
    #define TWO_PHASE 1
    #undef THREE_PHASE
#elif (6 == WINDING_COUNT)
    /*
        THREE_PHASE

        If defined, this macro indicates that the controller is running in
        three-phase mode, with six windings spaced π/3 radians apart.
    */
    #define THREE_PHASE 1
    #undef TWO_PHASE
#else
    /*
        If some other number of windings are defined, this is unhandled
        behaviour, so raise a compiler error.
    */
    #error "WINDING_COUNT Macro contains an illegal value. This Macro can only be defined as either 4 or 6!"
#endif

/*
    DegreesToRadians

    This function-like macro takes in a value, casts it to a `double`, and then
    converts it from being in units of degrees to being in units of radians. The
    result of this macro is a `double` value containing the converted value.
*/
#define DegreesToRadians(_Degrees) ((double)(_Degrees) * (PI / 180.0))

/*
    RadiansToDegrees

    This function-like macro takes in a value, casts it to a `double`, and then
    converts it from being in units of radians to being in units of degrees. The
    result of this macro is a `double` value containing the converted value.
*/
#define RadiansToDegrees(_Radians) ((double)(_Radians) * (180.0 / PI))

#if defined(CORRECT_DELAY)

/* +++ Begin Corrected Custom Implementations of millis(), micros(), delay(), and delayMicroseconds() +++ */

#define millis_corrected() (DelayTimerRescalingExponent >= 0) ? \
    (millis() >> DelayTimerRescalingExponent) : \
    (millis() << abs(DelayTimerRescalingExponent))

#define micros_corrected() (DelayTimerRescalingExponent >= 0) ? \
    (micros() >> DelayTimerRescalingExponent) : \
    (micros() << abs(DelayTimerRescalingExponent))

#define delay_corrected(_milliseconds) (DelayTimerRescalingExponent >= 0) ? \
    (delay(_milliseconds << DelayTimerRescalingExponent)) : \
    (delay(_milliseconds >> abs(DelayTimerRescalingExponent)))

#define delayMicroseconds_corrected(_microseconds) (DelayTimerRescalingExponent >= 0) ? \
    (delayMicroseconds(_microseconds << DelayTimerRescalingExponent)) : \
    (delayMicroseconds(_microseconds >> abs(DelayTimerRescalingExponent)))

/* --- End Corrected Custom Implementations of millis(), micros(), delay(), and delayMicroseconds() --- */

#else

/* +++ Existing Implementations of millis(), micros(), delay(), and delayMicroseconds() +++ */

#define millis_corrected() millis()
#define micros_corrected() micros()
#define delay_corrected(_milliseconds) delay(_milliseconds)
#define delayMicroseconds_corrected(_microseconds) delayMicroseconds(_microseconds)

/* --- Existing Implementations of millis(), micros(), delay(), and delayMicroseconds() --- */

#endif

/*
    ARRAY_SIZE

    This function-like macro implements the common C idiom for computing the
    size of a static fixed-size Array. This works on any fixed-size array where
    the value provided has not decayed to a pointer. The result of this a value
    of type `size_t`, and can be used in expressions transparently.
*/
#define ARRAY_SIZE(_Array) (sizeof(_Array) / sizeof((_Array)[0]))

// ...

/* --- End Program Macro Definitions --- */

/* +++ Begin Program Typedefs +++ */
/*
    This section contains the full set of custom typedefs used by this program.
    This includes the full set of struct's, enum's, and any typedefs of existing
    standard types.
*/

/*
    Pin_t

    This type...
*/
typedef uint8_t Pin_t;

/*
    PWMPrescalerMask_t

    This enumeration defines the full set of masks to use with the Arduino Uno
    to set the PWM Frequency pre-scaler values to modify exactly what the
    baseband PWM frequency is for a particular counter. These masks are derived
    from the specification sheet of the ATmega328 chip, while the frequency
    noted in the mask name is experimentally measured by applying the mask and
    directly measuring the period of the PWM signal at a 50% duty cycle using an
    oscilloscope.

    For the Timer Control Registers, the lower 3 bits correspond to the PWM
    frequency pre-scaler quantity. Unfortunately, this isn't a simple linear or
    even exponential mapping, and isn't consistent across all 3 timers of the
    Arduino Uno. Therefore, these masks are simply pre-defined in accordance
    with the specification sheet to set the relevant bits of these registers to
    achieve a PWM frequency roughly similar to the noted value.
*/
typedef enum PWMPrescalerMask_t: uint8_t {

    /*
        Pins 5 & 6 are controlled by the hardware register TCCR0B
    */

   Pin_5_6_62500Hz = 0b00000001,
   Pin_5_6_7812Hz  = 0b00000010,
   Pin_5_6_976Hz   = 0b00000011,
   Pin_5_6_244Hz   = 0b00000100,
   Pin_5_6_61Hz    = 0b00000101,


    /*
        Pins 9 & 10 are controlled by the hardware register TCCR1B
    */

   Pin_9_10_62500Hz = 0b00000001,
   Pin_9_10_7812Hz  = 0b00000010,
   Pin_9_10_976Hz   = 0b00000011,
   Pin_9_10_244Hz   = 0b00000100,
   Pin_9_10_61Hz    = 0b00000101,

    /*
        Pins 3 & 11 are controlled by the hardware register TCCR2B
    */

    Pin_3_11_62500Hz = 0b00000001,
    Pin_3_11_7812Hz  = 0b00000010,
    Pin_3_11_976Hz   = 0b00000100,
    Pin_3_11_244Hz   = 0b00000110,
    Pin_3_11_61Hz    = 0b00000111,

} PWMPrescalerMask_t;

/*
    PWMFrequency_t

    This enumeration type defines the set of PWM baseband frequencies available
    for all three of the Arduino Uno. These enumeration values are used with the
    PWM_SetFrequency() function to set the PWM frequency for all of the output
    loads of the Emitter. This is the expected higher-level API for modifying
    the PWM frequency, rather than the lower-level SetPWMFrequencyPrescaler().

    NOTE:
        These frequency values correspond to the baseband PWM frequency if and
        when the corresponding Timer is configured to operate in "8-bit FastPWM"
        mode. For other modes, like Phase-Correct PWM, the resulting frequency
        will be multiplied by (255/511). Other modes may introduce different
        re-scaling factors. See the documentation of the particular PWMMode_t
        enumeration constants for the exact details.
*/
typedef enum PWMFrequency_t: uint8_t {
    PWM_62500Hz,
    PWM_7812Hz,
    PWM_976Hz,
    PWM_244Hz,
    PWM_61Hz,
} PWMFrequency_t;

/*
    PWMTimer_t

    This enumeration defines the set of internal Timer/Counter blocks accessible
    to the Arduino Uno for modifying how the PWM output signals are generated.
*/
typedef enum PWMTimer_t: uint8_t {
    Timer0, //  Corresponds to registers [TCCR0A, TCCR0B], and controls pins 5 & 6
    Timer1, //  Corresponds to registers [TCCR1A, TCCR1B], and controls pins 9 & 10
    Timer2, //  Corresponds to registers [TCCR2A, TCCR2B], and controls pins 3 & 11
} PWMTimer_t;

/*
    PWMMode_t

    This enumeration defines the available set of "Waveform Generation Mode"s
    available on the Timer subsystems of the Arduino. This enumeration
    specifically defines only those modes which are achievable on ALL of the
    Timer subsystems of the Arduino. This enumeration is intended to be used
    with the higher-level PWM_SetWaveformMode API to configure the Waveform
    Generation Mode for all available Timers.

    For specifics of what these modes mean, see the documentation of the
    ATmega328p chip.
*/
typedef enum PWMMode_t: uint8_t {

    /*
        PWMMode_FastPWM_8Bit

        This mode operates in "single-slope", where the counter register simply
        counts up to form the duty cycle, this is the fastest available PWM
        mode, suitable for applications where the highest frequency is desirable
        due to the frequency response of the output device.
    */
    PWMMode_FastPWM_8Bit,

    // Define further modes below if desired and necessary.
    // ...

} PWMMode_t;

/*
    PWMModeMask_t

    This enumeration type defines the set of masks for modifying the
    Timer/Counter Control Registers (TCCRnA, TCCRnB) to achieve a desired
    Waveform Generation Mode. For each mode, and for each timer, a set of four
    masks are defined. The "*AndMask*" is defined to clear the specific bits
    required to be set low, while the "*OrMask*" is defined to set the specific
    bits required to be high.

    This type is defined for use with the SetPWMTimerMode() API, the low-level
    function for directly modifying the timer control registers of the Arduino.
    When using the higher-level APIs, the correct masks will be chosen for the
    requested PWM mode.
*/
typedef enum PWMModeMask_t: uint8_t {

    Timer0_FastPWM_8Bit_AndMask_A   = 0b11111111,
    Timer0_FastPWM_8Bit_OrMask_A    = 0b00000011,
    Timer0_FastPWM_8Bit_AndMask_B   = 0b00110111,
    Timer0_FastPWM_8Bit_OrMask_B    = 0x00000000,

    Timer1_FastPWM_8Bit_AndMask_A   = 0b11111101,
    Timer1_FastPWM_8Bit_OrMask_A    = 0b00000001,
    Timer1_FastPWM_8Bit_AndMask_B   = 0b00101111,
    Timer1_FastPWM_8Bit_OrMask_B    = 0b00001000,

    Timer2_FastPWM_8Bit_AndMask_A   = 0b11111111,
    Timer2_FastPWM_8Bit_OrMask_A    = 0b00000011,
    Timer2_FastPWM_8Bit_AndMask_B   = 0b00110111,
    Timer2_FastPWM_8Bit_OrMask_B    = 0b00000000,

    // Define further masks for the additional modes here, if desired and required.

} PWMModeMask_t;

typedef struct WelfordAccumulator_t {

    /*
        Count

        ...
    */
    uint16_t Count;

    /*
        MaxCount

        ...
    */
    uint16_t MaxCount = WelfordAccumulator_DefaultMaxCount;

    /*
        Mean

        ...
    */
    double Mean;

    /*
        M2

        ...
    */
    double M2;

    /*
        WelfordAccumulator_t

        This function...
    */
    WelfordAccumulator_t() = default;

    WelfordAccumulator_t(uint16_t MaxCount);

    /*
        Reset

        This function...
    */
    void Reset();

    /*
        TestUpdate

        This function...
    */
    void TestUpdate(double Value, double* NewMean, double* NewVariance);

    /*
        TestUpdate

        This function...
    */
    void TestUpdate(uint32_t Value, double* NewMean, double* NewVariance);

    /*
        Update

        This function...
    */
    void Update(double Value);

    /*
        Update

        This function...
    */
    void Update(uint32_t Value);

    /*
        Variance

        This function...
    */
    double Variance();

    /*
        SampleVariance

        This function...
    */
    double SampleVariance();

} WelfordAccumulator_t;

/*
    LoadPolarity_t

    This enumeration defines the polarity of the signal to apply to a particular
    load element. This enum is defined to be implemented using the uint8_t as
    the base storage type, shrinking from 4-bytes to 1-byte per value. The
    ordering of the enumeration constants is also important, and plays a role in
    the ordering of the pin array within the LoadElement_t struct, where
    LoadElement_t->Pin[PositivePolarity] WILL correspond to the output pin to
    drive high in order to apply a positive polarity signal to that load
    element.
*/
typedef enum LoadPolarity_t: uint8_t {
    PositivePolarity = 0,
    NegativePolarity = 1
} LoadPolarity_t;

/*
    Duration_t

    This represents a duration of time, corresponding to the difference between
    two Timestamp_t values. The precision of this type is defined by:
        1) The fundamental main clock speed of the Arduino, and
        2) The frequency pre-scaler of Timer2.

    For an Arduino Uno, operating with a 16MHz main clock, and a Timer2
    pre-scaler of 1 (indicting Timer2 increments once per clock cycle), this
    corresponds to a precision of 62.5ns. Each increment to a Duration_t value
    corresponds to a time interval of 62.5ns.

    For the precise conversion factor between increments of this type and units
    of seconds, the global constant "Second" provides this conversion factor,
    and is based off the F_CPU macro to provide the nominal CPU clock frequency.
*/
typedef uint32_t Duration_t;

/*
    SignedDuration_t

    This type...
*/
typedef int32_t SignedDuration_t;

/*
    Timestamp_t

    This type provides a 32-bit timestamp, where each increment of this value
    corresponds to a number of clock cycles equal to the Timer2 pre-scaler
    value. This has a maximum precision corresponding to 62.5ns for the 16MHz
    clock found standard on an Arduino Uno. This provides a proper
    implementation of timestamping functionality, well beyond what is provided
    by the default millis() and micros() functions.

    The precision of this timestamp, when used to compute a Duration_t, can be
    set  by modifying the Timer2 pre-scaler. Regardless of the actual value of
    this pre-scaler, the Duration_t will be computed with the correct
    correspondence with real time.
*/
typedef struct Timestamp_t {

    /*
        Ticks

        This defines the number of "ticks" of the "clock" the Timestamp_t is
        using as the driving element. The "clock" corresponds to Timer2 of the
        Arduino, and each "tick" corresponds to an increment of the TCNT2
        register. This register is incremented once each time the main CPU clock
        cycles an amount equal to the pre-scaler factor for Timer2. Depending on
        the value of the lower three bits of the TCCR2B register (CS2[0:2]),
        this can be one of:
            - 1
            - 8
            - 32
            - 64
            - 128
            - 256
            - 1024
        After this many clock cycles, the TCNT2 register increments once, and
        this counts as one tick of the Timestamp_t.
    */
    uint32_t Ticks;

    /*
        Default Constructor

        The default constructor is generated by the compiler.

        Return (Timestamp_t):
            Returns a default-initialized and ready-to-use Timestamp_t value.
            This is initialized to a zero-value suitable for use as-is.
    */
    Timestamp_t(void) = default;

    /*
        Copy Constructor

        The copy-constructor is generated by the compiler.

        Other:
            The existing Timestamp_t instance to copy into the current instance.

        Return (Timestamp_t):
            A new Timestamp_t value, set equal to the provided Other instance.
    */
    Timestamp_t(const Timestamp_t& Other) = default;

    /*
        Copy Constructor - From volatile const

        Other:
            The existing Timestamp_t to copy the contents from.

        Return (Timestamp_t):
            The newly copied-into Timestamp_t value.
    */
    Timestamp_t(const volatile Timestamp_t& Other);

    /*
        Custom Constructor

        This custom constructor allows generating a Timestamp_t value from the
        current TCNT2 counter value, and the current Timer2 Overflow Count
        value.

        OverflowCount:
            The current 32-bit value indicating the number of overflows of the
            Timer2 counter. All 32-bits are used to construct the upper 32-bits
            of the full 40-bit timestamp
        Counter:
            The current TCNT2 register value, as read out immediately prior to
            constructing the timestamp.

        Return (Timestamp_t):
            A constructed and initialized Timestamp_t value, set to the time
            corresponding to the provide tick count values.
    */
    Timestamp_t(const uint32_t OverflowCount, const uint8_t Counter);

    /*
        Const-to-Volatile Assignment Operator

        This is an overload of the basic assignment operator to allow assignment
        from a non-volatile const to a non-const volatile Timestamp_t.

        Other:
            The existing const Timestamp_t to assign from.

        Return (Timestamp_t):
            The volatile Timestamp_t to assign into.
    */
    volatile Timestamp_t& operator = (const Timestamp_t& Other) volatile;

    /*
        Const-Volatile-to-Volatile Assignment Operator

        This is an overload of the basic assignment operator to allow assignment
        from a volatile const to a non-const volatile Timestamp_t.

        Other:
            The existing const-volatile Timestamp_t to assign from.

        Return (Timestamp_t):
            The volatile Timestamp_t to assign into.
    */
    volatile Timestamp_t& operator=(const volatile Timestamp_t& Other) volatile;

    /*
        DurationSince

        This function computes a 32-bit duration between two Timestamp_t values.
        Do note that the result of this is only 32-bits, while the full
        Timestamp_t value is a 40-bit quantity.  The results will be truncated
        to only the lower 32-bits. If the value is too large to represent, a
        sentinel value of (-1 = 0xFFFFFFFF) will be returned to indicate this
        out-of-range error.

        Since:
            The Timestamp_t value in the past to compute how long the current
            Timestamp_t is in the future.

        Return (Duration_t):
            The resulting 32-bit difference between the two Timestamp_t values.
    */
    Duration_t DurationSince(const Timestamp_t& Since) const;

} Timestamp_t;

/* +++ Begin Duration_t Standard Duration Definitions +++ */

/*
    Second

    This global constant represents the Duration_t value of exactly 1 second.
*/
static constexpr Duration_t Second = (Duration_t)(F_CPU);

/*
    Millisecond

    This global constant represents the Duration_t value of exactly 1
    millisecond.
*/
static constexpr Duration_t Millisecond = (Duration_t)(Second / 1000.0);

/*
    Microsecond

    This global constant represents the Duration_t value of exactly 1
    microsecond.
*/
static constexpr Duration_t Microsecond = (Duration_t)(Millisecond / 1000.0);

/* --- End Duration_t Standard Duration Definitions --- */

/* +++ Begin Constructor Default Parameter Definitions +++ */

/*
    InterruptFrequencyDefaultUpdatePeriod

    This variable represents the default value for the update period of the
    InterruptFrequency_t type, in units of µ-seconds. This value can be changed
    to any sensible value corresponding to a meaningful timescale over which the
    frequency stability of the oscillator driving the interrupt signal is
    uncertain.

    NOTE:
        This value must be forward-declared here in order to be properly used in
        the parameterized constructor of the InterruptFrequency_t type.
*/
constexpr Duration_t InterruptFrequencyDefaultUpdatePeriod = (Duration_t)(2 * Second);

/* --- End Constructor Default Parameter Definitions --- */

/*
    LoadElement_t

    This struct provides the definition of a load element to be controlled by
    this program.  A load element consists of the pair of output pins to drive
    either a positive or negative signal to the load, the desired duty cycle for
    the PWM signal to apply, and the polarity of the signal to apply to the
    load.

    As noted in the documentation for the LoadPolarity_t type, the OutputPins
    array is initialized such that OutputPin[PositivePolarity] is the output pin
    to drive high in order to apply a positive polarity signal to the load.
*/
typedef struct LoadElement_t {

    /*
        OutputPins

        This represents the pair of output pins of the µ-controller to drive in
        order to apply power to the load element itself. These are ordered such
        that the LoadPolarity_t value indexes the "active" pin, while the other
        must be driven low.
    */
    Pin_t OutputPins[2];

    /*
        DutyCycle

        The duty-cycle for the PWM signal to apply to the power electronics
        which drive current into the load element. This is a raw value on the
        full range [0,255], where a 0 indicates that the phase should be turned
        off, while 255 indicates maximum available power to be applied to that
        phase. This simply provides the magnitude of the duty cycle to apply,
        but contains to information regarding the polarity.
    */
    uint8_t DutyCycle;

    /*
        Polarity

        See the documentation for the LoadPolarity_t type.

        This represents the polarity of the signal to apply to the load element.
        This can either be positive or negative. A zero-value for duty-cycle
        covers the case of the load element being completely de-energized.
    */
    LoadPolarity_t Polarity;

    /*
        Default Constructor

        This struct requires a default constructor, i.e. a constructor which
        accepts no arguments. Simply use the default constructor generated by
        the compiler, and note that this signature is never called.
    */
    LoadElement_t() = default;

    /*
        Constructor

        This is the constructor function for an instance of a LoadElement_t
        struct.  This is the function which is called to prepare and ensure a
        LoadElement_t instance is ready for use.

        PositivePin:
            The pin number of the output pin to use for applying a positive
            polarity output signal to the load.
        NegativePin:
            The pin number of the output pin to use for applying a negative
            polarity output signal to the load.

        Return (LoadElement_t):
            This function returns the allocated and initialized LoadElement_t
            instance, ready to be used.

        Note:
            This function does not explicitly set the `pinMode()` or set the
            output level (digitalWrite()) of the given pins. By default on
            reset, all pins are configured as pinMode(Pin, OUTPUT), and have
            level equivalent to digitalWrite(Pin, LOW);
    */
    LoadElement_t(const Pin_t PositivePin, const Pin_t NegativePin);

} LoadElement_t;

/*
    FieldEmitter_t

    This struct represents the full magnetic field emitter device. This consists
    of a set of pairs of electromagnet windings, arranged in either a 2-phase or
    3-phase arrangement. Each of the winding pairs are represented with as a
    LoadElement_t instance.
*/
typedef struct FieldEmitter_t {

    /*
        Phases

        This array of LoadElement_t values contains the LoadElement_t instances
        corresponding to the pairs of electromagnetic windings for each of the
        phases of the emitter. This array is ordered such that index 0
        corresponds to Phase A, index 1 to Phase B, and index 2 to Phase C (if
        this third phase is present).
    */
    LoadElement_t Phases[(WINDING_COUNT >> 1)];

    /*
        TriggerPin

        This is the pin which the µ-controller should toggle high for a short
        period when a "significant" change to the field orientation occurs. This
        is to be used as the primary "production-level" output signal of some
        particular noteworthy event, and to be used to trigger additional
        functionality such as the start of a camera video capture.

        Only the rising edge of this pin corresponds to the occurrence of the
        event in question, the falling edge may occur at some arbitrary later
        time, with no guarantees provided for the duration the pin is held high.
    */
    Pin_t TriggerPin;

    /*
        DesiredFieldOrientation

        This represents the desired orientation of the field to be generated by
        the emitter. This value is provided as an integer unsigned value,
        covering the half-open range [0, 360) degrees. The special global
        constant value `FIELD_ORIENTATION_OFF` is provided as a special-case
        value to indicate that the field should be turned off completely, with
        all available phases de-energized.

        This is declared as a volatile variable, as the value is computed online
        during the interrupt service routine "ComputeNextFieldOrientation()". We
        must declare this volatile since this may be written to asynchronously
        to the standard program flow.
    */
    volatile uint16_t DesiredFieldOrientation;

    /* +++ Begin Struct Methods +++ */
    /*
        2-Phase Constructor

        This constructor prepares a FieldEmitter_t instance for two-phase
        operation.  This will set up the load windings, the trigger pin, and set
        the desired initial field orientation to meaningful values, returning a
        ready-to-use FieldEmitter_t instance.
    */
    FieldEmitter_t(LoadElement_t&& PhaseA, LoadElement_t&& PhaseB, const uint8_t TriggerPin);

    /*
        3-Phase Constructor

        This constructor prepares a FieldEmitter_t instance for three-phase
        operation.  This will set up the load windings, the trigger pin, and set
        the desired initial field orientation to meaningful values, returning a
        ready-to-use FieldEmitter_t instance.
    */
    FieldEmitter_t(LoadElement_t&& PhaseA, LoadElement_t&& PhaseB, LoadElement_t&& PhaseC, const uint8_t TriggerPin);

    /*
        UpdateFieldOrientation

        This function is the top-level method to compute and apply the necessary
        duty cycles to the corresponding phase windings. This is the primary API
        for accessing and modifying the direction in which the field points.

        Return (void):
            This function returns nothing to the caller. The relevant
            calculations and pins are modified in order to set the output phase
            currents to achieve the best-match to the desired angle.
    */
    void UpdateFieldOrientation(void);

    /*
        ConfigurePins

        This function configures and sets the pins associated with this
        FieldEmitter_t to be explicitly OUTPUT pins. This counts for all of the
        PWM output pins, as well as the Trigger pin.

        Return (void):
            This function returns nothing to the caller. The full set of pins
            associated with the phase windings of this FieldEmitter_t are
            configured as pinMode(*, OUTPUT), as well as the Trigger pin.
    */
    void ConfigurePins(void) const;

    /*
        ToggleTrigger

        This function will toggle the output trigger pin of the FieldEmitter_t.
        This is useful for sending an output signal when a particular event
        occurs, like the field orientation changing a noteworthy amount. This
        can be used to synchronize timing of some additional circuitry or
        equipment with the event in question.

        Return (void):
            This function returns nothing to the caller. The trigger pin is
            turned on if it is currently off, and off if it is currently on.
    */
    void ToggleTrigger(void) const;

    /*
        Trigger

        This function will assert that the output trigger pin of the
        FieldEmitter_t is set to output HIGH.

        Return (void):
            This function returns nothing to the caller. The configured pin is
            set to output HIGH.
    */
    void Trigger(void) const;

    /*
        TriggerOff

        This function will assert that the output trigger pin of the
        FieldEmitter_t is set to output LOW.

        Return (void):
            This function returns nothing to the caller. The configured pin is
            set to output LOW.
    */
    void TriggerOff(void) const;

private:

    /*
        ComputeDutyCycles

        This function computes and updates the duty cycles to apply to each of
        the phases of the emitter in order to achieve the desired field
        orientation. This function makes some assumptions about the rotational
        symmetry and phase-winding layout of the emitter, but otherwise is
        generalized to either the two-phase or three-phase expected emitter
        layouts.

        Return(void):
            This function returns nothing to the caller. The relevant duty
            cycles and polarities for each of the phases are updated within the
            Emitter_t itself.
    */
    void ComputeDutyCycles(void);

    /*
        ApplyDutyCycles

        This function is what actually modifies the output pin PWM duty cycles
        in order to apply the desired phase currents to the phase windings of
        the emitter. This always ensures that the H-Bridge controlling the
        windings runs through the OFF-OFF configuration, rather than making
        assumptions about the phase of the PWM signal and risking a dead-short
        through the power MOSFETs of the H-Bridge.

        Return(void):
            This function returns nothing to the caller, and modifies nothing
            within the FieldEmitter_t itself. The requisite output pins have
            their output signals modified only.
    */
    void ApplyDutyCycles(void) const;

    /* --- End Struct Methods --- */

} FieldEmitter_t;

/*
    InterruptFrequency_t

    This type represents an object which is used to directly measure the
    frequency of the interrupt signal used to drive the "ticks" of the internal
    field orientation angle computation. This holds the necessary values and
    variables to measure and compute the frequency of this interrupt signal, as
    well as some additional logic and behaviours to minimize the ISR computation
    costs for keeping this frequency value up-to-date.

    Standard usage of an instance of this type is to initialize a global
    instance with the provided non-default constructor, calling the .Tick()
    method in the ISR attached to the input signal you wish to measure the
    frequency of, and calling the .Update() function at some point during the
    "loop()" function. When using the frequency value, simply utilize the DeltaT
    member variable, as this is asserted to be the most recent computed value at
    all points in time. To ensure a meaningful value for the DeltaT during the
    ISR, simply check whether this value is non-zero. Once the instance returns
    a non-zero DeltaT value, it is ready for use.

    NOTE:
        Due to needing to see two subsequent interrupts in order to accurately
        compute the DeltaT between them, the first interrupt must not utilize
        the DeltaT value. This is set be default to a sentinel value of 0.0f to
        indicate that it has not yet been computed.
*/
typedef struct InterruptFrequency_t {

    /*
        CurrentTimeStamp

        This represents the Timestamp_t associated with the most recent
        interrupt. This is set by the Tick() function, which should be called
        as the first operation of the interrupt to be timed.
    */
    volatile Timestamp_t CurrentTimeStamp;

    /*
        PreviousTimeStamp

        This represents the Timestamp_t associated with the immediately previous
        interrupt. This value is set by the Tick() function, in the same atomic
        operation as the CurrentTimeStamp value.
    */
    volatile Timestamp_t PreviousTimeStamp;

    /*
        LastUpdateTimeStamp

        This value represents Timestamp_t at which the DeltaT value was last
        computed. This is used to reduce the computational overhead associated
        with constantly recomputing this value, only requiring a full
        re-computation once per specified update period.
    */
    Timestamp_t LastUpdateTimeStamp;

    /*
        UpdatePeriod

        This value defines how frequently to actually re-compute the DeltaT value.
    */
    Duration_t UpdatePeriod;

    /*
        DeltaT

        This value holds the most up-to-date computed value of the time interval
        between successive interrupts being measured. This is a floating-point
        value in units of seconds. This is volatile as it is written to in the
        main loop, but read in the ISR.
    */
    volatile double DeltaT;

    /*
        Count

        To allow for computation of an online average of some past number of
        measurements, this instance keeps track of the number of values it has
        seen and included in the average measurement. This helps to smooth out
        the measured value of the interrupt frequency and remove the
        high-frequency noise that comes from the multiple potentially
        overlapping interrupts.
    */
    uint16_t Count;

    /*
        MaxCount

        This defines the maximum number of values to retain when computing the
        average interrupt interval. Once this many samples are included, the
        oldest values will drop off the end of the computation and should no
        longer be relevant.
    */
    uint16_t MaxCount = 64;

    /*
        IntervalMean

        The current computed value for the average-smoothed duration between
        interrupts calling the Tick() function. This is smoothed out with an
        online algorithm to remove the highest-frequency variations caused by
        the sporadic multi-level interrupts introducing slightly different
        delays into the Tick() timing.
    */
    Duration_t IntervalMean;

    /*
        IntervalVariance

        The current computed value for the online variance of the measured
        interrupt period values, as recorded by the Tick() function. This is used,
        along with the IntervalMean value to perform online filtering of the
        measurements, to account for the known stability of the interrupt signal.
    */
    double IntervalVariance;

    /*
        IsSet

        This value indicates whether or not the initial value of DeltaT has
        been computed.This is only useful in the initialization of this struct,
        and once initialized has no further use.
    */
    bool IsSet;

    /* +++ Begin Member Functions +++ */

    /*
        Constructor

        This constructor requires specification of the update period for
        computing a new DeltaT value with this instance of the
        InterruptFrequency_t value. This value is given in units of µ-seconds,
        and the value of DeltaT will be updated on the next call to .Update()
        which leads to (CurrentTimeStamp - LastUpdateTimeStamp > UpdatePeriod)
        being true.

        UpdatePeriod:
           This represents how long to wait between actually performing the
           computation of a new value of the DeltaT value, in units of
           µ-seconds. This should be chosen to represent a reasonable time-scale
           over which the frequency stability of the interrupt signal is
           expected to vary.

            DefaultValue = InterruptFrequencyDefaultUpdatePeriod:
                A default value is provided as a sensible default should no
                other information be provided, or a "better" value is not known.

        Return (InterruptFrequency_t):
            This function returns an allocated and initialized instance of an
            InterruptFrequency_t, ready to be used.
    */
    InterruptFrequency_t(Duration_t UpdatePeriod = InterruptFrequencyDefaultUpdatePeriod);

    /*
        Reset

        This is the top-level function to reset and re-initialize an instance of
        an InterruptFrequency_t value, during an event such as a Watchdog Timer
        timeout.  This resets the value to a state equivalent to a new value,
        ready to be reused.

        Return (void):
            This function returns nothing to the caller. The
            InterruptFrequency_t instance this is called upon is reset as if it
            were freshly created.
    */
    void Reset(void);

    /*
        Initialize

        This is the top-level initialization function for this
        InterruptFrequency_t value. This function should be called during the
        setup() function to ensure the initial value of DeltaT is set to a
        sensible value before actually computing field orientations.

        Return (void):
            This function returns nothing to the caller. The initial value
            of DeltaT is computed and available for calculations.
    */
    void Initialize(void);

    /*
        Tick

        This function performs the necessary Timestamp_t operations to "tick"
        forward by one time interval. This stores the "current" timestamp into
        the previous, and takes a new timestamp. For best performance, this
        operation should happen as soon as possible within the interrupt you
        wish to measure the period of.

        Return (void):
            This function returns nothing to the caller. The internal
            Timestamp_t values are updated accordingly.
    */
   void Tick(void);

    /*
        Update

        This function will attempt to update the currently computed value of
        DeltaT, assuming that the interval since the last update satisfies the
        UpdatePeriod requirement. If not enough time has elapsed, this function
        simply updates the Current and Previous timestamps and returns without
        modifying the DeltaT value.

        Return (void):
            This function returns nothing to the caller. If a new value of
            DeltaT is ready to be computed, the value will be computed and the
            local volatile member variable will be set to the new value.
    */
    void Update(void);

    private:

        /*
            SetDeltaT

            This function contains the actual inner logic for computing and
            setting a new value of DeltaT within this instance. This safely
            computes the duration between the timestamps, error-checks, and
            sets the flags and update timestamp values accordingly.

            Now:
                The most recent Timestamp_t value, corresponding to "now".
            Previous:
                The immediately previous Timestamp_t value, corresponding to
                the previous interrupt time.

            Return (void):
                This function returns nothing to the caller. The DeltaT value
                is updated and the other necessary flags and values updated
                accordingly.
        */
        void SetDeltaT(const Timestamp_t& Now, const Timestamp_t& Previous);

    /* --- End Member Functions --- */

} InterruptFrequency_t;

/*
    WatchdogTimeout_t

    This enumeration defines the watchdog timer pre-scaler mask values
    corresponding (roughly) to specific watchdog timeout durations. These
    durations are not guaranteed exactly, but should be quite close unless
    extreme variations in Vcc or ambient temperature are experienced. These
    masks set the following bits of the Watchdog Timer Control Register
    (WDTCSR):

        WDTP[3:0] = WDTCSR[5,2:0]

    Values other than these enumeration constants are implementation-reserved,
    and not guaranteed to operate as expected.
*/
typedef enum WatchdogTimeout_t {
    WatchdogTimeout_16ms   = 0b00000000,
    WatchdogTimeout_32ms   = 0b00000001,
    WatchdogTimeout_64ms   = 0b00000010,
    WatchdogTimeout_125ms  = 0b00000011,
    WatchdogTimeout_250ms  = 0b00000100,
    WatchdogTimeout_500ms  = 0b00000101,
    WatchdogTimeout_1000ms = 0b00000110,
    WatchdogTimeout_2000ms = 0b00000111,
    WatchdogTimeout_4000ms = 0b00100000,
    WatchdogTimeout_8000ms = 0b00100001,
} WatchdogTimeout_t;

/*
    WatchdogMode_t

    This enumeration defines the available set of modes the Watchdog Timer can
    operate in, and provides the bit-masks to apply to the Watchdog Timer
    Control Register (WDTCSR) in order to set the particular mode.

    In Interrupt mode, the ISR "WTD_vect" is called when the watchdog times out.
    The hardware implicitly clears the WDIE bit of the WDTCSR register when this
    occurs, so if interrupts are desired on each timeout, this bit must be
    re-written.

    In Reset mode, this triggers a hardware reset, setting the Watchdog Reset
    Flag (WDRF) of the MCU Status Register (MCUSR) high. No user-defined
    interrupt is called, only the implementation-defined RESET_vect.

    In Interrupt-Reset mode, this first calls the user-provided WDT_vect ISR,
    followed by a system reset. This is simply the concatenation of the two
    prior modes.

        WDIE = Interrupt Enable (bit 6)
        WDE = Watchdog Enable (bit 3)

    [WDIE,WDE] = WDTCSR[6,3]
*/
typedef enum WatchdogMode_t {
    WatchdogMode_Off            = 0b00000000,
    WatchdogMode_Interrupt      = 0b01000000,
    WatchdogMode_Reset          = 0b00001000,
    WatchdogMode_InterruptReset = (WatchdogMode_Interrupt | WatchdogMode_Reset),
} WatchdogMode_t;

/*
    ResetTriggerMask_t

    This enumeration defines the set of masks for interrogating the MCU Status
    Register (MCUSR) for the cause of the most recent reset.
*/
typedef enum ResetTriggerMask_t {
    ResetTrigger_PowerOn  = 0b00000001,
    ResetTrigger_External = 0b00000010,
    ResetTrigger_BrownOut = 0b00000100,
    ResetTrigger_Watchdog = 0b00001000,
} ResetTriggerMask_t;

// ...

/* --- End Program Typedefs --- */

/* +++ Begin Function Prototype Forward Declarations +++ */
/*
    This section contains only the forward declarations of the functions to be
    used by this program. The definitions of these functions is provided in a
    later section.
*/

/*
    SystemReset

    This function...

    Return (void):
        ...

    NOTE:
        ...
*/
void (*SystemReset)(void) = 0x0000;

/*
    Pin_Trigger

    This function...

    Pin:
        ...

    Return (void):
        ...
*/
void Pin_Trigger(const Pin_t Pin);

/*
    Pin_TriggerOff

    This function...

    Pin:
        ...

    Return (void):
        ...
*/
void Pin_TriggerOff(const Pin_t Pin);

/*
    Pin_ToggleTrigger

    This function...

    Pin:
        ...

    Return (void):
        ...
*/
void Pin_ToggleTrigger(const Pin_t Pin);

/*
    CheckResetState

    This function interrogates the MCU Status Register (MCUSR) for the cause of
    the most recent reset condition. This explicitly resets the MCUSR to 0,
    nominally allowing interrogation of multiple resets in quick succession.
    This allows the user to define particular additional behaviour to execute if
    and when a particular reset condition occurs.

    For optimal behaviour, this should be called as the first operation of the
    setup() function.

    Return (void):
        This function returns nothing to the caller. Any user-provided
        functionality for the handled reset states is performed and the MCUSR is
        set back to 0.
*/
void CheckResetState(void);

/*
    ConfigureStatusLEDPins

    This function...

    Return (void):
        ...
*/
void ConfigureStatusLEDPins(void);

/*
    Watchdog_Initialize

    This function is a re-implementation of the platform-provided
    "watchdog_enable()", but where the Timeout and Watchdog Mode are explicitly
    provided as enumerations, and can be set together in one consistent
    operation. This also ensures that the watchdog timer is reset prior to
    returning.

    Timeout:
        The timeout duration to request of the watchdog timer. A value selected
        from the WatchdogTimeout_t enumeration.
    Mode:
        The watchdog timeout mode to operate under. A value selected from the
        WatchdogMode_t enumeration.

    Return (void):
        This function returns nothing to the caller. The watchdog timer is
        initialized with the requested timeout and operation mode, and is reset
        to give the caller one full period before calling watchdog_reset().
*/
void Watchdog_Initialize(WatchdogTimeout_t Timeout, WatchdogMode_t Mode);

/*
    PrepareDutyCycleLUT

    This function pre-computes and fills in the global DutyCycleLUT and
    PolarityLUT look-up tables. By pre-computing these values only on reset, we
    trade (360 * 2) bytes of memory for the full cost of the emulated
    floating-point math operations required to actually compute the 2 (or 3)
    cosine values for a particular field orientation during the actual control
    loop. Basic measurements comparing an online calculation versus a look-up
    table indicate a control loop timing difference of roughly 150 vs.
    50µs/phase. This is well-worth the memory footprint, unless and until memory
    constraints make this impractical.

    Return (void):
        This function returns nothing to the caller. The global look-up tables
        are filled in, with the orientation angle as the index into the array
        for simple lookups.
*/
void PrepareDutyCycleLUT(void);

/*
    ConfigureTimingInterrupt

    This function configures the logic necessary for enabling and attaching an
    interrupt function to the provided pin. This is a simple wrapper around the
    library provided "attachInterrupt()" API.

    InterruptPin:
        The input pin of the Arduino Uno to configure and attach the interrupt
        service routine for.

    Return (void):
        This function returns nothing to the caller. The pin is configured and
        the interrupt is attached. See the implementation of the function for
        the function provided as the Interrupt Service Routine, and for details
        on exactly what the interrupt triggers on.
*/
void ConfigureTimingInterrupt(const Pin_t InterruptPin);

/*
    ConfigurePushButtonInterrupt

    This function...

    InterruptPin:
        ...

    Return (void):
        ...
*/
void ConfigurePushButtonInterrupt(const Pin_t InterruptPin);

/*
    ConfigurePWM

    This function configures all three of the available Arduino Uno
    Timer/Counter blocks to operate in the same mode, with the same waveform
    generation algorithm and pre-scaler frequency. This API artificially limits
    the available operations to only those which can be satisfied by all three
    of the Arduino Uno timers. For lower-level API's capable of setting
    individual timers to the full range of modes, see the "SetPWMTimerMode()"
    function.

    Frequency:
        The PWMFrequency_t enumeration constant indicating the PWM frequency to
        configure the Arduino to operate with. The name of these enumeration
        constants provide a close match to the actual measured output frequency
        generated for the constant.
    WaveformGenerationMode:
        The PWMMode_t enumeration constant indicating which hardware algorithms
        to use when generating the actual PWM output waveform. See the
        documentation of the PWMMode_t enumeration type for more details on this
        waveform generation.

    Return (void):
        This function returns nothing to the caller. The required Timer/Counter
        Control Registers (TCCRnA, TCCRnB) are modified to set the waveform
        generation algorithm and desired pre-scaler values.
*/
void ConfigurePWM(const PWMFrequency_t Frequency, const PWMMode_t WaveformGenerationMode);

/*
    PWM_SetWaveformMode

    This function is the high-level API for configuring the waveform generation
    mode for all of the available timers of the Arduino Uno. This will configure
    all of the timers to use the same mode, maximizing consistency of the
    waveforms generated by any and all such timers.

    To modify the underlying baseband frequency of the PWM output signals, see
    the PWM_SetFrequency() API.

    WaveformGenerationMode
        The PWMMode_t enumeration constant indicating which hardware algorithms
        to use when generating the actual PWM output waveform. See the
        documentation of the PWMMode_t enumeration type for more details on this
        waveform generation.

    Return (void):
        This function returns nothing to the caller. The relevant Timer/Counter
        Control Registers are modified to set the desired waveform generation
        mode.
*/
void PWM_SetWaveformMode(const PWMMode_t WaveformGenerationMode);

/*
    PWM_SetFrequency

    This function sets the PWM baseband frequency for all 2 (or 3) output loads
    of the emitter, ensuring that they are all set to nominally the same
    frequency.  This is the mid-level API for configuring the PWM baseband
    frequency, such that all outputs are using the same available values.

    Frequency:
        One value from the PWMFrequency_t enumeration, which defines the full
        set of PWM frequencies which can be set for ALL of the available PWM
        timers.

    Return (void):
        This function returns nothing to the caller. The relevant Timer
        Controller registers are modified and the PWM frequency of the Arduino
        is modified.

    NOTE:
        This function assumes that all 2 (or 3) PWM output pairs of the Arduino
        are all being used for controlling the individual phases of the emitter,
        and therefore should share the same PWM frequency to best match the
        impedance characteristics of the windings. If this is not the case, and
        the PWM outputs are being used for distinct, unrelated loads, then use
        of the lower-level SetPWMFrequencyPrescaler() function is warranted.
*/
void PWM_SetFrequency(const PWMFrequency_t Frequency);

/*
    SetPWMTimerMode

    This function is the low-level hardware accessing API for modifying the PWM
    Waveform Generation Mode for a particular Timer subsystem of the Arduino
    Uno. This function directly modifies the TCCRnA and TCCRnB registers, where
    "n" is one of [0, 1, 2], and corresponds to a particular Timer block.

    Timer:
        The PWMTimer_t enumeration value denotes which of the three timer
        subsystems of the Arduino Uno to modify.
    TCCRnA_AND_Mask:
        This defines the 8-bit mask to logically AND with the TCCRnA register, to
        clear particular bits when setting the waveform generation mode.
    TCCRnA_OR_Mask:
        This defines the 8-bit mask to logically OR with the TCCRnA register, to
        clear particular bits when setting the waveform generation mode.
    TCCRnB_AND_Mask:
        This defines the 8-bit mask to logically AND with the TCCRnB register, to
        clear particular bits when setting the waveform generation mode.
    TCCRnB_OR_Mask:
        This defines the 8-bit mask to logically OR with the TCCRnB register, to
        clear particular bits when setting the waveform generation mode.

    Return (void):
        This function returns nothing to the caller. The Timer/Counter Control
        Registers are modified directly and take effect immediately.
*/
void SetPWMTimerMode(const PWMTimer_t Timer, const PWMModeMask_t TCCRnA_AND_Mask, const PWMModeMask_t TCCRnA_OR_Mask, const PWMModeMask_t TCCRnB_AND_Mask, const PWMModeMask_t TCCRnB_OR_Mask);

/*
    SetPWMFrequencyPrescaler

    This function modifies the required hardware registers for the given pins in
    order to set the PWM frequency pre-scaler value to modify PWM baseband
    frequency to something other than the default value of ~500Hz (~1000Hz on
    some pins for hardware reasons).

    On the Arduino Uno, only pins 3, 5, 6, 9, 10, and 11 allow for PWM output,
    and the rough output frequencies of the various pre-scaler values are noted
    in the enumeration constant names of the PWMPrescalerMask_t type.

    Timer:
        The specific Timer subsystem to modify the PWM frequency for. See the
        documentation for the PWMTimer_t type for the mapping of which
        PWMTimer_t corresponds to which output pins.
    PrescalerMask:
        The particular mask to apply to the register. These are contained within
        the PWMPrescalerMask_t enumeration as a simplification to computing or
        remembering what mask corresponds to what output frequency for which
        pin.

    Return (void):
        This function returns nothing to the caller. If the pin is valid, the
        provided mask is applied to the relevant register.
*/
void SetPWMFrequencyPrescaler(const PWMTimer_t Timer, const PWMPrescalerMask_t Mask);

/*
    InitializeTimestamp

    This function performs the top-level initialization for using Timer2 for the
    Timestamp_t API. This removes all existing Timer interrupts from Timers 0
    through 2, and enables only the Timer 2 Overflow interrupt.

    Return (void):
        This function returns nothing to the caller. This sets the relevant
        Timer register values for all available timers and returns.
*/

void InitializeTimestamp(void);
/*
    GetTimestamp

    This function takes a snapshot of the current Timer2 TCNT2 and
    Timer2OverflowCount values to determine the number of clock cycles since
    the Arduino has booted (or these counters overflowed). This is used to
    construct a Timestamp_t value corresponding to that number of main clock
    ticks.

    Return (Timestamp_t):
        A newly created Timestamp_t value, with the most up-to-date count of
        total clock ticks since reboot.
*/

Timestamp_t GetTimestamp(void);

/*
    Duration_ToSeconds

    This function converts a Duration_t value from the internal representation
    units into a floating-point value corresponding to an equivalent number of
    seconds. This is necessary to convert Timestamp_t and Duration_t values into
    proper units for calculations.

    Duration:
        The Duration_t value to compute and convert into a measure of seconds.

    Return (double):
        The number of seconds which the Duration_t corresponds to, as a
        floating-point value.
*/
double Duration_ToSeconds(const Duration_t& Duration);

/*
    Duration_ToMilliseconds

    This function converts a Duration_t value from the internal representation
    units into a floating-point value corresponding to an equivalent number of
    milliseconds. This is necessary when converting Timestamp_t and Duration_t
    values for use in calculations where physical units are required.

    Duration:
        The Duration_t value to convert to units of milliseconds.

    Return (double):
        The number of milliseconds corresponding to the Duration_t, as a
        floating-point value
*/
double Duration_ToMilliseconds(const Duration_t& Duration);

/*
    Duration_ToMicroseconds

    This function converts a Duration_t value from the internal representation
    units into a floating-point value corresponding to an equivalent number of
    microseconds. This is necessary when converting Timestamp_t and Duration_t
    values for use in calculations where physical units are required.

    Duration:
        The Duration_t value to convert to units of microseconds.

    Return (double):
        The number of microseconds corresponding to the Duration_t, as a
        floating-point value
*/
double Duration_ToMicroseconds(const Duration_t& Duration);

/* +++ Begin Interrupt Service Routine Function Forward Declarations +++ */

/*
    ComputeNextFieldOrientation

    This is the function provided as the Interrupt Service Routine (ISR) to be
    called when the external signal triggers the Arduino. This function computes
    the desired "next" orientation of the magnetic field from some pre-defined
    analytical expression or trajectory. This functionality is extracted out
    into an ISR in order to assert that the time-step between subsequent calls
    remains constant, regardless of the particular control-loop timing
    experienced by the "loop()" function.

    The amount of code within this function MUST be kept small, not only does
    this need to complete before the NEXT interrupt occurs, but there is limited
    memory associated with ISRs for their stack (local variables & potential
    function calls).

    This function tracks a variable which corresponds to the current,
    instantaneous and floating-point precision desired field orientation angle,
    as well as the amount of real time which has passed since the first
    interrupt. These are used to compute the "increment" to the field angle, or
    how much further to turn. This asserts that the rotational frequency of the
    field actually matches the desired value, up to the Nyquist Limit (more
    practically, up to the point where the RL time-constant of the
    phase-windings is long enough to prevent changing their current fast
    enough.)

    Return (void):
        This function returns nothing to the caller. This required that the
        global Emitter instance be created and writable. This function directly
        writes the current desired field orientation angle to this instance,
        ensuring that it is bounded to the half-open range [0, 360).
*/
void ComputeNextFieldOrientation(void);

/*
    Timer2Overflow_Timestamp

    This function is called when the Timer2 Counter Register (TCNT2) overflows.
    This is used to track the number of overflows, i.e. blocks of 256 increments
    of this counter, for use in the Timestamp_t API for measuring time intervals
    with the Arduino.

    Return (void):
        This function returns nothing to the caller. A global value for tracking
        the count of Timer2 overflow events is incremented.

    NOTE:
        This function is actually implemented within the "ISR(TIMER2_OVF_vect)"
        block. This forward declaration and comment-block is solely for
        documentation purposes.
*/
void Timer2Overflow_Timestamp(void);

// ...

/* --- End Interrupt Service Routine Function Forward Declarations --- */

// ...

/* +++ Begin Template Function Prototype Forward Declarations +++ */
/*
    This subsection contains the forward declarations for any templated
    functions used by this program.
*/

/*
    RoundTo

    This function provides a templated rounding function, to allow rounding
    values to something other that just the nearest integer. This is templated
    to allow for its use with arbitrary integral types, and returns a value of
    the same type as was given.

    Value:
        The original, un-rounded value to operate on. This must be an integer
        type.

    Modulus:
        The modulus, or the multiple to round the Value towards. This must be a
        positive integer type, greater than 0.

    Return:
        The resulting rounded value, with ties always rounding to the next
        multiple up.
*/
template<typename N>
N RoundTo(const N Value, const N Modulus);

/*
    RoundDown

    This function provides a templated rounding function, to allow rounding
    values to something other that just the nearest integer. This is templated
    to allow for its use with arbitrary integral types, and returns a value of
    the same type as was given.

    Value:
        The original, un-rounded value to operate on. This must be an integer
        type.

    Modulus:
        The modulus, or the multiple to round the Value towards. This must be a
        positive integer type, greater than 0.

    Return:
        The resulting rounded value, always rounded down.
*/
template<typename N>
N RoundDown(const N Value, const N Modulus);

/* --- End Template Function Prototype Forward Declarations --- */

/* +++ Begin Debugging Function Prototype Forward Declarations +++ */

/*
    WarnInvalidPWMPrescalerValue

    This function prints a warning message out to the Serial port if and when
    the PWM pre-scaler value is requested to be an invalid pin.

    Timer:
        The Timer subsystem requested to be modified. This will not be a valid
        enumeration constant value if this function is being called.
    Mask:
        An instance of the PWMPrescalerMask_t enumeration type, containing the
        mask for the lower 3-bits of the TCCRnB register requested to set.

    Return (void):
        This function returns nothing to the caller. The warning message is
        printed out to the Serial port and control returns to the caller.
*/
void WarnInvalidPWMPrescalerValue(const PWMTimer_t Timer, const PWMPrescalerMask_t Mask);

/*
    LogTimingInterrupt

    This function provides debugger logging of the registration of the external
    timer interrupt used to drive the "ticks" of the internal clock for
    computing the desired field orientation angle as a function of real-time.
    This is solely used for instrumentation and logging, not to be used during
    production code.

    Pin:
        The pin number of the pin to which the interrupt service routine will be
        attached.
    InterruptFunctionName:
        The name of the function to register as the interrupt routine to run
        when the interrupt event occurs.
    TriggerType:
        The type of event on the pin which will trigger the interrupt to occur.

    Return (void):
        This function returns nothing to the caller. This makes no changes to
        the hardware or controller. This only writes out a log message to the
        Serial port.
*/
void LogTimingInterrupt(const Pin_t Pin, const char* InterruptFunctionName, const char* TriggerType);

/*
    LogDutyCycleTable

    This function prints out an individual row of the Duty Cycle and Polarity
    look-up tables, to be used as logging or debugging information while
    development of this controller is still underway. This will print out the
    field angle, the duty cycle, and whether this corresponds to a positive or
    negative polarity output condition.

    FieldOrientationAngle:
        The orientation angle of the field to print out the values from the
        global look-up tables, in units of whole degrees.

    Return (void):
        This function returns nothing to the caller. This function modifies
        nothing about either the hardware or software of the controller. This
        function only writes out a log message to the Serial Port.
*/
void LogDutyCycleTable(uint16_t FieldOrientationAngle);

/*
    WarnInvalidTimer_Waveform

    This function prints out an error log message if and when an invalid
    PWMTimer_t value is provided to the "SetPWMTimerMode()" function. This
    notifies which timer was supplied, as well as printing out the masks also
    requested to be applied for the particular timer.

    Timer:
        The PWMTimer_t value corresponding to the requested Timer to modify.
    TCCRnA_AND_Mask:
        The 8-bit mask requested to be logically AND'ed with the TCCRnA
        register.
    TCCRnA_OR_Mask:
        The 8-bit mask requested to be logically OR'ed with the TCCRnA register.
    TCCRnB_AND_Mask:
        The 8-bit mask requested to be logically AND'ed with the TCCRnB
        register.
    TCCRnB_OR_Mask:
        The 8-bit mask requested to be logically OR'ed with the TCCRnB register.

    Return (void):
        This function returns nothing to the caller. This function modifies
        nothing within the caller. This only writes out a log message to the
        Serial Port.
*/
void WarnInvalidTimer_Waveform(const PWMTimer_t Timer, const PWMModeMask_t TCCRnA_AND_Mask, const PWMModeMask_t TCCRnA_OR_Mask, const PWMModeMask_t TCCRnB_AND_Mask, const PWMModeMask_t TCCRnB_OR_Mask);

/*
    WarnInvalidPWMFrequency

    This function provides a logging error message if an invalid PWMFrequency_t
    value is provided to the "PWM_SetFrequency()" function. This will write out
    the provided enumeration constant and an error message for debugging
    purposes.

    Frequency:
        The PWMFrequency_t enumeration constant which was provided to the
        "PWM_SetFrequency()" function. This will correspond to an unknown or
        un-handled frequency value if this function is actually called.

    Return (void):
        This function returns nothing to the caller. This function modifies
        nothing in the hardware or software of the controller. This function
        only writes out a message to the Serial Port.
*/
void WarnInvalidPWMFrequency(PWMFrequency_t Frequency);

/*
    LogWatchdogReset

    This function is called on reset if and when a reset is triggered by the
    Watchdog Timer expiring. This is used to notify if and when the watchdog
    timer triggers a reset, for debugging and instrumentation purposes.

    Return (void):
        This function returns nothing to the caller. This prints out the log
        message and returns without modifying anything in the controller.
*/
void LogWatchdogReset(void);

/*
    LogBrownOutReset

    This function is called on reset if and when a reset is triggered by a power
    brown-out condition. This is used for instrumentation to notify about the
    potential reset triggers.

    Return (void):
        This function returns nothing to the caller. This prints out the log
        message and returns without modifying anything in the controller.
*/
void LogBrownOutReset(void);

/*
    LogExternalReset

    This function is called on reset if and when a reset is triggered by the
    external pin reset functionality. This is used for instrumentation to notify
    about the potential reset triggers.

    Return (void):
        This function returns nothing to the caller. This prints out the log
        message and returns without modifying anything in the controller.
*/
void LogExternalReset(void);

/*
    LogPowerOnReset

    This function is called on reset if and when the reset is caused by a
    power-cycle.  This is used for instrumentation of the different reset
    triggers.

    Return (void):
        This function returns nothing to the caller. This prints out the log
        message and returns without modifying anything in the controller.
*/
void LogPowerOnReset(void);

/*
    LogSetupComplete

    This function notifies when the "setup()" function completes, to indicate
    that control then falls to the "loop()" function. This is used for
    instrumentation of the control flow.

    Return (void):
        This function returns nothing to the caller. This prints a log message
        to the Serial port and modifies nothing within the controller.
*/
void LogSetupComplete(void);

/*
    LogWatchdogInitialized

    This function prints a log message indicating that the Watchdog Timer has
    been initialized, and with what timeout and execution mode.

    Timeout:
        The WatchdogTimeout_t enumeration constant provided to the
        Watchdog_Initialize() function.
    Mode:
        The WatchdogMode_t enumeration constant provided to the
        Watchdog_Initialize() function.

    Return (void):
        This function returns nothing to the caller. This prints the log message
        to the Serial port and modifies nothing of the controller.
*/
void LogWatchdogInitialized(WatchdogTimeout_t Timeout, WatchdogMode_t Mode);

/*
    LogPWMWaveformMode

    This function prints a log message to show the PWM Waveform Generation mode
    being used by the controller.

    Mode:
        The PWMMode_t waveform generation mode passed to the ConfigurePWM()
        function.

    Return (void):
        This function returns nothing to the caller. This prints the log message
        out to the Serial port and modifies nothing in the controller.
*/
void LogPWMWaveformMode(PWMMode_t Mode);

/*
    LogPWMFrequency

    This function prints a log message to show the PWM baseband frequency being
    used by this controller.

    Frequency:
        The PWMFrequency_t value provided to the ConfigurePWM() function.

    Return (void):
        This function returns nothing to the caller. This prints the log message
        out to the Serial port and modifies nothing in the controller.
*/
void LogPWMFrequency(PWMFrequency_t Frequency);

/*
    LogPinMode

    This function prints a log message indicating the pinMode() being set for a
    particular pin, used for instrumenting of the controller.

    Pin:
        The pin which has had the pinMode modified.
    PinMode:
        The mode passed to the pinMode function for the given pin.

    Return (void):
        This function returns nothing to the caller. This prints the log message
        out to the Serial port and modifies nothing in the controller.
*/
void LogPinMode(Pin_t Pin, uint8_t PinMode);

/*
    LogInterruptInterval

    This function...

    AverageInterval:
        ...
    CurrentInterval:
        ...

    Return (void):
        ...
*/
void LogInterruptInterval(const Duration_t& AverageInterval, const Duration_t& CurrentInterval);

// ...

/* --- End Debugging Function Prototype Forward Declarations --- */

// ...

/* --- End Function Prototype Forward Declarations ---*/

/* +++ Begin Program Global Constant Definitions +++ */
/*
    This section contains the full set of globally accessible constants used by
    the program. These should be implemented as `constexpr` expressions to the
    fullest extent possible. All of the values in this section MUST be constant
    and read-only.
*/

/*
    PhaseCount defines the number of distinct phases within the stator. This is
    used in computation of the duty-cycles for mapping a particular desired
    field orientation angle to the set of phase currents to apply to each of the
    phase windings.
*/
constexpr uint8_t PhaseCount = (WINDING_COUNT >> 1);

/*
    PhaseSeparationRadians defines the angular separation of the phase windings
    around the ring of the Field Emitter. For the two-phase design this is π/2
    radians, and for the three-phase design this is 2π/3 radians. This is used
    in computing the duty cycles to apply to the phase windings.
*/
constexpr double PhaseSeparationRadians = (PI - (PI / PhaseCount));

/*
    PhaseSeparationDegrees defines the angular separation of the phase windings
    around the ring of the Field Emitter. For the two-phase design this is 90
    degrees, and for the three-phase design this is 120 degrees. This is used
    in computing the duty cycles to apply to the phase windings.
*/
constexpr uint16_t PhaseSeparationDegrees = (180 - (180 / PhaseCount));



/* +++ Begin Pin Definitions +++ */
/*
    Define the pairs of output pins for each of the two (or three) phases for
    the emitter.  The positive and negative pins for each phase MUST be both
    based on the same internal hardware timer register. This way the internal
    counter for the pins is consistent regardless of directionality, and the
    overall PWM behaviour is invariant to the polarity of the phase.
*/
constexpr Pin_t Phase_A_Positive = 5;
constexpr Pin_t Phase_A_Negative = 6;

constexpr Pin_t Phase_B_Positive = 3;
constexpr Pin_t Phase_B_Negative = 11;

constexpr Pin_t Phase_C_Positive = 9;
constexpr Pin_t Phase_C_Negative = 10;

/*
    EmitterTriggerPin

    Define the output pin to be toggled high when the orientation of the field
    changes by some minimum threshold. This can be used as an output signal or
    trigger for a camera or other circuitry to be based off the changing field
    orientation.
*/
constexpr Pin_t EmitterTriggerPin = 8;

/*
    PushButtonInputPin

    Define the pin connected to the PushButton which is used to enable actual
    execution of the programmed field trajectory. Until this button is pressed
    and a falling-event occurs, the controller will ensure the magnetic field is
    off. Once a falling-edge occurs, the field will begin to follow the
    programmed trajectory. Once the trajectory is completed, the controller will
    again enter a state where it waits for the button to be pressed and another
    falling-edge to be present on this pin.

    NOTE:
        This pin MUST be on a separate Port from any of the other pins used by
        the controller, so that we can utilize the pin-change interrupt for this
        behaviour, without needing additional complex logic to differentiate and
        dispatch on the actual exact pin which changed.
*/
constexpr Pin_t PushButtonInputPin = A0;

/*
    TimingInterruptPin

    This defines the input pin to monitor for an external clock signal to drive
    the "ComputeNextFieldOrientation()" interrupt service routine (ISR). This
    external interrupt is used to assert a constant ∆t when computing the "next"
    orientation of the field from the given function φ(t) for the field
    orientation as a function of time. See the documentation for the function
    "ComputeNextFieldOrientation()" for more details on this process.
*/
constexpr Pin_t TimingInterruptPin = 2;

/*
    DeviceInitializedPin

    This pin...
*/
constexpr Pin_t DeviceInitializedPin = 12;

/*
    FieldTrajectoryEnablePin

    This pin...
*/
constexpr Pin_t FieldTrajectoryEnablePin = 13;

/*
    WatchdogResetNotificationPin

    This defines the pin to set high to indicate a watchdog timeout event has
    occurred. This is primarily used for instrumentation purposes, to know when
    this timer expires.
*/
constexpr Pin_t WatchdogResetNotificationPin = 7;

/* --- End Pin Definitions --- */

/*
    FIELD_ORIENTATION_OFF

    This constant defines a mask to allow indicating that the magnetic field
    should be turned off, rather than to some particular angle. In order to turn
    off the field, one must ensure that:

        (0 != (LoadEmitter_t.DesiredFieldOrientation & FIELD_ORIENTATION_OFF))

    i.e. the upper 4 bits of the desired orientation are non-zero.

    How in particular this is computed is irrelevant, when the emitter computes
    the duty cycles to apply, it makes the above check explicitly and if the
    result is non-zero sets the duty cycles of all phases to 0.
*/
constexpr uint16_t FIELD_ORIENTATION_OFF = (uint16_t)0xF000;

/* +++ Begin Timer Control Register Declarations +++ */
/*
    This section contains declarations for the timer control registers
    for timers 0, 1, and 2 of the Arduino Uno. Proper declarations and definitions
    of these are provided during compilation by the tool-chain, but these
    declarations here prevent Intellisense erroring about the "unknown identifiers"
    in the functionality regarding the PWM Frequency Pre-scaler.
*/
#if !defined(TCCR0A)
    extern volatile uint8_t TCCR0A;
#endif
#if !defined(TCCR0B)
    extern volatile uint8_t TCCR0B;
#endif

#if !defined(TCCR1A)
    extern volatile uint8_t TCCR1A;
#endif
#if !defined(TCCR1B)
    extern volatile uint8_t TCCR1B;
#endif

#if !defined(TCCR2A)
    extern volatile uint8_t TCCR2A;
#endif
#if !defined(TCCR2B)
    extern volatile uint8_t TCCR2B;
#endif
/* --- End Timer Control Register Defintions --- */

/* +++ Begin Timer Interrupt Mask Registers +++ */
/*
    This section defines global symbols for the Timer0-2 Mask registers. These
    are used with the interrupts available for the timers, and are potentially
    modified in the Timer, PWM, and Timestamp_t setup functionality.
*/
#if !defined(TIMSK0)
    extern volatile uint8_t TIMSK0;
#endif
#if !defined(TIMSK1)
    extern volatile uint8_t TIMSK1;
#endif
#if !defined(TIMSK2)
    extern volatile uint8_t TIMSK2;
#endif
/* --- End Timer Interrupt Mask Registers --- */

/*
    TCNT2

    This defines the Timer2 Counter register, used in the Timestamp_t API for
    measuring clock cycles of the Arduino to provide a consistent and raw view
    of durations of time on the Arduino, without the additional correction logic
    in the platform-provided millis() and micros() functions.
*/
#if !defined(TCNT2)
    extern volatile uint8_t TCNT2;
#endif

// ...

/* --- End Program Global Constant Definitions --- */

/* +++ Begin Program Global Variable Definitions +++ */
/*
    This section contains the full set of global variables accessible and used
    by the program. These are distinct from the global constants above, as these
    are intended to be fully read/write accessible.
*/

#if defined(CORRECT_DELAY)
    /*
        DelayTimerRescalingExponent

        This global variable is used to correct for the rescaling of the Timer-0
        which may occur as a result of modifying the PWM baseband frequency for
        pins 5 & 6. This timer also controls the timer used in the delay()
        function, so we need to modify the argument of the delay() call to
        correct for the different pre-scaler actually in use.
    */
    long DelayTimerRescalingExponent = 0;

#endif

/*
    Emitter

    This defines the global variable for the overall Emitter device. This is
    the instance to operate on in order to modify the magnetic field to be
    generated, manipulate the field-orientation trigger pin, or anything
    else directly related to the magnetic field generation.

    This is initialized as the 3-phase design.
*/
FieldEmitter_t Emitter = FieldEmitter_t(
    LoadElement_t(Phase_A_Positive, Phase_A_Negative),
    LoadElement_t(Phase_B_Positive, Phase_B_Negative),
#if defined(THREE_PHASE)
    LoadElement_t(Phase_C_Positive, Phase_C_Negative),
#endif
    EmitterTriggerPin
);

/*
    InterruptFrequency

    This is the global instance of the InterruptFrequency_t value used to
    compute, store, and report the directly measured interrupt frequency of the
    signal used to drive the "ticks" of the "ComputeNextFieldOrientation()"
    function. This is directly measured to account for frequency drift, and to
    prevent measurement errors of a hardcoded frequency value biasing the
    calculations of the field orientation angle.
*/
InterruptFrequency_t InterruptFrequency = InterruptFrequency_t();

/*
    InterruptFrequency_DeratingFactor

    This value represents the number of interrupt events which must occur
    between calls of the InterruptFrequency_t.Tick() method. This is used to
    account for the case where the interrupt frequency is higher than can be
    handled by the Arduino.
*/
uint8_t InterruptFrequency_DeratingFactor = 1;

/*
    WelfordAccumulator_DefaultMaxCount

    This value...
*/
uint16_t WelfordAccumulator_DefaultMaxCount = 64;

/*
    DutyCycleLUT

    This array defines the look-up table between the field orientation angle
    (index) and the corresponding duty cycle to apply for the phase winding
    (value). This look-up table is provided so as to not require computing these
    values online during the control loop, as the Arduino platform does not
    contain hardware floating-point support and the necessary cosine
    calculations are extremely slow.

    NOTE:
        This look-up table contains only the magnitude of the duty-cycle to
        apply, and assumes that a field orientation angle of 0 is perfectly
        aligned with the positive polarity of Phase A.
*/
uint8_t DutyCycleLUT[360];

/*
    PolarityLUT

    This array defines the look-up table between the field orientation angle
    (index) and the corresponding duty cycle polarity to apply for the phase
    winding (value). This look-up table is provided so as to not require
    computing these values online during the control loop, as the Arduino
    platform does not contain hardware floating-point support and the necessary
    cosine calculations are extremely slow.

    NOTE:
        This look-up table contains only the polarity of the duty-cycle to
        apply, and assumes that a field orientation angle of 0 is perfectly
        aligned with the positive polarity of Phase A.
*/
LoadPolarity_t PolarityLUT[360];

/*
    Timer2OverflowCount

    This counter represents the number of times the Timer2 counter has
    overflowed.  This corresponds to 256 discrete increment operations, each of
    which takes a number of clock-cycles equal to the pre-scaler value applied
    to Timer2.  This is used as a replacement to the platform implemented
    overflow counter in order to generalize the timestamp behaviour away from
    assumptions of a particular Timer2 pre-scaler value as is used in the
    platform implementation.
*/
volatile unsigned long Timer2OverflowCount = 0;

/*
    Timer2Prescaler

    This value corresponds to the pre-scaler value currently applied to Timer2.
    This represents the number of clock cycles which must occur before the
    Timer/Counter Count Register (TCNT2) is actually incremented by one. This
    therefore provides the scaling factor between the value of this counter (and
    the corresponding Timer2OverflowCount variable), and the number of actual
    clock cycles the Arduino has experienced.

    This value must be set to the correct pre-scaler value if and when this is
    modified for Timer2. See the SetPWMFrequencyPrescaler() function for when
    and where to modify this variable.
*/
uint16_t Timer2Prescaler = 64;

/*
    Timer2PrescalerExponent

    This value corresponds to the base-2 logarithm of the Timer2 pre-scaler
    value. This is used in correcting times from Timer2 increment events to CPU
    clock cycles as the shift size to apply, since the pre-scaler is ALWAYS a
    power of two.
*/
uint8_t Timer2PrescalerExponent = 6;

/*
    FieldTrajectory_Enable

    This boolean...
*/
uint8_t FieldTrajectory_Enable = 0;

// ...

/* --- End Program Global Variable Definitions --- */

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
void setup() {

#if defined(DEBUG)
    // When running in debug/instrumented mode, enable the Serial port for
    // output communications.
    Serial.begin(115200);
    Serial.println("Starting setup()...");
#endif

    // Check for whether the controller is powering up for the first time, or a
    // reset event has been triggered. Handle the different potential reset
    // conditions accordingly.
    CheckResetState();

    // Configure the pins associated with the status LEDs
    ConfigureStatusLEDPins();

    // Configure the PWM mode & Frequency for all three timers of the Arduino
    // Uno.
    ConfigurePWM(PWM_7812Hz, PWMMode_FastPWM_8Bit);

    // Configure all of the pins related to the Emitter itself as either input
    // or output as required.
    Emitter.ConfigurePins();

    // Pre-compute and store the global look-up tables for phase-angle to duty
    // cycle and polarity with an angular resolution of 1 degree.
    PrepareDutyCycleLUT();

    // Initialize the watchdog timer, with a timeout of 500ms and in a mode such
    // that it executes the user-defined WDT_vect interrupt and then a system
    // reset. This must be done after the slow operations of setup(), but prior
    // to any interrupt registration which it is used to break out of.
    Watchdog_Initialize(WatchdogTimeout_500ms, WatchdogMode_Reset);

    // Prepare the necessary ISR and global variables for the custom Timer2
    // based Timestamp_t API
    InitializeTimestamp();

    // ...
    InterruptFrequency.Reset();

    // Configure the Interrupt Service Routine (ISR) used to update the field
    // orientation angle with a constant time-step.
    ConfigureTimingInterrupt(TimingInterruptPin);

    // Initialize the InterruptFrequency_t measurement of the period of the
    // Interrupt signal attached to TimingInterruptPin.
    InterruptFrequency.Initialize();

    // ...
    ConfigurePushButtonInterrupt(PushButtonInputPin);

    // Notify that the setup() function is now complete.
    Pin_Trigger(DeviceInitializedPin);
    LogSetupComplete();

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
void loop() {

    /*
        ...
    */
    wdt_reset();

    /*
        Update the duty cycles applied to the phase windings of the Emitter.
        The orientation angle to point the field in is computed by the ISR
        "ComputeNextFieldOrientation()", and so this control loop simply needs
        to utilize the value computed there.
    */
    Emitter.UpdateFieldOrientation();

    /*
        Update the computed value of the interrupt signal frequency. This is
        done to account for frequency drift over time, as well as component
        variability in the timing components of the oscillator driving this
        signal.

        This function is called in the main "loop()" function rather than the
        ISR in order to limit the amount of computation required for the ISR.
        Additionally, if a particular iteration of this loop takes longer than
        standard, the field orientation angle keeps ticking around regardless,
        so while we may lose one position, this should still optimally track the
        desired orientation.
    */
    InterruptFrequency.Update();

    return;
}
/* --- End Arduino Implementation Functions --- */

/* +++ Begin Program Function Definitions +++ */
/*
    This section contains the function definitions (function bodies) for the
    custom functionality required by the Arduino. For each function definition
    in this section, there must be a corresponding documented forward
    declaration in the designated section above. This section should ONLY
    contain those functions which are not related to debugging or
    instrumentation of the controller, so that any logging, debugging, or
    instrumentation can be trivially removed with the top-level DEBUG macro.
*/

void Pin_Trigger(const Pin_t Pin) {

#if defined(DEBUG)
    Serial.print("Setting Pin ");
    Serial.print(Pin);
    Serial.println(" to HIGH.");
#endif
    digitalWrite(Pin, HIGH);

    return;
}

void Pin_TriggerOff(const Pin_t Pin) {

#if defined(DEBUG)
    Serial.print("Setting Pin ");
    Serial.print(Pin);
    Serial.println(" to LOW.");
#endif
    digitalWrite(Pin, LOW);

    return;
}

void Pin_ToggleTrigger(const Pin_t Pin) {

    uint8_t NewPinState = !digitalRead(Pin);

#if defined(DEBUG)
    Serial.print("Setting Pin ");
    Serial.print(Pin);
    Serial.print(" to ");
    Serial.print(NewPinState ? "HIGH" : "LOW");
    Serial.print(".");
#endif
    digitalWrite(Pin, !digitalRead(Pin));

    return;
}

void CheckResetState(void) {

#if !defined(MCUSR)
    extern volatile uint8_t MCUSR;
#endif

    uint8_t OldMCUSR = MCUSR;
    MCUSR = 0;

    if ( 0 != (OldMCUSR & ResetTrigger_PowerOn )) {
        // On a power-cycle, we want to reset the field orientation interrupt
        // de-rating factor, as well as logging a message to notify this as the
        // reset trigger.
        LogPowerOnReset();
        InterruptFrequency_DeratingFactor = 1;
    }

    if ( 0 != (OldMCUSR & ResetTrigger_External )) {
        // On external-triggered resets, we just want to notify and continue on.
        LogExternalReset();
    }

    if ( 0 != (OldMCUSR & ResetTrigger_BrownOut )) {
        // On a brown-out reset, we just want to notify and continue on.
        LogBrownOutReset();
    }

    if ( 0 != (OldMCUSR & ResetTrigger_Watchdog )) {
        // On a watchdog reset, this is likely triggered by an overload of
        // interrupts. Increase the de-rating factor applied to the field
        // orientation timing interrupt to reduce the computational load this
        // exerts and continue on.
        LogWatchdogReset();

        // Reset all of the previously computed information about the frequency
        // of the interrupt signal.
        InterruptFrequency.Reset();

        // If the watchdog triggered, then we want to increase the derating
        // factor for the 555-Timer ISR.
        InterruptFrequency_DeratingFactor <<= 1;
    }

    return;
}

void ConfigureStatusLEDPins(void) {

    /*
        ...
    */
    pinMode(DeviceInitializedPin, OUTPUT);
    LogPinMode(DeviceInitializedPin, OUTPUT);
    Pin_TriggerOff(DeviceInitializedPin);

    /*
        ...
    */
    pinMode(FieldTrajectoryEnablePin, OUTPUT);
    LogPinMode(FieldTrajectoryEnablePin, OUTPUT);
    Pin_TriggerOff(FieldTrajectoryEnablePin);

    return;
}

void Watchdog_Initialize(WatchdogTimeout_t Timeout, WatchdogMode_t Mode) {

#if !defined(WDTCSR)
    extern volatile uint8_t WDTCSR;
#endif
#if !defined(MCUSR)
    extern volatile uint8_t MCUSR;
#endif

    noInterrupts();

    // Reset the watchdog timer at the beginning, to prevent an accidental
    // expiration from occurring when we modify the timeout and mode bits next.
    wdt_reset();

    // Write the Watchdog Change Enable (WDCE) bit, to allow modifications of
    // the other bits of the WDTCSR register. Then, set the timeout and mode
    // bits to the values requested. We don't have to set the WDCE bit low, as
    // the hardware explicitly does this after 4 clock cycles.
    MCUSR &= ~ResetTrigger_Watchdog;
    WDTCSR |= 0b00011000;
    WDTCSR |= Timeout | Mode;
    interrupts();

    wdt_reset();

    // Set the pin used to outwardly signal a watchdog reset as an output pin,
    // and write it low.
    pinMode(WatchdogResetNotificationPin, OUTPUT);
    LogPinMode(WatchdogResetNotificationPin, OUTPUT);
    Pin_TriggerOff(WatchdogResetNotificationPin);

    // Log the watchdog timer settings.
    LogWatchdogInitialized(Timeout, Mode);

    return;
}

void ConfigurePushButtonInterrupt(const Pin_t InterruptPin) {

    pinMode(InterruptPin, INPUT);
    LogPinMode(InterruptPin, INPUT);

#if !defined(PCICR)
    extern volatile uint8_t PCICR;
#endif

#if !defined(PCMSK1)
    extern volatile uint8_t PCMSK1;
#endif

    switch (InterruptPin) {
        case A0:
            PCICR |= 0b00000010;
            PCMSK1 |= (1 << (InterruptPin - A0));
            break;
        case A1:
            PCICR |= 0b00000010;
            PCMSK1 |= (1 << (InterruptPin - A0));
            break;
        case A2:
            PCICR |= 0b00000010;
            PCMSK1 |= (1 << (InterruptPin - A0));
            break;
        case A3:
            PCICR |= 0b00000010;
            PCMSK1 |= (1 << (InterruptPin - A0));
            break;
        case A4:
            PCICR |= 0b00000010;
            PCMSK1 |= (1 << (InterruptPin - A0));
            break;
        case A5:
            PCICR |= 0b00000010;
            PCMSK1 |= (1 << (InterruptPin - A0));
            break;
        default:
            // Log an error, this pin is unsupported.
            // ...
            return;
    }

    // Notify that the interrupt has been attached.
    LogTimingInterrupt(InterruptPin, "PCINT1_vect", "Change");
    // ...

    return;
}

void ConfigureTimingInterrupt(const Pin_t InterruptPin) {

    /*
        Attach the function "ComputeNextFieldOrientation()" to run when the
        InterruptPin sees a rising edge transition. This signal comes from an
        external oscillator, and is used to provide a constant ∆t during
        computation of the desired field orientation.
    */
    attachInterrupt(digitalPinToInterrupt(InterruptPin), ComputeNextFieldOrientation, RISING);

    LogPinMode(InterruptPin, INPUT);
    LogTimingInterrupt(InterruptPin, "ComputeNextFieldOrientation", "Rising");

    return;
}

void PrepareDutyCycleLUT(void) {

    /*
        For each of the possible orientation angles (in units of whole degrees),
        compute the duty cycle corresponding to that particular angle. This is
        all relative to Phase A, but we can apply these same values to
        additional phases by simply shifting the orientation angle to account
        for the PHYSICAL angular separation of the phase with respect to Phase
        A. This way we can compute a single look-up table and apply it to all
        phases symmetrically.
    */
    for ( uint16_t Index = 0; Index < ARRAY_SIZE(DutyCycleLUT); Index++ ) {

        /*
            Convert the duty cycle to radians, compute the cosine, and then
            re-scale to the range [-255, 255].
        */
        const int16_t DutyCycle = (int16_t)(255.0 * cos(DegreesToRadians(Index)));

        /*
            The Polarity is just the sign of the duty cycle, while the
            duty-cycle look-up table only records the magnitude.
        */
        PolarityLUT[Index] = (DutyCycle >= 0) ? (PositivePolarity) : (NegativePolarity);
        DutyCycleLUT[Index] = (uint8_t)abs(DutyCycle);

        // LogDutyCycleTable(Index);
    }

    return;
}

void ConfigurePWM(const PWMFrequency_t Frequency, const PWMMode_t WaveformGenerationMode) {

    /*
        Configure all of the timer subsystems to operate in the same Waveform
        Generation Mode. This ensures that all of them are as similar as
        possible, to minimize the differences in signals applied to the phase
        windings of the Emitter.
    */
    PWM_SetWaveformMode(WaveformGenerationMode);
    LogPWMWaveformMode(WaveformGenerationMode);

    /*
        Configure all of the timer subsystems to operate at the same baseband
        frequency. This ensures that all of them are as similar as possible, to
        minimize the differences in signals applied to the phase windings of the
        Emitter.
    */
    PWM_SetFrequency(Frequency);
    LogPWMFrequency(Frequency);

    return;
}

void PWM_SetWaveformMode(const PWMMode_t WaveformGenerationMode) {

    /*
        Dispatch on the particular waveform generation mode to set the
        Timer/Counter Control Registers (TCCRnA, TCCRnB) to the correct value
        for the requested PWM waveform generation mode. Set the registers for
        all two (or three) of the PWM timers, to ensure they are all as similar
        as possible.
    */

    switch (WaveformGenerationMode) {
        case PWMMode_FastPWM_8Bit:
            SetPWMTimerMode(Timer0, Timer0_FastPWM_8Bit_AndMask_A, Timer0_FastPWM_8Bit_OrMask_A, Timer0_FastPWM_8Bit_AndMask_B, Timer0_FastPWM_8Bit_OrMask_B);
            SetPWMTimerMode(Timer2, Timer2_FastPWM_8Bit_AndMask_A, Timer2_FastPWM_8Bit_OrMask_A, Timer2_FastPWM_8Bit_AndMask_B, Timer2_FastPWM_8Bit_OrMask_B);
#if defined(THREE_PHASE)
            SetPWMTimerMode(Timer1, Timer1_FastPWM_8Bit_AndMask_A, Timer1_FastPWM_8Bit_OrMask_A, Timer1_FastPWM_8Bit_AndMask_B, Timer1_FastPWM_8Bit_OrMask_B);
#endif
            /*
                Currently, only 8-bit FastPWM is implemented, as this is the
                mode that all three timers can operate in, where the output PWM
                baseband frequency is the maximum value.

                If more or different modes are required or desired, then add
                further cases (and corresponding enumeration entries)
                analogously to the one above.
            */
        default:
            break;
    }

    return;
}

void SetPWMTimerMode(const PWMTimer_t Timer, const PWMModeMask_t TCCRnA_AND_Mask, const PWMModeMask_t TCCRnA_OR_Mask, const PWMModeMask_t TCCRnB_AND_Mask, const PWMModeMask_t TCCRnB_OR_Mask) {

    /*
        Dispatch on the specific Timer, to modify the required control registers
        using the provided masks. For efficiency, perform only one read and one
        write, while performing the masks sequentially on the intermediate
        value.

        This reduces the memory-access requirements for this operation, and
        makes it easier for the compiler to optimize out no-op masks
    */

    switch (Timer) {
        case Timer0:
            TCCR0A = (TCCR0A & TCCRnA_AND_Mask) | TCCRnA_OR_Mask;
            TCCR0B = (TCCR0B & TCCRnB_AND_Mask) | TCCRnB_OR_Mask;
            break;
        case Timer1:
            TCCR1A = (TCCR1A & TCCRnA_AND_Mask) | TCCRnA_OR_Mask;
            TCCR1B = (TCCR1B & TCCRnB_AND_Mask) | TCCRnB_OR_Mask;
            break;
        case Timer2:
            TCCR2A = (TCCR2A & TCCRnA_AND_Mask) | TCCRnA_OR_Mask;
            TCCR2B = (TCCR2B & TCCRnB_AND_Mask) | TCCRnB_OR_Mask;
            break;
        default:
            WarnInvalidTimer_Waveform(Timer, TCCRnA_AND_Mask, TCCRnA_OR_Mask, TCCRnB_AND_Mask, TCCRnB_OR_Mask);
            break;
    }

    return;
}

void PWM_SetFrequency(const PWMFrequency_t Frequency) {

    /*
        Dispatch on the requested PWM baseband frequency, configuring all two
        (or three) PWM timers with the necessary masks to achieve the desired
        frequency by adjusting the pre-scaler value.
    */

    switch (Frequency) {
        case PWM_62500Hz:
            SetPWMFrequencyPrescaler(Timer0, Pin_5_6_62500Hz);
            SetPWMFrequencyPrescaler(Timer2, Pin_3_11_62500Hz);
#if defined(THREE_PHASE)
            SetPWMFrequencyPrescaler(Timer1, Pin_9_10_62500Hz);
#endif
            break;
        case PWM_7812Hz:
            SetPWMFrequencyPrescaler(Timer0, Pin_5_6_7812Hz);
            SetPWMFrequencyPrescaler(Timer2, Pin_3_11_7812Hz);
#if defined(THREE_PHASE)
            SetPWMFrequencyPrescaler(Timer1, Pin_9_10_7812Hz);
#endif
            break;
        case PWM_976Hz:
            SetPWMFrequencyPrescaler(Timer0, Pin_5_6_976Hz);
            SetPWMFrequencyPrescaler(Timer2, Pin_3_11_976Hz);
#if defined(THREE_PHASE)
            SetPWMFrequencyPrescaler(Timer1, Pin_9_10_976Hz);
#endif
            break;
        case PWM_244Hz:
            SetPWMFrequencyPrescaler(Timer0, Pin_5_6_244Hz);
            SetPWMFrequencyPrescaler(Timer2, Pin_3_11_244Hz);
#if defined(THREE_PHASE)
            SetPWMFrequencyPrescaler(Timer1, Pin_9_10_244Hz);
#endif
            break;
        case PWM_61Hz:
            SetPWMFrequencyPrescaler(Timer0, Pin_5_6_61Hz);
            SetPWMFrequencyPrescaler(Timer2, Pin_3_11_61Hz);
#if defined(THREE_PHASE)
            SetPWMFrequencyPrescaler(Timer1, Pin_9_10_61Hz);
#endif
            break;
        default:
            WarnInvalidPWMFrequency(Frequency);
            break;
    }
}

void SetPWMFrequencyPrescaler(const PWMTimer_t Timer, const PWMPrescalerMask_t Mask) {

    /*
        Dispatch on the particular pin to modify the PWM Baseband Frequency for
    */

    switch (Timer) {
        case Timer0:
            TCCR0B = (TCCR0B & 0b11111000) | Mask;
#if defined(CORRECT_DELAY)
            /*
                Since Timer0 also plays a role in the implementation-provided
                `delay()` function, we need to determine the correction factor
                to "undo" our changes to Timer0 here and account for the assumed
                Timer0 pre-scaler which `delay()` and the other timing functions
                use.

            */
            switch (Mask) {
                case Pin_5_6_61Hz:
                    DelayTimerRescalingExponent = -4;
                    break;
                case Pin_5_6_244Hz:
                    DelayTimerRescalingExponent = -2;
                    break;
                case Pin_5_6_976Hz:
                    DelayTimerRescalingExponent = 0;
                    break;
                case Pin_5_6_7812Hz:
                    DelayTimerRescalingExponent = 3;
                    break;
                case Pin_5_6_62500Hz:
                    DelayTimerRescalingExponent = 6;
                    break;
            }
#endif
            break;
        case Timer1:
            TCCR1B = (TCCR1B & 0b11111000) | Mask;
            break;
        case Timer2:
            TCCR2B = (TCCR2B & 0b11111000) | Mask;
            switch (Mask) {
                case Pin_3_11_62500Hz:
                    Timer2Prescaler = 1;
                    Timer2PrescalerExponent = 0;
                    break;
                case Pin_3_11_7812Hz:
                    Timer2Prescaler = 8;
                    Timer2PrescalerExponent = 3;
                    break;
                case Pin_3_11_976Hz:
                    Timer2Prescaler = 64;
                    Timer2PrescalerExponent = 6;
                    break;
                case Pin_3_11_244Hz:
                    Timer2Prescaler = 256;
                    Timer2PrescalerExponent = 8;
                    break;
                case Pin_3_11_61Hz:
                    Timer2Prescaler = 1024;
                    Timer2PrescalerExponent = 10;
                    break;
            }
            break;
        default:
            WarnInvalidPWMPrescalerValue(Timer, Mask);
            break;
    }

    return;
}

void InitializeTimestamp(void) {

#if defined(DEBUG)
    Serial.println("Disabling all Timer interrupts except TIMER2_OVF");
#endif

    /*
        Within Timers 0, 1, and 2, there is a Timer Interrupt Mask Register,
        TIMSKn.  This register contains bits which, if set, act to enable
        execution of a corresponding ISR when the interrupt trigger condition is
        met.

        For Timer0, the lower three bits correspond to the following interrupts:
            bit 0: Timer 0 Overflow
            bit 1: Timer 0 Compare Output A
            bit 2: Timer 0 Compare Output B

        For Timer 1, bits 0, 1, 2, and 5 correspond to the following interrupts:
            bit 0: Timer 1 Overflow
            bit 1: Timer 1 Compare Output A
            bit 2: Timer 1 Compare Output B
            bit 5: Timer 1 Input Capture

        For Timer 2, bits 0, 1, and 2 correspond to the following interrupts:
            bit 0: Timer 2 Overflow
            bit 1: Timer 2 Compare Output A
            bit 2: Timer 2 Compare Output B

        To minimize the amount of processor time spent on computations not
        relevant for this controller, we explicitly disable ALL of these
        interrupts, with the exception of Timer 2 Overflow, which we use as the
        overflow counter for the Timestamp_t API.
    */
    TIMSK0 &= 0b11111000;
    TIMSK1 &= 0b11011000;
    TIMSK2 |= 0b00000001;
    TIMSK2 &= 0b11111001;

    return;
}

Timestamp_t GetTimestamp(void) {

    /*
        Why are we not explicitly checking for a Timer2 Overflow here?
    */
    uint8_t Timer2Count = TCNT2;
    unsigned long OverflowCount = Timer2OverflowCount;

    return Timestamp_t(OverflowCount, Timer2Count);
}

// ...

/* +++ Begin Struct/Class Methods +++ */
LoadElement_t::LoadElement_t(const Pin_t Positive, const Pin_t Negative) {

    this->OutputPins[PositivePolarity] = Positive;
    this->OutputPins[NegativePolarity] = Negative;

    this->DutyCycle = 0;
    this->Polarity = PositivePolarity;

    return;
};

FieldEmitter_t::FieldEmitter_t(LoadElement_t&& PhaseA, LoadElement_t&& PhaseB, const uint8_t TriggerPin) {

    this->Phases[0] = PhaseA;
    this->Phases[1] = PhaseB;

    this->TriggerPin = TriggerPin;
    this->DesiredFieldOrientation = FIELD_ORIENTATION_OFF;

    return;
}

FieldEmitter_t::FieldEmitter_t(LoadElement_t&& PhaseA, LoadElement_t&& PhaseB, LoadElement_t&& PhaseC, const uint8_t TriggerPin) {

    this->Phases[0] = PhaseA;
    this->Phases[1] = PhaseB;
    this->Phases[2] = PhaseC;

    this->TriggerPin = TriggerPin;
    this->DesiredFieldOrientation = FIELD_ORIENTATION_OFF;

    return;
}

void FieldEmitter_t::UpdateFieldOrientation(void) {

#if defined(DEBUG)
    // Serial.print("Desired Angle: ");
    // this->DesiredFieldOrientation == FIELD_ORIENTATION_OFF ? Serial.println("Off") : Serial.println(this->DesiredFieldOrientation);
#endif

    this->ComputeDutyCycles();

    /*
        Define custom logic in this block to define the conditions in which the
        output trigger pin is set or unset.
    */
    // if (( this->DesiredFieldOrientation == FIELD_ORIENTATION_OFF ) || ( this->DesiredFieldOrientation == 0 )) {
    //     this->Trigger();
    // } else {
    //     this->TriggerOff();
    // }

    this->ApplyDutyCycles();

    return;
}

void FieldEmitter_t::ToggleTrigger(void) const {

    /*
        Perform a digital read of the current output pin, returning either a
        1 or 0 (HIGH or LOW). Logically invert that to get the opposite state,
        and then write that back out to the pin.
    */
    Pin_ToggleTrigger(this->TriggerPin);

    return;
}

void FieldEmitter_t::Trigger(void) const {

    Pin_Trigger(this->TriggerPin);

    return;
}

void FieldEmitter_t::TriggerOff(void) const {

    Pin_TriggerOff(this->TriggerPin);

    return;
}

void FieldEmitter_t::ConfigurePins(void) const {

    /*
        For each of the phases, we need to set the PWM pins as outputs.
        Iterate over the phases and set each pin as an output.
    */
    for ( uint8_t PhaseIndex = 0; PhaseIndex < PhaseCount; PhaseIndex++ ) {

        const LoadElement_t& PhaseWinding = this->Phases[PhaseIndex];

        pinMode(PhaseWinding.OutputPins[PositivePolarity], OUTPUT);
        LogPinMode(PhaseWinding.OutputPins[PositivePolarity], OUTPUT);
        Pin_TriggerOff(PhaseWinding.OutputPins[PositivePolarity]);

        pinMode(PhaseWinding.OutputPins[NegativePolarity], OUTPUT);
        LogPinMode(PhaseWinding.OutputPins[NegativePolarity], OUTPUT);
        Pin_TriggerOff(PhaseWinding.OutputPins[NegativePolarity]);
    }

    pinMode(this->TriggerPin, OUTPUT);
    LogPinMode(this->TriggerPin, OUTPUT);
    Pin_TriggerOff(this->TriggerPin);

    return;
}

void FieldEmitter_t::ComputeDutyCycles(void) {

    /*
        Explicitly make a local copy of the desired field orientation at the
        beginning of this function. With the asynchronous interrupt
        "ComputeNextFieldOrientation" running, it's assured that at some point
        that that the interrupt will trigger during the execution of this
        function, between setting the duty cycles for the each of the phases. To
        prevent this race condition, make a local copy here and accept that this
        field orientation value may be "slightly" out of date by the time it's
        applied to the phase windings.
    */
    const uint16_t DesiredFieldOrientation = this->DesiredFieldOrientation;

    /*
        If the desired orientation indicates to turn the field off, set the duty
        cycle of each pin to 0.
    */
    if ( 0 != (FIELD_ORIENTATION_OFF & DesiredFieldOrientation) ) {
        for ( uint8_t PhaseIndex = 0; PhaseIndex < PhaseCount; PhaseIndex++ ) {
            this->Phases[PhaseIndex].DutyCycle = 0;
        }
    } else {
        /*
            If the desired orientation is not indicating to turn the field off,
            then apply a modulus operator to ensure the orientation angle is
            bounded within the range [0, 360).
        */
        for ( uint8_t PhaseIndex = 0; PhaseIndex < PhaseCount; PhaseIndex++ ) {

            const uint16_t PhaseOrientation = (DesiredFieldOrientation + (PhaseIndex * PhaseSeparationDegrees)) % 360;
            LoadElement_t& PhaseWinding = this->Phases[PhaseIndex];

            PhaseWinding.DutyCycle = DutyCycleLUT[PhaseOrientation];
            PhaseWinding.Polarity = PolarityLUT[PhaseOrientation];
        }
    }

    return;
}

void FieldEmitter_t::ApplyDutyCycles(void) const {

    for ( uint8_t PhaseIndex = 0; PhaseIndex < PhaseCount; PhaseIndex++ ) {

        /*
            Get a constant reference to the current phase winding to apply the
            new PWM duty cycles for.
        */
        const LoadElement_t& PhaseWinding = this->Phases[PhaseIndex];

        /*
            Extract out the duty cycle and polarity for this particular phase
            into local variables for readability. This should all be optimized
            out by the compiler anyway.
        */
        const LoadPolarity_t Polarity = PhaseWinding.Polarity;
        const uint8_t DutyCycle = PhaseWinding.DutyCycle;

        /*
            Extract out and determine which of the output pins of the Phase
            winding should be active and which should be inactive.

            If we perform an exclusive OR operation between 1 and the polarity
            value, this acts to toggle the value between 1 and 0, exactly what
            we want to know.
        */
        const Pin_t InactivePin = PhaseWinding.OutputPins[(0x01 ^ Polarity)];
        const Pin_t ActivePin = PhaseWinding.OutputPins[Polarity];

        /*
            Apply the required duty cycles to the output pins for this phase.

            We find the inactive output pin as being the OPPOSITE polarity as
            what the duty cycle computed earlier requires. To find this, we can
            simply XOR the polarity with 1 to toggle between the two
            LoadPolarity_t values.

            First, set the inactive pin to low, to ensure that there is never a
            chance for the H-bridge to short through the power MOSFETs.

            Then, apply the computed duty cycle to the active pin, either
            enabling the PWM signal or modifying the duty cycle.
        */
        digitalWrite(InactivePin, LOW);
        analogWrite(ActivePin, DutyCycle);
    }

    return;
}

WelfordAccumulator_t::WelfordAccumulator_t(uint16_t MaxCount) {

    this->Count = 0;
    this->MaxCount = MaxCount;
    this->Mean = 0.0f;
    this->M2 = 0.0f;

    return;
}

void WelfordAccumulator_t::Reset() {

    this->Count = 0;
    this->Mean = 0.0f;
    this->M2 = 0.0f;

    return;
}

void WelfordAccumulator_t::TestUpdate(double Value, double* NewMean, double* NewVariance) {

    uint16_t Count = (this->Count < this->MaxCount) ? this->Count + 1 : this->Count;
    double Delta = Value - this->Mean;
    *NewMean = this->Mean + Delta;

    double Delta2 = Value - *NewMean;
    double M2 = Delta * Delta2;
    *NewVariance = M2 / double(Count);

    return;
}

void WelfordAccumulator_t::TestUpdate(uint32_t Value, double* NewMean, double* NewVariance) {
    return this->TestUpdate(double(Value), NewMean, NewVariance);
}

void WelfordAccumulator_t::Update(double Value) {

    if ( this->Count < this->MaxCount ) {
        this->Count++;
    }

    double Delta = Value - this->Mean;
    this->Mean += Delta / double(this->Count);

    double Delta2 = Value - this->Mean;
    this->M2 += Delta * Delta2;

    return;
}

void WelfordAccumulator_t::Update(uint32_t Value) {
    this->Update(double(Value));
    return;
}

double WelfordAccumulator_t::Variance() {
    if ( this->Count == 0 ) {
        return 0.0f;
    }

    return this->M2 / double(this->Count);
}

double WelfordAccumulator_t::SampleVariance() {
    if ( this->Count <= 1 ) {
        return 0.0f;
    }

    return this->M2 / double(this->Count - 1);
}

InterruptFrequency_t::InterruptFrequency_t(Duration_t UpdatePeriod = InterruptFrequencyDefaultUpdatePeriod) {

    /*
        Initialize all of the Timestamp_t values to 0's, so we have known values
        to compare against.
    */
    this->CurrentTimeStamp = Timestamp_t();
    this->PreviousTimeStamp = Timestamp_t();
    this->LastUpdateTimeStamp = Timestamp_t();

    /*
        Set the DeltaT value to a sentinel of 0.0f to indicate that it has not
        yet been set and to allow the ISR to quickly check whether to use it or
        not.
    */
    this->DeltaT = 0.0f;

    /*
        Set the update period to the requested value.
    */
    this->UpdatePeriod = UpdatePeriod;

    this->IsSet = false;


    /*
        At initialization, this has recorded 0 interrupt interval values into
        the online average calculation.
    */
    this->Count = 0;
    this->IntervalMean = 0;
    this->IntervalVariance = 0;

    return;
}

void InterruptFrequency_t::Reset(void) {

#if defined(DEBUG)
    Serial.println("Resetting InterruptFrequency_t");
#endif

    this->CurrentTimeStamp = Timestamp_t();
    this->PreviousTimeStamp = Timestamp_t();
    this->LastUpdateTimeStamp= Timestamp_t();

    if ( 0 == this->UpdatePeriod ) {
        this->UpdatePeriod = InterruptFrequency_t::UpdatePeriod;
    }

    if ( 0 == this->MaxCount ) {
        this->MaxCount = InterruptFrequency_t::MaxCount;
    }

    this->Count = 0;
    this->IntervalMean = 0;
    this->IntervalVariance = 0.0f;

    this->DeltaT = 0.0f;

    this->IsSet = false;

    return;
}

void InterruptFrequency_t::Initialize(void) {

    if ( this->UpdatePeriod == 0 ) {
        this->UpdatePeriod = InterruptFrequencyDefaultUpdatePeriod;
    }

    /*
        To initialize this value, we just keep trying to compute the interrupt
        interval until we actually get a value. This may take up to 1
        UpdatePeriod to do so, but this ensures that by the time the main loop()
        actually executes, the interrupt period is a known good value.
    */
    while ( !this->IsSet ) {
#if defined(DEBUG)
    Serial.println("Initializing InterruptFrequency_t...");
#endif
        this->Update();
    }

#if defined(DEBUG)
    Serial.print("Successfully determined interrupt frequency: ");
    Serial.print(1.0 / this->DeltaT);
    Serial.println("Hz");
#endif

    return;
}

void InterruptFrequency_t::Tick(void) {

    /*
        To "tick", we need to capture the current timestamp, push the old "now"
        into the "previous" Timestamp_t, and then record the current
        Timestamp_t. We split apart the "get current" and "write current"
        timestamp values in case some other interrupt triggers between shifting
        the existing value. This gives us insensitivity to other interrupts
        introducing timing jitter into this measurement.
    */

    Timestamp_t Now = GetTimestamp();

    this->PreviousTimeStamp = this->CurrentTimeStamp;
    this->CurrentTimeStamp = Now;

    return;
}

void InterruptFrequency_t::Update(void) {

    /*
        Disable interrupts, so that we read both the Previous and Current
        timestamp values atomically.

        Why do we not worry about Timer2 Overflow events here?
    */
    noInterrupts();
    Timestamp_t Previous = this->PreviousTimeStamp;
    Timestamp_t Now = this->CurrentTimeStamp;
    interrupts();

    Timestamp_t LastUpdated = this->LastUpdateTimeStamp;
    Duration_t UpdatePeriod = Now.DurationSince(LastUpdated);

    // If the interrupt period has been set before, make sure we've waiting long
    // enough for one update period to pass. Otherwise just return early.
    if ( this->IsSet ) {
        if ( UpdatePeriod < this->UpdatePeriod ) {
            return;
        }
    } else {
        // If this has not been set yet, just assert that the Previous timestamp
        // has been set, returning if not.
        if ( 0 == Previous.DurationSince(Timestamp_t())) {
            return;
        }
    }

    // Compute and set the time interval between the previous and current
    // timestamp.
    this->SetDeltaT(Now, Previous);
    return;
}

void InterruptFrequency_t::SetDeltaT(const Timestamp_t& Now, const Timestamp_t& Previous) {

    // Compute the duration between the current and previous timestamps.
    // If this is zero for some reason, then ignore them and continue on.
    const Duration_t InterruptInterval = Now.DurationSince(Previous);

    /*
        Using Welford's Algorithm:
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        we compute the mean and variance of the stream of values for the
        interrupt period.  We do this to apply some filtering to the value we
        report back to the actual interrupt, since we have some apriori
        knowledge about the stability of the frequency of the interrupt signal
        itself. We know that we can ignore measurements which are significantly
        different from the average, while still allowing for adjustments to
        match slow frequency drift over time due to temperature and Vcc
        dependencies in the interrupt source.

        The recurrence relation(s) we rely on for this algorithm are as follows:

        Sample Mean of the first n samples:
            x(bar)_n = x(bar)_{n-1} + ((x_{n} - x(bar)_{n-1}) / n)

        Sum of Squares of Differences from Current Mean:
            M_{2,n} = M_{2,n-1} + (x_{n} - x(bar)_{n-1})(x_{n} - x(bar)_{n})

        Biased Sample Variance of the first n samples:
            σ^{2}_{n} = M_{2,n} / n

        Unbiased Sample Variance of the first n samples:
            s^{2}_{n} = M_{2,n} / (n-1), n > 1

        We need to perform these calculations in such a way that we do not
        disturb these recurrence relations if we reach a sample which we reject.
        Therefore, we introduce temporary variables for computing the
        "increment" for the current sample, and only if the value is
        satisfactory do we update the actual recurrence values.
    */
    static Duration_t NextIntervalMean = 0;
    static double NextIntervalVariance = 0.0f;
    double VarianceIncrement = 0.0f;

    if ( this->IsSet ) {

        // Until we reach the maximum number of measurements for the online-average
        // calculation, record how many measurements we've taken so far.
        if ( this->Count < this->MaxCount ) {
            this->Count += 1;
        }

        double MeanIncrement = ((SignedDuration_t)InterruptInterval - (SignedDuration_t)this->IntervalMean) / this->Count;
        SignedDuration_t MeanIncrementI = (Duration_t)((MeanIncrement > 0) ? ceil(MeanIncrement) : floor(MeanIncrement));
        VarianceIncrement = ((((SignedDuration_t)InterruptInterval - (SignedDuration_t)this->IntervalMean) * ((SignedDuration_t)InterruptInterval - (SignedDuration_t)(this->IntervalMean + MeanIncrementI))) - this->IntervalVariance) / this->Count;

        NextIntervalMean += MeanIncrementI;
        NextIntervalVariance += VarianceIncrement;

    } else {

        bool Initialized = false;
        static constexpr uint8_t nClusters = 3;
        static Duration_t Means[nClusters] = { 0 };
        static double Variances[nClusters] = { 0.0f };
        static uint16_t ClusterCounts[nClusters] = { 0 };

        for ( uint8_t ClusterIndex = 0; ClusterIndex < nClusters; ClusterIndex++ ) {

            ClusterCounts[ClusterIndex] += 1;

            double MeanIncrement = ((SignedDuration_t)InterruptInterval - (SignedDuration_t)Means[ClusterIndex]) / (double)ClusterCounts[ClusterIndex];
            SignedDuration_t MeanIncrementI = (Duration_t)((MeanIncrement > 0) ? ceil(MeanIncrement) : floor(MeanIncrement));
            VarianceIncrement = ((((SignedDuration_t)InterruptInterval - (SignedDuration_t)Means[ClusterIndex]) * ((SignedDuration_t)InterruptInterval - (SignedDuration_t)(Means[ClusterIndex] + MeanIncrementI))) - Variances[ClusterIndex]) / ClusterCounts[ClusterIndex];

            if ( VarianceIncrement <= ( 5 * Means[ClusterIndex] )) {

                Means[ClusterIndex] += MeanIncrementI;
                Variances[ClusterIndex] += VarianceIncrement;

                if ( ClusterCounts[ClusterIndex] < this->MaxCount ) {
                    return;
                }

                this->IntervalMean = Means[ClusterIndex];
                this->IntervalVariance = Variances[ClusterIndex];

                NextIntervalMean = Means[ClusterIndex];
                NextIntervalVariance = Variances[ClusterIndex];

                this->Count = ClusterCounts[ClusterIndex];
                Initialized = true;

                break;

            } else {
                ClusterCounts[ClusterIndex] -= 1;
            }
        }

        if ( !Initialized ) {
            return;
        }
    }

    if ( VarianceIncrement > ( 5 * this->IntervalMean )) {
        NextIntervalMean = this->IntervalMean;
        NextIntervalVariance = this->IntervalVariance;
        return;
    }

    // Convert the Duration into units of seconds, record it, and update the
    // other required fields of this instance are updated accordingly.
    this->IntervalMean = NextIntervalMean;
    this->IntervalVariance = NextIntervalVariance;

    this->DeltaT = Duration_ToSeconds(this->IntervalMean);
    this->LastUpdateTimeStamp = Now;

    this->IsSet = true;

    // Display how long the interrupt interval is computed to be.
    LogInterruptInterval(this->IntervalMean, InterruptInterval);

    return;
}

Timestamp_t::Timestamp_t(const volatile Timestamp_t& Other) {
    this->Ticks = Other.Ticks;
    return;
}

volatile Timestamp_t& Timestamp_t::operator= (const Timestamp_t& Other) volatile {
    this->Ticks = Other.Ticks;
    return;
}

volatile Timestamp_t& Timestamp_t::operator= (const volatile Timestamp_t& Other) volatile {
    this->Ticks = Other.Ticks;
    return;
}

Timestamp_t::Timestamp_t(const uint32_t OverflowCount, const uint8_t Counter) {
    this->Ticks = ((OverflowCount << 8) + Counter);
    return;
}

Duration_t Timestamp_t::DurationSince(const Timestamp_t& Since) const {
    return ((Duration_t)(this->Ticks - Since.Ticks) << Timer2PrescalerExponent);
}

double Duration_ToSeconds(const Duration_t& Duration) {
    return ((double)(Duration) / (Second));
}

double Duration_ToMilliseconds(const Duration_t& Duration) {
    return ((double)Duration / Millisecond);
}

double Duration_ToMicroseconds(const Duration_t& Duration) {
    return ((double)Duration / Microsecond);
}

/* --- End Struct/Class Methods --- */

/* +++ Begin Template Function Definitions +++ */

template<typename N>
N RoundTo(const N Value, const N Modulus) {

    N Remainder = Value % Modulus;

    // If the value is already divisible, return it.
    if ( 0 == Modulus ) {
        return Value;
    }

    // If the original value is closer to the next multiple
    // of the modulus above, round up. Otherwise round down.
    // In this implementation, ties round up always.
    if ( Remainder >= ( Modulus >> 1 )) {
        return (Value + Modulus - Remainder);
    } else {
        return (Value - Remainder);
    }
}

template<typename N>
N RoundDown(const N Value, const N Modulus) {
    return (Value - (Value % Modulus));
}

/* --- End Template Function Definitions --- */

/* +++ Begin Interrupt Service Routine Function Definitions +++ */

void ComputeNextFieldOrientation(void) {

    /*
        Handle the de-rating factor of the ISR in case the 555-Timer
        is running too fast.
    */
    static uint8_t InterruptCount = 0;
    if (++InterruptCount < InterruptFrequency_DeratingFactor) {
        return;
    }
    InterruptCount = 0;

    /*
        ...
    */
    interrupts();

    /*
        This interrupt is the primary point to modify the trajectory and overall
        behaviour of the magnetic field generated by the Emitter this controller
        is connected to. This function is called routinely, with a fixed and
        known time interval between each call, allowing the field trajectory to
        be specified as some analytical function of time.

        What this means in practice, is that how you would like the orientation
        of the field (φ) to change over time can should be expressed as some
        function of time, i.e. φ(t). This function is then solved at each
        time-step to recover the precise orientation angle at that moment in
        time, allowing for complex field trajectories without explicitly
        providing a list of orientation angles and time delays between each.

        As an example, if you want to produce a magnetic field which rotates
        with some fixed rotational rate, f, as measured in Hertz, you would
        proceed as follows:

            -   φ(0) corresponds to the initial field angle you would like to
                start at.

            -   Compute, by hand, the discrete time derivative, i.e.
                    φ(t) = 2πft + φ(0)
                    ∆φ/∆t = 2πf∆t/∆t
                    ∆φ = 2πf∆t

            -   At each time-step, compute φ(t_{n}) = φ(t_{n-1}) + 2πf∆t

        You would only have to implement this last step in this function, i.e.
        the calculation of φ += 2πf∆t. Relying on the compiler for
        optimizations, this typically only requires a single floating-point
        multiplication and addition, fairly efficient for the behaviour it
        implements.

        In full generality, the standard operation of this controller expects a
        field orientation angle function of the form:

            φ(t) = RoundTo( f(t) mod 2π, (2π / AngularResolution) )
            φ(t+P) = φ(t), for some period P
            (Optional: φ(t) = FIELD_OFF, t ∈ {t_{n}, ...})

        Where f(t) is some function of time, definining what the field
        orientation should be. This can be a continuous function, piecewise
        function, a state machine, or anything else computable within this
        interrupt.

        In order to turn the field off, a special sentinel value of:

            -   FIELD_ORIENTATION_OFF

        is provided as a global variable in order to indicate that the field
        should be turned off entirely, rather than set to some particular
        orientation angle around the circle.

        At the end of this function, the computed value of φ(t) is written "out"
        to the Emitter_t global instance as the desired field orientation value
        for the current moment in time.  On the next iteration through the main
        loop() function, it will read this value (or potentially a more
        up-to-date value, if this interrupt is executed a second time before the
        main loop() comes back around!), compute the necessary duty cycles for
        the phase windings, and then update the PWM outputs.

        In order to modify the field trajectory, you only need to look through
        the functionality within the block-commented section delimited with:

            "Begin Analytical Expression..."

        Everything outside of this small section of this interrupt is required
        bookkeeping and other required implementation details to provide the
        timing guarantees for the ∆t value between invocations of this function.
        You need not modify this without good reason or a strong reason.

        The values provided by this bookkeeping for your use in computing the
        field orientation angle are as follows:
            1)  Phi - The field orientation angle, as computed by the previous
                        invocation of this function.
            2)  t   - The total time elapsed for the field trajectory
                        calculations.
            3)  dt  - The time interval between successive interrupt
                        invocations.
    */

    /*
        Within this interrupt, you will find many values defined with qualifiers
        using some combination of "static" and "constexpr". These additional
        qualifiers are both necessary for proper functionality, and to allow the
        optimizer of the compiler the most opportunity to perform some of the
        calculations at compile-time, rather than run-time.

        The values declared with "static" are local to this function, but are
        not re-initialized or reset on the next invocation of the function; they
        retain their value across multiple calls to this function. Retaining
        value across multiple calls would be achievable using global variables,
        but this would also mean that these values were visible outside of the
        scope of this function, something we very much do not want.

        The values declared with "const" or "constexpr" are values which are
        guaranteed never to change their value during execution of the Sketch.
        Being declared as either "const" or "constexpr" allows the compiler to
        know this, and to perform calculations using only "const" or "constexpr"
        values AT COMPILE TIME.  This way, the calculations run on YOUR
        computer, and not the slow Arduino. This lets you write larger,
        complicated functions for the field orientation angle in a
        mathematically convenient manner, while not forcing the Arduino to
        perform the difficult and slow floating-point mathematical operations.
    */

    /*
        Implement the actual logic for the frequency de-rating functionality. If
        the number of times this interrupt has been called since the last time
        is less than the count implied by the de-rating factor, just skip
        executing this iteration and return "immediately".

        If this is equal to the count implied by the de-rating factor, actually
        continue on with the standard logic.
    */

    /*
        Tick the clock of the InterruptFrequency_t instance, to measure the
        interval between this interrupt and the previous one.
    */
    InterruptFrequency.Tick();

    /*
        Phi

        This variable represents the current, instantaneous, field orientation
        angle, in units of radians. This value assumes that the 0-angle position
        is exactly aligned with the positive Phase A winding. In order to set an
        initial field orientation, this value should be set to the orientation
        desired, in units of radians.
    */
    static double Phi = 0;

    /*
        t

        This value represents the accumulated time which the µ-controller has
        been running since the last "reset event". This "Reset Event" can be a
        explicit hardware reset, triggered either by power-cycling the
        µ-controller or pressing the hardware reset button.  Alternatively, and
        typically much more often, this "Reset Event" will simply be a
        well-defined end-time condition where the behaviour of field orientation
        should repeat. This allows for whatever behaviour is defined for the
        phase-angle to be a time-periodic function, with some maximal period T,
        beyond which the full behaviour repeats.

        This value cannot be const or constexpr since this MUST vary as the
        interrupts occur to reflect the accumulation of time.

        NOTE:
            The total time which the µ-controller has run is typically only
            required when computing an orientation function which includes an
            angular acceleration term. This is not a strict requirement, any
            definable and computable function (which fits within the memory and
            time constraints of this interrupt) can be provided as the
            definition of the field orientation angle, and can use any
            information or variables accessible from this interrupt.
    */
    static double t = 0;

    // Don't trigger until all setup is completed.
    if ( 0 == FieldTrajectory_Enable ) {
        Phi = 0;
        t = 0;
        Emitter.DesiredFieldOrientation = FIELD_ORIENTATION_OFF;
        Emitter.Trigger();
        return;
    }

    Emitter.TriggerOff();

    /*
        dt

        This value represents the amount of time which has elapsed since the
        previous interrupt was called. This is commonly used for computing the
        incremental rotation of the field in the case where a constant rotation
        rate is desired.

        NOTE:
            This value is provided in units of seconds.
    */
    double dt = InterruptFrequency.DeltaT;

    /* +++
        Begin Analytical Expression for computing the incremental change to the
        field orientation angle.
    +++ */

    static constexpr uint16_t Segments = 12; // How many segments to divide the 360 degrees of the circle into?
    static constexpr uint16_t Theta_deg = (uint16_t)(360.0 / Segments);

    static constexpr double Omega = (1.0 / 30.0); // Hz, rotational frequency
    static constexpr double Alpha = 0.0;  // Hz^2, rotational acceleration

    static constexpr double Period = 180.0;
    static constexpr double OffDelay = 0.0; //Period / Segments;

    if ( t <= Period ) {
        Phi += ((TWO_PI * Omega * dt) + (TWO_PI * 2 * Alpha * t * dt));

        Emitter.DesiredFieldOrientation = RoundDown(
            (uint16_t)(((uint16_t)RadiansToDegrees(Phi)) % 360),
            Theta_deg
        );
    } else {
        Emitter.DesiredFieldOrientation = FIELD_ORIENTATION_OFF;
    }

    /* ---
        End Analytical Expression for computing the incremental change to the
        field orientation angle.
    --- */

    // Update the total elapsed duration experienced by the interrupt. At some
    // point, this must repeat. If it's been long enough to repeat, reset both
    // the field angle and the total elapsed time.
    t += dt;
    if ( t > (Period + OffDelay)) {
        t = 0;
        Phi = 0;
        FieldTrajectory_Enable = 0;
        Emitter.DesiredFieldOrientation = FIELD_ORIENTATION_OFF;
        Pin_TriggerOff(FieldTrajectoryEnablePin);
    }

    return;
}

void Timer2Overflow_Timestamp(void);
#if defined(TIM2_OVF_vect)
ISR(TIM2_OVF_vect) {

    interrupts();
    Timer2OverflowCount += 1;

    return;
}
#elif defined(TIMER2_OVF_vect)
ISR(TIMER2_OVF_vect){

    interrupts();
    Timer2OverflowCount += 1;

    return;
}
#else
    #warning "Unknown or missing Timer2 Overflow Vector definition!"
#endif

/*
    ...
*/
#if defined(PCINT1_vect)
ISR(PCINT1_vect) {

    interrupts();
    if ( LOW == digitalRead(PushButtonInputPin)) {
        FieldTrajectory_Enable = 0xFF;
        Pin_Trigger(FieldTrajectoryEnablePin);
    }

    return;
}
#endif

// ...

/* --- End Interrupt Service Routine Function Definitions --- */

/* --- End Program Function Definitions --- */

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

void WarnInvalidPWMPrescalerValue(const PWMTimer_t Timer, const PWMPrescalerMask_t Mask) {

    Serial.print("Error: Attempting to set PWM pre-scaler value ");
    Serial.print(Mask, 2);
    Serial.print(" for Timer ");
    Serial.println(Timer);

    return;
}

void LogTimingInterrupt(const Pin_t Pin, const char* InterruptFunctionName, const char* TriggerType) {

    Serial.print("Attaching interrupt function ");
    Serial.print(InterruptFunctionName);
    Serial.print(" to pin ");
    Serial.print(Pin);
    Serial.print(" on a ");
    Serial.print(TriggerType);
    Serial.println(" event.");

    return;
}

void LogDutyCycleTable(uint16_t FieldOrientationAngle) {

    Serial.print("Field Orientation Angle ");
    Serial.print(FieldOrientationAngle);
    Serial.print(" degrees has duty cycle: ");
    (PolarityLUT[FieldOrientationAngle] == PositivePolarity) ? Serial.print("+") : Serial.print("-");
    Serial.println(DutyCycleLUT[FieldOrientationAngle]);

    return;
}

void WarnInvalidTimer_Waveform(const PWMTimer_t Timer, const PWMModeMask_t TCCRnA_AND_Mask, const PWMModeMask_t TCCRnA_OR_Mask, const PWMModeMask_t TCCRnB_AND_Mask, const PWMModeMask_t TCCRnB_OR_Mask) {

    Serial.print("Error: Timer ");
    Serial.print(Timer);
    Serial.print("is not available for waveform generation mode changes with masks: ");
    Serial.print("(TCCRnA & ");
    Serial.print(TCCRnA_AND_Mask);
    Serial.print(") | ");
    Serial.print(TCCRnA_OR_Mask);
    Serial.print(" - TCCRnB & ");
    Serial.print(TCCRnB_AND_Mask);
    Serial.print(") | ");
    Serial.println(TCCRnB_OR_Mask);

    return;
}

void WarnInvalidPWMFrequency(PWMFrequency_t Frequency) {

    Serial.print("Error: Unknown PWMFrequency_t value: ");
    Serial.println(Frequency);

    return;
}

void LogWatchdogReset(void) {

    Serial.println("Watchdog Reset");

    return;
}

void LogBrownOutReset(void) {

    Serial.println("BrownOut Reset");

    return;
}

void LogExternalReset(void) {

    Serial.println("External Pin Reset");

    return;
}

void LogPowerOnReset(void) {

    Serial.println("Power On/Reset");

    return;
}

void LogSetupComplete(void) {

    Serial.println("Completed setup().");
    Serial.flush();

    return;
}

void LogWatchdogInitialized(WatchdogTimeout_t Timeout, WatchdogMode_t Mode) {

    Serial.print("Enabling watchdog timeout: ");
    switch(Timeout) {
        case WatchdogTimeout_16ms:
            Serial.print("16");
            break;
        case WatchdogTimeout_32ms:
            Serial.print("32");
            break;
        case WatchdogTimeout_64ms:
            Serial.print("64");
            break;
        case WatchdogTimeout_125ms:
            Serial.print("125");
            break;
        case WatchdogTimeout_250ms:
            Serial.print("250");
            break;
        case WatchdogTimeout_500ms:
            Serial.print("500");
            break;
        case WatchdogTimeout_1000ms:
            Serial.print("1000");
            break;
        case WatchdogTimeout_2000ms:
            Serial.print("2000");
            break;
        case WatchdogTimeout_4000ms:
            Serial.print("4000");
            break;
        case WatchdogTimeout_8000ms:
            Serial.print("8000");
            break;
        default:
            Serial.print("UNKNOWN");
            break;
    }

    Serial.print("ms and Mode: ");
    switch (Mode) {
        case WatchdogMode_Off:
            Serial.println("Off");
            break;
        case WatchdogMode_Interrupt:
            Serial.println("Interrupt");
            break;
        case WatchdogMode_Reset:
            Serial.println("Reset");
            break;
        case WatchdogMode_InterruptReset:
            Serial.println("Interrupt & Reset");
            break;
        default:
            Serial.println("UNKNOWN");
            break;
    }

    return;
}

void LogPWMFrequency(PWMFrequency_t Frequency) {

    Serial.print("Setting PWM Frequency to ");

    switch (Frequency) {
        case PWM_62500Hz:
            Serial.print("62500");
            break;
        case PWM_7812Hz:
            Serial.print("7812");
            break;
        case PWM_976Hz:
            Serial.print("976");
            break;
        case PWM_244Hz:
            Serial.print("244");
            break;
        case PWM_61Hz:
            Serial.print("61");
            break;
        default:
            Serial.print("UNKNOWN");
            break;
    }

    Serial.println("Hz");

    return;
}

void LogPWMWaveformMode(PWMMode_t Mode) {

    Serial.print("Setting PWM Waveform Generation Mode: ");

    switch (Mode) {
        case PWMMode_FastPWM_8Bit:
            Serial.println("8-bit FastPWM");
            break;
        default:
            Serial.println("UNKNOWN");
            break;
    }

    return;
}

void LogPinMode(Pin_t Pin, uint8_t PinMode) {

    Serial.print("Setting pin ");
    Serial.print(Pin, 10);
    Serial.print(" as ");
    switch (PinMode) {
        case INPUT:
            Serial.println("INPUT");
            break;
        case INPUT_PULLUP:
            Serial.println("INPUT_PULLUP");
            break;
        case OUTPUT:
            Serial.println("OUTPUT");
            break;
        default:
            Serial.println("UNKNOWN");
            break;
    }

    return;
}

void LogInterruptInterval(const Duration_t& AverageInterval, const Duration_t& CurrentInterval) {

    Serial.print("Average Interrupt Frequency: ");
    Serial.print((1.0 / Duration_ToSeconds(AverageInterval)), 2);
    Serial.print("Hz - Current Frequency: ");
    Serial.print((1.0 / Duration_ToSeconds(CurrentInterval)), 2);
    Serial.println("Hz");

    return;
}

// ...

#else
/*
    Place empty function definition for the debug-only functions in this
    section.  These should all be void functions with a function definition
    consisting of only a single return statement. These will be safely optimized
    out during compilation and will produce no runtime penalty for calling when
    running in non-debug mode.
*/
void WarnInvalidPWMPrescalerValue(const PWMTimer_t Timer, const PWMPrescalerMask_t Mask) { return; }

void LogTimingInterrupt(const Pin_t Pin, const char* InterruptFunctionName, const char* TriggerType) { return; }

void LogDutyCycleTable(uint16_t FieldOrientationAngle) { return; }

void WarnInvalidTimer_Waveform(const PWMTimer_t Timer, const PWMModeMask_t TCCRnA_AND_Mask, const PWMModeMask_t TCCRnA_OR_Mask, const PWMModeMask_t TCCRnB_AND_Mask, const PWMModeMask_t TCCRnB_OR_Mask) { return; }

void WarnInvalidPWMFrequency(PWMFrequency_t Frequency) { return; }

void LogWatchdogReset(void) { return; }

void LogBrownOutReset(void) { return; }

void LogExternalReset(void) { return; }

void LogPowerOnReset(void) { return; }

void LogSetupComplete(void) { return; }

void LogWatchdogInitialized(WatchdogTimeout_t Timeout, WatchdogMode_t Mode) { return; }

void LogPWMFrequency(PWMFrequency_t Frequency) { return; }

void LogPWMWaveformMode(PWMMode_t Mode) { return; }

void LogPinMode(Pin_t Pin, uint8_t PinMode) { return; }

void LogInterruptInterval(const Duration_t& AverageInterval, const Duration_t& CurrentInterval) { return; }

// ...

#endif
/* --- End Debugging Function Definitions --- */
