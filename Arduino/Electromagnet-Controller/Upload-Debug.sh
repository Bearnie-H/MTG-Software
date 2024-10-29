#!/usr/bin/env bash

echo "Compiling and uploading [ DEBUG ] mode firmware for the electromagnet controller..."

echo "Checking for available serial ports..."
SerialPorts=$(python3 << EOF
import serial
import serial.tools.list_ports
import serial.tools.list_ports_common

Devices = serial.tools.list_ports.comports()
for Index, Device in enumerate(Devices, start=1):
    print(Device.device)
EOF
)

ArduinoPort=
for SerialPort in ${SerialPorts[@]}; do
    echo "Serial Port: $SerialPort"
    if [[ $SerialPort == *cu.usbmodem* ]]; then
        echo "Assuming this corresponds to the Arduino..."
        ArduinoPort="$SerialPort"
    else
        echo "This doesn't look like an Arduino serial port..."
    fi
done

if [ -z "$ArduinoPort" ]; then
    echo "Failed to find what looks like the Arduino Serial Port. Just compiling sketch..."
    arduino-cli compile -b arduino:avr:uno --build-property "build.extra_flags=-DDEBUG" --clean "Electromagnet-Controller.ino"
else
    arduino-cli compile -b arduino:avr:uno --build-property "build.extra_flags=-DDEBUG" --clean --port "$ArduinoPort" -u "Electromagnet-Controller.ino"
fi
