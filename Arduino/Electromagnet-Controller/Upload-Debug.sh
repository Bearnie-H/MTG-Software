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

ArduinoPort="$1"
for SerialPort in ${SerialPorts[@]}; do
    echo "Serial Port: $SerialPort"
    if [[ $SerialPort == *cu.usbmodem* ]]; then
        echo "Assuming this corresponds to the Arduino..."
        if [ ! -z $ArduinoPort ]; then
            echo "Already found a serial port!"
        else
            ArduinoPort="$SerialPort"
        fi
    else
        echo "This doesn't look like an Arduino serial port..."
    fi
done

if [ -z "$ArduinoPort" ]; then
    echo "Failed to find what looks like the Arduino Serial Port. Just compiling sketch..."
    arduino-cli compile -b arduino:avr:uno --build-property "build.extra_flags=-DDEBUG" --build-property "build.extra_flags=-DLOG_LEVEL_HIGH" --clean "Electromagnet-Controller.ino"
else
    arduino-cli compile -b arduino:avr:uno --build-property "build.extra_flags=-DDEBUG" --build-property "build.extra_flags=-DLOG_LEVEL_HIGH" --clean --port "$ArduinoPort" -u "Electromagnet-Controller.ino"
    echo "Starting to monitor Serial Port..."
    arduino-cli monitor -b arduino:avr:uno --config 115200 --port "$ArduinoPort"
fi
