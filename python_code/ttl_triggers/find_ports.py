# https://stefanappelhoff.com/usb-to-ttl/code-examples.html

import serial  # imports pyserial
import serial.tools.list_ports as list_ports

# List all comports
all_ports = list_ports.comports()
print([(port.device, port.description) for port in all_ports]) 

# Each entry in the `all_ports` list is a serial device. Check it's
# description and device attributes to learn more
first_serial_device = all_ports[0]
print(first_serial_device.device)  # the `port_name`
print(first_serial_device.description)  # perhaps helpful to know if this is your device

# continue until you found your device, then note down the `port_name`
