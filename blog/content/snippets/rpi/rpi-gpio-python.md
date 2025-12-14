---
title: "Raspberry Pi GPIO - Python"
date: 2024-12-12
draft: false
category: "rpi"
tags: ["rpi-knowhow", "python", "gpio", "i2c", "spi", "i2s", "pwm"]
---


Python examples for accessing Raspberry Pi GPIO, I2C, SPI, I2S, and PWM.

---

## GPIO (Digital I/O)

### Installation

```bash
# Install RPi.GPIO
sudo apt install python3-rpi.gpio

# Or use pip
pip3 install RPi.GPIO

# Alternative: gpiozero (higher-level)
pip3 install gpiozero
```

### Basic GPIO (RPi.GPIO)

```python
import RPi.GPIO as GPIO
import time

# Set mode (BCM or BOARD)
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering

# Setup pins
LED_PIN = 18
BUTTON_PIN = 23

GPIO.setup(LED_PIN, GPIO.OUT)  # Output
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Input with pull-up

# Digital output
GPIO.output(LED_PIN, GPIO.HIGH)
time.sleep(1)
GPIO.output(LED_PIN, GPIO.LOW)

# Digital input
button_state = GPIO.input(BUTTON_PIN)
if button_state == GPIO.LOW:
    print("Button pressed!")

# Cleanup
GPIO.cleanup()
```

### GPIO with gpiozero (Recommended)

```python
from gpiozero import LED, Button
from signal import pause

# LED
led = LED(18)
led.on()
led.off()
led.toggle()
led.blink(on_time=1, off_time=1)  # Blink every second

# Button
button = Button(23)

def button_pressed():
    print("Button pressed!")
    led.toggle()

button.when_pressed = button_pressed

# Keep program running
pause()
```

---

## PWM (Pulse Width Modulation)

### Software PWM

```python
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
LED_PIN = 18
GPIO.setup(LED_PIN, GPIO.OUT)

# Create PWM instance (pin, frequency in Hz)
pwm = GPIO.PWM(LED_PIN, 1000)

# Start PWM (duty cycle 0-100%)
pwm.start(0)

# Fade in
for duty in range(0, 101, 5):
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.1)

# Fade out
for duty in range(100, -1, -5):
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.1)

pwm.stop()
GPIO.cleanup()
```

### Hardware PWM (pigpio)

```python
import pigpio
import time

# Connect to pigpio daemon
pi = pigpio.pi()

if not pi.connected:
    exit()

PWM_PIN = 18

# Set PWM frequency (Hz)
pi.set_PWM_frequency(PWM_PIN, 1000)

# Set PWM duty cycle (0-255)
pi.set_PWM_dutycycle(PWM_PIN, 128)  # 50%

# Fade
for i in range(0, 256, 5):
    pi.set_PWM_dutycycle(PWM_PIN, i)
    time.sleep(0.01)

pi.set_PWM_dutycycle(PWM_PIN, 0)
pi.stop()
```

---

## I2C

### Installation

```bash
# Enable I2C
sudo raspi-config
# Interface Options > I2C > Enable

# Install tools
sudo apt install i2c-tools python3-smbus

# Check I2C devices
i2cdetect -y 1
```

### I2C Communication

```python
import smbus2
import time

# I2C bus (1 for newer Pi, 0 for very old models)
bus = smbus2.SMBus(1)

# Device address (e.g., 0x48 for ADS1115)
DEVICE_ADDR = 0x48

# Write byte
bus.write_byte_data(DEVICE_ADDR, 0x01, 0x83)

# Read byte
data = bus.read_byte_data(DEVICE_ADDR, 0x00)
print(f"Read: {data}")

# Read block
block = bus.read_i2c_block_data(DEVICE_ADDR, 0x00, 2)
print(f"Block: {block}")

# Write block
bus.write_i2c_block_data(DEVICE_ADDR, 0x01, [0x83, 0x00])

bus.close()
```

### I2C Example: BMP280 Sensor

```python
import smbus2
import time

class BMP280:
    def __init__(self, address=0x76):
        self.bus = smbus2.SMBus(1)
        self.address = address
        
        # Read calibration data
        self.dig_T1 = self.read_uint16(0x88)
        self.dig_T2 = self.read_int16(0x8A)
        self.dig_T3 = self.read_int16(0x8C)
        
        # Configure sensor
        self.bus.write_byte_data(self.address, 0xF4, 0x27)
    
    def read_uint16(self, reg):
        data = self.bus.read_i2c_block_data(self.address, reg, 2)
        return data[0] | (data[1] << 8)
    
    def read_int16(self, reg):
        val = self.read_uint16(reg)
        return val if val < 32768 else val - 65536
    
    def read_temperature(self):
        # Read raw temperature
        data = self.bus.read_i2c_block_data(self.address, 0xFA, 3)
        adc_T = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        
        # Compensate
        var1 = ((adc_T / 16384.0) - (self.dig_T1 / 1024.0)) * self.dig_T2
        var2 = (((adc_T / 131072.0) - (self.dig_T1 / 8192.0)) ** 2) * self.dig_T3
        t_fine = var1 + var2
        temperature = t_fine / 5120.0
        
        return temperature

# Usage
sensor = BMP280()
temp = sensor.read_temperature()
print(f"Temperature: {temp:.2f}°C")
```

---

## SPI

### Installation

```bash
# Enable SPI
sudo raspi-config
# Interface Options > SPI > Enable

# Install library
pip3 install spidev
```

### SPI Communication

```python
import spidev
import time

# Open SPI bus
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Device 0 (CE0)

# Configure
spi.max_speed_hz = 1000000  # 1 MHz
spi.mode = 0  # SPI mode 0

# Transfer data
to_send = [0x01, 0x02, 0x03]
received = spi.xfer2(to_send)
print(f"Received: {received}")

# Read data
data = spi.readbytes(3)
print(f"Read: {data}")

# Write data
spi.writebytes([0x01, 0x02, 0x03])

spi.close()
```

### SPI Example: MCP3008 ADC

```python
import spidev
import time

class MCP3008:
    def __init__(self):
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 1350000
    
    def read_channel(self, channel):
        if channel < 0 or channel > 7:
            return -1
        
        # Build command
        cmd = [1, (8 + channel) << 4, 0]
        
        # Send command and read response
        reply = self.spi.xfer2(cmd)
        
        # Extract 10-bit value
        value = ((reply[1] & 3) << 8) + reply[2]
        
        return value
    
    def read_voltage(self, channel, vref=3.3):
        value = self.read_channel(channel)
        voltage = (value * vref) / 1023.0
        return voltage
    
    def close(self):
        self.spi.close()

# Usage
adc = MCP3008()

while True:
    value = adc.read_channel(0)
    voltage = adc.read_voltage(0)
    print(f"Channel 0: {value} ({voltage:.2f}V)")
    time.sleep(1)
```

---

## I2S (Audio)

### Installation

```bash
# Enable I2S
sudo raspi-config
# Interface Options > I2S > Enable

# Install libraries
pip3 install pyaudio numpy
```

### I2S Audio Playback

```python
import pyaudio
import wave
import numpy as np

# Play WAV file
def play_wav(filename):
    chunk = 1024
    
    wf = wave.open(filename, 'rb')
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    
    data = wf.readframes(chunk)
    
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# Generate tone
def generate_tone(frequency=440, duration=1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    wave_data = (wave_data * 32767).astype(np.int16)
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        output=True
    )
    
    stream.write(wave_data.tobytes())
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# Usage
play_wav('sound.wav')
generate_tone(440, 2)  # A4 note for 2 seconds
```

---

## Complete Example: Weather Station

```python
from gpiozero import LED, Button
import smbus2
import time

class WeatherStation:
    def __init__(self):
        self.led = LED(18)
        self.button = Button(23)
        self.bus = smbus2.SMBus(1)
        self.bmp280_addr = 0x76
        
        # Setup BMP280
        self.bus.write_byte_data(self.bmp280_addr, 0xF4, 0x27)
        
        # Button callback
        self.button.when_pressed = self.read_and_display
    
    def read_temperature(self):
        # Simplified BMP280 reading
        data = self.bus.read_i2c_block_data(self.bmp280_addr, 0xFA, 3)
        adc_T = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        # ... (calibration code omitted for brevity)
        return adc_T / 1000.0  # Simplified
    
    def read_and_display(self):
        self.led.on()
        temp = self.read_temperature()
        print(f"Temperature: {temp:.2f}°C")
        time.sleep(0.5)
        self.led.off()
    
    def run(self):
        print("Weather Station Running. Press button to read temperature.")
        while True:
            time.sleep(0.1)

# Usage
station = WeatherStation()
station.run()
```

---

## Best Practices

1. **Always cleanup GPIO** - Use `try/finally` or context managers
2. **Use pull-up/pull-down resistors** - Prevent floating inputs
3. **Check voltage levels** - RPi GPIO is 3.3V, not 5V tolerant!
4. **Use level shifters** - For 5V devices
5. **Limit current** - Use resistors with LEDs
6. **Enable interfaces** - I2C, SPI, I2S must be enabled in raspi-config

---