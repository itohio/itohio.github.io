---
title: "Raspberry Pi GPIO - Go"
date: 2024-12-12
draft: false
category: "rpi"
tags: ["rpi-knowhow", "go", "golang", "gpio", "i2c", "spi", "pwm"]
---


Go examples for accessing Raspberry Pi GPIO, I2C, SPI, and PWM using periph.io.

---

## Installation

```bash
# Install Go (if not already installed)
wget https://go.dev/dl/go1.21.5.linux-arm64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-arm64.tar.gz

# Add to PATH
echo 'export PATH=$PATH:/usr/local/bin/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify
go version
```

---

## GPIO (Digital I/O)

### Setup Project

```bash
mkdir rpi-gpio && cd rpi-gpio
go mod init rpi-gpio
go get periph.io/x/conn/v3/gpio
go get periph.io/x/conn/v3/gpio/gpioreg
go get periph.io/x/host/v3
```

### Basic GPIO

```go
package main

import (
    "fmt"
    "log"
    "time"

    "periph.io/x/conn/v3/gpio"
    "periph.io/x/conn/v3/gpio/gpioreg"
    "periph.io/x/host/v3"
)

func main() {
    // Initialize periph
    if _, err := host.Init(); err != nil {
        log.Fatal(err)
    }

    // Get GPIO pin (BCM numbering)
    ledPin := gpioreg.ByName("GPIO18")
    if ledPin == nil {
        log.Fatal("Failed to find GPIO18")
    }

    // Set as output
    if err := ledPin.Out(gpio.Low); err != nil {
        log.Fatal(err)
    }

    // Blink LED
    for i := 0; i < 10; i++ {
        ledPin.Out(gpio.High)
        fmt.Println("LED ON")
        time.Sleep(500 * time.Millisecond)

        ledPin.Out(gpio.Low)
        fmt.Println("LED OFF")
        time.Sleep(500 * time.Millisecond)
    }
}
```

### GPIO Input with Button

```go
package main

import (
    "fmt"
    "log"
    "time"

    "periph.io/x/conn/v3/gpio"
    "periph.io/x/conn/v3/gpio/gpioreg"
    "periph.io/x/host/v3"
)

func main() {
    if _, err := host.Init(); err != nil {
        log.Fatal(err)
    }

    // LED output
    ledPin := gpioreg.ByName("GPIO18")
    ledPin.Out(gpio.Low)

    // Button input with pull-up
    buttonPin := gpioreg.ByName("GPIO23")
    if err := buttonPin.In(gpio.PullUp, gpio.BothEdges); err != nil {
        log.Fatal(err)
    }

    fmt.Println("Press button to toggle LED")

    ledState := false

    for {
        // Wait for button press
        buttonPin.WaitForEdge(-1)

        // Debounce
        time.Sleep(50 * time.Millisecond)

        // Check if button is pressed (active low with pull-up)
        if buttonPin.Read() == gpio.Low {
            ledState = !ledState
            if ledState {
                ledPin.Out(gpio.High)
                fmt.Println("LED ON")
            } else {
                ledPin.Out(gpio.Low)
                fmt.Println("LED OFF")
            }
        }
    }
}
```

---

## PWM

```go
package main

import (
    "log"
    "time"

    "periph.io/x/conn/v3/gpio"
    "periph.io/x/conn/v3/gpio/gpioreg"
    "periph.io/x/conn/v3/physic"
    "periph.io/x/host/v3"
)

func main() {
    if _, err := host.Init(); err != nil {
        log.Fatal(err)
    }

    // Get PWM-capable pin
    pwmPin := gpioreg.ByName("GPIO18")
    if pwmPin == nil {
        log.Fatal("Failed to find GPIO18")
    }

    // Check if pin supports PWM
    if pwm, ok := pwmPin.(gpio.PinPWM); ok {
        // Set PWM frequency (1kHz)
        freq := 1000 * physic.Hertz

        // Fade in
        for duty := gpio.Duty(0); duty <= gpio.DutyMax; duty += gpio.DutyMax / 100 {
            if err := pwm.PWM(duty, freq); err != nil {
                log.Fatal(err)
            }
            time.Sleep(10 * time.Millisecond)
        }

        // Fade out
        for duty := gpio.DutyMax; duty >= 0; duty -= gpio.DutyMax / 100 {
            if err := pwm.PWM(duty, freq); err != nil {
                log.Fatal(err)
            }
            time.Sleep(10 * time.Millisecond)
        }

        // Turn off
        pwm.PWM(0, freq)
    } else {
        log.Fatal("Pin does not support PWM")
    }
}
```

---

## I2C

### Setup

```bash
# Enable I2C
sudo raspi-config
# Interface Options > I2C > Enable

# Add dependencies
go get periph.io/x/conn/v3/i2c
go get periph.io/x/conn/v3/i2c/i2creg
```

### I2C Communication

```go
package main

import (
    "fmt"
    "log"

    "periph.io/x/conn/v3/i2c"
    "periph.io/x/conn/v3/i2c/i2creg"
    "periph.io/x/host/v3"
)

func main() {
    if _, err := host.Init(); err != nil {
        log.Fatal(err)
    }

    // Open I2C bus
    bus, err := i2creg.Open("")
    if err != nil {
        log.Fatal(err)
    }
    defer bus.Close()

    // Device address
    const deviceAddr = 0x48

    // Create device
    dev := &i2c.Dev{Bus: bus, Addr: uint16(deviceAddr)}

    // Write byte
    if err := dev.Tx([]byte{0x01, 0x83}, nil); err != nil {
        log.Fatal(err)
    }

    // Read byte
    read := make([]byte, 1)
    if err := dev.Tx([]byte{0x00}, read); err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Read: 0x%02X\n", read[0])

    // Read block
    block := make([]byte, 2)
    if err := dev.Tx([]byte{0x00}, block); err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Block: %v\n", block)
}
```

### I2C Example: BMP280 Sensor

```go
package main

import (
    "encoding/binary"
    "fmt"
    "log"

    "periph.io/x/conn/v3/i2c"
    "periph.io/x/conn/v3/i2c/i2creg"
    "periph.io/x/host/v3"
)

type BMP280 struct {
    dev    *i2c.Dev
    digT1  uint16
    digT2  int16
    digT3  int16
}

func NewBMP280(bus i2c.Bus, addr uint16) (*BMP280, error) {
    dev := &i2c.Dev{Bus: bus, Addr: addr}
    
    bmp := &BMP280{dev: dev}
    
    // Read calibration data
    calib := make([]byte, 6)
    if err := dev.Tx([]byte{0x88}, calib); err != nil {
        return nil, err
    }
    
    bmp.digT1 = binary.LittleEndian.Uint16(calib[0:2])
    bmp.digT2 = int16(binary.LittleEndian.Uint16(calib[2:4]))
    bmp.digT3 = int16(binary.LittleEndian.Uint16(calib[4:6]))
    
    // Configure sensor
    if err := dev.Tx([]byte{0xF4, 0x27}, nil); err != nil {
        return nil, err
    }
    
    return bmp, nil
}

func (b *BMP280) ReadTemperature() (float64, error) {
    // Read raw temperature
    data := make([]byte, 3)
    if err := b.dev.Tx([]byte{0xFA}, data); err != nil {
        return 0, err
    }
    
    adcT := int32(data[0])<<12 | int32(data[1])<<4 | int32(data[2])>>4
    
    // Compensate
    var1 := (float64(adcT)/16384.0 - float64(b.digT1)/1024.0) * float64(b.digT2)
    var2 := ((float64(adcT)/131072.0 - float64(b.digT1)/8192.0) * 
             (float64(adcT)/131072.0 - float64(b.digT1)/8192.0)) * float64(b.digT3)
    
    tFine := var1 + var2
    temperature := tFine / 5120.0
    
    return temperature, nil
}

func main() {
    if _, err := host.Init(); err != nil {
        log.Fatal(err)
    }

    bus, err := i2creg.Open("")
    if err != nil {
        log.Fatal(err)
    }
    defer bus.Close()

    sensor, err := NewBMP280(bus, 0x76)
    if err != nil {
        log.Fatal(err)
    }

    temp, err := sensor.ReadTemperature()
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Temperature: %.2f°C\n", temp)
}
```

---

## SPI

### Setup

```bash
# Enable SPI
sudo raspi-config
# Interface Options > SPI > Enable

# Add dependencies
go get periph.io/x/conn/v3/spi
go get periph.io/x/conn/v3/spi/spireg
```

### SPI Communication

```go
package main

import (
    "fmt"
    "log"

    "periph.io/x/conn/v3/physic"
    "periph.io/x/conn/v3/spi"
    "periph.io/x/conn/v3/spi/spireg"
    "periph.io/x/host/v3"
)

func main() {
    if _, err := host.Init(); err != nil {
        log.Fatal(err)
    }

    // Open SPI port
    port, err := spireg.Open("")
    if err != nil {
        log.Fatal(err)
    }
    defer port.Close()

    // Configure SPI
    conn, err := port.Connect(1*physic.MegaHertz, spi.Mode0, 8)
    if err != nil {
        log.Fatal(err)
    }

    // Transfer data
    write := []byte{0x01, 0x02, 0x03}
    read := make([]byte, len(write))

    if err := conn.Tx(write, read); err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Sent: %v\n", write)
    fmt.Printf("Received: %v\n", read)
}
```

### SPI Example: MCP3008 ADC

```go
package main

import (
    "fmt"
    "log"
    "time"

    "periph.io/x/conn/v3/physic"
    "periph.io/x/conn/v3/spi"
    "periph.io/x/conn/v3/spi/spireg"
    "periph.io/x/host/v3"
)

type MCP3008 struct {
    conn spi.Conn
}

func NewMCP3008(port spi.Port) (*MCP3008, error) {
    conn, err := port.Connect(1350*physic.KiloHertz, spi.Mode0, 8)
    if err != nil {
        return nil, err
    }
    return &MCP3008{conn: conn}, nil
}

func (m *MCP3008) ReadChannel(channel int) (int, error) {
    if channel < 0 || channel > 7 {
        return 0, fmt.Errorf("invalid channel: %d", channel)
    }

    // Build command
    cmd := []byte{
        0x01,
        byte((8 + channel) << 4),
        0x00,
    }

    // Send command and read response
    reply := make([]byte, 3)
    if err := m.conn.Tx(cmd, reply); err != nil {
        return 0, err
    }

    // Extract 10-bit value
    value := int(reply[1]&0x03)<<8 | int(reply[2])

    return value, nil
}

func (m *MCP3008) ReadVoltage(channel int, vref float64) (float64, error) {
    value, err := m.ReadChannel(channel)
    if err != nil {
        return 0, err
    }

    voltage := float64(value) * vref / 1023.0
    return voltage, nil
}

func main() {
    if _, err := host.Init(); err != nil {
        log.Fatal(err)
    }

    port, err := spireg.Open("")
    if err != nil {
        log.Fatal(err)
    }
    defer port.Close()

    adc, err := NewMCP3008(port)
    if err != nil {
        log.Fatal(err)
    }

    for {
        value, err := adc.ReadChannel(0)
        if err != nil {
            log.Fatal(err)
        }

        voltage, err := adc.ReadVoltage(0, 3.3)
        if err != nil {
            log.Fatal(err)
        }

        fmt.Printf("Channel 0: %d (%.2fV)\n", value, voltage)
        time.Sleep(1 * time.Second)
    }
}
```

---

## Complete Example: Weather Station

```go
package main

import (
    "fmt"
    "log"
    "time"

    "periph.io/x/conn/v3/gpio"
    "periph.io/x/conn/v3/gpio/gpioreg"
    "periph.io/x/conn/v3/i2c"
    "periph.io/x/conn/v3/i2c/i2creg"
    "periph.io/x/host/v3"
)

type WeatherStation struct {
    led    gpio.PinOut
    button gpio.PinIn
    sensor *BMP280
}

func NewWeatherStation() (*WeatherStation, error) {
    if _, err := host.Init(); err != nil {
        return nil, err
    }

    // Setup LED
    led := gpioreg.ByName("GPIO18")
    led.Out(gpio.Low)

    // Setup button
    button := gpioreg.ByName("GPIO23")
    button.In(gpio.PullUp, gpio.FallingEdge)

    // Setup I2C sensor
    bus, err := i2creg.Open("")
    if err != nil {
        return nil, err
    }

    sensor, err := NewBMP280(bus, 0x76)
    if err != nil {
        return nil, err
    }

    return &WeatherStation{
        led:    led,
        button: button,
        sensor: sensor,
    }, nil
}

func (ws *WeatherStation) Run() {
    fmt.Println("Weather Station Running. Press button to read temperature.")

    for {
        // Wait for button press
        ws.button.WaitForEdge(-1)

        // Debounce
        time.Sleep(50 * time.Millisecond)

        if ws.button.Read() == gpio.Low {
            ws.led.Out(gpio.High)

            temp, err := ws.sensor.ReadTemperature()
            if err != nil {
                log.Println("Error reading temperature:", err)
            } else {
                fmt.Printf("Temperature: %.2f°C\n", temp)
            }

            time.Sleep(500 * time.Millisecond)
            ws.led.Out(gpio.Low)
        }
    }
}

func main() {
    station, err := NewWeatherStation()
    if err != nil {
        log.Fatal(err)
    }

    station.Run()
}
```

---

## Best Practices

1. **Always initialize host** - Call `host.Init()` first
2. **Check pin availability** - Verify pins exist before use
3. **Handle errors properly** - Don't ignore errors
4. **Close resources** - Use `defer` for cleanup
5. **Use appropriate voltage** - RPi GPIO is 3.3V
6. **Debounce inputs** - Add delays for button presses
7. **Use goroutines** - For concurrent GPIO operations

---