---
title: "ELRS Configuration — Bind Phrase, FCC vs CECC"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "elrs", "expressLRS", "radio", "link", "bind", "fcc", "lbt"]
---

ExpressLRS (ELRS) is an open-source RC link protocol known for ultra-low latency and long range. Two critical settings must match between transmitter module (TX) and receiver (RX): **regulatory domain** and **bind phrase**.

---

## Regulatory Domain: FCC vs LBT (ELRS/CE)

ELRS firmware is compiled for a specific radio regulatory domain that determines how it hops frequencies and what transmission behavior is allowed.

| Domain | Used in                          | Behavior                                         |
|--------|----------------------------------|--------------------------------------------------|
| **FCC**| USA, Canada, most of the Americas| Full frequency hopping, no listen-before-talk    |
| **LBT**| EU, UK, Australia, Lithuania 🇱🇹 | Listen-Before-Talk — checks channel before TX    |
| **CE** | Synonym for LBT in many contexts | Same as LBT; ELRS uses "LBT" in firmware         |

**LBT (Listen Before Talk)** adds a brief channel check before each transmission to comply with EU spectrum regulations. This adds a small amount of latency compared to FCC mode.

### Why They Must Match

TX and RX must be flashed with the **same regulatory domain**. A FCC TX will not bind to an LBT RX, and vice versa. They operate on different channel sequences and timing assumptions.

Mismatched domains = **no link**, even at close range.

If you import a module from the USA and use it in the EU, reflash both TX and RX to LBT firmware.

---

## Checking Your Domain

In ELRS Configurator (during a flash or update):
- Look for the `Regulatory Domain` dropdown — it will show `FCC_915_US`, `EU_868`, `ISM_2400` etc.
- 2.4 GHz ELRS (`ISM_2400`) is generally **not subject to LBT** in most regions and is the same firmware globally — making it simpler to use internationally.

For 900 MHz ELRS, domain matters most. For 2.4 GHz, you can mostly ignore it (same firmware everywhere), but verify with your specific hardware's docs.

---

## Bind Phrase

The bind phrase is a user-chosen passphrase compiled into both TX and RX firmware. It replaces traditional button-press binding.

- Both TX module and RX **must be flashed with the same bind phrase**.
- The phrase is set at flash time, not at runtime — you cannot change it without reflashing.
- There is no "default" phrase — every compiled firmware has one embedded.

**Setup:**
1. Open [ELRS Configurator](https://github.com/ExpressLRS/ExpressLRS-Configurator)
2. Select your hardware target
3. Enter your chosen bind phrase in the `Binding Phrase` field
4. Select regulatory domain
5. Flash both TX module and all receivers with **identical settings**

```
# Example bind phrase (use something unique to you)
MY_UNIQUE_PHRASE_2024
```

The phrase is hashed — it doesn't need to be secret, just consistent.

---

## Packet Rate & Telemetry

After binding, set your preferred packet rate and telemetry ratio on the TX module LUA script (run from your radio's tools menu).

| Packet rate | Latency  | Range trade-off           |
|-------------|----------|---------------------------|
| 50 Hz       | ~20 ms   | Best range                |
| 150 Hz      | ~6.7 ms  | Good balance              |
| 250 Hz      | ~4 ms    | Low latency, shorter range|
| 500 Hz      | ~2 ms    | Lowest latency            |
| F1000 / D500| ~1 ms    | Race mode, very short range|

For cinematic and general flying: **150 Hz** is a solid default. For racing: **250 Hz or higher**.

Telemetry ratio (`1:n`) controls how often the RX sends data back. Higher ratio = less TX time for telemetry = better link budget on RX → TX path.

---

## Common ELRS Issues

| Symptom                   | Likely cause                                  |
|---------------------------|-----------------------------------------------|
| No bind after flashing     | Different bind phrase or regulatory domain    |
| Intermittent link drops    | Antenna orientation; TX and RX domains differ |
| Telemetry not working      | Telemetry ratio set to `Off`; wrong serial port |
| RSSI shows but no control  | UART not configured in Betaflight; CRSF not set as receiver type |

In Betaflight CLI:
```
set serialrx_provider = CRSF
set serialrx_inverted = OFF
save
```
Assign the UART connected to the ELRS RX as `Serial RX` in the Ports tab.
