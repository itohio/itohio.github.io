---
title: "Rate Profile Selection & Persistent Stats"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "osd", "persistent-stats", "switch"]
---

Switching rate profiles mid-session via a transmitter switch and saving your personal-best stats to the OSD are both useful features — but they interact in a surprising way.

---

## Persistent Stats

Betaflight can track flight statistics across power cycles and display them on the OSD:

- Max altitude, max speed, max G-force, max current, max throttle
- Total armed time, total flights

**Why they're useful:** persistent stats let you see your personal bests in the OSD while flying — useful for reviewing footage and knowing whether a flight was your fastest/hardest. They survive disarm and power-off so you can review them between sessions.

### Enabling Persistent Stats

In Betaflight Configurator → **OSD** tab → scroll to the Stats section, enable the stats you want. They appear on the post-flight stats screen after disarm.

Via CLI:
```
set stats = ON
set stats_total_flights = ON
set stats_total_time = ON
save
```

---

## Rate Profile Selection via Switch

Betaflight supports up to 6 rate profiles. You can switch between them in flight using a transmitter channel mapped to a mode condition.

In Configurator → **Modes** tab, add a `RATES PROFILE 1`, `RATES PROFILE 2`, `RATES PROFILE 3` condition on your chosen AUX channel ranges. Each profile stores its own RC Rate / Super Rate / Expo values.

Typical setup: 3-position switch → one rate profile per position (e.g. 533 → 633 → 733).

---

## The Conflict: Why Stats Don't Save on Disarm

**When rate profile switching is enabled, persistent stats will not save on disarm.**

Here's why: switching a rate profile modifies the flight controller's configuration (it's a config change, not just a runtime value). For that change to survive a reboot, Betaflight would need to write the config to flash. Betaflight does **not** do an automatic config save on disarm — by design, to avoid flash wear and accidental overwrites mid-session.

Because the same save mechanism is shared between "which rate profile is active" and "current stat counters," and because no auto-save fires on disarm, your stat counters reset on the next power cycle.

---

## Workarounds

### Option 1 — Save config manually after landing

Run the `save` command from Betaflight Configurator or the OSD stick command immediately after landing, before removing the battery. This writes the current config (including which rate profile is active and the latest stat counters) to flash.

```
# In CLI after each session
save
```

### Option 2 — Disable rate profile switching

If you don't need in-flight rate profile changes, remove the `RATES PROFILE` mode conditions from the Modes tab. With profile switching disabled, the active profile never changes the config at runtime, and stats save normally on disarm.

### Option 3 — Use a single profile and tune rates per style

Instead of switching profiles in flight, keep one profile set to the rates you fly most often. If you want to test different rates, change the profile on the bench between sessions and save manually.

---

## Summary

| Feature combination             | Stats persist on disarm? |
|---------------------------------|--------------------------|
| Persistent stats, no profile switch | ✅ Yes               |
| Persistent stats + profile switch   | ❌ No (save manually)|
| Profile switch, stats disabled      | N/A                  |
