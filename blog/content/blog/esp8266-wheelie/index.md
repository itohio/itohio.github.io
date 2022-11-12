---
title: "ESP8266 Wheelie"
subtitle: "Using Mouse Sensor as a camera with Web interface"
author: admin
date: 2018-07-02T01:01:28+02:00
tags: ["arduino","esp8266", "Differential Drive", "Robots"]
categories: ["robots"]
thumbnail: https://github.com/foxis/ESPWheelie/raw/master/schematics/robot.jpg
---


![Wheelie finished](https://github.com/foxis/ESPWheelie/raw/master/schematics/robot-finished.jpg)

As I've mentioned in the last blog post, I was trying to build a four wheel drive robot using arduino that utilizes two A3080 mouse sensors for odometry. Sadly that project never got finished and I disassembled the robot and built a balancing robot instead. However later I decided to not abandon the four wheel drive robot and ordered several ADNS3080 sensors with proper optics, obtained an electric wheelie toy, which lead to the birth of this WiFi controlled toy car.

The source code for this project can be found [here](https://github.com/foxis/ESPWheelie).

# How it works

Chinese HBridge driver and ADNS3080 mouse sensor module are connected to Wemos D1 mini which serves a simple web page that allows to control the robot using touch gestures. Image retrieved using ADNS3080 is transferred to the web page using WebSockets as well as commands sent to the robot.

![Web UI](https://github.com/foxis/ESPWheelie/raw/master/schematics/Capture.PNG)

The robot is powered using two LiPo 1200mAh batteries forming a 2S1P battery. A potentiometer connected to GND and Battery Plus scales battery voltage which is fed to Analog pin of the Wemos board and subsequently is displayed on the web page.

![Wheelie guts](https://github.com/foxis/ESPWheelie/raw/master/schematics/robot.jpg)

Mechanically the robot consists of four wheels with each sides spinning independently. Interestingly enough, this toy has two wheels of the same side connected to one motor. So that all two pairs of wheels rotate independently like in a tank.

After finishing this project I decided to refactor differential drive and ADNS3080 sensor code into a separate library, that I call [EasyLocomotion](https://github.com/itohi/EasyLocomotion). This little library has different driving algorithms for wheeled as well as legged robots and is still in development.