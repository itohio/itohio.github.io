---
title: "Mermaid Sankey Diagrams"
date: 2024-12-12T23:30:00Z
draft: false
description: "Create Sankey diagrams for flow visualization with Mermaid"
tags: ["mermaid", "sankey", "flow", "data-visualization", "diagram", "diagrams"]
category: "diagrams"
---

Sankey diagrams show flow between nodes, with link width proportional to flow quantity. Perfect for visualizing energy flows, data transfers, and resource movements.

## Use Case

Use Sankey diagrams when you need to:
- Visualize flow quantities
- Show energy or resource flows
- Display data transfers
- Map process flows
- Visualize conversions

## Code

````markdown
```mermaid
sankey-beta
    Source A,Target A,100
    Source A,Target B,50
    Source B,Target A,30
    Source B,Target B,70
```
````

**Result:**

```mermaid
sankey-beta
    Source A,Target A,100
    Source A,Target B,50
    Source B,Target A,30
    Source B,Target B,70
```

## Examples

### Example 1: Energy Flow

````markdown
```mermaid
sankey-beta
    Solar,Grid,80
    Solar,Battery,20
    Wind,Grid,60
    Wind,Battery,40
    Grid,Home,100
    Battery,Home,30
```
````

**Result:**

```mermaid
sankey-beta
    Solar,Grid,80
    Wind,Grid,60
    Grid,Home,100
    Battery,Home,30
```

### Example 2: Data Pipeline

````markdown
```mermaid
sankey-beta
    Source A,Processing,1000
    Source B,Processing,500
    Processing,Database,1200
    Processing,Analytics,300
    Database,Archive,800
    Analytics,Reports,300
```
````

**Result:**

```mermaid
sankey-beta
    Source A,Processing,1000
    Source B,Processing,500
    Processing,Database,1200
    Processing,Analytics,300
    Database,Archive,800
    Analytics,Reports,300
```

### Example 3: Large Energy System Sankey with Config

You can also provide a diagram-specific config in the frontmatter to control Sankey behaviour, for example to hide numeric values on the links:

````markdown
```mermaid
---
config:
  sankey:
    showValues: false
---
sankey-beta

Agricultural 'waste',Bio-conversion,124.729
Bio-conversion,Liquid,0.597
Bio-conversion,Losses,26.862
Bio-conversion,Solid,280.322
Bio-conversion,Gas,81.144
Biofuel imports,Liquid,35
Biomass imports,Solid,35
Coal imports,Coal,11.606
Coal reserves,Coal,63.965
Coal,Solid,75.571
District heating,Industry,10.639
District heating,Heating and cooling - commercial,22.505
District heating,Heating and cooling - homes,46.184
Electricity grid,Over generation / exports,104.453
Electricity grid,Heating and cooling - homes,113.726
Electricity grid,H2 conversion,27.14
Electricity grid,Industry,342.165
Electricity grid,Road transport,37.797
Electricity grid,Agriculture,4.412
Electricity grid,Heating and cooling - commercial,40.858
Electricity grid,Losses,56.691
Electricity grid,Rail transport,7.863
Electricity grid,Lighting & appliances - commercial,90.008
Electricity grid,Lighting & appliances - homes,93.494
Gas imports,NGas,40.719
Gas reserves,NGas,82.233
Gas,Heating and cooling - commercial,0.129
Gas,Losses,1.401
Gas,Thermal generation,151.891
Gas,Agriculture,2.096
Gas,Industry,48.58
Geothermal,Electricity grid,7.013
H2 conversion,H2,20.897
H2 conversion,Losses,6.242
H2,Road transport,20.897
Hydro,Electricity grid,6.995
Liquid,Industry,121.066
Liquid,International shipping,128.69
Liquid,Road transport,135.835
Liquid,Domestic aviation,14.458
Liquid,International aviation,206.267
Liquid,Agriculture,3.64
Liquid,National navigation,33.218
Liquid,Rail transport,4.413
Marine algae,Bio-conversion,4.375
NGas,Gas,122.952
Nuclear,Thermal generation,839.978
Oil imports,Oil,504.287
Oil reserves,Oil,107.703
Oil,Liquid,611.99
Other waste,Solid,56.587
Other waste,Bio-conversion,77.81
Pumped heat,Heating and cooling - homes,193.026
Pumped heat,Heating and cooling - commercial,70.672
Solar PV,Electricity grid,59.901
Solar Thermal,Heating and cooling - homes,19.263
Solar,Solar Thermal,19.263
Solar,Solar PV,59.901
Solid,Agriculture,0.882
Solid,Thermal generation,400.12
Solid,Industry,46.477
Thermal generation,Electricity grid,525.531
Thermal generation,Losses,787.129
Thermal generation,District heating,79.329
Tidal,Electricity grid,9.452
UK land based bioenergy,Bio-conversion,182.01
Wave,Electricity grid,19.013
Wind,Electricity grid,289.366
```
````

**Result:**

```mermaid
---
config:
  sankey:
    showValues: false
---
sankey-beta

Agricultural 'waste',Bio-conversion,124.729
Bio-conversion,Liquid,0.597
Bio-conversion,Losses,26.862
Bio-conversion,Solid,280.322
Bio-conversion,Gas,81.144
Biofuel imports,Liquid,35
Biomass imports,Solid,35
Coal imports,Coal,11.606
Coal reserves,Coal,63.965
Coal,Solid,75.571
District heating,Industry,10.639
District heating,Heating and cooling - commercial,22.505
District heating,Heating and cooling - homes,46.184
Electricity grid,Over generation / exports,104.453
Electricity grid,Heating and cooling - homes,113.726
Electricity grid,H2 conversion,27.14
Electricity grid,Industry,342.165
Electricity grid,Road transport,37.797
Electricity grid,Agriculture,4.412
Electricity grid,Heating and cooling - commercial,40.858
Electricity grid,Losses,56.691
Electricity grid,Rail transport,7.863
Electricity grid,Lighting & appliances - commercial,90.008
Electricity grid,Lighting & appliances - homes,93.494
Gas imports,NGas,40.719
Gas reserves,NGas,82.233
Gas,Heating and cooling - commercial,0.129
Gas,Losses,1.401
Gas,Thermal generation,151.891
Gas,Agriculture,2.096
Gas,Industry,48.58
Geothermal,Electricity grid,7.013
H2 conversion,H2,20.897
H2 conversion,Losses,6.242
H2,Road transport,20.897
Hydro,Electricity grid,6.995
Liquid,Industry,121.066
Liquid,International shipping,128.69
Liquid,Road transport,135.835
Liquid,Domestic aviation,14.458
Liquid,International aviation,206.267
Liquid,Agriculture,3.64
Liquid,National navigation,33.218
Liquid,Rail transport,4.413
Marine algae,Bio-conversion,4.375
NGas,Gas,122.952
Nuclear,Thermal generation,839.978
Oil imports,Oil,504.287
Oil reserves,Oil,107.703
Oil,Liquid,611.99
Other waste,Solid,56.587
Other waste,Bio-conversion,77.81
Pumped heat,Heating and cooling - homes,193.026
Pumped heat,Heating and cooling - commercial,70.672
Solar PV,Electricity grid,59.901
Solar Thermal,Heating and cooling - homes,19.263
Solar,Solar Thermal,19.263
Solar,Solar PV,59.901
Solid,Agriculture,0.882
Solid,Thermal generation,400.12
Solid,Industry,46.477
Thermal generation,Electricity grid,525.531
Thermal generation,Losses,787.129
Thermal generation,District heating,79.329
Tidal,Electricity grid,9.452
UK land based bioenergy,Bio-conversion,182.01
Wave,Electricity grid,19.013
Wind,Electricity grid,289.366
```

## Notes

- Format: `Source,Target,Value`
- Link width proportional to value
- Nodes auto-created from connections
- Values should be positive numbers
- You can control Sankey behaviour via `config.sankey` frontmatter (for example, `showValues: false` to hide numeric labels)

## Gotchas/Warnings

- ⚠️ **Syntax**: Use `sankey-beta` (experimental)
- ⚠️ **Format**: Must be `Source,Target,Value` (comma-separated)
- ⚠️ **Values**: Should be positive numbers
- ⚠️ **Config**: Frontmatter `config` support depends on your Mermaid integration
- ⚠️ **Beta**: Feature is in beta - syntax may change

