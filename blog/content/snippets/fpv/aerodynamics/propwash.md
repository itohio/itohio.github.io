---
title: "Propwash — What It Is and Why It Happens"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "aerodynamics", "propwash", "pid", "tuning", "airflow", "motors"]
---

Propwash is the turbulence a multirotor flies through when it descends into its own rotor downwash. It is the dominant cause of the characteristic oscillation felt during punch-outs, split-S exits, and any recovery from a dive. Understanding the airflow geometry makes the tuning response obvious.

---

## What Is Rotor Downwash?

Each spinning prop accelerates air downward through a pressure differential — high pressure above the disk, low below. The resulting column of accelerated air is called **downwash**. In a hover, this column extends several prop diameters below the craft and disperses gradually.

<div style="display:flex;justify-content:center;margin:2rem 0;">
<canvas id="downwash-canvas" width="520" height="360" style="border-radius:8px;background:#111;display:block;"></canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.4/p5.min.js"></script>
<script>
(function(){
  // Hover / downwash visualization
  var sketch = function(p){
    var particles = [];
    var W = 520, H = 360;
    var propY = 70, propHalf = 90;
    var NUM = 80;

    p.setup = function(){
      p.createCanvas(W, H);
      for(var i=0;i<NUM;i++){
        particles.push(newParticle(p));
      }
    };

    function newParticle(p){
      var side = p.random()<0.5 ? -1 : 1;
      var r = p.random(8, propHalf);
      return {
        x: W/2 + side * r,
        y: propY + p.random(-20, 20),
        vy: p.random(0.8, 2.2),
        vx: side * p.random(0.0, 0.5),
        life: p.random(0.3, 1.0),
        age: 0,
        maxAge: p.random(80, 160)
      };
    }

    p.draw = function(){
      p.background(17, 17, 17, 60);

      // Quad body
      p.fill(50, 50, 60);
      p.noStroke();
      p.rectMode(p.CENTER);
      p.rect(W/2, propY - 18, 60, 14, 4);
      // Arms
      p.stroke(60,60,70); p.strokeWeight(6);
      p.line(W/2 - 55, propY - 14, W/2 - 105, propY - 40);
      p.line(W/2 + 55, propY - 14, W/2 + 105, propY - 40);
      p.line(W/2 - 55, propY - 22, W/2 - 105, propY + 2);
      p.line(W/2 + 55, propY - 22, W/2 + 105, propY + 2);
      // Motors
      p.noStroke();
      p.fill(80,80,90);
      p.ellipse(W/2 - 105, propY - 40, 18, 18);
      p.ellipse(W/2 + 105, propY - 40, 18, 18);
      p.ellipse(W/2 - 105, propY + 2, 18, 18);
      p.ellipse(W/2 + 105, propY + 2, 18, 18);
      // Props
      p.stroke(100,180,255); p.strokeWeight(3); p.noFill();
      var t = p.frameCount * 0.12;
      for(var m=0;m<4;m++){
        var mx = [W/2-105, W/2+105, W/2-105, W/2+105][m];
        var my = [propY-40, propY-40, propY+2, propY+2][m];
        var dir = [1,-1,-1,1][m];
        p.push();
        p.translate(mx, my);
        p.rotate(t * dir);
        p.line(-22, 0, 22, 0);
        p.pop();
      }

      // Downwash column boundary (faint)
      p.stroke(80,140,255,25); p.strokeWeight(1); p.noFill();
      p.beginShape();
      p.vertex(W/2 - propHalf, propY);
      p.vertex(W/2 - propHalf * 0.55, H - 30);
      p.endShape();
      p.beginShape();
      p.vertex(W/2 + propHalf, propY);
      p.vertex(W/2 + propHalf * 0.55, H - 30);
      p.endShape();

      // Particles
      for(var i=particles.length-1;i>=0;i--){
        var pt = particles[i];
        pt.x += pt.vx;
        pt.y += pt.vy;
        pt.age++;

        // Expand outward as they fall (spreading wake)
        var spread = (pt.y - propY) / H * 0.6;
        pt.vx += (pt.x < W/2 ? -1 : 1) * spread * 0.02;

        var frac = pt.age / pt.maxAge;
        var alpha = 200 * (1 - frac);
        var speed = p.sqrt(pt.vx*pt.vx + pt.vy*pt.vy);
        var g = p.map(speed, 0.8, 2.2, 80, 220);
        p.noStroke();
        p.fill(60, g, 255, alpha);
        p.ellipse(pt.x, pt.y, 4, 4);

        if(pt.age > pt.maxAge || pt.y > H) {
          particles[i] = newParticle(p);
        }
      }

      // Label
      p.fill(180); p.noStroke(); p.textSize(11); p.textAlign(p.CENTER);
      p.text("Downwash column (hover)", W/2, H - 10);
    };
  };
  new p5(sketch, document.getElementById('downwash-canvas').parentElement).canvas = document.getElementById('downwash-canvas');
  // p5 auto-creates canvas; let's target it properly
})();
</script>

---

## How Propwash Oscillation Happens

During normal forward flight the quad is flying into clean air. When it pitches level after a dive — or punches up into a descent — it flies back into the disturbed air column it just came through. Each prop then ingests turbulent, non-uniform inflow instead of smooth laminar air.

```mermaid
flowchart TD
    A[Quad dives<br/>descends fast] --> B[Downwash column<br/>moves relative to craft]
    B --> C[Craft levels out<br/>or climbs through own wake]
    C --> D[Props ingest turbulent<br/>non-uniform inflow]
    D --> E[Asymmetric thrust<br/>per-prop, per-blade]
    E --> F[Rapid attitude disturbance<br/>before FC can correct]
    F --> G{PIDs respond}
    G -->|D term too low| H[Oscillation — slow to damp]
    G -->|D term well-tuned| I[Quick correction<br/>clean recovery]
    G -->|D term too high| J[Motor noise amplified<br/>overheating risk]
```

The disturbance is primarily felt as a pitch/roll wobble on exit from dives and during throttle-down recovery. It is **not** a PID instability — it is an external aerodynamic input that the PID loop has to reject. Tuning helps, but it cannot eliminate the physics.

---

## The Airflow Geometry — Live

<div style="display:flex;justify-content:center;margin:2rem 0;">
<canvas id="propwash-canvas" width="520" height="400" style="border-radius:8px;background:#111;display:block;"></canvas>
</div>
<script>
(function(){
  var sketch2 = function(p){
    var W=520, H=400;
    var particles=[], groundParticles=[];
    var NUM=90, GNUM=50;
    var quadY=60, propHalf=88;
    var time=0;
    // Phase: 0=hover, 1=diving, 2=recovering
    var phase=0, phaseTimer=0;
    var phaseDuration=[200,120,200];
    var phaseLabel=["Hovering — clean downwash","Diving — quad outruns its wake","Recovery — flying into own turbulence ⚡"];
    var quadVY=0, quadActualY=quadY;

    p.setup=function(){
      p.createCanvas(W,H);
      p.textFont('monospace');
      for(var i=0;i<NUM;i++) particles.push(makeP(p,true));
      for(var i=0;i<GNUM;i++) groundParticles.push(makeGP(p,true));
    };

    function makeP(p,init){
      var side=p.random()<0.5?-1:1;
      var r=p.random(5,propHalf-5);
      return {
        x:W/2+side*r,
        y:(init?p.random(quadY,H-60):quadActualY+p.random(-10,10)),
        vy:p.random(1.0,2.8)+(phase===1?2.5:0),
        vx:side*p.random(0,0.4),
        age:init?p.random(0,120):0,
        maxAge:p.random(90,180),
        turb:0
      };
    }

    function makeGP(p,init){
      return {
        x:W/2+p.random(-propHalf*0.7,propHalf*0.7),
        y:H-40+p.random(-8,8),
        vx:p.random(-1.2,1.2),
        vy:p.random(-1.5,-0.3),
        age:init?p.random(0,60):0,
        maxAge:p.random(40,90)
      };
    }

    p.draw=function(){
      p.background(17,17,17,55);
      time++;
      phaseTimer++;

      // Phase advance
      if(phaseTimer>phaseDuration[phase]){
        phaseTimer=0;
        phase=(phase+1)%3;
      }

      // Quad motion
      if(phase===1){ quadVY=p.lerp(quadVY,4,0.08); }
      else if(phase===2){ quadVY=p.lerp(quadVY,-2,0.06); }
      else { quadVY=p.lerp(quadVY,0,0.06); }
      quadActualY=p.constrain(quadActualY+quadVY,quadY,H*0.55);

      // Ground
      p.stroke(60,60,60); p.strokeWeight(1);
      p.line(30,H-30,W-30,H-30);

      // Downwash wake trail (turbulence zone during recovery)
      if(phase===2){
        var turbAlpha=p.map(phaseTimer,0,40,0,60);
        p.noStroke(); p.fill(255,120,0,turbAlpha);
        p.ellipse(W/2, H*0.3+20, propHalf*1.5, H*0.4);
      }

      // Ground bounce particles
      for(var i=groundParticles.length-1;i>=0;i--){
        var gp=groundParticles[i];
        gp.x+=gp.vx; gp.y+=gp.vy; gp.age++;
        var f2=gp.age/gp.maxAge;
        p.noStroke(); p.fill(80,180,255,150*(1-f2));
        p.ellipse(gp.x,gp.y,3,3);
        if(gp.age>gp.maxAge) groundParticles[i]=makeGP(p,false);
      }

      // Flow particles
      for(var i=particles.length-1;i>=0;i--){
        var pt=particles[i];
        // Turbulence injection during recovery
        if(phase===2 && pt.y > H*0.25 && pt.y < H*0.65){
          pt.turb=p.lerp(pt.turb, p.random(-1.2,1.2), 0.15);
        } else {
          pt.turb=p.lerp(pt.turb,0,0.1);
        }
        pt.vx+=pt.turb*0.04;
        var spread=(pt.y-quadActualY)/H*0.5;
        pt.vx+=(pt.x<W/2?-1:1)*spread*0.015;
        pt.x+=pt.vx; pt.y+=pt.vy+(phase===1?2:0);
        pt.age++;

        var frac=pt.age/pt.maxAge;
        var alpha=200*(1-frac);
        var turbMag=p.abs(pt.turb);
        var r2=p.map(turbMag,0,1.2,60,255);
        var g2=p.map(turbMag,0,1.2,200,100);
        var b2=p.map(turbMag,0,1.2,255,60);
        p.noStroke(); p.fill(r2,g2,b2,alpha);
        p.ellipse(pt.x,pt.y,4,4);
        if(pt.age>pt.maxAge||pt.y>H-28) particles[i]=makeP(p,false);
      }

      // Draw quad
      var qy=quadActualY;
      p.fill(50,50,60); p.noStroke();
      p.rectMode(p.CENTER);
      p.rect(W/2,qy-18,60,14,4);
      p.stroke(60,60,70); p.strokeWeight(6);
      p.line(W/2-55,qy-14,W/2-105,qy-42);
      p.line(W/2+55,qy-14,W/2+105,qy-42);
      p.line(W/2-55,qy-22,W/2-105,qy-2);
      p.line(W/2+55,qy-22,W/2+105,qy-2);
      p.noStroke(); p.fill(80,80,90);
      p.ellipse(W/2-105,qy-42,18,18);
      p.ellipse(W/2+105,qy-42,18,18);
      p.ellipse(W/2-105,qy-2,18,18);
      p.ellipse(W/2+105,qy-2,18,18);
      p.stroke(100,180,255); p.strokeWeight(3); p.noFill();
      var t2=time*0.14;
      var dirs=[1,-1,-1,1];
      var mxs=[W/2-105,W/2+105,W/2-105,W/2+105];
      var mys=[qy-42,qy-42,qy-2,qy-2];
      for(var m=0;m<4;m++){
        p.push(); p.translate(mxs[m],mys[m]); p.rotate(t2*dirs[m]);
        p.line(-22,0,22,0); p.pop();
      }

      // Phase label
      var col=phase===2?p.color(255,140,40):p.color(120,200,255);
      p.fill(col); p.noStroke(); p.textSize(12); p.textAlign(p.CENTER);
      p.text(phaseLabel[phase], W/2, H-8);
      p.fill(80); p.textSize(10);
      p.text("blue = laminar  |  orange = turbulent", W/2, H+8-2);
    };
  };
  new p5(sketch2, 'propwash-canvas-host');
})();
</script>
<div id="propwash-canvas-host" style="display:none"></div>

**Orange = turbulent inflow.** During recovery the quad descends into the disturbed column it just pushed downward. Each blade encounters varying angle of attack across the disk, producing asymmetric thrust.

---

## Why Ground Effect Adds to It

Close to the ground (within ~1 prop diameter altitude), the downwash cannot fully develop — it spreads radially outward along the surface and wraps back up, re-entering the rotor disk from outside. This **ground recirculation** reduces effective thrust and adds another turbulent input. Combined with prop wash during low-altitude descents, this is why slow hover-in-ground-effect landings can feel mushy.

---

## What Tuning Can and Can't Fix

| Symptom | Tuning fix | Limit |
|---------|-----------|-------|
| Mild oscillation on dive exit, damps in 1–2 cycles | Increase D (Roll/Pitch) 5–10% | Fully fixable |
| Wobble on every throttle-down | Increase D, verify RPM filter | Largely fixable |
| Violent oscillation on aggressive split-S | D + reduce P slightly, check filtering | Partially — extreme moves always have propwash |
| Oscillation hot motors | D is too high — back off | Don't chase propwash with excessive D |
| Still wobbling after D is at thermal limit | Accept it — aerodynamics win | Not a tuning problem |

**The goal is not to eliminate propwash — it is to reject it quickly without overheating motors.** An aggressive freestyle quad will always have some propwash. A well-tuned one damps it within one to two oscillation cycles.

---

## Related

- [PID Basics](../../tuning/pid-basics/)
- [BBL-Based PID Tuning Protocol](../../tuning/bbl-pid-tuning-protocol/)
- [Blackbox Logging](../../tuning/blackbox-logging/)
