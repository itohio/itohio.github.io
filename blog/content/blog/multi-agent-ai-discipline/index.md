---
title: "Building Multi-Agent AI That Gets Better Over Time"
date: 2026-09-14
description: "Without structure, multi-agent AI systems degrade on the second iteration. With the right architecture — skill files, self-reflective agents, production signal feedback — they improve instead."
draft: false
toc: true
categories:
  - Software Engineering
  - AI
tags:
  - ai
  - machine-learning
  - agentic-ai
  - multi-agent
  - software-engineering
  - llm
  - orchestration
  - prompt-engineering
keywords: ["multi-agent AI systems", "AI orchestration architecture", "self-reflective AI agents", "AI discipline production", "vibe coding problems AI", "multi-agent LLM architecture", "AI systems that improve", "skill files AI", "AI context drift"]
---

I've watched developers rebuild the same AI agent three times in a sprint. Different model, different prompt, same degrading output by iteration two. The code works. Once.

That's vibe coding. It looks productive. The demo is great. By the fourth change request, nobody knows why anything works and the context window is a garbage heap of contradictions.

With AI systems specifically, this failure mode is invisible until it isn't. The model doesn't tell you when its context has drifted. It doesn't flag when a previous decision has been silently overridden. It just produces output that was technically coherent at turn 3 and confidently wrong at turn 23.

## What AI does to itself without structure

Single-turn AI is well understood. Multi-agent systems, run without discipline, do something more interesting: they accumulate contradictions. An agent in turn 8 overrides a constraint set in turn 2 without noticing. A specialized subagent optimizes for its own goal and produces output that undermines the coordinator above it. Goals drift. Rules get forgotten. The system "works" in the sense that it produces output, and the output looks reasonable until you look closely.

The real problem is that this looks like success. Especially on the first iteration.

Without observable state and explicit, persistent constraints, an AI orchestration system is a condom between production and an LLM. It's a filter. A very expensive one.

## The architecture that makes improvement possible

The insight I arrived at was embarrassingly mundane: the AI needs to be able to read its own past decisions.

Not as conversation history. As structured, durable knowledge. Markdown files with rules. Domain knowledge in readable form. Decisions logged with the reasoning that produced them. A record of what worked, what didn't, and why — maintained by the system itself.

This isn't prompt engineering. Prompt engineering is rewriting the question. What I'm describing is building the AI a working memory that survives beyond a context window.

In practice, this looks like single-responsibility skill files: one file per concern, each containing rules, constraints, and accumulated learning. The AI reads them, executes against them, and updates them based on outcomes. Not every run. When something breaks, or when the output diverges from what it should have been. The file becomes a living record.

Multi-agent hierarchies built on this foundation behave differently than flat orchestration graphs. A specialized LLM with a well-maintained domain knowledge file gets better at its job over repeated runs. The coordinator above it can rely on consistent behavior and focus on coordination, not error-correction. The goals and constraints survive past 20 turns because they're not in the context window. They're in the files.

## Closing the feedback loop with production signals

The second piece is observability. Not for humans — for the AI manager.

Sentry tells you what broke. Grafana tells you how the system is behaving. Logs tell you what actually happened. Mixpanel tells you what users are doing. Intercom or Canny tells you what they're complaining about and asking for.

These aren't monitoring dashboards anymore. They're inputs. The AI manager reads production signals, identifies patterns, and decides whether current behavior is correct relative to the goals. When it isn't, it updates the relevant skill file. The next run is better informed than the last.

This is the part that looks like magic until you understand the mechanism. A system that reads its own failure signals and improves its rules is not particularly sophisticated. It's just disciplined.

## A different model of what an AI manager does

Most AI orchestration I've seen is defensive. The AI manager sits between a user request and the underlying LLMs and tries not to break anything. It manages context, it routes, it retries. It's a moderately intelligent proxy.

What I'm describing is different. The AI manager as actual orchestrator: reading signals from across the production stack, holding constraints that survive beyond any single session, coordinating specialized LLMs that each maintain their own domain expertise, driving improvement instead of just maintaining coherence. Hierarchical. Self-reflective. The goals don't drift because they're stored, not inferred from whatever happens to be in the context window this turn.

This isn't the agentic graph model. It's not about the number of agents or the complexity of the routing. It's about whether the system accumulates knowledge or throws it away at the end of every conversation.

## Where it started

A folder of markdown files. Company rules. Project notes. Client context. A few prompts. Research results.

No architecture decision. No orchestration framework. Just the habit of writing things down in a way the AI could read back. That habit, scaled and formalized — structured skill files, self-evaluation loops, production signal ingestion — is the foundation of everything above.

The architecture wasn't designed. It emerged from doing the obvious thing consistently.

A few steps from truly autonomous AI systems. Currently at: a folder of markdown files and enough discipline to not let the AI lie to itself.
