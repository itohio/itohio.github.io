---
title: "Research Interview Questions - Hard"
date: 2025-12-13
tags: ["research", "interview", "hard", "advanced-methods"]
---

Hard-level research interview questions covering advanced methodologies and complex analysis.

## Q1: Explain Bayesian vs. Frequentist approaches to statistics.

**Answer**:

```mermaid
graph TB
    subgraph Frequentist["Frequentist Approach"]
        A1[Fixed Parameters] --> B1[Probability of Data<br/>given Parameters]
        B1 --> C1[P-values<br/>Confidence Intervals]
        C1 --> D1[Long-run frequency<br/>interpretation]
        style A1 fill:#87CEEB
    end
    
    subgraph Bayesian["Bayesian Approach"]
        A2[Parameters are<br/>Distributions] --> B2[Probability of Parameters<br/>given Data]
        B2 --> C2[Posterior Distribution<br/>Credible Intervals]
        C2 --> D2[Degree of belief<br/>interpretation]
        style A2 fill:#90EE90
    end
```

**Bayes' Theorem**:
$$P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)}$$

```mermaid
graph LR
    A[Prior<br/>P θ] --> B[Likelihood<br/>P D|θ]
    B --> C[Posterior<br/>P θ|D]
    
    D[Data] --> B
    
    style A fill:#FFE4B5
    style B fill:#87CEEB
    style C fill:#90EE90
```

**When to Use**:
- **Frequentist**: Large samples, no prior knowledge
- **Bayesian**: Small samples, incorporate prior knowledge, sequential updating

---

## Q2: Design a randomized controlled trial with complex interventions.

**Answer**:

```mermaid
graph TB
    A[Complex RCT<br/>Design] --> B[Cluster<br/>Randomization]
    A --> C[Stepped-Wedge<br/>Design]
    A --> D[Adaptive<br/>Design]
    A --> E[Factorial<br/>Design]
    
    style A fill:#FFD700
```

### Cluster Randomized Trial

```mermaid
graph TB
    A[Population] --> B[Randomize<br/>by Cluster]
    
    B --> C1[Cluster 1<br/>Hospital A<br/>Intervention]
    B --> C2[Cluster 2<br/>Hospital B<br/>Control]
    B --> C3[Cluster 3<br/>Hospital C<br/>Intervention]
    
    C1 --> D[Measure<br/>Outcomes]
    C2 --> D
    C3 --> D
    
    D --> E[Account for<br/>Clustering<br/>ICC adjustment]
    
    style B fill:#FFD700
    style E fill:#87CEEB
```

**Intraclass Correlation (ICC)**: Similarity within clusters
- Requires larger sample size than individual randomization
- Design effect = 1 + (m-1) × ICC

### Stepped-Wedge Design

```mermaid
graph TB
    A[All Start<br/>Control] --> B[Time 1:<br/>Group 1 → Intervention]
    B --> C[Time 2:<br/>Group 2 → Intervention]
    C --> D[Time 3:<br/>Group 3 → Intervention]
    D --> E[All Receive<br/>Intervention]
    
    style A fill:#FFB6C1
    style E fill:#90EE90
```

**Advantages**: Ethical (all get treatment), controls for time trends
**Disadvantages**: Complex analysis, longer duration

---

## Q3: Explain structural equation modeling (SEM).

**Answer**:

```mermaid
graph TB
    A[SEM Components] --> B[Measurement Model<br/>Latent variables]
    A --> C[Structural Model<br/>Relationships]
    
    B --> D1[Observed<br/>Variables]
    B --> D2[Latent<br/>Variables]
    
    C --> E1[Direct Effects]
    C --> E2[Indirect Effects]
    C --> E3[Mediation]
    C --> E4[Moderation]
    
    style A fill:#FFD700
    style B fill:#87CEEB
    style C fill:#90EE90
```

### Path Diagram

```mermaid
graph LR
    A[X1] --> L1((Latent 1))
    B[X2] --> L1
    C[X3] --> L1
    
    D[X4] --> L2((Latent 2))
    E[X5] --> L2
    F[X6] --> L2
    
    L1 --> L2
    L1 --> Y[Outcome]
    L2 --> Y
    
    style L1 fill:#87CEEB
    style L2 fill:#87CEEB
    style Y fill:#90EE90
```

**Fit Indices**:
- **CFI** (Comparative Fit Index): > 0.95 good
- **RMSEA** (Root Mean Square Error): < 0.06 good
- **SRMR** (Standardized Root Mean Square Residual): < 0.08 good

---

## Q4: How do you handle multiple testing problems?

**Answer**:

```mermaid
graph TB
    A[Multiple Testing<br/>Problem] --> B[Inflated Type I<br/>Error Rate]
    
    B --> C[If α = 0.05<br/>and 20 tests]
    C --> D[Expected false<br/>positives = 1]
    
    D --> E[Correction<br/>Methods]
    
    E --> F1[Bonferroni<br/>α/n]
    E --> F2[Holm-Bonferroni<br/>Sequential]
    E --> F3[FDR<br/>Benjamini-Hochberg]
    E --> F4[Permutation<br/>Tests]
    
    style B fill:#FF6B6B
    style E fill:#FFD700
    style F3 fill:#90EE90
```

### Family-Wise Error Rate (FWER)

**Bonferroni Correction**:
$$\alpha_{adjusted} = \frac{\alpha}{n}$$

```mermaid
graph LR
    A[20 tests<br/>α = 0.05] --> B[Bonferroni<br/>α = 0.05/20<br/>= 0.0025]
    
    B --> C[Very<br/>Conservative]
    
    style B fill:#87CEEB
    style C fill:#FFB6C1
```

### False Discovery Rate (FDR)

**Benjamini-Hochberg Procedure**:

```mermaid
sequenceDiagram
    participant T as Tests
    participant P as P-values
    participant R as Rank
    participant D as Decision
    
    T->>P: Calculate all p-values
    P->>R: Sort ascending
    R->>R: Find largest k where<br/>p(k) ≤ (k/n) × α
    R->>D: Reject H0 for tests 1 to k
```

**Less conservative than Bonferroni**, controls proportion of false discoveries

---

## Q5: Explain time series analysis and forecasting.

**Answer**:

```mermaid
graph TB
    A[Time Series<br/>Components] --> B[Trend<br/>Long-term direction]
    A --> C[Seasonality<br/>Repeating patterns]
    A --> D[Cyclical<br/>Non-fixed cycles]
    A --> E[Irregular<br/>Random noise]
    
    style A fill:#FFD700
```

### Decomposition

```mermaid
graph TB
    A[Observed<br/>Time Series] --> B[Decompose]
    
    B --> C1[Trend<br/>Component]
    B --> C2[Seasonal<br/>Component]
    B --> C3[Residual<br/>Component]
    
    C1 --> D[Additive:<br/>Y = T + S + R]
    C2 --> D
    C3 --> D
    
    C1 --> E[Multiplicative:<br/>Y = T × S × R]
    C2 --> E
    C3 --> E
    
    style B fill:#FFD700
```

### ARIMA Models

```mermaid
graph LR
    A[ARIMA p,d,q] --> B[AR p<br/>Autoregressive<br/>Past values]
    A --> C[I d<br/>Integrated<br/>Differencing]
    A --> D[MA q<br/>Moving Average<br/>Past errors]
    
    style A fill:#FFD700
```

**Model Selection**:
- **ACF/PACF plots**: Identify p, q
- **AIC/BIC**: Compare models
- **Stationarity tests**: Determine d

---

## Q6: Design a mixed-methods research study.

**Answer**:

```mermaid
graph TB
    A[Mixed Methods<br/>Designs] --> B[Convergent<br/>Parallel]
    A --> C[Explanatory<br/>Sequential]
    A --> D[Exploratory<br/>Sequential]
    A --> E[Embedded]
    
    style A fill:#FFD700
```

### Convergent Parallel Design

```mermaid
graph TB
    A[Research Question] --> B1[Quantitative<br/>Phase]
    A --> B2[Qualitative<br/>Phase]
    
    B1 --> C1[Survey<br/>n=500]
    B2 --> C2[Interviews<br/>n=20]
    
    C1 --> D[Merge &<br/>Compare]
    C2 --> D
    
    D --> E[Integrated<br/>Interpretation]
    
    style D fill:#FFD700
    style E fill:#90EE90
```

**Collect both simultaneously**, compare and integrate

### Explanatory Sequential Design

```mermaid
sequenceDiagram
    participant Q1 as Quantitative Phase
    participant A as Analysis
    participant Q2 as Qualitative Phase
    participant I as Integration
    
    Q1->>A: Survey (n=1000)
    A->>A: Find unexpected result
    A->>Q2: Design interviews<br/>to explain finding
    Q2->>I: Interviews (n=15)
    I->>I: Explain quantitative<br/>results with qualitative
```

**Quant first**, then qual to explain

---

## Q7: Implement machine learning for causal inference.

**Answer**:

```mermaid
graph TB
    A[ML for Causal<br/>Inference] --> B[Propensity Score<br/>with ML]
    A --> C[Causal Forests]
    A --> D[Double/Debiased<br/>ML]
    A --> E[Instrumental<br/>Variables + ML]
    
    style A fill:#FFD700
```

### Double/Debiased Machine Learning

```mermaid
graph TB
    A[Treatment Effect<br/>Estimation] --> B[Step 1:<br/>Predict Y with ML]
    
    B --> C[Get residuals<br/>Ỹ = Y - Ŷ]
    
    A --> D[Step 2:<br/>Predict T with ML]
    
    D --> E[Get residuals<br/>T̃ = T - T̂]
    
    C --> F[Step 3:<br/>Regress Ỹ on T̃]
    E --> F
    
    F --> G[Unbiased<br/>Treatment Effect]
    
    style A fill:#FFE4B5
    style F fill:#87CEEB
    style G fill:#90EE90
```

**Advantages**:
- Flexible modeling of confounders
- Reduces bias from model misspecification
- Valid inference

### Causal Forests

```mermaid
graph TB
    A[Random Forest<br/>for Heterogeneous<br/>Treatment Effects] --> B[Split Data<br/>Randomly]
    
    B --> C[For each split:<br/>Estimate treatment<br/>effect in subgroups]
    
    C --> D[Aggregate across<br/>trees]
    
    D --> E[Individual-level<br/>treatment effects]
    
    style A fill:#FFD700
    style E fill:#90EE90
```

---

## Q8: Explain survival analysis and competing risks.

**Answer**:

```mermaid
graph TB
    A[Survival Analysis] --> B[Kaplan-Meier<br/>Non-parametric]
    A --> C[Cox Regression<br/>Semi-parametric]
    A --> D[Parametric Models<br/>Weibull, etc.]
    A --> E[Competing Risks]
    
    style A fill:#FFD700
```

### Kaplan-Meier Curve

```mermaid
graph LR
    A[Time 0<br/>100% survive] --> B[Event 1<br/>95% survive]
    B --> C[Event 2<br/>88% survive]
    C --> D[Event 3<br/>75% survive]
    D --> E[Censored<br/>75% survive]
    E --> F[Event 4<br/>65% survive]
    
    style A fill:#90EE90
    style F fill:#FFB6C1
```

**Censoring**: Participant lost to follow-up or study ends

### Competing Risks

```mermaid
graph TB
    A[Patient<br/>at Risk] --> B{Outcome}
    
    B --> C[Event of<br/>Interest]
    B --> D[Competing<br/>Event 1]
    B --> E[Competing<br/>Event 2]
    B --> F[Censored]
    
    style C fill:#90EE90
    style D fill:#FFB6C1
    style E fill:#FFB6C1
```

**Example**: Studying death from disease
- **Event of interest**: Death from disease
- **Competing risk**: Death from other causes

**Cumulative Incidence Function (CIF)**: Accounts for competing risks

---

## Q9: Design and analyze network experiments.

**Answer**:

```mermaid
graph TB
    A[Network<br/>Experiments] --> B[Spillover Effects<br/>Treatment affects<br/>neighbors]
    
    B --> C[Cluster<br/>Randomization]
    B --> D[Ego-Network<br/>Randomization]
    B --> E[Graph Cluster<br/>Randomization]
    
    style A fill:#FFD700
    style B fill:#FF6B6B
```

### Network Structure

```mermaid
graph TB
    A((User A<br/>Treated)) --> B((User B<br/>Control))
    A --> C((User C<br/>Control))
    B --> D((User D<br/>Control))
    
    Note[Spillover: A's treatment<br/>may affect B and C]
    
    style A fill:#90EE90
    style B fill:#FFB6C1
    style C fill:#FFB6C1
```

### Graph Cluster Randomization

```mermaid
graph TB
    A[Network] --> B[Detect<br/>Communities]
    
    B --> C1[Community 1<br/>Treatment]
    B --> C2[Community 2<br/>Control]
    B --> C3[Community 3<br/>Treatment]
    
    C1 --> D[Minimize<br/>Between-cluster<br/>Connections]
    C2 --> D
    C3 --> D
    
    style B fill:#FFD700
    style D fill:#90EE90
```

**Analysis Considerations**:
- Direct effects vs. spillover effects
- Network autocorrelation
- Exposure mapping (who affects whom)

---

## Q10: Implement Bayesian hierarchical models.

**Answer**:

```mermaid
graph TB
    A[Hierarchical<br/>Bayesian Model] --> B[Level 1:<br/>Individual<br/>Observations]
    
    B --> C[Level 2:<br/>Group<br/>Parameters]
    
    C --> D[Level 3:<br/>Hyperparameters]
    
    D --> E[Priors on<br/>Hyperparameters]
    
    style A fill:#FFD700
    style C fill:#87CEEB
    style E fill:#FFE4B5
```

### Model Structure

```mermaid
graph TB
    A[Hyperprior<br/>μ, τ] --> B1[Group 1<br/>θ₁ ~ N μ,τ]
    A --> B2[Group 2<br/>θ₂ ~ N μ,τ]
    A --> B3[Group k<br/>θₖ ~ N μ,τ]
    
    B1 --> C1[Observations<br/>y₁ⱼ ~ N θ₁,σ]
    B2 --> C2[Observations<br/>y₂ⱼ ~ N θ₂,σ]
    B3 --> C3[Observations<br/>yₖⱼ ~ N θₖ,σ]
    
    style A fill:#FFE4B5
    style B1 fill:#87CEEB
    style B2 fill:#87CEEB
    style B3 fill:#87CEEB
```

**Advantages**:
- **Partial pooling**: Borrow strength across groups
- **Shrinkage**: Pull extreme estimates toward mean
- **Uncertainty quantification**: Full posterior distributions

### MCMC Sampling

```mermaid
sequenceDiagram
    participant I as Initialize
    participant S as Sample
    participant A as Accept/Reject
    participant C as Converge
    
    I->>S: Start with initial values
    loop MCMC iterations
        S->>S: Propose new parameters
        S->>A: Calculate acceptance probability
        A->>S: Accept or reject
    end
    S->>C: Check convergence (R-hat)
    
    alt Converged
        C->>C: Use samples for inference
    else Not converged
        C->>S: Continue sampling
    end
```

**Diagnostics**:
- **Trace plots**: Visual convergence check
- **R-hat**: < 1.01 indicates convergence
- **Effective sample size**: > 1000 recommended

---

## Summary

Hard research topics:
- **Bayesian vs. Frequentist**: Different statistical philosophies
- **Complex RCTs**: Cluster, stepped-wedge, adaptive designs
- **SEM**: Latent variables and structural relationships
- **Multiple Testing**: FWER and FDR control
- **Time Series**: ARIMA, decomposition, forecasting
- **Mixed Methods**: Integrating qual and quant
- **ML for Causality**: Double ML, causal forests
- **Survival Analysis**: Competing risks, censoring
- **Network Experiments**: Spillover effects
- **Hierarchical Bayesian**: Partial pooling, MCMC

These advanced methods enable tackling complex research questions with rigor.

