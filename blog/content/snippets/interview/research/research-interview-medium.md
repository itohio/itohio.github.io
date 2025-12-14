---
title: "Research Interview Questions - Medium"
date: 2025-12-13
tags: ["research", "interview", "medium", "methodology", "analysis"]
---

Medium-level research interview questions covering advanced methodologies and analysis techniques.

## Q1: Explain different experimental designs and when to use each.

**Answer**:

```mermaid
graph TB
    A[Experimental<br/>Designs] --> B[Between-Subjects]
    A --> C[Within-Subjects]
    A --> D[Factorial]
    A --> E[Quasi-Experimental]
    
    B --> F1[Different participants<br/>per condition]
    C --> F2[Same participants<br/>all conditions]
    D --> F3[Multiple independent<br/>variables]
    E --> F4[No random<br/>assignment]
    
    style A fill:#FFD700
    style B fill:#87CEEB
    style C fill:#90EE90
    style D fill:#DDA0DD
    style E fill:#FFB6C1
```

### Between-Subjects Design

```mermaid
graph LR
    A[100 Participants] --> B[Random<br/>Assignment]
    B --> C1[Group 1: 50<br/>Algorithm A]
    B --> C2[Group 2: 50<br/>Algorithm B]
    
    C1 --> D[Measure<br/>Performance]
    C2 --> D
    D --> E[Compare<br/>Groups]
    
    style B fill:#FFD700
```

**Pros**: No learning effects, simpler analysis
**Cons**: Need more participants, individual differences

### Within-Subjects Design

```mermaid
graph TB
    A[50 Participants] --> B[All try<br/>Algorithm A]
    B --> C[All try<br/>Algorithm B]
    C --> D[Compare<br/>Performance]
    
    Note[Counterbalancing:<br/>Half do A then B<br/>Half do B then A]
    
    style A fill:#FFE4B5
    style D fill:#90EE90
```

**Pros**: Fewer participants, control individual differences
**Cons**: Learning effects, fatigue

### Factorial Design

```mermaid
graph TB
    A[2x2 Factorial] --> B[Factor 1:<br/>Algorithm A vs B]
    A --> C[Factor 2:<br/>Dataset Small vs Large]
    
    B --> D1[A + Small]
    B --> D2[A + Large]
    C --> D3[B + Small]
    C --> D4[B + Large]
    
    D1 --> E[Test<br/>Interactions]
    D2 --> E
    D3 --> E
    D4 --> E
    
    style A fill:#FFD700
    style E fill:#90EE90
```

**Pros**: Test multiple factors, find interactions
**Cons**: Complex, need more participants

---

## Q2: How do you handle confounding variables?

**Answer**:

```mermaid
graph TB
    A[Confounding<br/>Variable] --> B[Affects both<br/>IV and DV]
    
    B --> C[Strategies to<br/>Control]
    
    C --> D1[Randomization<br/>Distribute evenly]
    C --> D2[Matching<br/>Pair similar participants]
    C --> D3[Statistical Control<br/>ANCOVA, regression]
    C --> D4[Blocking<br/>Group by confounder]
    C --> D5[Standardization<br/>Keep constant]
    
    style A fill:#FF6B6B
    style C fill:#FFD700
    style D1 fill:#90EE90
```

**Example - Testing Algorithm Performance**:

```mermaid
graph LR
    A[Algorithm Type<br/>Independent Variable] --> C[Performance<br/>Dependent Variable]
    
    B[Hardware<br/>Confound] --> C
    B -.->|Also affects| A
    
    style B fill:#FF6B6B
```

**Solution**: Use same hardware for all tests (standardization)

---

## Q3: Explain power analysis and sample size calculation.

**Answer**:

```mermaid
graph TB
    A[Power Analysis] --> B[Determines<br/>Sample Size]
    
    B --> C1[Effect Size<br/>How big is difference?]
    B --> C2[Alpha α<br/>Significance level<br/>Usually 0.05]
    B --> C3[Power 1-β<br/>Usually 0.80]
    B --> C4[Sample Size n<br/>Calculate this]
    
    C1 --> D{Effect Size}
    D --> E1[Small: 0.2<br/>Need 200+ participants]
    D --> E2[Medium: 0.5<br/>Need 64 participants]
    D --> E3[Large: 0.8<br/>Need 26 participants]
    
    style A fill:#FFD700
    style C4 fill:#90EE90
```

**Power**: Probability of detecting effect if it exists

```mermaid
graph TB
    A[True State] --> B1[Effect Exists]
    A --> B2[No Effect]
    
    B1 --> C1[Detect: Power<br/>1-β = 0.80]
    B1 --> C2[Miss: Type II Error<br/>β = 0.20]
    
    B2 --> D1[Correctly Accept<br/>1-α = 0.95]
    B2 --> D2[False Positive<br/>Type I Error<br/>α = 0.05]
    
    style C1 fill:#90EE90
    style C2 fill:#FF6B6B
    style D1 fill:#90EE90
    style D2 fill:#FF6B6B
```

---

## Q4: How do you conduct meta-analysis?

**Answer**:

```mermaid
graph TB
    A[Define Research<br/>Question] --> B[Search Literature<br/>Systematically]
    
    B --> C[Screen Studies<br/>Inclusion criteria]
    
    C --> D[Extract Data<br/>Effect sizes]
    
    D --> E[Assess Quality<br/>Risk of bias]
    
    E --> F[Calculate<br/>Pooled Effect]
    
    F --> G[Test<br/>Heterogeneity]
    
    G --> H{Heterogeneous?}
    
    H -->|Yes| I[Subgroup Analysis<br/>Meta-regression]
    H -->|No| J[Report Combined<br/>Effect]
    
    I --> J
    
    style A fill:#FFE4B5
    style F fill:#87CEEB
    style J fill:#90EE90
```

**Forest Plot Visualization**:

```mermaid
graph LR
    A[Study 1: 0.5 ± 0.1] --> D[Combined<br/>Effect]
    B[Study 2: 0.6 ± 0.15] --> D
    C[Study 3: 0.4 ± 0.12] --> D
    
    D --> E[Pooled: 0.52<br/>95% CI: 0.45-0.59]
    
    style D fill:#FFD700
    style E fill:#90EE90
```

**Heterogeneity Tests**:
- **I²**: Percentage of variation due to heterogeneity
  - I² < 25%: Low
  - I² 25-75%: Moderate
  - I² > 75%: High

---

## Q5: Explain different types of bias in research.

**Answer**:

```mermaid
graph TB
    A[Research Bias] --> B[Selection Bias]
    A --> C[Measurement Bias]
    A --> D[Reporting Bias]
    A --> E[Confirmation Bias]
    
    B --> F1[Non-random<br/>sampling]
    C --> F2[Systematic error<br/>in measurement]
    D --> F3[Selective<br/>publishing]
    E --> F4[Favoring expected<br/>results]
    
    style A fill:#FFD700
    style B fill:#FF6B6B
    style C fill:#FF6B6B
    style D fill:#FF6B6B
    style E fill:#FF6B6B
```

### Selection Bias

```mermaid
graph LR
    A[Target Population<br/>All users] --> B[Sample<br/>Only power users]
    
    B --> C[Results not<br/>generalizable]
    
    style B fill:#FF6B6B
    style C fill:#FF6B6B
```

**Mitigation**: Random sampling, stratified sampling

### Publication Bias

```mermaid
graph TB
    A[10 Studies Conducted] --> B[5 Positive Results]
    A --> C[5 Negative Results]
    
    B --> D[All 5 Published]
    C --> E[Only 1 Published]
    
    D --> F[Literature appears<br/>more positive than reality]
    E --> F
    
    style F fill:#FF6B6B
```

**Mitigation**: Pre-registration, publish all results

---

## Q6: How do you perform A/B testing correctly?

**Answer**:

```mermaid
graph TB
    A[Define Metric] --> B[Calculate<br/>Sample Size]
    
    B --> C[Random<br/>Assignment]
    
    C --> D1[Group A: Control<br/>50% traffic]
    C --> D2[Group B: Treatment<br/>50% traffic]
    
    D1 --> E[Collect Data]
    D2 --> E
    
    E --> F[Statistical Test]
    
    F --> G{Significant?}
    
    G -->|Yes| H[Implement B]
    G -->|No| I[Keep A]
    
    style A fill:#FFE4B5
    style C fill:#FFD700
    style F fill:#87CEEB
```

**Common Pitfalls**:

```mermaid
graph TB
    A[A/B Testing<br/>Pitfalls] --> B1[Peeking<br/>Check results early]
    A --> B2[Multiple Testing<br/>Test many variants]
    A --> B3[Novelty Effect<br/>Initial excitement]
    A --> B4[Sample Ratio<br/>Mismatch]
    
    B1 --> C1[Solution:<br/>Pre-determine duration]
    B2 --> C2[Solution:<br/>Bonferroni correction]
    B3 --> C3[Solution:<br/>Run longer test]
    B4 --> C4[Solution:<br/>Check randomization]
    
    style A fill:#FFD700
    style B1 fill:#FF6B6B
    style B2 fill:#FF6B6B
    style B3 fill:#FF6B6B
    style B4 fill:#FF6B6B
```

**Sequential Testing**:

```mermaid
sequenceDiagram
    participant T as Test
    participant A as Analysis
    participant D as Decision
    
    Note over T: Week 1
    T->>A: Check results
    A->>D: Not significant, continue
    
    Note over T: Week 2
    T->>A: Check results
    A->>D: Not significant, continue
    
    Note over T: Week 3
    T->>A: Check results
    A->>D: Significant! Stop test
    
    Note over D: Implement winner
```

---

## Q7: Explain regression analysis and when to use it.

**Answer**:

```mermaid
graph TB
    A[Regression<br/>Analysis] --> B[Linear Regression]
    A --> C[Multiple Regression]
    A --> D[Logistic Regression]
    
    B --> E1[One predictor<br/>Continuous outcome]
    C --> E2[Multiple predictors<br/>Continuous outcome]
    D --> E3[Predict<br/>Binary outcome]
    
    style A fill:#FFD700
    style B fill:#87CEEB
    style C fill:#90EE90
    style D fill:#DDA0DD
```

### Simple Linear Regression

```mermaid
graph LR
    A[X: Training Time] --> B[y = β₀ + β₁X]
    B --> C[Y: Model Accuracy]
    
    style B fill:#FFD700
```

**Equation**: $y = \beta_0 + \beta_1 x + \epsilon$

**Interpretation**:
- $\beta_0$: Intercept (accuracy with 0 training)
- $\beta_1$: Slope (accuracy increase per hour)
- $R^2$: Proportion of variance explained

### Multiple Regression

```mermaid
graph TB
    A1[X₁: Training Time] --> D[y = β₀ + β₁X₁ + β₂X₂ + β₃X₃]
    A2[X₂: Dataset Size] --> D
    A3[X₃: Model Complexity] --> D
    
    D --> E[Y: Model Accuracy]
    
    style D fill:#FFD700
```

**Use Cases**:
- Predict continuous outcomes
- Understand relationships
- Control for confounds
- Feature importance

---

## Q8: How do you handle missing data?

**Answer**:

```mermaid
graph TB
    A[Missing Data<br/>Patterns] --> B[MCAR<br/>Missing Completely<br/>At Random]
    A --> C[MAR<br/>Missing At Random]
    A --> D[MNAR<br/>Missing Not<br/>At Random]
    
    B --> E1[Safe to delete<br/>or impute]
    C --> E2[Can impute<br/>based on other vars]
    D --> E3[Problematic<br/>Biased results]
    
    style A fill:#FFD700
    style B fill:#90EE90
    style C fill:#FFD700
    style D fill:#FF6B6B
```

**Handling Strategies**:

```mermaid
graph TB
    A[Missing Data<br/>Solutions] --> B1[Deletion<br/>Listwise/Pairwise]
    A --> B2[Imputation<br/>Fill in values]
    A --> B3[Model-Based<br/>ML methods]
    
    B2 --> C1[Mean/Median<br/>Simple]
    B2 --> C2[Regression<br/>Predict from others]
    B2 --> C3[Multiple Imputation<br/>Create several datasets]
    B2 --> C4[KNN<br/>Similar cases]
    
    style A fill:#FFD700
    style B2 fill:#87CEEB
    style C3 fill:#90EE90
```

**Multiple Imputation Process**:

```mermaid
sequenceDiagram
    participant D as Original Data
    participant I as Imputation
    participant A as Analysis
    participant P as Pooling
    
    D->>I: Create imputed dataset 1
    D->>I: Create imputed dataset 2
    D->>I: Create imputed dataset n
    
    I->>A: Analyze dataset 1
    I->>A: Analyze dataset 2
    I->>A: Analyze dataset n
    
    A->>P: Combine results
    P->>P: Pool estimates
```

---

## Q9: Explain causal inference and methods.

**Answer**:

```mermaid
graph TB
    A[Correlation ≠<br/>Causation] --> B[Establish<br/>Causality]
    
    B --> C1[Randomized<br/>Controlled Trial<br/>Gold standard]
    B --> C2[Natural<br/>Experiments]
    B --> C3[Instrumental<br/>Variables]
    B --> C4[Regression<br/>Discontinuity]
    B --> C5[Difference-in-<br/>Differences]
    
    style A fill:#FF6B6B
    style C1 fill:#90EE90
```

### Causal Diagrams (DAG)

```mermaid
graph LR
    A[Treatment] --> C[Outcome]
    B[Confounder] --> A
    B --> C
    D[Mediator] --> C
    A --> D
    
    style A fill:#87CEEB
    style C fill:#90EE90
    style B fill:#FFD700
```

### Propensity Score Matching

```mermaid
graph TB
    A[Observational Data] --> B[Calculate<br/>Propensity Scores]
    
    B --> C[Probability of<br/>receiving treatment]
    
    C --> D[Match treated<br/>with untreated]
    
    D --> E[Similar propensity<br/>scores paired]
    
    E --> F[Compare outcomes<br/>in matched pairs]
    
    style B fill:#FFD700
    style F fill:#90EE90
```

---

## Q10: How do you conduct reproducible research?

**Answer**:

```mermaid
graph TB
    A[Reproducible<br/>Research] --> B[Version Control<br/>Git]
    A --> C[Documentation<br/>README, comments]
    A --> D[Environment<br/>Docker, conda]
    A --> E[Data Management<br/>Raw + processed]
    A --> F[Code Organization<br/>Modular, tested]
    
    B --> G[Track all changes]
    C --> G
    D --> G
    E --> G
    F --> G
    
    G --> H[Others can<br/>replicate results]
    
    style A fill:#FFD700
    style H fill:#90EE90
```

**Project Structure**:

```mermaid
graph TB
    A[project/] --> B[data/<br/>raw/, processed/]
    A --> C[src/<br/>analysis scripts]
    A --> D[notebooks/<br/>exploration]
    A --> E[results/<br/>figures, tables]
    A --> F[README.md]
    A --> G[requirements.txt]
    A --> H[.gitignore]
    
    style A fill:#FFE4B5
```

**Reproducibility Checklist**:

```mermaid
graph LR
    A[Checklist] --> B1[✓ Code versioned]
    A --> B2[✓ Dependencies listed]
    A --> B3[✓ Data available]
    A --> B4[✓ Seeds set]
    A --> B5[✓ Steps documented]
    A --> B6[✓ Results match]
    
    style A fill:#FFD700
    style B1 fill:#90EE90
    style B2 fill:#90EE90
    style B3 fill:#90EE90
    style B4 fill:#90EE90
    style B5 fill:#90EE90
    style B6 fill:#90EE90
```

---

## Summary

Medium research topics:
- **Experimental Designs**: Between, within, factorial
- **Confounding Variables**: Control strategies
- **Power Analysis**: Sample size calculation
- **Meta-Analysis**: Combining study results
- **Research Bias**: Types and mitigation
- **A/B Testing**: Proper implementation
- **Regression Analysis**: Prediction and relationships
- **Missing Data**: Handling strategies
- **Causal Inference**: Establishing causality
- **Reproducibility**: Version control, documentation

These techniques enable conducting rigorous, reliable research.

