# Chapter 6 — Benchmark Against Rule-Based System

## Techniques and experiments described

- Build a **rule-based benchmark** from operational heuristics (amount, time-window risk, region change, etc.).
- Compare ML/DL systems against rules on:
  - recall, precision, F1,
  - practical cost outcomes,
  - interpretability and deployment trade-offs.
- Discuss hybrid operation where rules and learned models complement each other.

## Implementation check

- Implemented:
  - deterministic rule score in pipeline,
  - benchmark row as a first-class result,
  - cost-based savings metric (`net_savings_rel_rule_based`) for all learned methods.
- Ensemble track now provides clearer evidence for “hybridized” learned systems vs rules in dedicated outputs.
