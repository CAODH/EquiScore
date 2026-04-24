# AI Agent Guidelines for EquiScore Development

This document guides AI agents working on EquiScore to be **proactive**, **collaborative**, and **decision-aware**. Future agents should actively seek clarification and surface assumptions early.

---

## Core Principles

### 1. **Be Proactive, Not Passive**
- **Don't assume** implementation details; **ask first** when ambiguity exists
- Surface **trade-offs** and **design alternatives** explicitly before implementing
- Propose **2-3 approaches** with pros/cons and ask which resonates with your intent
- Flag **risks**, **edge cases**, and **dependencies** before writing code

### 2. **Respect Established Patterns**
All EquiScore outputs follow strict conventions:
- **Output schemas are tidy**: method, dataset_name, sna as leading columns (no wide format)
- **Metrics are median-only**: no means; see [evaluate_predictions.py](workflow/scripts/evaluate_predictions.py#L179)
- **Assay data is minimal**: only standard_value_filled (median-imputed); no one-hots or missingness flags (see [score_methods.py](workflow/scripts/score_methods.py#L51))
- **EDA focuses on train+val splits**: test sets held out for final validation
- **Multi-method pipeline**: 7 methods enable systematic ablation (ligand → assay → fingerprint)

### 3. **Prioritize Clarity & Parsimony**
- Code should read like a senior engineer wrote it (conventional patterns, DRY principles, no redundancy)
- One visualization per insight; delete duplicates
- Document **why** decisions were made (e.g., "standard_value_filled only to avoid leakage")
- Use type hints and docstrings for key functions

### 4. **Plan Before Implementing**
When tasked with a feature or fix:
1. **Create a structured plan** (in session memory at `/memories/session/plan.md`)
2. **Show the plan to the user** and ask for explicit approval
3. **Implement only after approval**

---

## Collaboration Workflow

### Phase 1: **Discovery**
When asked to implement a feature or investigate an issue:

1. **Explore the codebase** for related patterns
   - Use `Explore` subagent for quick context gathering
   - Check `config.yaml` for method/parameter definitions
   - Review function signatures and docstrings in key modules
   - Look for existing test data or example runs

2. **Identify key code anchors**
   - Link to specific functions and line ranges
   - Example: "[_feature_cols() at line 36](workflow/scripts/score_methods.py#L36)"
   - Document what each anchor does and why it matters

3. **Surface ambiguities immediately**
   - "Should we analyze test sets in EDA, or keep them held out?"
   - "Do fingerprint visualizations need PCA, or are 1D histograms sufficient?"
   - "Is this a new feature or replacing an existing method?"

### Phase 2: **Alignment**
If the task has multiple valid approaches:

1. **Present 2–3 options** with clear trade-offs
   - Option A: Simpler implementation, narrower scope
   - Option B: More comprehensive, higher effort
   - Option C: Minimal viable version with optional enhancements

2. **Ask explicitly for direction**
   - "Which approach aligns with your vision?"
   - "Any constraints on compute time or data size?"
   - "Should I prioritize speed or interpretability?"

3. **Surface hidden assumptions**
   - "I'm assuming SNA=true and SNA=false should be analyzed separately. Correct?"
   - "Should missing features be imputed or dropped?"

### Phase 3: **Planning**
Before writing code:

1. **Create a detailed plan** in `/memories/session/plan.md`
   ```markdown
   ## Task: [Feature Name]
   
   ### TL;DR
   [One-line summary]
   
   ### Approach
   [Selected approach and why]
   
   ### Steps
   1. [Step 1 - file, function, expected output]
   2. [Step 2 - dependencies on Step 1]
   ...
   
   ### Verification
   - [ ] [Test 1]
   - [ ] [Test 2]
   
   ### Open Questions
   - [Question 1?]
   ```

2. **Show the plan to the user**
   - Ask: "Does this match your intent?"
   - Ask: "Any changes before I start implementing?"
   - Ask: "Should I proceed?"

### Phase 4: **Implementation**
Once approved:

1. **Implement systematically**
   - Follow existing code patterns (Polars over pandas, parquet for I/O, polars.DataFrame pipelines)
   - Write clean, readable code with docstrings for all functions
   - Minimize scope creep; stick to the plan

2. **Verify as you go**
   - Run sample commands with actual data
   - Check error messages and logs
   - Validate outputs against the plan

3. **Document decisions**
   - Add comments explaining non-obvious choices
   - Link to related code sections
   - Note any deviations from the plan and why

---

## Technical Context

### Output Architecture
- **Predictions**: Long format (one row = protein + ligand + method)
  - Columns: protein_id, ligand_smiles, label, method, score, rank_within_protein, sna
- **Summaries**: Method-level aggregation (median metrics only)
  - Leading columns: method; no per-dataset breakout
- **SNA Comparison**: Tidy format with explicit sna column
  - Leading columns: method, dataset_name, sna (no wide format like sna_true/sna_false)

### Feature Usage at Inference
Methods use different feature sets:

| Method | Features |
|--------|----------|
| `lambdamart_ligand` | Descriptors (TPSA, MW, HBD/HBA, rotatable_bonds, etc.) |
| `logistic_regression_ligand` | Descriptors |
| `tanimoto_random_ref` | Morgan fingerprints (radius=2, 2048 bits) vs random active ref |
| `lambdamart_ligand_assay` | Descriptors + standard_value_filled |
| `logistic_regression_ligand_assay` | Descriptors + standard_value_filled |
| `lambdamart_fingerprint` | ECFP bits only (2048-bit Morgan) |
| `logistic_regression_fingerprint` | ECFP bits only |

**Note**: Assay methods show perfect/near-perfect metrics → likely data leakage (intended for ablation, not production).

### Manifest Parquets
Located in `dataset/`:
- `bigbind_manifest.parquet` – Full BigBind (train+val+test) with assay metadata + SNA
- `pdbscreen_manifest.parquet` – PDBScreen (train+val+test) without assay data

**Columns available**:
- Ligand: protein_id, ligand_smiles, label, split, sna (BigBind only)
- Descriptors: tpsa, molecular_weight, rotatable_bonds, ring_count, aromatic_ring_count, hbd, hba
- Compliance: lipinski_compliant, verber_compliant
- Fingerprints: ligand_fingerprint
- Assay (BigBind): standard_type, standard_value, standard_value_filled

**Usage**: Pre-computed; all scoring scripts load from these. Avoid recomputing features.

### Data Splits
- **Train**: Used for model training, median imputation baseline (standard_value_filled), tanimoto reference selection
- **Val**: Used for hyperparameter tuning, intermediate evaluation
- **Test**: Held out intentionally; not in EDA notebooks; only for final benchmarking
- **Clarification**: If you need test data, ask explicitly; we keep it separate by design

---

## Questions to Ask Before Starting

Always ask these before implementing:

1. **Scope & Goals**
   - "Is this a new feature, bug fix, or enhancement?"
   - "How does it fit into the broader roadmap?"
   - "What's the intended use case?"

2. **Validation & Testing**
   - "How will we know the change works?"
   - "Any benchmarks or regression tests I should run?"
   - "Should I test on both BigBind and PDBScreen?"

3. **Trade-offs & Constraints**
   - "Is there a simpler version that gets 80% of the value?"
   - "Any limits on compute time, memory, or dependencies?"
   - "Should we prioritize speed or interpretability?"

4. **Patterns & Conventions**
   - "Should this follow existing pattern X, or is a new pattern OK?"
   - "Do you want the output in tidy or wide format?"
   - "Median or mean aggregation for this metric?"

5. **Edge Cases**
   - "What should happen with missing features?"
   - "Should we handle test sets separately?"
   - "Any special logic for proteins with <5 ligands?"

---

## When in Doubt

1. **Ask, don't guess**. The cost of clarification << cost of re-implementing.
2. **Show your work**. Document the plan *before* coding.
3. **Keep it simple**. Prefer conventional patterns over clever abstractions.
4. **Test early**. Validate assumptions with data/code asap.
5. **Be explicit about decisions**. Leave comments explaining the "why".

---

## Common Patterns to Reuse

### Loading Data
```python
import polars as pl
from pathlib import Path

# Load manifest
df = pl.read_parquet(Path("dataset/bigbind_manifest.parquet"))

# EDA: filter to train+val only
df = df.filter(pl.col("split").is_in(["train", "val"]))
```

### Feature Columns (Ligand-Only Methods)
```python
def _feature_cols():
    """Ligand descriptor features."""
    return ["tpsa", "molecular_weight", "rotatable_bonds", "ring_count", 
            "aromatic_ring_count", "hbd", "hba"]
```

### Median Aggregation
```python
summary = df.group_by("method").agg([
    pl.col("roc_auc").median().alias("roc_auc_median"),
    pl.col("map").median().alias("map_median"),
    pl.len().alias("n_queries"),
])
```

### Tidy Output (SNA Comparison)
```python
# NOT wide format (sna_true_roc_auc, sna_false_roc_auc)
# USE tidy format with explicit sna column
output = df.select(["method", "dataset_name", "sna", "ef@1%_median", "map_median"])
```

### Polars Pipelines
```python
result = (
    df
    .filter(pl.col("split").is_in(["train", "val"]))
    .group_by("method")
    .agg([pl.col("score").median()])
    .sort("method")
)
```

---

## Session Memory Structure

Use `/memories/session/plan.md` to track task progress:

```markdown
## Task: [Feature/Issue Name]

### Status
- [ ] Discovery
- [ ] Alignment
- [ ] Planning
- [ ] Implementation
- [ ] Verification

### TL;DR
[One-line summary]

### Key Insights
- [Finding 1]
- [Finding 2]

### Proposed Approach
[Option selected and why]

### Steps (Sequential or Parallel)
1. [Step 1 - file/function/expected output]
   - Depends on: [none or prior step]
   - Verification: [command or test]

### Open Questions
- [Question 1?]
- [Question 2?]

### Blockers
- [None / list any blockers]

### Next Action
[What comes next when user approves]
```

---

## AI Agent Workflow Checklist

Before implementing anything:
- [ ] Explored related code and patterns
- [ ] Identified key code anchors (links to files/lines)
- [ ] Surfaced ambiguities or assumptions
- [ ] Proposed 2–3 approaches (if applicable)
- [ ] Created detailed plan in session memory
- [ ] Showed plan to user and got approval
- [ ] Ready to implement

During implementation:
- [ ] Following existing code patterns
- [ ] Writing readable, documented code
- [ ] Running validation tests
- [ ] Sticking to the approved plan

After implementation:
- [ ] All verification tests pass
- [ ] Edge cases handled
- [ ] Output formats validated
- [ ] Documented any deviations from plan

---

## Questions? Need Clarification?

If something feels ambiguous, **ask first**. The goal is collaborative development where:
- Assumptions are surfaced early
- Decisions are made together
- Code quality and correctness are prioritized over speed
