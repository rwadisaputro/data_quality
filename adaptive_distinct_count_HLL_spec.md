# Adaptive Distinct Count + HyperLogLog: Quick-Read Implementation Requirements

## Goal

Implement a cardinality profiler that works when the distinct count of an input column is unknown.

Default behavior:

```text
start exact
    ↓
exact state still cheap?
    ├─ yes → continue exact → return exact count
    └─ no  → promote once to HLL → return approximate count
```

The governing rule is:

$$
\boxed{\text{exact when cheap, approximate when necessary}}
$$

HyperLogLog (HLL) is **not sampling**. It normally processes every included row. Its advantage is bounded retained state and cheap mergeability.

---

## 1. Execution policies

Support:

```text
ADAPTIVE_PREFERRED   # default
EXACT_ONLY
HLL_ONLY
TWO_STAGE            # fallback when adaptive state is impossible
```

### Adaptive mode

Begin with an exact distinct set.

Promote to HLL only if the exact state exceeds a configured limit.

### Two-stage fallback

If the execution engine cannot dynamically promote aggregate state:

```text
pass 1: HLL estimate + row/null metrics
    ↓
decide whether exact counting is safe
    ↓
pass 2: exact count only for selected low-cardinality columns
```

Never calculate a complete exact distinct set and then run HLL afterward.

---

## 2. Equality and canonicalization

Exact counting and HLL must use the **same definition of equality**.

Define a deterministic canonicalization function:

$$
C(x)
$$

and guarantee:

$$
C_{\text{exact}}(x)=C_{\text{HLL}}(x)
$$

Use type-aware encoding so values such as:

```text
integer 1
string "1"
boolean true
```

do not accidentally become equivalent.

Define explicit handling for:

```text
null
NaN
±infinity
-0.0 / +0.0
datetime/timezone values
Unicode/string normalization
```

Recommended null policy:

```text
exclude null from cardinality
```

Keep `null_count` separately.

---

## 3. Hash requirements

HLL hashing must be:

```text
deterministic
stable across runs/workers
well distributed
fixed width
seeded consistently
```

Store/version:

```text
hash_algorithm_id
hash_width_bits
hash_seed
canonicalisation_version
```

Prefer a mature HLL/HLL++ implementation with its own tested hash/estimator behavior.

Do not merge sketches whose hash or canonicalization semantics differ.

---

## 4. Exact state

Maintain:

```text
mode = EXACT | HLL

total_row_count
non_null_count
null_count

exact_values
exact_state_bytes
peak_exact_unique_count
peak_exact_state_bytes

hll_sketch

promotion_reason
promotion_row_index
promotion_unique_count
```

For each non-null value in `EXACT` mode:

```text
canonicalize value
→ insert into exact set
→ if new, update exact memory/accounting
→ test promotion thresholds
```

Promotion occurs when either:

$$
|S_{\text{exact}}| > U_{\max}
$$

or:

$$
B_{\text{exact}} > B_{\max}
$$

where:

- $U_{\max}$ = maximum exact unique values;
- $B_{\max}$ = maximum exact-state memory.

Use **both** thresholds. Cardinality alone is insufficient because value sizes vary.

Suggested initial defaults:

```text
exact_unique_limit = 50,000
exact_memory_budget = 32 MiB
memory_safety_factor = 1.25
```

These are tuning defaults, not universal constants.

---

## 5. Promotion to HLL

When exact state crosses a threshold:

```text
1. Create empty HLL.
2. Insert every retained exact unique value into HLL once.
3. Record promotion telemetry.
4. Release the exact set.
5. Set mode = HLL.
6. Feed all remaining values directly to HLL.
```

Do **not** replay all previously processed rows.

The retained unique values already contain all distinct information observed before promotion.

Once promoted, never return to exact mode.

---

## 6. HLL mechanics

With precision $p$:

$$
m=2^p
$$

where $m$ is the register count.

For each canonical input:

```text
hash value
→ first p bits select register
→ remaining bits determine leading-zero rank
→ update register with maximum observed rank
```

Conceptually:

$$
M_j \leftarrow \max(M_j,\rho(w))
$$

where:

$$
\rho(w)=1+\text{leading-zero count}(w)
$$

A conventional HLL estimate is based on all registers:

$$
\hat U
=
\alpha_m m^2
\left(
\sum_{j=1}^{m}2^{-M_j}
\right)^{-1}
$$

Use a mature implementation rather than manually implementing estimator corrections.

---

## 7. Precision and error

For conventional HLL:

$$
\operatorname{RSE}
\approx
\frac{1.04}{\sqrt{m}}
=
\frac{1.04}{2^{p/2}}
$$

Typical values:

| $p$ | Registers | Approx. RSE |
|---:|---:|---:|
| 12 | 4,096 | 1.63% |
| 14 | 16,384 | 0.81% |
| 16 | 65,536 | 0.41% |

Suggested initial default:

```text
hll_precision = 14
```

If the chosen implementation exposes native lower/upper bounds or a different documented error model, use those.

RSE is not a hard maximum-error guarantee.

---

## 8. Final result

If the state finishes in exact mode:

$$
U=|S_{\text{exact}}|
$$

Return an exact result.

If it finishes in HLL mode:

$$
\hat U=\operatorname{estimate}(\text{HLL})
$$

Return an approximate result.

Never return only a bare integer.

Minimum result contract:

```text
cardinality
estimate
method                  # EXACT | HLL
is_exact

total_row_count
non_null_count
null_count

distinct_ratio
duplicate_ratio

hll_precision
hll_register_count
nominal_rse
lower_bound
upper_bound

promotion_reason
promotion_row_index
promotion_unique_count

peak_exact_unique_count
peak_exact_state_bytes

hash/canonicalization metadata
source_pass_count
```

For exact results:

```text
estimate = cardinality
lower_bound = cardinality
upper_bound = cardinality
is_exact = true
```

For HLL results:

```text
is_exact = false
estimate = raw HLL estimate
cardinality = rounded/display value
```

---

## 9. Derived metrics

Let:

$$
U^*
=
\begin{cases}
U, & \text{exact}\\
\hat U, & \text{HLL}
\end{cases}
$$

Then:

$$
\text{distinct ratio}
=
\frac{U^*}{N_{\text{non-null}}}
$$

$$
\text{duplicate ratio}
=
1-\text{distinct ratio}
$$

If $U^*$ comes from HLL, mark dependent metrics as estimated.

---

## 10. Distributed merge behavior

All cardinality states must be mergeable.

### EXACT + EXACT

Union the sets.

Remain exact if the merged state stays within resource limits; otherwise promote.

### EXACT + HLL

Insert exact unique values into HLL once. Result is HLL.

### HLL + HLL

Merge compatible sketches.

Conceptually:

$$
M_j^{A\cup B}
=
\max(M_j^A,M_j^B)
$$

All HLL states in one logical aggregation should use the same precision.

Reject incompatible sketches unless explicit, tested conversion support exists.

Compatibility includes:

```text
algorithm/version
precision
hash algorithm
hash width
hash seed
canonicalisation version
null/equality semantics
```

---

## 11. Two-stage routing rule

When adaptive state is unavailable, use HLL first.

If the HLL implementation provides an upper bound, use it for planning.

Otherwise a conservative routing heuristic may be:

$$
U_{\text{planning}}
=
\hat U(1+z\cdot\operatorname{RSE})
$$

Estimate exact-state memory:

$$
B_{\text{planning}}
=
U_{\text{planning}}
\left(
\bar B_{\text{value}}
+
B_{\text{entry overhead}}
\right)
$$

Run exact count only when both are satisfied:

$$
U_{\text{planning}}\le U_{\max}
$$

$$
B_{\text{planning}}\le B_{\max}
$$

This fallback requires replayable input.

---

## 12. Complexity expectations

Exact mode:

$$
T=O(N), \qquad M=O(U)
$$

HLL mode:

$$
T=O(N)+O(m), \qquad M=O(m)
$$

HLL therefore does **not** avoid reading the rows. It prevents aggregation state from growing with the true number of unique values.

---

## 13. Telemetry

Collect:

```text
row/non-null/null counts

exact completion rate
HLL promotion rate
promotion reason
promotion row
promotion unique count

peak exact unique count
peak exact-state bytes

HLL precision
sketch bytes
estimate
bounds

source pass count
elapsed time
peak memory
spill/shuffle/network bytes where available
```

Use telemetry to tune:

```text
exact_unique_limit
exact_memory_budget
memory safety factor
HLL precision
```

---

## 14. Validation

For controlled datasets where exact cardinality $U$ is known, measure:

$$
\text{relative error}
=
\frac{|\hat U-U|}{U}
$$

Test across different:

```text
cardinalities
row counts
distinct ratios
value lengths/types
HLL precisions
partition counts
merge depths
input orderings
```

If native confidence bounds are available, verify empirical coverage.

---

## 15. Minimum tests

Required cases:

```text
empty input
all-null input
one repeated value
small low-cardinality input remains exact
unique threshold boundary
memory-triggered promotion
count-triggered promotion
promotion equivalence to full-stream HLL
EXACT + EXACT merge
EXACT + HLL merge
HLL + HLL merge
incompatible sketch rejection
duplicate-heavy input
long-string input
stable canonicalization/hash vectors
input-order invariance
```

---

## 16. Persisted sketch compatibility

If HLL sketches are stored for later reuse, persist:

```text
sketch_format_version
algorithm_id
algorithm_version
precision
hash_algorithm_id
hash_width_bits
hash_seed
canonicalisation_version
null_policy
sketch_payload
```

Never persist anonymous sketch bytes without enough metadata to validate future merges.

---

## 17. Core invariants

The implementation must guarantee:

```text
Exact results are always labelled exact.

HLL results are always labelled approximate.

Exact and HLL use identical equality/canonicalization semantics.

Exact state stops growing after promotion.

All retained exact uniques are transferred to HLL exactly once.

All subsequent included values update HLL.

Merge operations preserve union semantics.

Incompatible sketches are never silently merged.

Approximate results expose precision/error provenance.

A completed exact cardinality is never followed by redundant HLL computation.
```

---

## 18. Recommended component boundaries

Keep backend-specific dataframe code outside the core mechanics.

Suggested abstractions:

```text
CardinalityProfiler
CardinalityState
ExactDistinctState
HyperLogLogState
CanonicalValueEncoder
HashConfiguration
CardinalityDecisionPolicy
CardinalityMergePolicy
CardinalityConfiguration
CardinalityResult
CardinalityTelemetry
```

---

## 19. Implementation sequence

```text
1. Define canonical equality/null semantics.
2. Define configuration and result models.
3. Implement bounded exact state.
4. Wrap a mature HLL/HLL++ implementation.
5. Implement exact → HLL promotion.
6. Implement finalization/provenance.
7. Implement merge rules.
8. Add memory accounting.
9. Add telemetry.
10. Add persisted-sketch compatibility metadata.
11. Add two-stage fallback.
12. Benchmark and tune thresholds.
13. Validate HLL error against exact counts.
14. Integrate backend adapters last.
```

---

## 20. Acceptance criterion

The feature is complete when the library can receive an unknown-cardinality column and safely produce:

```text
an exact count when exact state remains inexpensive

OR

an explicitly approximate HLL count when exact state would exceed
configured resource limits
```

without needing prior knowledge of the column's cardinality.

---

## References

- Philippe Flajolet et al., **HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm**  
  https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf

- Stefan Heule, Marc Nunkesser, Alex Hall, **HyperLogLog in Practice**  
  https://research.google/pubs/hyperloglog-in-practice-algorithmic-engineering-of-a-state-of-the-art-cardinality-estimation-algorithm/

- Apache DataSketches, **HyperLogLog Sketches Overview**  
  https://datasketches.apache.org/docs/HLL/HllSketches.html

- Apache DataSketches, **HLL Maximum Sketch Size & Error Table**  
  https://datasketches.apache.org/docs/HLL/HllMaxSizeAndErrorTable.html
