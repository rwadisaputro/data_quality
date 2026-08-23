# Adaptive Distinct Counting with HyperLogLog: Non-Technical Implementation Guide

## Purpose

Build a distinct-value counting feature that works safely even when we do not know in advance how many unique values a column contains.

The system should behave like this:

```text
Start by counting exactly
    ↓
Is the exact count still cheap to maintain?
    ├─ Yes → keep going exactly
    └─ No  → switch to HyperLogLog
```

The guiding principle is:

> **Use an exact count when it is cheap, and use HyperLogLog only when exact counting becomes too expensive.**

HyperLogLog (HLL) is an approximate distinct-counting technique. It is useful because it can estimate very large numbers of unique values while keeping its memory usage small and predictable.

It does **not** normally skip rows or work from a sample. It still processes the values it is given. The saving comes from how little information it keeps in memory.

---

## 1. What problem are we solving?

For some columns, exact distinct counting is easy.

Example:

```text
Australia
Australia
New Zealand
Australia
New Zealand
```

There are only two unique values:

```text
Australia
New Zealand
```

Keeping those two values in memory is trivial.

But another column might contain millions of different customer IDs, transaction IDs, URLs, or email addresses.

An exact distinct count may then require the system to remember millions of unique values at once.

We therefore need a method that works well for both cases without knowing beforehand which type of column we are dealing with.

---

## 2. Preferred behaviour

The preferred approach is to begin with exact counting.

The system keeps a set of unique values seen so far:

```text
value arrives
    ↓
have we seen it before?
    ├─ Yes → nothing new needs to be stored
    └─ No  → add it to the exact unique-value set
```

As long as this set stays small enough, continue exact counting.

If the set becomes too large, switch to HLL.

This means:

```text
small-cardinality column
→ exact result

large-cardinality column
→ approximate HLL result
```

The system does not need to know the final cardinality in advance.

---

## 3. When should the system switch to HLL?

Use two limits:

1. **Maximum number of exact unique values**
2. **Maximum memory allowed for the exact set**

Example starting values:

```text
maximum exact unique values = 50,000
maximum exact-state memory = 32 MiB
HLL precision = 14
```

These values should be configurable and tuned later using real workloads.

Both limits matter.

For example:

```text
50,000 short integer IDs
```

may use much less memory than:

```text
50,000 very long strings
```

So the system should not look only at the number of unique values.

---

## 4. What happens when the limit is reached?

Suppose the exact limit is 50,000 unique values.

The system may process a column like this:

```text
row 1
row 2
row 3
...
```

Eventually it reaches:

```text
50,001 unique values
```

At that point:

```text
1. Create an HLL sketch.
2. Add the 50,001 exact unique values already collected into HLL.
3. Delete the exact unique-value set.
4. Continue processing all remaining rows through HLL.
5. Return an approximate distinct count at the end.
```

There is no need to reread all earlier rows.

Only the distinct values already retained in the exact set need to be transferred.

After the switch, the system should remain in HLL mode.

Do not switch back to exact counting later.

---

## 5. Important rule: do not run HLL after finishing an exact count

Avoid this:

```text
read entire column
→ calculate complete exact distinct count
→ calculate HLL
```

If the exact distinct count has already been completed successfully, HLL provides no benefit.

The switch to HLL must happen **before** exact counting becomes too expensive.

---

## 6. What HLL is doing after the switch

HLL does not store every unique value.

Instead, each value is:

```text
converted into a consistent internal representation
    ↓
hashed
    ↓
used to update a small statistical sketch
```

The sketch contains a fixed number of small internal buckets called **registers**.

The precision setting controls the number of registers.

If the precision is:

```text
p = 14
```

then the sketch uses:

$$
2^{14}=16,384
$$

registers.

More registers generally mean:

```text
better accuracy
but
more memory
```

Fewer registers mean:

```text
less memory
but
more estimation error
```

A precision of 14 corresponds to a commonly cited conventional HLL relative standard error of roughly 0.8%.

That is a statistical error level, not a guarantee that every result will fall within exactly 0.8% of the true count.

Modern HLL/HLL++ libraries may use improved estimators and small-cardinality handling internally.

---

## 7. Exact and HLL counting must agree on what "same value" means

Before counting, define clearly what makes two values equal.

For example:

```text
"Australia"
"australia"
" Australia "
```

could either be treated as:

```text
three raw distinct values
```

or:

```text
one normalized value
```

depending on the library's normalization policy.

The cardinality component itself should not silently decide this.

Instead, values should first be converted into a consistent internal form.

The exact counter and HLL must use the **same converted representation**.

Otherwise they could disagree about what counts as the same value.

---

## 8. Null handling

Use one consistent rule across all backends.

Recommended default:

```text
null values are excluded from distinct cardinality
```

Track nulls separately.

For example:

```text
total rows      = 1,000,000
non-null rows   =   950,000
null rows       =    50,000
distinct values =       120
```

This makes the cardinality result easier to interpret.

---

## 9. Hashing rules

HLL relies on hashing, so hashing must be stable and consistent.

The implementation should use:

```text
the same hash algorithm
the same hash width
the same seed
the same value-conversion rules
```

whenever sketches might later be compared or merged.

Store this information as metadata.

Do not merge two HLL sketches if they were created using incompatible hashing or normalization rules.

Use a mature HLL/HLL++ implementation rather than building the hashing and estimator logic from scratch unless there is a strong engineering reason to do so.

---

## 10. What should the result contain?

Do not return only:

```text
4829311
```

Return enough information to tell the user what that number means.

For an exact result:

```text
cardinality: 3
method: exact
is_exact: true
```

For an HLL result:

```text
cardinality: 4,829,311
method: hyperloglog
is_exact: false
precision: 14
estimated_error: approximately 0.8%
```

Recommended result information:

```text
cardinality
raw estimate
method
is_exact

total row count
non-null count
null count

distinct ratio
duplicate ratio

HLL precision
HLL error information
lower/upper estimate bounds if available

whether promotion occurred
why promotion occurred
when promotion occurred

peak exact unique-value count
peak exact-state memory
```

This allows downstream consumers to distinguish an exact answer from an approximation.

---

## 11. Useful derived metrics

Once cardinality is known or estimated, derive:

### Distinct ratio

```text
distinct values
÷
non-null rows
```

Example:

```text
100 unique values
÷
1,000 rows
=
10% distinct
```

### Duplicate ratio

```text
1 - distinct ratio
```

If HLL produced the cardinality, these derived metrics should also be labelled approximate.

---

## 12. What if the data is processed in several partitions or workers?

The design must support merging partial results.

Different workers may process different parts of the same column.

Possible combinations are:

### Exact + Exact

Combine the two exact unique-value sets.

If the combined set still fits within the exact limits, remain exact.

If the merged set becomes too large, switch the combined result to HLL.

### Exact + HLL

Add the exact unique values into the HLL sketch.

The combined result is HLL.

### HLL + HLL

Merge the two compatible HLL sketches.

This mergeability is one of HLL's major strengths in distributed systems. HLL sketches can be combined without collecting all original distinct values in one place. Google describes HLL++ as parallelizing naturally and computing its estimate in a single pass, and Apache DataSketches highlights HLL as useful when distinct counting and merging are needed with very small space requirements.

---

## 13. Fallback when the backend cannot switch from exact to HLL mid-process

Some execution systems may not allow an aggregation to start as exact and dynamically become HLL.

For those systems, use a two-stage fallback:

```text
Stage 1
Run HLL to estimate cardinality
and collect row/null metrics
    ↓
Decide whether exact counting is likely to be cheap
    ↓
Stage 2
Run exact distinct count only for safe low-cardinality columns
```

This can require reading selected columns a second time.

Therefore it is a fallback, not the preferred design.

The preferred design remains:

```text
one pass
start exact
promote to HLL only when needed
```

---

## 14. What should be monitored?

Collect telemetry so the thresholds can be improved later.

Useful measurements include:

```text
percentage of columns completed exactly
percentage of columns promoted to HLL

promotion because of:
    unique-value limit
    memory limit

number of unique values at promotion
memory used at promotion

HLL precision
HLL sketch size

processing time
number of source passes
peak memory
```

These measurements will show whether the limits are too high or too low.

For example:

```text
many columns promote very late
```

may suggest that the exact threshold is too generous.

Conversely:

```text
many columns use HLL even though exact counting would have been cheap
```

may suggest that the limits are too conservative.

---

## 15. Validation

Periodically test HLL against exact results on datasets where exact counting is affordable.

Compare:

```text
exact cardinality
versus
HLL estimated cardinality
```

Calculate relative error:

$$
\text{relative error}
=
\frac{|\text{HLL estimate}-\text{exact count}|}
{\text{exact count}}
$$

Test across:

```text
small cardinalities
medium cardinalities
very large cardinalities

short values
long values

duplicate-heavy columns
mostly-unique columns

different HLL precision settings
different partition layouts
```

This gives real evidence for whether the selected HLL precision and promotion thresholds are suitable.

---

## 16. Minimum implementation tests

At minimum test:

```text
empty column
all-null column
one repeated value
small low-cardinality column
exact unique-value threshold
memory threshold
promotion from exact to HLL
duplicate-heavy data
very long values

exact + exact merge
exact + HLL merge
HLL + HLL merge

incompatible sketch rejection
stable hashing across runs
same result semantics regardless of input order
```

---

## 17. Persisted HLL sketches

If HLL sketches may be saved and reused later, store metadata with them.

Include:

```text
HLL algorithm/version
precision
hash algorithm
hash width
hash seed
normalization/canonicalization version
null policy
sketch payload
```

This prevents incompatible sketches from being merged accidentally.

Persisted sketches can be useful for pre-aggregated reporting.

For example:

```text
Monday HLL ┐
Tuesday HLL├── merge → weekly distinct estimate
Wednesday  ┘
```

The weekly calculation can then use the stored sketches instead of rereading all original rows.

---

## 18. Key rules to preserve

The implementation should always follow these rules:

1. **Use exact counting while it remains cheap.**
2. **Switch to HLL before the exact state becomes too expensive.**
3. **Do not run HLL after a complete exact count has already been obtained.**
4. **Exact and HLL must use the same definition of equality.**
5. **Once promoted to HLL, discard the exact state.**
6. **Do not describe HLL as sampling.**
7. **Clearly label approximate results as approximate.**
8. **Store precision and error information with HLL results.**
9. **Do not silently merge incompatible HLL sketches.**
10. **Keep backend-specific dataframe logic separate from the core counting policy.**

---

## 19. Recommended implementation order

Build the feature in this order:

```text
1. Define how values are normalized and compared.
2. Define null handling.
3. Define the result structure.
4. Implement bounded exact counting.
5. Integrate a mature HLL/HLL++ implementation.
6. Implement automatic promotion from exact to HLL.
7. Add memory tracking.
8. Implement merge behavior.
9. Add telemetry.
10. Add HLL persistence metadata if needed.
11. Add the two-stage fallback.
12. Benchmark and tune thresholds.
13. Validate HLL accuracy against exact counts.
14. Add backend-specific implementations last.
```

---

## 20. Final expected behaviour

The user should be able to provide a column without knowing anything about its cardinality beforehand.

The profiler should then automatically produce one of two outcomes:

```text
Exact result
when the unique-value state remains inexpensive
```

or:

```text
Approximate HLL result
when exact distinct tracking would become too expensive
```

That gives the library a safe default for arbitrary user-supplied data while preserving exact answers wherever practical.

---

## References

- Philippe Flajolet, Éric Fusy, Olivier Gandouet, Frédéric Meunier, **HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm**  
  https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf

- Stefan Heule, Marc Nunkesser, Alex Hall, **HyperLogLog in Practice: Algorithmic Engineering of a State of the Art Cardinality Estimation Algorithm**  
  https://research.google/pubs/hyperloglog-in-practice-algorithmic-engineering-of-a-state-of-the-art-cardinality-estimation-algorithm/

- Apache DataSketches, **DataSketches Library / HLL overview**  
  https://datasketches.apache.org/

