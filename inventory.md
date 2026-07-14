# Kernel Inventory

Comparison against [`arrow::compute`](https://docs.rs/arrow/59.1.0/arrow/compute/index.html) — **arrow-rs 59.1.0**.

`arrow::compute` is a re-export facade: it does not implement kernels itself but
re-exports them from `arrow-arith`, `arrow-ord`, `arrow-select`, `arrow-string`,
and `arrow-cast`. Rows below are derived from those crate sources at 59.1.0,
**scoped to what is actually re-exported under `arrow::compute`** (so `arrow-cast`
display / pretty-printing / base64 / string-parsing helpers, which live under
`arrow::util` rather than `arrow::compute`, are excluded), cross-checked against
the `arrow::compute` module docs and this repo's public surface
(`src/arrow_interface.rs` plus the kernels exported from `src/compiled_kernels/`).
Nothing here is inferred from what a compute engine "usually" has.

> Note: the repo currently pins arrow-rs **55.2** in `Cargo.toml`. The compute
> surface between 55.2 and 59.1 is nearly identical — 59.1 adds `product`,
> `merge`/`merge_n`, `union_extract_by_id`/`by_type`, `eq_ignore_ascii_case`,
> `num_cast`, `rescale_decimal`, and the `Decimal32`/`Decimal64` types; the
> functions that appear "removed" (standalone temporal accessors,
> `regexp_is_match_utf8*`, `build_compare`, `build_filter`) were consolidated,
> not dropped. None of the differences change the rows below.

---

## Supported — `arrow::compute` operations covered by a JIT kernel

| Arrow operation | Kernel in this repo | Notes |
|---|---|---|
| `add`, `sub`/`sub_wrapping`, `mul`/`mul_wrapping`, `div`, `rem` | `BinOpKernel` | array-vs-array or array-vs-scalar, all numeric types |
| `neg_wrapping` | `UnaryOpKernel` | arithmetic negation, all numeric types; wrapping semantics only (no checked `neg`) |
| `abs` | `UnaryOpKernel` | absolute value; value semantics match pyarrow's C++ `AbsoluteValue` on shared types — signed-int overflow wraps (`abs(i32::MIN) == i32::MIN`), unsigned is identity, floats use IEEE `fabs`; wrapping only (no checked `abs`). Type-domain differs from pyarrow: **no** decimal128/256 or duration (currently panics via `for_arrow_type`), and **does** accept float16 (pyarrow does not register `abs` for it). |
| `eq`, `neq`, `lt`, `lt_eq`, `gt`, `gt_eq` | `ComparisonKernel` | numeric + string |
| `cast`, `cast_with_options` | `CastKernel` | numeric↔numeric, binary↔utf8, boolean↔numeric, primitive↔dict, dict→StringView, REE value cast, FixedSizeList element cast |
| `filter`, `filter_record_batch` | `FilterKernel` | all array types |
| `take`, `take_arrays`, `take_record_batch` | `TakeKernel` | all array types, all integer index types |
| `concat`, `concat_batches` | `concat_all` | all array types |
| `interleave`, `interleave_record_batch` | `InterleaveKernel` | all array types |
| `partition` | `PartitionKernel` | groups consecutive equal rows |
| `sort_to_indices` | `sort_col` | single-column sort → index array |
| `lexsort_to_indices` | `sort_multi_col` | multi-column sort → index array |
| `sort_limit` / `partial_sort` | `top_k` | K smallest/largest as indices |
| `min`, `max` | `ReductionKernel`, `MinMaxAggKernel` | ungrouped and grouped |
| `sum` | `ReductionKernel`, `SumAggregator` | ungrouped and grouped |
| `like` | `compile_string_like` | GLOB-style matching with escape char (case-sensitive only) |
| `contains` | `string_contains` | substring search |
| `starts_with`, `ends_with` | `StringStartEndKernel` | prefix / suffix match |

---

## Extensions — kernels in this repo with no `arrow::compute` equivalent

| Operation | Kernel | Description |
|---|---|---|
| Between / bounds check | `BetweenKernel`, `BoundsKernel` | fused `lo <= x <= hi` in one pass (Arrow has no `between`) |
| Argmin / argmax | `ReductionKernel` (`argmin`, `argmax`) | index of the min/max — Arrow has no argmin/argmax |
| Grouped aggregation | `CountAggregator`, `SumAggregator`, `MinMaxAggKernel`, `MostRecentAggregator` (+ their `*MergeKernel`s) | SQL-style GROUP BY with mergeable partial states for parallel aggregation — Arrow has no group-by |
| Hash / grouping | `HashKernel` | Murmur and unchained CRC32 hashing into a ticket table |
| Sort key normalization | `normalize_columns` | maps raw sort keys to a canonical ordinal encoding |
| Binary search | `lower_bound` | bisect on a sorted column |
| Run-length estimate | `approx_max_run_length` | estimates the largest run for REE planning |
| Vector — dot product | `DotKernel` | dot product of two fixed-size-list columns |
| Vector — L2 norm | `NormVecKernel` | norm of a fixed-size-list column |
| Vector — nearest neighbor | `NearestNeighborKernel` | nearest-neighbor search over a fixed-size-list column |

---

## Not yet supported — `arrow::compute` 59.1 operations with no JIT kernel

### Logical / boolean (`arrow-arith::boolean`)
| Function | Description |
|---|---|
| `and`, `and_kleene` | element-wise AND (with Kleene null logic) |
| `or`, `or_kleene` | element-wise OR (with Kleene null logic) |
| `not` | element-wise NOT |
| `and_not` | element-wise AND NOT |

### Null handling
| Function | Description |
|---|---|
| `is_null`, `is_not_null` | boolean mask of null / non-null positions |
| `nullif` | null out entries where a condition holds |

### Bitwise, element-wise (`arrow-arith::bitwise`)
| Function | Description |
|---|---|
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `bitwise_and_not` | element-wise bit ops (+ `_scalar` variants) |
| `bitwise_shift_left`, `bitwise_shift_right` | element-wise shifts (+ `_scalar` variants) |

### Aggregation (`arrow-arith::aggregate`)
| Function | Description |
|---|---|
| `product`, `product_checked` | multiply all elements to one scalar |
| `sum_checked`, `sum_array`, `sum_array_checked` | overflow-checked / list-element sums |
| `bit_and`, `bit_or`, `bit_xor` | reduce an array via bitwise AND/OR/XOR |
| `bool_and`, `bool_or` | all-true / any-true reductions |

### Comparison / set membership / rank (`arrow-ord`)
| Function | Description |
|---|---|
| `distinct`, `not_distinct` | null-aware equality |
| `in_list`, `in_list_utf8` | membership test against a value set |
| `rank` | assign a rank to each element by sort order |

### Array manipulation (`arrow-select`)
| Function | Description |
|---|---|
| `shift` | shift elements left/right, filling with null |
| `zip` | select from one array or another per a boolean mask |
| `merge`, `merge_n` | merge pre-sorted runs into one sorted output |
| `union_extract`, `union_extract_by_id`, `union_extract_by_type` | extract a child from a union array |

### String / binary (`arrow-string`)
| Function | Description |
|---|---|
| `length`, `bit_length` | character / byte length |
| `substring`, `substring_by_char` | substring extraction |
| `concat_elements_*` | element-wise string/binary concatenation |
| `ilike`, `nlike`, `nilike` | case-insensitive LIKE, negated LIKE, negated case-insensitive LIKE |
| `eq_ignore_ascii_case` | case-insensitive ASCII equality |
| `regexp_is_match`, `regexp_is_match_scalar`, `regexp_match` | regex match / capture |

### Temporal (`arrow-arith::temporal`)
| Function | Description |
|---|---|
| `date_part` | extract year/month/day/hour/… from a temporal array |

### Cast / decimal (`arrow-cast`, re-exported into `arrow::compute`)
| Function | Description |
|---|---|
| `num_cast`, `cast_num_to_bool` | scalar / numeric-to-bool conversions |
| `multiply_fixed_point`, `rescale_decimal` | decimal-specific arithmetic and rescaling |

---

## Not supported, but unnecessary — functions that don't benefit from JIT

| Function / family | Reason |
|---|---|
| `unary`, `binary`, `try_unary`, `try_binary` (+ `_mut`) | low-level building blocks for writing kernels, not end-user ops |
| `can_cast_types` | type-level check, touches no data |
| `filter_record_batch`, `take_record_batch`, `concat_batches`, `interleave_record_batch`, `take_arrays` | thin per-column wrappers; the column kernels are what matter |
| `prep_null_mask_filter`, `partition_validity` | trivial bitmask fixups |
| `FilterBuilder` / `optimize`, `BatchCoalescer`, `garbage_collect_dictionary` | internal optimization / allocation machinery, not compute |
| `make_comparator` | builds a comparator closure; the sort kernels JIT this inline |
| Schema / metadata / builder APIs | struct manipulation and allocation, not compute-bound |
