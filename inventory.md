# Kernel Inventory

Comparison against [`arrow::compute`](https://docs.rs/arrow/59.1.0/arrow/compute/index.html) — **arrow-rs 59.1.0**.

> Note: this repo now pins Arrow crates at **59.1** in `Cargo.toml`, matching
> the comparison target used here.

---

## Supported — `arrow::compute` operations covered by a JIT kernel

| Arrow operation | Kernel in this repo | Notes |
|---|---|---|
| `add`, `sub`/`sub_wrapping`, `mul`/`mul_wrapping`, `div`, `rem` | `BinOpKernel` | array-vs-array or array-vs-scalar, all numeric types |
| `neg_wrapping` | `UnaryOpKernel` | arithmetic negation, all numeric types; wrapping semantics only (no checked `neg`) |
| `abs` | `UnaryOpKernel` | absolute value; value semantics match pyarrow's C++ `AbsoluteValue` on shared types — signed-int overflow wraps (`abs(i32::MIN) == i32::MIN`), unsigned is identity, floats use IEEE `fabs`; wrapping only (no checked `abs`). Type-domain differs from pyarrow: **no** decimal128/256 or duration (currently panics via `for_arrow_type`), and **does** accept float16 (pyarrow does not register `abs` for it). |
| `sqrt` | `UnaryOpKernel` | square root; non-checked (negative → `NaN`, no `sqrt_checked` variant). Float input preserves type (`f32 → f32`, `f64 → f64`); integer input is promoted to `float64` (matching pyarrow). IEEE special cases follow `llvm.sqrt` (`sqrt(inf) = inf`, `sqrt(NaN) = NaN`, `sqrt(-0.0) = -0.0`). Type-domain differs from pyarrow: **no** decimal128/256 (panics via `for_arrow_type`); float16 input is accepted and kept as `f16` here (pyarrow does not compute sqrt natively on half-float). |
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
| `min`, `max` | `ReductionKernel`, `MinMaxAggKernel` | ungrouped and grouped; nulls skipped |
| `sum` | `SumAggregator` | grouped; ungrouped via `Aggregator::ingest_ungrouped`; nulls skipped |
| `product` | `ProductAggregator` | grouped; ungrouped via `Aggregator::ingest_ungrouped`; nulls skipped; wrapping semantics only (no `product_checked`) |
| `like` | `compile_string_like` | GLOB-style matching with escape char (case-sensitive only) |
| `contains` | `string_contains` | substring search |
| `starts_with`, `ends_with` | `StringStartEndKernel` | prefix / suffix match |

---

## Extensions — kernels in this repo with no `arrow::compute` equivalent

| Operation | Kernel | Description |
|---|---|---|
| Between / bounds check | `BetweenKernel`, `BoundsKernel` | fused `lo <= x <= hi` in one pass (Arrow has no `between`) |
| Argmin / argmax | `ReductionKernel` (`argmin`, `argmax`) | index of the min/max — Arrow has no argmin/argmax |
| Grouped aggregation | `CountAggregator`, `SumAggregator`, `ProductAggregator`, `MinMaxAggKernel`, `MostRecentAggregator` (+ their `*MergeKernel`s) | SQL-style GROUP BY with mergeable partial states for parallel aggregation — Arrow has no group-by |
| Hash / grouping | `HashKernel` | Murmur and unchained CRC32 hashing into a ticket table |
| Sort key normalization | `normalize_columns` | maps raw sort keys to a canonical ordinal encoding |
| Binary search | `lower_bound` | bisect on a sorted column |
| Run-length estimate | `approx_max_run_length` | estimates the largest run for REE planning |
| Vector — dot product | `DotKernel` | dot product of two fixed-size-list columns |
| Vector — L2 norm | `NormVecKernel` | norm of a fixed-size-list column |
| Vector — nearest neighbor | `NearestNeighborKernel` | nearest-neighbor search over a fixed-size-list column |

---

## Not yet supported — `arrow::compute` 59.1 operations with no direct Arrow-compatible JIT kernel

This section is organized by the module names used under
`arrow::compute::kernels`. The "candidate existing implementation" column is
intentionally strict: it only lists code that is close enough to reuse for a
direct implementation. Otherwise it says no close implementation was found.

### `arrow::compute::kernels::boolean`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `and` | element-wise AND | `DSLBitwiseBinOp::And` is the same value operation internally; needs a public wrapper and normal null propagation. |
| `and_kleene` | element-wise AND with Kleene null logic | Partial: `DSLBitwiseBinOp::And` covers the value operation, but Kleene validity logic still needs to be added. |
| `or` | element-wise OR | `DSLBitwiseBinOp::Or` is the same value operation internally; needs a public wrapper and normal null propagation. |
| `or_kleene` | element-wise OR with Kleene null logic | Partial: `DSLBitwiseBinOp::Or` covers the value operation, but Kleene validity logic still needs to be added. |
| `not` | element-wise NOT | `DSLExpr::bit_not` supports booleans internally; needs a public wrapper. |
| `and_not` | element-wise AND NOT | Directly expressible as `DSLBitwiseBinOp::And` plus `DSLExpr::bit_not`; needs a public wrapper and null semantics. |
| `is_null`, `is_not_null` | boolean mask of null / non-null positions | `logical_nulls` already computes logical validity/null buffers; implementation would mostly wrap/invert that into a `BooleanArray`. |

### `arrow::compute::kernels::nullif`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `nullif` | null out entries where a condition holds | `replace_nulls` can attach the final null buffer once it is derived from the input validity and condition mask. |

### `arrow::compute::kernels::bitwise`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `bitwise_and_not` | element-wise bit ops (+ `_scalar` variants) | `DSLBitwiseBinOp::{And, Or, Xor}` and `DSLExpr::bit_not` are the same internal operations; needs public kernels and scalar dispatch. |
| `bitwise_shift_left`, `bitwise_shift_right` | element-wise shifts (+ `_scalar` variants) | No close implementation found. |

### `arrow::compute::kernels::aggregate`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `product_checked` | overflow-checked product | Partial: `ProductAggregator` has the same accumulation structure for wrapping product; checked multiply/overflow handling would be new. |
| `sum_checked` | overflow-checked sum | Partial: `SumAggregator` has the same accumulation structure for wrapping/widened sum; checked addition/overflow handling would be new. |
| `sum_array`, `sum_array_checked` | list-element sums | No close implementation found. `SumAggregator` reduces rows/groups, not elements inside list rows. |
| `bit_and`, `bit_or`, `bit_xor` | reduce an array via bitwise AND/OR/XOR | No close reduction implementation found. Existing DSL bitwise ops are element-wise only. |
| `bool_and`, `bool_or` | all-true / any-true reductions | `DSLReductionType::{And, Or}` already implements the reduction core; needs a public Arrow aggregate wrapper and null semantics. |

### `arrow::compute::kernels::comparison`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `distinct`, `not_distinct` | null-aware equality | `ComparisonKernel` provides the equality/inequality value operation; null-aware distinct semantics can be layered with `logical_nulls`. |
| `in_list`, `in_list_utf8` | membership test against a value set | No close implementation found. |
| `nlike` | negated LIKE | `compile_string_like` plus boolean `bit_not` is a direct implementation path. |
| `ilike`, `nilike` | case-insensitive LIKE / negated case-insensitive LIKE | Partial: `compile_string_like` has the LIKE pattern engine, but case-insensitive matching/case folding is not implemented. |
| `eq_ignore_ascii_case` | case-insensitive ASCII equality | No equivalent kernel found; no lower/upper-case normalization kernel is present. |

### `arrow::compute::kernels::rank`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `rank` | assign a rank to each element by sort order | `sort_col` / `sort_multi_col` produce the sorted order; rank can be built by assigning rank values back through those indices. Tie handling would be new. |

### `arrow::compute::kernels::window`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `shift` | shift elements left/right, filling with null | `TakeKernel` can gather shifted element indices; implementation mainly needs index construction plus a null-padding validity mask. |

### `arrow::compute::kernels::zip`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `zip` | select from one array or another per a boolean mask | `InterleaveKernel` can implement this after converting the boolean mask to source-array indices and row indices. |

### `arrow::compute::kernels::merge`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `merge`, `merge_n` | merge pre-sorted runs into one sorted output | No close implementation found. `InterleaveKernel` can only materialize a precomputed merge order; it does not compute that order. |

### `arrow::compute::kernels::union_extract`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `union_extract`, `union_extract_by_id`, `union_extract_by_type` | extract a child from a union array | No close implementation found. |

### `arrow::compute::kernels::length`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `length`, `bit_length` | character / byte length | String/binary iterators expose byte slices, so byte length and bit length are straightforward. UTF-8 character length still needs character counting. |

### `arrow::compute::kernels::substring`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `substring` | byte/offset substring extraction | String/binary iterators plus string writers are close enough for a byte-slice substring kernel. |
| `substring_by_char` | character-aware substring extraction | Partial: string iterators/writers are reusable, but UTF-8 character-boundary scanning would be new. |

### `arrow::compute::kernels::concat_elements`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `concat_elements_*` | element-wise string/binary concatenation | String/binary writers are reusable for the output, but `concat_all` is whole-array concatenation and is not the row-wise operation. |

### `arrow::compute::kernels::regexp`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `regexp_is_match`, `regexp_is_match_scalar`, `regexp_match` | regex match / capture | No close implementation found. `compile_string_like` and `string_contains` are not regex engines. |

### `arrow::compute::kernels::temporal`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `date_part` | extract year/month/day/hour/... from a temporal array | No close implementation found. |

### `arrow::compute::kernels::numeric`
| Function | Description | Candidate existing implementation |
|---|---|---|
| `multiply_fixed_point`, `rescale_decimal` | decimal-specific arithmetic and rescaling | No close implementation found. `BinOpKernel` covers primitive numeric arithmetic, not decimal fixed-point scale handling. |

---

## Not supported, but unnecessary — functions that don't benefit from JIT

| Function / family | Reason |
|---|---|
| `unary`, `binary`, `try_unary`, `try_binary` (+ `_mut`) | low-level building blocks for writing kernels, not end-user ops |
| `can_cast_types` | type-level check, touches no data |
| `num_cast`, `cast_num_to_bool` | scalar cast helpers; array-level numeric and numeric-to-bool casts are already covered by `CastKernel` |
| `filter_record_batch`, `take_record_batch`, `concat_batches`, `interleave_record_batch`, `take_arrays` | thin per-column wrappers; the column kernels are what matter |
| `prep_null_mask_filter`, `partition_validity` | trivial bitmask fixups |
| `FilterBuilder` / `optimize`, `BatchCoalescer`, `garbage_collect_dictionary` | internal optimization / allocation machinery, not compute |
| `make_comparator` | builds a comparator closure; the sort kernels JIT this inline |
| Schema / metadata / builder APIs | struct manipulation and allocation, not compute-bound |
