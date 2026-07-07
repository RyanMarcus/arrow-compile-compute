# Kernel Inventory

Comparison against [arrow::compute](https://docs.rs/arrow/latest/arrow/compute/index.html) (arrow-rs 55.2.0).

---

## Supported — kernels in this repo that cover Arrow compute operations

| Operation | Kernel | Notes |
|---|---|---|
| **Arithmetic** | `BinOpKernel` | add, sub, mul, div, rem — array vs array or scalar, all numeric types |
| **Comparison** | `ComparisonKernel` | eq, ne, lt, lte, gt, gte — numeric + string |
| **Cast** | `CastKernel` | numeric↔numeric, binary↔utf8, boolean↔numeric, primitive→dict, dict→primitive, dict→StringView, REE value cast, FixedSizeList element cast |
| **Filter** | `FilterKernel` | all Arrow array types |
| **Take** | `TakeKernel` | all Arrow array types, all integer index types |
| **Concat** | `concat_all` | all Arrow array types |
| **Interleave** | `InterleaveKernel` | all Arrow array types |
| **Partition** | `PartitionKernel` | group consecutive equal rows, all Arrow array types |
| **Sort** | `sort_col`, `sort_multi_col` | single-column and multi-column sort, returns index array |
| **Top-K** | `top_k` | partial sort returning the K smallest/largest elements |
| **Reduce** | `ReductionKernel` | min, max, argmin, argmax — ungrouped |
| **String predicates** | `compile_string_like`, `string_contains`, `StringStartEndKernel` | like (GLOB-style), ilike, contains, starts_with, ends_with |

---

## Extensions — kernels in this repo with no Arrow compute equivalent

These operations don't exist in Arrow compute at all; they are additional capabilities this repo provides.

| Operation | Kernel | Description |
|---|---|---|
| **Bounds check** | `BoundsKernel` | Tests whether each value falls in \[lo, hi\] — a single fused kernel vs two comparisons |
| **Lower bound** | `lower_bound` | Binary search (bisect) on a sorted column |
| **Grouped aggregation** | `CountAggregator`, `SumAggregator`, `MinAggregator`, `MaxAggregator`, `MostRecentAggregator` | Hash-table grouped aggregation (SQL-style GROUP BY) |
| **Hash / grouping** | `HashKernel` | Murmur and unchained CRC32 hashing into a ticket table for dictionary building |
| **Sort normalization** | `normalize_columns` | Maps raw sort keys to a canonical ordinal representation |
| **Vector — dot product** | `DotKernel` | Dot product of two fixed-size list columns |
| **Vector — norm** | `NormVecKernel` | L2 norm of a fixed-size list column |
| **Vector — nearest neighbor** | `NearestNeighborKernel` | Nearest-neighbor search over a fixed-size list column |

---

## Not yet supported — Arrow compute operations not covered by this repo

### Logical / Boolean
| Arrow function | Description |
|---|---|
| `and`, `and_kleene` | Element-wise boolean AND (with Kleene null logic) |
| `or`, `or_kleene` | Element-wise boolean OR (with Kleene null logic) |
| `not` | Element-wise boolean NOT |
| `and_not` | Element-wise AND NOT |

### Unary arithmetic (arrow-arith)
| Arrow function | Description |
|---|---|
| `negate` | Arithmetic negation |
| `abs` | Absolute value |
| `pow` | Element-wise power |
| `sqrt`, `exp`, `ln`, `log2`, `log10` | Exponential / logarithm |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh` | Trigonometric functions |
| `floor`, `ceil`, `round`, `trunc` | Rounding |
| `signum` | Sign of value |

### Bitwise aggregation
| Arrow function | Description |
|---|---|
| `bit_and`, `bit_or`, `bit_xor` | Reduce an array to a single value via bitwise AND/OR/XOR |

### Boolean aggregation
| Arrow function | Description |
|---|---|
| `bool_and` | True if all non-null inputs are true |
| `bool_or` | True if any non-null input is true |

### String operations (arrow-string)
| Arrow function | Description |
|---|---|
| `length` / `bit_length` | Character / byte length of each string |
| `upper`, `lower` | Case conversion |
| `ltrim`, `rtrim`, `trim` | Whitespace trimming |
| `pad_left`, `pad_right` | String padding |
| `substr` / `substring` | Substring extraction |
| `repeat` | Repeat a string N times |
| `replace` | Find-and-replace within strings |
| `reverse` | Reverse string characters |
| `regexp_is_match`, `regexp_match` | Regular-expression matching and capture group extraction |

### Null handling
| Arrow function | Description |
|---|---|
| `is_null`, `is_not_null` | Returns boolean array marking null / non-null positions |
| `nullif` | Sets entries to null where a condition holds |

### Sorting
| Arrow function | Description |
|---|---|
| `rank` | Assigns a rank to each element based on sort order |

### Array manipulation
| Arrow function | Description |
|---|---|
| `shift` | Shifts array elements left or right, filling with null |

### Date / time
| Arrow function | Description |
|---|---|
| `date_part` | Extracts a component (year, month, day, hour, …) from a timestamp or date |

### Decimal arithmetic
| Arrow function | Description |
|---|---|
| `multiply_fixed_point`, `rescale_decimal` | Decimal-specific multiplication and precision rescaling |

### Union arrays
| Arrow function | Description |
|---|---|
| `union_extract` | Extracts the value at a named field from a union array |

---

## Not supported, but unnecessary — Arrow compute functions that don't benefit from JIT kernels

| Function | Reason |
|---|---|
| Schema / metadata operations | Pure Rust struct manipulation, no data computation |
| Array construction / builders | Allocation-bound, not compute-bound |
| `can_cast_types` | Type-level check with no data touched |
| `concat_batches`, `filter_record_batch` | Thin wrappers that call column-by-column; the per-column kernels are what matters |
| `unary`, `binary`, `try_unary`, `try_binary` | Low-level building blocks for writing kernels, not end-user operations |
| `prep_null_mask_filter` | Trivial bitwise fixup |
| `partition_validity` | Separates null/non-null index sets; memory-bound, not compute-bound |
