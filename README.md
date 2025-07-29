[![Rust](https://github.com/RyanMarcus/arrow-compile-compute/actions/workflows/rust.yml/badge.svg)](https://github.com/RyanMarcus/arrow-compile-compute/actions/workflows/rust.yml)

An experimental project for testing LLVM-compiled kernels for Arrow.

Compared to `arrow`, this crate:

* ➕ Operates on *recursive* Arrow arrays, for example, a dictionary-of-dictionary array.
* ➕ Casting to and from arbitrary Arrow array types, like casting a dictionary to a run-end encoded array.
* ➕ Always uses the SIMD operations available at runtime, no recompilation required.
* ➖ Requires LLVM as a dependency.
* ➖ Startup overhead the first time a kernel is invoked (~10s of ms).
* ➖ Not production ready -- this is a research project.
