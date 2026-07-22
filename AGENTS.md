## Common commands

This is a standard Rust project. 

* Run tests: `cargo test`
* Run benchmarks: `cargo bench`
* Check for build errors: `cargo check`

## Code style

* Use `unsafe` only when it leads to large performance improvements or is otherwise required (e.g., JIT).
* Do not introduce new helper functions that are only called from a single place.

## Large changes

If you are making a large change that might be difficult for a human to review,
and the user requested you to use `jj`, following the below instructions. Do not
use `jj` if the user did not request it.

Use `jj`, not mutating Git commands.

Implement the work as a linear stack of reviewable jj changes:

1. Inspect the stack with `jj log`.
2. Propose the ordered changes before editing.
3. Implement one semantic change at a time.
4. Finish each change with `jj commit -m "<description>"`.
5. Keep every change independently buildable and testable.
6. Do not collapse the stack into one change.
7. At the end, show each change ID, description, diff summary, and tests run.

Each change should be a reviewable unit of code. The first change does not need
to fully implement the user's request. 

Use `jj split`, `jj squash`, or `jj edit` when changes need to be
redistributed. Do not use `git commit`, `git rebase`, `git reset`, or
`git checkout`.
