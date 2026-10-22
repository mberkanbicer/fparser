# fparser-rs

This crate is the first Rust implementation unit for the fparser migration.
It is a dependency-free, `f64` parser/evaluator with the original expression
precedence, built-in real functions, degree mode, units, constants, variables,
inline variables, callbacks, epsilon comparisons, conditional evaluation, and
the original evaluator error codes. Logical operators evaluate both operands
as in the default C++ evaluator.

The existing C++ implementation remains the behavior oracle. This crate does
not yet implement the C++ bytecode format or the full generated optimizer.
`optimize()` currently performs constant folding and a measured subset of C++
rewrites. Verified rewrites can eliminate or coalesce callback invocations,
matching the original optimizer's observed behavior. Optional float,
long-double, integer, complex, MPFR, and GMP variants are out of scope for this
crate; the original C++ implementation remains their compatibility path.
It reproduces the original UTF-8
identifier and whitespace scanner without copying its generated function-name
table. Identifier classification is checked across all Unicode scalar values,
with selected C++ differential cases for multibyte identifiers and whitespace.
Malformed UTF-8 byte behavior is out of scope because the public API accepts
valid UTF-8 `str` inputs.

`add_parser_function` links an already-parsed child through `Arc<Parser>`.
Shared ownership prevents dangling links and keeps the child immutable after
it is linked; the original C++ API instead uses a non-owning parser pointer.
The child parser's evaluation errors propagate through the parent.
`Parser` is also `Clone`; clones have independent parse state and identifier
tables while sharing registered callback closures.

Run the crate checks from this directory with `cargo test`. With CMake and a
C++ compiler available, run `python tests/run_differential.py` to build the
pinned C++ implementation as an oracle and compare the shared regression
corpus against this crate. The runner also generates deterministic operator,
power, and built-in function matrices, including NaN and infinities, for
optimized and unoptimized evaluation,
plus all pairwise unary
built-in compositions over a six-value input grid in both modes and an
additional 12-value stress grid in both modes, plus 393 targeted optimizer
regressions. It also checks the optimized
`abs(tanh(1e100))` overflow case against the C++ implementation. The
118,098-case optimized triple-composition sweep passes on Windows/MSVC and
Ubuntu WSL/GCC after measured `cot(atan(x))` and `sqrt(exp2(x))` lowerings.
The separate 236,196-case 12-value triple stress audit passes on both systems.
The audit is broad but does not prove equivalence for every generated optimizer
rule. Reproduce the passing audit with
`python tests/run_differential.py --triple-audit` and the stress audit with
`python tests/run_differential.py --triple-stress-audit`; the latter reports
every mismatch and exits nonzero while gaps remain.

## API and compatibility

`Parser` is an idiomatic Rust API, not a source-compatible translation of
`FunctionParserBase`. Parsing and evaluation return typed `Result` errors
instead of updating mutable error state; `eval` takes a slice and reports a
variable-count mismatch. Variable deduction returns the sorted variable names.
Callbacks are registered as `Send + Sync + 'static` closures, and linked
parsers use `Arc<Parser>` to keep children alive. Cloning copies parser state
and shares callback closures and linked parsers; there is no `ForceDeepCopy`
equivalent.

Custom C++ `FunctionWrapper` hooks and debug bytecode injection/printing have
no direct Rust equivalent. This crate supports `f64` only; the C++ float,
long-double, integer, complex, MPFR, and GMP variants are not implemented.
The crate also defines no C ABI or Python/Matlab bindings; those integration
contracts belong to the later scripting-interface migration task.
