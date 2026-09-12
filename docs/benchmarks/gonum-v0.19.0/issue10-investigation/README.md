# Issue #10: arm64 regression investigation

Investigated 2026-09-12 on Apple M1 Pro (darwin/arm64), GOMAXPROCS=8.
[Issue #10](https://github.com/jamestjsp/gonum/issues/10) reports downstream
regressions between v0.17.7-fork and v0.19.0-fork.

## Findings

The large Go 1.26.4 regressions reproduce. Their first substantial increase is
at `ebe982daa841625254b141121265738410ba42ee`, the worker-aware DGEMM dispatch
change. The immediately preceding commit `7fd5fefe` does not show the large
slowdowns. The subsequent Darwin threshold calibration preserves them.

The evidence strongly supports executable-layout sensitivity, not an incorrect
parallel-work threshold:

- The MatLog_N50 trace contains no Dgemv calls, including setup. Its GEMM shapes
  are 51×51×51 (setup), 50×50×50 and 36×36×64. All remain serial under both
  releases because they occupy fewer than four output tiles.
- Triangular solves include a 64×36 panel with lda=ldb=100 and a 100×50 solve
  with lda=100, ldb=50. Dtrsm source is unchanged between releases. Normalized
  disassembly has the same 888 instructions and identical hot-body operations;
  differing operands are addresses of panic strings/types in cold paths.
  Dtrsm starts at 0x100127980 in the original baseline executable and
  0x100128410 in the candidate, changing its position modulo 64 from 0 to 16.
  Other unchanged functions move too. This does not identify Dtrsm alone as
  the cause of the whole-program effect.
- Relinking the same Go 1.26.4 code with `-ldflags=-funcalign=64` removes almost
  all of the three largest regressions. This changes layout without reverting
  dispatch, changing arithmetic, or undoing the scaling fixes.
- With default Go 1.27.1 builds, none of the seven reported regression cases
  has a statistically significant slowdown in this ten-sample follow-up.
  Absence of significance is not proof of equivalence; smaller effects and
  other workloads remain possible.

No production numerical code was changed. Go 1.27.1 was installed as the host's
default toolchain. The older Go 1.26.4 installation was preserved for diagnosis.
The global alignment flag is an experiment, not a proposed build requirement.
The issue remains reproducible with the older compiler's default layout.

## Reproduction and bisection

All entries below are medians from ten interleaved samples per binary.
The release reproduction uses 300ms per case; bisection uses 200ms.

| Benchmark | Go 1.26.4 old release | New release | Change |
| --- | ---: | ---: | ---: |
| MatLog_N50 | 335.2 us | 377.7 us | +12.67% |
| D2C_ZOH_N50 | 413.1 us | 453.2 us | +9.69% |
| Stabsep_N100 | 9.030 ms | 9.626 ms | +6.60% |
| Reduce | 46.47 us | 49.19 us | +5.85% |
| DiscretizeZOH | 21.35 us | 22.58 us | +5.74% |
| Modsep_N50 | 1.453 ms | 1.535 ms | +5.63% |

All six comparisons have p≤0.005. The FRD timing regression did not reproduce
(p=0.579), although its median allocation count rose from 60 to 61. Current
master also reproduces the large timing regressions with Go 1.26.4.

| Go 1.26.4 revision/build | MatLog_N50 | D2C_ZOH_N50 | Stabsep_N100 |
| --- | ---: | ---: | ---: |
| Old release `f43007ad` | 327.7 us | 409.2 us | 8.857 ms |
| Merge `7fd5fefe` | 330.1 us | 411.9 us | 8.846 ms |
| Worker-aware gate `ebe982da` | 370.1 us | 450.3 us | 9.398 ms |
| Darwin calibration `777029ff` | 369.5 us | 450.2 us | 9.452 ms |
| New release `a78cf83e` | 369.7 us | 451.1 us | 9.421 ms |
| Old release, function alignment 64 | 327.1 us | 408.2 us | 8.892 ms |
| New release, function alignment 64 | 329.2 us | 410.6 us | 8.874 ms |

`bisect-go1.26.4/benchstat.txt` contains all uncertainty and allocation results.
This is attribution of the trigger in these executables, not evidence that
changing the threshold back is the appropriate repair. GEMM reversion loses the
confirmed simulation and regulator gains.

## Go 1.27.1 and allocation specialization

Both release endpoints and current master were rebuilt with Go 1.27.1.
The five-way comparison also includes isolated reversions of the old Dgemv
implementation and old dgemm.go into the new release. These are experimental
ablation builds, not proposed source changes.

| Benchmark | Old release | New release | Current master |
| --- | ---: | ---: | ---: |
| MatLog_N50 | 335.4 us | 340.0 us | 338.7 us |
| D2C_ZOH_N50 | 416.1 us | 418.7 us | 419.1 us |
| Stabsep_N100 | 8.672 ms | 8.697 ms | 8.674 ms |
| Simulate_DCMotor | 44.44 us | 38.12 us | 37.99 us |
| Reg_N100_M5_P5 | 80.56 us | 61.20 us | 60.60 us |

The new release retains the simulation improvement (-14.22%) and regulator
improvement (-24.02%), with the same 45→11 and 55→25 allocation reductions
observed under Go 1.26.4. These allocation-count changes come from the Gonum
comparison, not from installing the new compiler. Some smaller allocation
increases remain; consult the complete tables rather than timing alone.

Go 1.27 adds size-specialized allocation calls for small objects; see the
[official release notes](https://go.dev/doc/go1.27#runtime). A separate paired
comparison uses identical new-release source with the feature enabled by
default and disabled with `GOEXPERIMENT=nosizespecializedmalloc`:

- Disabled Reduce is 1.88% slower (p<0.001): the default feature helps this case.
- Disabled MatLog and D2C are 1.25% and 1.41% faster (p<0.001).
- Stabsep is inconclusive (p=0.579); other results are in
  `allocator-go1.27.1/benchstat.txt`.

The feature therefore does not explain the disappearance of the large original
regressions. Toggling it changes emitted code/layout as well as allocation
calls, so these are whole-program effects, not isolated allocator timings.
No persistent GOEXPERIMENT override was set.

## Evidence and limitations

- `go1.26.4/`: original three-way release/master comparison, 10×300ms.
- `go1.27.1/`: five-way release/master/ablation comparison, 10×300ms.
- `allocator-go1.27.1/`: allocator default versus disabled, 10×300ms.
- `bisect-go1.26.4/`: five commit states plus two alignment builds, 10×200ms.
- Each directory contains full raw output, benchstat results, binary SHA-256
  hashes, resolved commits, toolchain versions and alternating run order.
- `matlog-trace.txt`: instrumented one-iteration call-shape trace, including
  untimed fixture setup. Its timing is not performance evidence.
- `code-comparison.txt`: normalized instruction-count and address diagnostics.
  The reported diff counts include cold panic-address materialization.
- `profiles/`: separate five-second MatLog CPU profiles with Go 1.26.4. The
  macOS profiles are dominated by runtime/system samples and are not used to
  make a quantitative CPU attribution.

The unchanged controlsys fixture source is commit
`54d5d6b119c8b4351db0cd0bc0ebb4122bde94d4`; all comparisons replace only its
Gonum dependency. The Gonum endpoints are commits, not annotated tag objects:
`f43007ad8a2d208bc8f47c338e85f05381d9643b` and
`a78cf83eff160df0e423b08af0c7642ade964150`. Current master is
`1c42629c5cb53235d9ae074bb1be6da6471771da`.

Benchstat: `golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`.

Normal desktop host, no affinity or power-state control; native default Go
BLAS/LAPACK, no build tags or PGO. All binaries in a timed comparison were
prebuilt, with no concurrent builds, tests or profiling. The interrupted
Go 1.26.4 ablation run was discarded. Comparisons across the two toolchains'
separate sessions are descriptive; each source/feature comparison within a
session is interleaved. No other platforms or worker counts were measured.
This focused investigation does not replace the original 236-case suite or
establish a regression-free release on all workloads.

## Run again

Requires Python 3.12+, Go, and local repositories containing the pinned commits.
The harness uses fresh snapshots and refuses to overwrite its output directory.
It checks exit status and matching nonempty benchmark sets. Builds finish before
any timed runs. Use an absolute older Go executable for the old-toolchain test;
do not change the host default back.

```sh
uv run python3 docs/benchmarks/gonum-v0.19.0/issue10-investigation/run.py \
  --gonum /path/to/gonum \
  --controlsys /path/to/controlsys --go /path/to/go1.27.1/bin/go \
  --suite releases --output /tmp/issue10-releases
uv run python3 docs/benchmarks/gonum-v0.19.0/issue10-investigation/run.py \
  --gonum /path/to/gonum \
  --controlsys /path/to/controlsys --go /path/to/go1.27.1/bin/go \
  --suite allocator --output /tmp/issue10-allocator
uv run python3 docs/benchmarks/gonum-v0.19.0/issue10-investigation/run.py \
  --gonum /path/to/gonum \
  --controlsys /path/to/controlsys --go /path/to/go1.26.4/bin/go \
  --suite bisect --benchtime 200ms --output /tmp/issue10-bisect
benchstat /tmp/issue10-releases/old.txt /tmp/issue10-releases/new.txt
```

The new `BenchmarkDtrsmSmallRectangular` in `blas/gonum/dtrsmbench_test.go`
provides bounded, known-solution coverage for the traced solve shapes. It times
RHS restoration plus the public solve and validates the fixture outside timing.
It is a kernel guardrail, not a replacement for the downstream executable when
investigating layout sensitivity.

Validation on Go 1.27.1:

- `go test ./blas/gonum ./lapack/gonum ./mat` passed.
- The unchanged controlsys snapshot with current Gonum master passed `go test ./...`.
- All four new benchmark fixtures passed a one-iteration smoke check, with zero
  timed allocations. These single samples are not performance claims.
- The retained harness completed its allocator suite with one iteration per
  benchmark and one round; this is a functionality check, excluded from tables.

This directory preserves the local Gonum investigation for the downstream PR.
The original Gonum checkout and its uncommitted benchmark are unchanged.
Pass `--gonum` explicitly when running the copied harness from controlsys.
