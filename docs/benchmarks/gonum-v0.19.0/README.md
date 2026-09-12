# Gonum v0.19.0-fork upgrade: performance evidence

Measured on 2026-09-12 for PR #171. The upgrade has mixed performance on this
machine. The complete 236-case timing geomean improves 0.75%, but reproducible
slowdowns remain. This is not a regression-free performance approval. Assess the
tradeoff or repair the affected backend paths before merging and publishing the
prepared v1.5.1 release.

## Comparison

- Baseline: `fdda4f150d13b34a969de7faed6ae896e827ce94`, Gonum `v0.17.7-fork`.
- Candidate: `54d5d6b119c8b4351db0cd0bc0ebb4122bde94d4`, Gonum `v0.19.0-fork`.
- The baseline was the exact PR parent and freshly fetched `origin/main`.
- Apple M1 Pro, darwin/arm64, Go 1.26.4, `GOMAXPROCS=8`.
- Default Gonum Go BLAS and LAPACK implementations; no backend overrides,
  custom build tags, GOFLAGS, or PGO profiles. Native compilation, no emulation.
- Same unchanged benchmark fixtures on both revisions. Both test binaries were
  built before timing. No agent builds, tests, or profiling ran concurrently.
  This was a normal desktop host, without CPU affinity or power-state control.
- Ten samples per case per revision, alternating baseline/candidate and
  candidate/baseline order. Full suite: 100ms per benchmark; focused checks:
  300ms. All benchmark processes exited successfully with matching case sets.
- `benchstat`: `golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`.

## Results

The longer follow-up covered the 14 full-sweep slowdowns of at least 3%, plus
DC motor simulation, regulator construction, and Lyapunov allocation behavior.
One nested benchmark needed a separate slash-separated selection. Medians below
come from the longer samples; all listed changes have p < 0.001, except the
10,000-point SISO FRD case (p=0.009). Full tables include confidence intervals.

| Benchmark | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: |
| MatLog_N50 | 330.3 us | 372.0 us | +12.62% |
| D2C_ZOH_N50 | 413.3 us | 454.3 us | +9.92% |
| Stabsep_N100 | 8.904 ms | 9.455 ms | +6.18% |
| SystemFRD_SISO_10000 | 220.2 us | 233.7 us | +6.15% |
| Reduce | 45.94 us | 48.43 us | +5.42% |
| Modsep_N50 | 1.429 ms | 1.504 ms | +5.29% |
| Stabsep_N50 | 1.385 ms | 1.458 ms | +5.27% |
| DiscretizeZOH | 21.48 us | 22.58 us | +5.15% |
| MatLog_N20 | 49.37 us | 51.78 us | +4.88% |
| ControllabilityStaircase | 36.36 us | 37.94 us | +4.36% |
| D2C_ZOH_N20 | 61.47 us | 63.94 us | +4.02% |
| Canon_Modal_N50 | 640.5 us | 660.9 us | +3.19% |
| Stabsep_N10 | 32.70 us | 33.47 us | +2.37% |
| Simulate_DCMotor | 46.69 us | 39.72 us | -14.93% |
| Reg_N100_M5_P5 | 79.72 us | 60.92 us | -23.58% |

The `FrequencySweepKernels/N4_W2/TransferFunction` slowdown did not reproduce
with longer samples (p=0.447). Lyap_N100 timing was inconclusive (p=0.353).
Absence of statistical significance does not establish equivalence. The full
sweep contains additional smaller changes; see its complete table rather than
using this selected follow-up as an aggregate performance score. Multiple case
comparisons can produce false positives, especially for small effects.

Allocations also vary: DC motor simulation drops from 45 to 11 allocs/op and
regulator construction from 55 to 25. The focused SISO FRD case rises from 60 to
61, and Stabsep_N100 from 481 to 482. Lyap_N100's full-sweep increase from 363 to
364 did not reproduce in the focused run (both medians 363), although its bytes
per operation increased 0.08%. Tables retain all memory measurements.

These data establish behavior only on this host at eight workers. They do not
establish performance on amd64, other worker counts, or arbitrary models, and do
not attribute a specific backend change as the cause. Correctness tests and race
tests passed separately; performance measurements do not replace those tests.

## Evidence and reproduction

- `full-benchstat.txt`: all 236 cases, timing and allocations.
- `focused-benchstat.txt`: 16 longer follow-up cases.
- `subcase-benchstat.txt`: the nested transfer-function follow-up.
- Matching `*-baseline.txt.gz` and `*-candidate.txt.gz`: complete raw samples.
- `metadata.json`: full-sweep binary hashes, flags, environment, and run order.

From the repository, build isolated copies of the exact revisions, then run the
existing Go benchmarks. Prebuilding is equivalent to `go test -run '^$'
-bench=. -benchmem -benchtime=100ms -count=1`, with compilation outside timing.

```bash
set -eu
bench_dir=$(mktemp -d)
mkdir "$bench_dir/baseline" "$bench_dir/candidate"
git archive fdda4f150d13b34a969de7faed6ae896e827ce94 | tar -x -C "$bench_dir/baseline"
git archive 54d5d6b119c8b4351db0cd0bc0ebb4122bde94d4 | tar -x -C "$bench_dir/candidate"
(cd "$bench_dir/baseline" && go test -c -o "$bench_dir/baseline.test")
(cd "$bench_dir/candidate" && go test -c -o "$bench_dir/candidate.test")
export GOMAXPROCS=8
for round in $(seq 1 10); do
  order='baseline candidate'
  if [ $((round % 2)) -eq 0 ]; then order='candidate baseline'; fi
  for label in $order; do
    "$bench_dir/$label.test" -test.run='^$' -test.bench=. \
      -test.benchmem -test.benchtime=100ms -test.count=1 \
      >> "$bench_dir/$label.txt"
  done
done
benchstat "$bench_dir/baseline.txt" "$bench_dir/candidate.txt"
```

For focused reproduction, use 300ms and select the top-level names in
`focused-benchstat.txt` with `^Benchmark(Name1|Name2|...)$`. For the nested case,
use `^BenchmarkFrequencySweepKernels$/^N4_W2$/^TransferFunction$` separately.
To reanalyze retained data, decompress each pair with `gzip -dc` and pass the
resulting files to the recorded benchstat version.
