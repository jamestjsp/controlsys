# Full Go 1.27.1 follow-up

Measured on 2026-09-12 for controlsys PR #171. The Go 1.27.1 configuration
substantially reduces the large Go 1.26.4 regressions and retains simulation and
regulator improvements. Residual slowdowns remain: this is a mixed-performance
upgrade, with a release decision still needed before merge/publication.

## Controlled comparison

Apple M1 Pro, darwin/arm64, Go 1.27.1, GOMAXPROCS=8, default Gonum Go BLAS/LAPACK,
no custom build tags, GOFLAGS, PGO, alignment flags, or GOEXPERIMENT overrides.
A normal desktop host was used without affinity or power-state control.

- Candidate source: controlsys `2027e5d`, Gonum `v0.19.0-fork`.
- Baseline source: controlsys `fdda4f150d13b34a969de7faed6ae896e827ce94`, Gonum
  `v0.17.7-fork`, with the same Go 1.27.1 directive and candidate `model_array.go`.
- Go 1.27's required `go fix` changed the reverse index loop to `slices.Backward`.
  That rewrite was applied identically to both variants. All 191 Go files were
  verified byte-identical between variants; candidate source matched the PR.
- Both binaries were built before timing. No local builds, tests, or profiling
  ran concurrently with acceptance measurements.
- Full suite: 236 cases, ten samples per version, 100ms per case. Follow-up:
  eleven cases, ten samples per version, 300ms per case. Revision order alternated
  baseline/candidate and candidate/baseline. All runs passed with matching cases.
- benchstat: `golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`.

This isolates the dependency under the final Go 1.27 configuration; it does not
isolate a compiler-only change against the earlier Go 1.26 run. The separate
[issue investigation](../issue10-investigation/README.md) examines that question.

## Results

The complete timing geomean improves **1.07%**. A geomean is not application
throughput and does not imply that every workload improves. The longer follow-up
covered all full-suite slowdowns of at least 3%, the seven original issue cases,
and representative gains. Follow-up medians:

| Benchmark | v0.17.7-fork | v0.19.0-fork | Time change | p |
| --- | ---: | ---: | ---: | ---: |
| SimulateWithDelay_SISO | 18.31 us | 18.93 us | +3.38% | <0.001 |
| SimulateInternalDelay | 20.52 us | 21.42 us | +4.38% | <0.001 |
| SystemFRD_SISO_10000 | 217.6 us | 231.0 us | +6.17% | <0.001 |
| MatLog_N50 | 331.9 us | 335.6 us | +1.09% | <0.001 |
| D2C_ZOH_N50 | 413.7 us | 417.9 us | +1.02% | 0.001 |
| Stabsep_N100 | 8.612 ms | 8.658 ms | +0.54% | 0.015 |
| Reduce | 45.33 us | 45.81 us | +1.07% | 0.009 |
| DiscretizeZOH | 21.23 us | 21.36 us | inconclusive | 0.305 |
| Modsep_N50 | 1.433 ms | 1.428 ms | -0.36% | 0.011 |
| Simulate_DCMotor | 43.88 us | 38.05 us | -13.29% | <0.001 |
| Reg_N100_M5_P5 | 79.78 us | 59.68 us | -25.20% | <0.001 |

The earlier 12.62%, 9.92%, and 6.18% regressions in MatLog, D2C, and Stabsep
are greatly reduced, but small significant differences remain in this final
configuration. The original focused investigation's absence of significance
should not be read as proof of equivalence. Different source/build layouts and
measurement sessions can change these small effects; no specific cause of the
remaining slowdowns has been established here.

Delay benchmarks have unchanged allocation counts. The SISO FRD case rises from
60 to 61 allocs/op, Stabsep_N100 from 480 to 484. DC motor simulation drops from
45 to 11 and regulator construction from 55 to 25. Full memory results and
confidence intervals are retained in the tables. The full sweep includes smaller
changes outside the selected follow-up; multiple comparisons can yield false
positives, particularly for small effects. Other CPUs and worker counts remain
unmeasured.

## Validation and release implications

Go 1.27.1 passes `go fix ./...`, `go vet ./...`, `go build ./...`, full uncached
tests, race tests, the local downstream-consumer action, and GitHub push/PR CI.
The current CLI `govulncheck` v1.8.0 reports no vulnerabilities. The bundled gopls
v0.22.0 scanner could not load the newer Go version; the CLI replaced that check.

The PR now requires Go 1.27.1 in go.mod and CI. This raises the minimum Go version
from 1.26.3 and must be disclosed in release notes. The Gonum pin remains the
released `v0.19.0-fork`, not master. The prepared release remains a draft: assess
or repair the residual 3–6% regressions before publication. This configuration
improves the performance assessment but does not close Gonum issue #10 or repair
the older-toolchain executable-layout behavior.

## Evidence and reproduction

`full-benchstat.txt` and `focused-benchstat.txt` retain complete tables. Matching
`*-baseline.txt.gz` and `*-candidate.txt.gz` contain the raw samples.
`metadata.json` records full-sweep hashes, flags, order, and source adjustments.

Run this Bash recipe from controlsys with Go 1.27.1 to reproduce the controlled
source comparison. Both variants use the final candidate source; only the Gonum
replacement differs. Compile before timing and keep other CPU-heavy work idle.

```bash
set -eu
bench_dir=$(mktemp -d)
for label in baseline candidate; do
  mkdir "$bench_dir/$label"
  git archive 2027e5d | tar -x -C "$bench_dir/$label"
done
(cd "$bench_dir/baseline" && go mod edit \
  -replace=gonum.org/v1/gonum=github.com/jamestjsp/gonum@v0.17.7-fork && go mod tidy)
for label in baseline candidate; do
  (cd "$bench_dir/$label" && go test -c -o "$bench_dir/$label.test")
done
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

For the focused follow-up, use 300ms and:

```text
^Benchmark(SimulateWithDelay_SISO|SimulateInternalDelay|SystemFRD_SISO_10000|MatLog_N50|D2C_ZOH_N50|Stabsep_N100|Reduce|Modsep_N50|DiscretizeZOH|Simulate_DCMotor|Reg_N100_M5_P5)$
```

For reanalysis without rerunning, decompress each retained pair using `gzip -dc`
and pass the resulting files to the recorded benchstat version.
