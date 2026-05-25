# Controlsys

For domain terminology, modeling assumptions, and project language, read
`CONTEXT.md` before making design, documentation, or behavior changes.

## Build & Test

```bash
go test -v -count=1
```

## Go Modernization

- When updating to a newer Go toolchain, start from a clean git state and run `go fix ./...`; repeat until it reaches a fixed point before reviewing the mechanical diff.
- Keep generated rewrites only after the normal test and benchmark gates stay clean; roll back hot-path rewrites that regress performance.

## Key Conventions

- gonum LAPACK uses **row-major** flat arrays (`a[row*n+col]`), not column-major like Fortran
- Always test with non-symmetric A matrices (diagonal A hides transposition bugs)
- No inline comments unless logic is counter-intuitive

## Performance

- Always benchmark before/after with `go test -bench=. -benchmem` and `benchstat`
- Prefer `m.RawMatrix().Data` with stride-aware indexing over `At(i,j)`/`Set(i,j)`
- Use `copy()` on raw slices for submatrix extraction, not element-wise loops
- Pre-allocate buffers outside loops; reuse LAPACK work arrays across calls
