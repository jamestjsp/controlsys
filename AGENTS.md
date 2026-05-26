# Controlsys

For domain terminology, modeling assumptions, and project language, read
`CONTEXT.md` before making design, documentation, or behavior changes.

## gopls MCP

Use `gopls mcp -instructions` as the source of truth. `gopls mcp` runs detached over stdio and only sees saved files.

When `gopls` MCP tools are available for Go work, start with `go_workspace`, use `go_symbol_references` before changing symbol definitions, and run `go_diagnostics` on edited Go files before tests. Omit `files` for workspace diagnostics; do not pass Markdown files. If `go.mod` changes, run `go_vulncheck`.

If the MCP tools are not exposed in Codex, say so and use CLI checks instead.

## Static Checks

Run `go fix` the same way you run `go vet`:

```bash
go fix ./...
go vet ./...
```

## Build & Test

```bash
go test -v -count=1
```

## Key Conventions

- gonum LAPACK uses **row-major** flat arrays (`a[row*n+col]`), not column-major like Fortran
- Always test with non-symmetric A matrices (diagonal A hides transposition bugs)
- No inline comments unless logic is counter-intuitive

## Performance

- Always benchmark before/after with `go test -bench=. -benchmem` and `benchstat`
- Prefer `m.RawMatrix().Data` with stride-aware indexing over `At(i,j)`/`Set(i,j)`
- Use `copy()` on raw slices for submatrix extraction, not element-wise loops
- Pre-allocate buffers outside loops; reuse LAPACK work arrays across calls
