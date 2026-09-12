#!/usr/bin/env python3
"""Build immutable controlsys/Gonum snapshots and compare prebuilt binaries."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

CONTROL = "54d5d6b119c8b4351db0cd0bc0ebb4122bde94d4"
OLD = "f43007ad8a2d208bc8f47c338e85f05381d9643b"
NEW = "a78cf83eff160df0e423b08af0c7642ade964150"
MASTER = "1c42629c5cb53235d9ae074bb1be6da6471771da"
CASES = "MatLog_N50|D2C_ZOH_N50|Stabsep_N100|SystemFRD_SISO_10000|Reduce|Modsep_N50|DiscretizeZOH|Simulate_DCMotor|Reg_N100_M5_P5"


def capture(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def archive(repo, revision, destination):
    destination.mkdir(parents=True)
    data = subprocess.check_output(["git", "-C", str(repo), "archive", revision])
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        source.extractall(destination, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gonum", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--controlsys", type=Path, required=True)
    parser.add_argument("--go", default="go")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["releases", "allocator", "bisect"], default="releases")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--benchtime", default="300ms")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    go = shutil.which(args.go)
    if not go:
        parser.error("Go executable not found")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    variants = [("old", OLD), ("new", NEW), ("master", MASTER),
                ("no_gemv", NEW), ("no_gemm", NEW)]
    cases = CASES
    if args.suite == "allocator":
        variants = [("allocator_default", NEW), ("nosizemalloc", NEW)]
    elif args.suite == "bisect":
        variants = [("old", OLD), ("merge", "7fd5fefe"), ("worker", "ebe982da"),
                    ("calibrated", "777029ff"), ("new", NEW),
                    ("old_align64", OLD), ("new_align64", NEW)]
        cases = "MatLog_N50|D2C_ZOH_N50|Stabsep_N100"
    metadata = {
        "controlsys": CONTROL, "go": capture([go, "version"]),
        "go_env": json.loads(capture([go, "env", "-json", "GOOS", "GOARCH", "GOFLAGS", "GOEXPERIMENT", "GOTOOLCHAIN"])),
        "runtime_env": {k: os.environ.get(k, "") for k in ["GODEBUG", "GOMEMLIMIT", "GOGC"]},
        "GOMAXPROCS": 8, "rounds": args.rounds, "benchtime": args.benchtime,
        "bench": "^Benchmark(" + cases + ")$", "variants": {}, "order": [],
    }
    for label, revision in variants:
        revision = capture(["git", "-C", str(args.gonum), "rev-parse", revision + "^{commit}"])
        directory = output / label
        gonum, control = directory / "gonum", directory / "controlsys"
        archive(args.gonum, revision, gonum)
        archive(args.controlsys, CONTROL, control)
        reverted = {"no_gemv": "blas/gonum/level2float64.go", "no_gemm": "blas/gonum/dgemm.go"}.get(label)
        if reverted:
            (gonum / reverted).write_bytes(subprocess.check_output(
                ["git", "-C", str(args.gonum), "show", OLD + ":" + reverted]))
        env = dict(os.environ)
        if label == "nosizemalloc":
            if env.get("GOEXPERIMENT"):
                raise RuntimeError("allocator comparison requires an unset GOEXPERIMENT")
            env["GOEXPERIMENT"] = "nosizespecializedmalloc"
        subprocess.run([go, "mod", "edit", "-replace=gonum.org/v1/gonum=" + str(gonum)], cwd=control, check=True)
        binary = directory / "controlsys.test"
        command = [go, "test", "-c", "-o", str(binary)]
        if label.endswith("_align64"):
            command.append("-ldflags=-funcalign=64")
        subprocess.run(command, cwd=control, env=env, check=True)
        metadata["variants"][label] = {
            "gonum": revision, "reverted_file": reverted, "build": command,
            "GOEXPERIMENT": env.get("GOEXPERIMENT", ""),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "binary_version": capture([go, "version", "-m", str(binary)]),
        }
        print("built", label, flush=True)
    expected = None
    labels = [label for label, _ in variants]
    for round_index in range(args.rounds):
        order = labels if round_index % 2 == 0 else labels[::-1]
        metadata["order"].append(order)
        for label in order:
            command = [str(output / label / "controlsys.test"), "-test.run=^$",
                       "-test.bench=" + metadata["bench"], "-test.benchmem",
                       "-test.benchtime=" + args.benchtime, "-test.count=1"]
            result = subprocess.run(command, cwd=output, env={**os.environ, "GOMAXPROCS": "8"},
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with (output / (label + ".txt")).open("a") as stream:
                stream.write(result.stdout)
            result.check_returncode()
            names = re.findall(r"^(Benchmark\S+)\s+\d+\s+", result.stdout, re.MULTILINE)
            if not names or (expected is not None and names != expected):
                raise RuntimeError("empty or mismatched benchmark selection: " + label)
            expected = names
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("round", round_index + 1, flush=True)


if __name__ == "__main__":
    main()
