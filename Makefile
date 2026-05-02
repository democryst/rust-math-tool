.PHONY: all test bench fuzz doc check example clean fmt

# ─── Config ──────────────────────────────────────────────────────────────────
FUZZ_TARGET  := fuzz_target_1
FUZZ_TIME    := 10
EXAMPLE      := xor_training

# ─── Default ──────────────────────────────────────────────────────────────────
all: check test

# ─── Build ────────────────────────────────────────────────────────────────────
build:
	cargo build

build-release:
	cargo build --release

# ─── Testing ──────────────────────────────────────────────────────────────────
## Run all unit and integration tests
test:
	cargo test

## Run a specific test by name (usage: make test-one NAME=test_autodiff_basic)
test-one:
	cargo test $(NAME)

# ─── Code Quality ─────────────────────────────────────────────────────────────
## Run clippy linter with warnings-as-errors for CI strictness
check:
	cargo clippy -- -D warnings 2>&1 || true

## Auto-format source code
fmt:
	cargo fmt

# ─── Documentation ────────────────────────────────────────────────────────────
## Build rustdoc and open in browser
doc:
	cargo doc --no-deps --open

## Build rustdoc without opening
doc-build:
	cargo doc --no-deps

# ─── Benchmarks ───────────────────────────────────────────────────────────────
## Run criterion performance benchmarks
bench:
	cargo bench

# ─── Examples ─────────────────────────────────────────────────────────────────
## Run the XOR training example
example:
	cargo run --example $(EXAMPLE) --release

# ─── Fuzzing ──────────────────────────────────────────────────────────────────
## Run a short fuzzing session (default: 10 seconds)
fuzz:
	cargo +nightly fuzz run $(FUZZ_TARGET) -- -max_total_time=$(FUZZ_TIME)

## Run a longer fuzzing session (1 minute)
fuzz-long:
	cargo +nightly fuzz run $(FUZZ_TARGET) -- -max_total_time=60

## Reproduce a specific fuzz crash artifact
fuzz-repro:
	cargo +nightly fuzz run $(FUZZ_TARGET) $(ARTIFACT)

# ─── CI Gate ──────────────────────────────────────────────────────────────────
## Run the full Integrity Gate pipeline (lint → test → fuzz)
gate: check test fuzz
	@echo ""
	@echo "✅ Integrity Gate PASSED — all checks cleared."

# ─── Cleanup ──────────────────────────────────────────────────────────────────
## Remove build artifacts
clean:
	cargo clean

## Remove fuzz corpus and artifacts
fuzz-clean:
	rm -rf fuzz/corpus fuzz/artifacts

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "rust-math-tool — Developer Commands"
	@echo "────────────────────────────────────────"
	@echo "  make test          Run all tests"
	@echo "  make check         Clippy lint"
	@echo "  make fmt           Auto-format"
	@echo "  make doc           Build + open rustdoc"
	@echo "  make bench         Run benchmarks"
	@echo "  make example       Run XOR training example"
	@echo "  make fuzz          Run 10s fuzz session"
	@echo "  make fuzz-long     Run 60s fuzz session"
	@echo "  make gate          Full Integrity Gate (lint+test+fuzz)"
	@echo "  make clean         Remove build artifacts"
	@echo ""
