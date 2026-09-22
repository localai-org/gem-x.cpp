# Native motion streaming validation — 2026-09-22

The native parts of the [specification](MOTION-STREAMING-SPEC.md) are implemented
in `gemx_stream.h`. The demo worker uses the same pipeline. LocalAI's public
Protobuf/WebSocket transport and independent transport client tests remain outside
this repository; specification acceptance item 8 is not claimed complete.

## Checks performed

- GCC 14.2 release/Vulkan build and all seven CTest contracts passed on Debian 13.
- Clang 19 ASan/UBSan build and all seven contracts passed. The new non-GGUF
  streaming fuzz target completed **500,000 executions** without a finding.
  It mutates decoded joint/rotation arrays, counts, source time, sequence and
  temporal reset transitions. This complements existing RGB/parser/API fuzzing;
  it does not claim to fuzz valid resident GPU inference or the GGUF loader.
- The native reference test matched **18 independent pinned upstream conversion
  fixtures**: maximum position component error **1.90735e-6 m**, quaternion
  component error **1.19209e-7**, compared modulo sign. Limits are 1e-5 / 1e-6.
- A direct native client loaded all three models, printed a parseable versioned
  definition, accepted three RGB frames with explicit source times, emitted two
  SMPL poses and reset/destroyed its pipeline. It ran without the demo, Python,
  network access or SAM3D Body. JSON topology, transform counts and schema IDs
  were checked independently.
- The weight-dependent integration test ran **two concurrent resident instances**
  with interleaved submissions and compared their outputs exactly. It checked
  result ownership after reset/destruction, repeated creation, caller identity
  changes, timestamp gaps, size changes, explicit reset, invalid capacities,
  overflowing stride, duplicate sequence/time and int64 times above 2^53.
  Local-transform FK reproduced emitted SOMA positions within 1e-5 m while the
  inferred identity/scale changed. A 120-frame configuration also created and
  ran successfully (this test does not fill all 120 observations).
- The integration test exercised real YOLOX loss on blank RGB, fresh detection
  after loss, reacquisition warm-up and crop-reuse flags. Deterministic box-state
  tests cover multiple-person ambiguity and rejected identity switches. These
  establish policy semantics, not re-identification accuracy or broad camera
  quality. No new identity-quality claim is made for the conservative policy.
- C11 header compatibility and existing demo Go tests plus timestamp tests
  passed, including `go test -race ./...`. They check int64 preservation,
  monotonicity, fixed clock mode and exact forwarding to the worker. Camera sampling
  time is recorded before JPEG encoding/upload. The binary preview format remains
  unchanged; no UI redesign was needed.

CPU builds and model-free tests were exercised; full resident/model-backed runs
in this audit used Linux/Vulkan. Windows, macOS and other GGML backends were not
tested. Existing low-level/offline behavior and its prior parity scope are
unchanged; this audit does not establish new long-sequence offline parity.

## Live parity and memory replay

Hardware: NVIDIA GeForce RTX 5070 Ti 16 GB, driver 595.71.05. Vulkan strict F32
(`GGML_VK_DISABLE_F16/COOPMAT/COOPMAT2=1`), validated default ViTPose optimizations,
eight CPU cores, 30-frame window, detection every five processed frames, upstream
largest-person/full-image policy. Source images are 480×270 from the existing
pinned football replay. The three GGUFs are the existing F32 release models.

The final 144-frame replay produced **143 byte-identical complete GEMPOSE2 outputs**
against the previously accepted resident-worker report. This includes warm-up and
rolling-window eviction. No numerical threshold was relaxed. The first 143 poses
of the longer replay also match that reference exactly.

The longer replay processed **1,440 frames** (72 source images repeated 20 times).
It uses explicitly declared synthetic source spacing of 83,333 µs to isolate
execution time from temporal reset decisions. This is a test input timeline,
not a claim about the original camera's capture clock. Ingestion-time replay can
correctly reset if shader compilation or a processing stall exceeds the configured
source gap; recorded replay should always provide source times.

| Measurement | Result |
| --- | ---: |
| Model creation and READY | 1.741 s |
| Entire replay, excluding model creation | 101.952 s / 14.12 processed frames/s |
| Steady worker processing mean | 64.47 ms |
| Steady worker processing p50 / p95 / p99 | 53.16 / 108.57 / 109.95 ms |
| Reciprocal of steady processing mean | 15.51 frames/s |
| Host RSS at frame 30 | 322.2 MiB |
| Peak host RSS | 326.9 MiB |
| Host RSS throughout final 720 frames | 326.875 MiB (min = max) |
| Detector calls | 294 |

The whole replay includes graph/shader warm-up and host file transfer; the steady
processing rows begin when the window reaches 30. These are processing durations,
not camera-to-display latency. The extra detector calls reflect immediate retries
after misses. No HTTP/JPEG/camera work is included. Model creation is measured
separately from first-use shader/graph compilation.

RSS is the worker's host resident memory, sampled after each frame; it is **not GPU
allocation size**. GPU/driver caches were not separately attributed per instance.
Each instance owns three weight sets and bounded graph caches; see the
[resource contract](MOTION-STREAMING.md#demo-and-resource-behavior). The stable tail
and bounded implementation show no accumulating frame queue in this replay, not a
proof of constant total driver memory for every shape/device combination.

A compact [measurement record](measurements/motion-streaming-20260922.json) is
committed. Raw reports, per-frame hashes and stages remain under
`generated/streaming-audit/{parity-final,long-replay}` in the validation workspace.
The new converter fixture and its provenance are committed, so normal CTest needs
no weights or Python.

## Reproduce

Use the normal release/Vulkan build instructions. All compilation and runtime
work in this audit was restricted to CPUs 0–7. Substitute suitable eight CPUs
for machines with different affinity assignments.

```sh
cmake --build --preset vulkan -j8
ctest --preset vulkan

GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
  taskset -c 0-7 build/vulkan/gemx-streaming-models-test \
  generated/reference/gem-x-contact-f32.gguf \
  generated/reference/vitpose-f32.gguf generated/reference/yolox-f32.gguf \
  build/vulkan/bin/libggml-vulkan.so Vulkan 0 \
  "$FIRST_PACKED_FRAME" "$SECOND_PACKED_FRAME"

# Two distinct human frames with valid source-space boxes in S3DIMG01 headers
# are needed for the optional shape/lifecycle test above.
taskset -c 0-7 python3 scripts/profile_live_worker.py \
  --frames "$PACKED_FRAME_DIRECTORY" --output generated/streaming-replay \
  --detect-interval 5 --repeats 20 --source-step-us 83333

# Add --compare <accepted-report.json> to a same-length replay to require
# byte-for-byte equality. Output directories must not already exist.
```

The native example and manifest format are in the [API guide](MOTION-STREAMING.md).
For sanitizer/fuzz builds follow the [development guide](DEVELOPMENT.md), then run:

```sh
taskset -c 0-7 build/fuzz/gemx-fuzz-streaming -runs=500000 -max_len=4096
```

To regenerate converter fixtures, fetch the four source files listed in
`reference/streaming-sources.json` from its **pinned revision** into one directory.
The generator checks their hashes and the SOMA rig hash, executes upstream
functions on CPU (not the native converter), and writes deterministic cases:

```sh
USE_JIT_TORCH_TRANSFORM=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  taskset -c 0-7 python3 scripts/generate_streaming_reference.py \
  "$PINNED_UPSTREAM_DIRECTORY" "$SOMA_NEUTRAL_NPZ"
```

The fixture was generated with CPU PyTorch 2.7.1+cu128. JIT decorators are removed
from the selected helper functions to avoid importing unrelated simulation
modules; their arithmetic and the upstream `SomaToSmpl.convert` body are unchanged.
A stub supplies decoded SOMA joints, so this is an adapter oracle, not an upstream
body-model inference test. Source SPDX attribution is retained in the native
adapter and `NOTICE`.
