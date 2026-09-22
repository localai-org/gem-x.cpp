# Development and validation

The README's build commands use a standard C++23 toolchain. The runtime has no
Python dependency. After fetching the pinned GGML submodule, native configure,
build and contract tests need no network or model files. Go 1.23+ builds the demo
without third-party modules. Use at most eight build/test jobs and inference
threads; Linux `taskset -c 0-7` can additionally restrict the entire process tree
when those CPU IDs are available.

## Native tests and installation

```sh
cmake --preset debug
cmake --build --preset debug -j8
ctest --preset debug
cmake --install build/release --prefix /your/install/prefix
```

Debug enables ASan/UBSan. Install a compiler with its sanitizer runtimes
(Debian: `clang libclang-rt-dev` for the Clang fuzz configuration). Model-backed
reference executables additionally require the verified converted files;
CTest's model-free contracts are not full end-to-end model validation.
CPU backend directories select compatible GGML variants; Vulkan loads its
explicit module. Deployment must include the shared libraries and matching
backend modules. The install smoke check is recorded in the release audit.

## Non-GGUF fuzzing

```sh
cmake -S . -B build/fuzz -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGEMX_FUZZ=ON -DGEMX_SANITIZERS=ON -DGGML_CPU_ALL_VARIANTS=OFF
cmake --build build/fuzz -j8
python3 scripts/seed_fuzz.py build/fuzz/corpus
build/fuzz/gemx-fuzz-inputs build/fuzz/corpus/inputs -max_total_time=60 -max_len=16384
build/fuzz/gemx-fuzz-api build/fuzz/corpus/api -max_total_time=60 -max_len=8192
build/fuzz/gemx-fuzz-streaming -max_total_time=60 -max_len=4096
(cd demo && GOMAXPROCS=8 go test -race -p 8 ./...)
(cd demo && GOMAXPROCS=8 go test -parallel 4 -fuzz=FuzzImage -fuzztime=30s -run='^$')
(cd demo && GOMAXPROCS=8 go test -parallel 4 -fuzz=FuzzPackedBox -fuzztime=30s -run='^$')
(cd demo && GOMAXPROCS=8 go test -parallel 4 -fuzz=FuzzCreateJSON -fuzztime=30s -run='^$')
(cd demo && GOMAXPROCS=8 go test -parallel 4 -fuzz=FuzzDetectorBoxes -fuzztime=30s -run='^$')
```

The native harness calls the production stream parsers for S3DIMG01 RGB frames,
S3DOUT01 Body token data, GEMPOSE1/2, GEMIMGS1, GEMSEQ01/02, GEMMAN01, and CLI
numbers. It never opens paths from fuzz input. The API harness covers observation
and camera arrays, motion denormalization, 6D rotations, RGB capacities/strides,
ViTPose crops and YOLOX preprocessing. Caller-owned memory is bounded; arbitrary
invalid C pointers are outside the API contract. The Go targets cover JPEG/PNG
decoding/packing, box patching, detector box responses and job-description JSON.

These are bounded sanitizer campaigns, not a guarantee that all malformed
inputs are safe. GGUF loading is explicitly outside this campaign. Browser
video codecs, GGML's parsers and conversion-only libraries have their own input
surfaces. Conversion tools require hash-verified upstream artifacts and should
not be used as general-purpose untrusted checkpoint loaders. A path manifest
is a local CLI interface, not a sandbox for arbitrary filesystem access.

## Optional tooling and portability

Python conversion dependencies: Python 3.10+, NumPy, ONNX and `gguf`; PyTorch is
needed only for the verified checkpoint/contact and identity extraction steps.
OpenCV and ONNX Runtime are reference/comparison dependencies, not native build
dependencies. The browser QA helper is bundled and uses Python's standard
library plus an installed Chromium executable (`--chrome`). No sibling checkout
is needed for native builds, live demo, or browser QA. Offline Body inference
uses the pinned `sam3d.cpp/` submodule. Build its worker and prepare the Body models using the [offline setup](../demo/README.md#offline-dependency-setup);
external installations remain supported through path overrides.

Audit/replay tools accept `--device` and optional `--device-name`; model/module
paths are explicit arguments or repository-relative defaults. Profiler paths
remain caller-supplied. `/work` and `/nsys` in profiler scripts are container
mount destinations, not requirements on the host. GPU-specific tile guards are
intentional validated dispatch restrictions; they do not select the device.
Dated raw JSON reports preserve the original machine paths as provenance, not
configuration. The README links the current measurements rather than rewriting
historical benchmark evidence.

## CI and release evidence

The CI workflow builds CPU and Vulkan, runs model-free contracts and Go tests,
and performs bounded sanitizer fuzz smoke runs. It requires no model downloads.
See [the release audit](RELEASE-AUDIT.md) for the independent checkout/toolchain,
actual fuzz execution counts, install check and remaining limitations.

The resident pose API, optional model-backed lifecycle test, native example and
pinned SOMA-to-SMPL fixtures are described in [Motion streaming](MOTION-STREAMING.md).
