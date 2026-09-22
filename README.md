# gem-x.cpp

Turn webcam movements or recorded videos into an animated 3D human skeleton.
Built on [NVIDIA GEM-X](https://github.com/NVlabs/GEM-X), gem-x.cpp runs locally
and includes a browser demo for viewing the results.

- **Live webcam:** see your movements as a continuously updated skeleton, either
  overlaid on the camera image or in a 3D view.
- **Recorded video:** upload a clip or record one with your webcam, inspect the
  reconstructed motion, and export an animated GLB for use in 3D tools.

Applications can also consume timestamped SOMA or SMPL poses through the
[native streaming API](docs/MOTION-STREAMING.md). Robot control and the LocalAI
network streaming adapter are separate integrations.

## Use the demo

Once installed using the steps below, start the demo from the repository root:

```sh
./demo/gemx-demo --threads 8
```

Open **http://localhost:8098** in your browser.

### Live webcam

1. Select **Live webcam**, press **Start live**, and allow camera access.
2. Keep the camera stationary and your full body in view.
3. Switch between the camera overlay and 3D skeleton view to inspect the motion.

**Detect person every N frames** controls how often the person detector runs.
The default is 5; use 1 when moving around, or a larger interval when staying
in the same area to reduce processing work. Pose inference still runs on each
processed frame.

**Stop inference** returns the preview to your webcam. **Close camera** releases
the camera. Camera access needs localhost or HTTPS when accessing the demo
from another device.

### Recorded video

Select **Offline video**, then upload a video or record a webcam clip. Press **Build motion**, inspect the skeleton alongside the video, then select
**Download animated GLB**. Clips are limited to 120 sampled frames.

Offline processing uses the included `sam3d.cpp` submodule and its Body model
assets. Live webcam mode does not need them.
See the [demo guide](demo/README.md) for configuration and controls.

## Install

Linux is the tested platform. You need Git, a C++23 compiler (GCC 13+ or
Clang 17+), CMake 3.24+, Ninja, and Go 1.23+ for the browser demo.

### 1. Build

From your checkout, fetch the dependencies:

```sh
git submodule update --init --recursive
```

For GPU inference, install the Vulkan development packages, `glslc`, and SPIR-V
headers (`libvulkan-dev glslc spirv-headers` on Debian/Ubuntu), then build:

```sh
cmake --preset vulkan
cmake --build --preset vulkan -j8
```

Build the browser demo:

```sh
(cd demo && GOMAXPROCS=8 CGO_ENABLED=0 go build -p 8 -o gemx-demo .)
```

For CPU inference, build with the `release` preset instead and launch with:

```sh
./demo/gemx-demo --threads 8 \
  --pipeline build/release/gemx-pipeline \
  --backend CPU --module build/release/bin
```

### 2. Prepare the models

Download the models from [LocalAI-io/GEM-X-GGUF](https://huggingface.co/LocalAI-io/GEM-X-GGUF)
using the Hugging Face CLI (`hf`, from the `huggingface_hub` Python package):

```sh
hf download LocalAI-io/GEM-X-GGUF gem-x-contact-f32.gguf vitpose-f32.gguf yolox-f32.gguf --local-dir generated/reference
```

The demo expects these files by default:

```text
generated/reference/gem-x-contact-f32.gguf
generated/reference/vitpose-f32.gguf
generated/reference/yolox-f32.gguf
```

You can override model locations and select a GPU with command-line flags;
run `./demo/gemx-demo --help` for the options. For offline video, also build the
included SAM3D worker and prepare its Body models using the [offline setup instructions](demo/README.md#offline-dependency-setup).

### 3. Start the demo

Run `./demo/gemx-demo --threads 8` from the repository root, then open
**http://localhost:8098**. The default configuration uses the Vulkan build.

## More information

- [Demo guide](demo/README.md): camera controls, offline setup and server options.
- [Model card](distribution/README.md): conversion, required assets and model licenses.
- [Quality comparisons](docs/LIVE-OFFLINE-PARITY.md): upstream parity measurements and limitations.
- [Performance](docs/LIVE-PIPELINING.md): measured throughput and latency.
- [Development guide](docs/DEVELOPMENT.md): tests, installation, fuzzing and tooling.
- [Release audit](docs/RELEASE-AUDIT.md): fresh-build checks and validation evidence.

## License

Original code contributions are Apache-2.0. Model weights and third-party
components have separate terms. See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[licensing details](docs/LICENSING.md).
