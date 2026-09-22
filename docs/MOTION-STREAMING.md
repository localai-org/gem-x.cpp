# Native motion streams

Use `libgemx` to turn timestamped RGB frames into human poses without running the
browser demo or a worker process. Include [`gemx_stream.h`](../include/gemx_stream.h)
and link `gemx`. Live inference needs the GEM-X, ViTPose and YOLOX F32 GGUFs;
it does not load SAM3D Body, Python, SONIC, Protobuf or network libraries.

This implements the native portion of the
[motion streaming specification](MOTION-STREAMING-SPEC.md). LocalAI's public
`/api/motion/...` endpoints, Protobuf schema, subscriber queues and transport
interoperability tests remain an **external integration**, not shipped here.
No robot controller, simulation or physical-base calibration is included.

## Lifecycle and ownership

1. `gemx_live_create` loads the three models once. All configuration is explicit:
   paths, backend module/directory, CPU or Vulkan backend, device index and optional
   expected description, 1–8 CPU threads, 2–120 observations, detector interval
   1–30, selection policy, precision policy, `GEMX_LIVE_FRAME_INDEX` temporal policy and
   maximum source gap. Other temporal policies are rejected.
2. Read `gemx_live_definition` once. Call with `NULL,0` to query the byte count,
   allocate that many bytes (including NUL), then copy its versioned JSON. It
   reports effective configuration, capabilities, topology, neutral reference
   transforms, units, bases, quaternion orders and channel sizes. Hosts should
   publish selected skeleton definitions, not private model filesystem paths.
3. `gemx_live_submit` borrows RGB bytes for one synchronous call. Supply dimensions,
   row stride and byte capacity. Frames are 8–32766 pixels on each side, at most
   16 million pixels. Optional subject boxes are four finite in-image `xyxy`
   coordinates. Caller subject IDs identify deliberate selection changes.
4. Read integer metadata with `gemx_live_result_info`. Copy desired float channels
   with `gemx_live_result_copy`, using the constants in the header. Its NULL/zero
   query returns element counts. Float arrays flatten joint-major, then component.
   Warm-up/loss results have no pose arrays; they are not all-zero poses.
5. Destroy each result explicitly. It owns its data and survives later submissions,
   resets and destruction of its pipeline. No internal pointers are borrowed.
6. Reset to begin a new epoch without reloading weights. Destroy the pipeline when
   done. Calls on a pipeline serialize; independent pipelines have independent
   state and model/backend allocations. Weight sharing is not implemented.

`GEMX_OK` means an accepted result, whose outcome is `WARMUP`, `POSE`, `LOST` or
`AMBIGUOUS`. `RESET` is a flag and may accompany any outcome. Invalid arguments
return `GEMX_INVALID_ARGUMENT`, leave temporal state unchanged and return no
result. Allocation/backend failures return an error and invalidate temporal state
if inference has begun. Do not retry a failed estimate as if continuity survived.
Buffer capacities must describe actual readable/writable allocations; the C API
cannot validate the physical allocation behind an arbitrary non-null pointer.

The first pose needs two usable observations. There is no native frame queue,
background inference or cancellation token. Reset waits for the current call;
destroy must not race a call. A host can stop admitting work and discard a result,
but cannot immediately interrupt GPU inference.

## Timing and selection

Sequence and nonnegative `int64_t` microsecond source time are returned unchanged.
The caller declares its stream origin (for example the first camera sample or
media time zero). Within an epoch both must strictly increase; sequence gaps are
allowed. Reset explicitly before seeking backward or changing calibration/camera.
Duplicate/out-of-order submissions are rejected before changing state.

A source gap greater than `max_gap_us`, changed image dimensions, switching between
automatic/caller selection, or a changed caller subject ID clears the window and
crop and advances the epoch. Intrinsics currently follow upstream live:
`fx=fy=max(width,height)`, principal point at the image centre. Calibrated
intrinsics are not an input to this first ABI.

The temporal policy is **accepted-frame-index**, not time-aware inference. It
processes the observations as supplied, reports actual source cadence, and does
not claim irregular input is 30 FPS. No smoothing, contact refinement, resampling
or extrapolation is added. Processing durations use a local monotonic clock;
they are not network end-to-end latency. The optional float metrics channel includes an
approximate source delta for profiling; use the int64 result timestamp for timing.

| Policy | Behavior |
| --- | --- |
| `GEMX_LIVE_PARITY` | Largest raw detection by area, then clip to image; full-image fallback on no detections. Matches the existing live worker. Track epoch zero for automatic selection means no identity-continuity claim. |
| `GEMX_LIVE_CONTINUITY` | Acquire only a unique detection; subsequently require exactly one detection with IoU ≥ 0.3 against the last crop. No matches means lost; multiple matches means ambiguous. Clear history on loss/ambiguity, and require fresh detection to reacquire. |
| Caller box | Bypass detection and use the supplied box each frame. Changing its subject ID clears history. The caller owns identification. |

A successful acquisition increments the track epoch. On loss, no held pose is
emitted. Reacquisition starts with warm-up. This is conservative box association,
not re-identification. It may lose a rapidly moving or crossing subject. In parity
mode, the largest person may change silently; it is deliberately not a tracker.
The parity demo retains this behavior. Continuity mode is available to direct
native clients; no comparative real-camera identity-quality claim is made.

Detection runs once every N **processed** frames after a successful detection.
Between detections `CROP_REUSED` is set. The detector cannot report someone leaving
the scene until it runs again; crop reuse does not assert fresh tracking evidence.
Misses and resets force a fresh detection. Flags also identify detector execution,
caller boxes and full-image fallback. Detector scores and 2D keypoint confidences
are separate from 3D confidence, which is unavailable.

## Coordinate contracts

`soma77` has fixed names/parents with pelvis index 0, right-handed native SOMA
Y-up coordinates and metre units. Positions are gravity-aligned with pelvis
translation removed **only**; root rotation is retained. `local_rotations` are
parent-local unit XYZW quaternions; `local_translations` are parent-local metres.
Their forward kinematics reconstruct the emitted positions. Root axis-angle is
in radians; root translation is explicitly zero. Camera-oriented root-relative
positions and the camera translation are separate projection channels.

Use the **emitted frame's** identity, scale and local translations. The neutral
reference uses identity zero, global scale one and other scale parameters zero;
it does not promise constant bone lengths. The underlying model's existing
identity/scale pooling is unchanged; the orchestration does not add averaging
across previously emitted results. Quaternions have canonical nonnegative `w`;
there is no temporal sign alignment or filtering.

Rolling-window translation is not a continuous world trajectory. This API
therefore publishes root-relative skeletons and an orientation anchor, not a
recovered locomotion path.

### Standalone SOMA-to-SMPL adapter

`gemx_soma_to_smpl` accepts 77 decoded SOMA positions with global orientation
already applied, plus the SOMA root axis-angle. The same function serves live and
recorded decoded poses. The mapped 24 positions use canonical SMPL joint order;
this does not fit a SMPL mesh or produce SMPL articulated joint rotations.

With WXYZ quaternion multiplication, `YtoZ(x,y,z)=(x,-z,y)`:

```text
q_anchor = normalize(Rx(pi/2) * q_root * conjugate([.5,.5,.5,.5]))
local[j] = rotate(inverse(q_anchor), YtoZ(soma[mapping[j]] - soma[0]))
YtoZ(soma[mapping[j]] - soma[0]) = rotate(q_anchor, local[j])
```

The anchor is unit WXYZ with canonical nonnegative `w`. The SMPL reference
local transforms use identity point-coordinate frames and neutral mapped bone
offsets; they are not estimates of articulated SMPL rotations. The definition
provides the exact 77→24 mapping and Y-up→Z-up matrix. No hidden display conversion,
heading alignment, physical-base subtraction or G1 wrist synthesis is applied.

The adapter follows NVIDIA's Apache-2.0 upstream converter at revision
`7f151314d4d1606544bf249d2a7a1cb754c64582`. See
[provenance and source hashes](../reference/streaming-sources.json).
The model-free test uses 18 independent outputs of the pinned upstream Python
converter. Synthetic decoded joints isolate adapter correctness, including
nonzero heading, asymmetric coordinates, scale changes and pelvis offsets.
Native model-backed tests separately check the SOMA FK/shape path. Rotation
comparison is modulo quaternion sign; limits are `1e-5` metres / `1e-6` quaternion
components. These tolerances do not change existing inference parity thresholds.

## Run a native client

Build normally, then run [`gemx-live-stream`](../examples/live_stream.cpp).
Its manifest contains width/height followed by sequence, source microseconds and
a path to tightly packed RGB bytes (paths relative to the working directory):

```text
640 480
1 0 frames/first.rgb
2 83333 frames/second.rgb
3 165900 frames/third.rgb
```

Use real capture/media times in your application. This example's sample times
illustrate irregular timing; they are not inferred from filenames or FPS.

```sh
GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
  build/vulkan/gemx-live-stream \
  generated/reference/gem-x-contact-f32.gguf \
  generated/reference/vitpose-f32.gguf generated/reference/yolox-f32.gguf \
  build/vulkan/bin/libggml-vulkan.so Vulkan 0 frames.txt
```

For CPU, use a release build, `build/release/bin` and `CPU 0`. The example prints
the definition once, then outcomes and SMPL poses, resets and releases resources.
It demonstrates both native pipeline channels and standalone conversion.

Strict Vulkan requires the three environment flags above **before any GGML
backend initialization in the process**. The API checks them but never changes
global environment variables. Keep backend precision configuration fixed for the
process: GGML caches its backend registry. `BACKEND_DEFAULT` accepts the host's
existing configuration and makes no strict-parity precision promise. F32 weights
are required in either mode. Only CPU/Vulkan are supported; Linux is tested.

## Demo and resource behavior

The demo's resident worker now calls this API; it no longer owns a duplicate
observation/inference pipeline. Its existing binary preview format stays the same.
The browser supplies `X-GEMX-Source-Time-Us` from its monotonic canvas sampling
clock relative to live-session start, before JPEG encoding/upload. This is browser
sampling time, not a hardware camera exposure timestamp. Go preserves it as int64
and forwards `FRAME sequence source_us`. Legacy bare `FRAME` clients use monotonic
file-ingestion time and cannot claim original capture timing. Clock mode cannot
change within a demo session. The HTTP demo is not the LocalAI public protocol.

Each instance owns all three weight sets, one ViTPose graph, one YOLOX graph and
an LRU GEM graph cache capped at `min(window,32)`. Observation history has at most
`window` entries; temporary decoding arrays scale with that window. RGB is not
retained. Host-owned results accumulate only if the host retains them. Reset
clears observations/crop but intentionally preserves model and graph allocations;
destroy frees per-instance resources. GGML backend registries and driver caches
remain process-wide. Memory grows while new window lengths build graphs, then
should stabilize; it is not multiplied by total stream duration.

See [validation measurements](MOTION-STREAMING-VALIDATION.md) for the tested
platform, memory scope, parity evidence and reproducible commands. This does not
extend the existing offline/long-sequence upstream parity claims.
