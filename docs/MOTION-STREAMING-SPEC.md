# Native motion pipeline and streaming integration specification

Status: native implementation available; LocalAI transport remains external.
See [API guide](MOTION-STREAMING.md) and [validation](MOTION-STREAMING-VALIDATION.md).
The requirements below retain their original scope; native completion does not
claim transport acceptance item 8 is complete.
Date: 2026-09-22.

## Objective

Make gem-x.cpp a reusable, resident source of timestamped human poses for local
applications and inference servers. LocalAI is an initial integration: it will
accept video/live frames and publish poses using Protobuf over WebSocket.
External consumers may render, record, or convert these poses into references for
SONIC. No SONIC controller or simulation is to be implemented in LocalAI.

The native library must be usable without the demo server, LocalAI, Protobuf,
a network connection, Python, or a SONIC installation. No application-specific
consumer is part of this contract.

## Existing implementation and gaps

The current C API in `include/gemx.h` exposes resident YOLOX, ViTPose and GEM
models, `gemx_session_create_live`, inference, motion decoding, skeleton building
and GLB export. Live callers currently assemble their own rolling observation
windows. The demo and native worker provide orchestration that should become
reusable library functionality rather than being copied into LocalAI.

Live uses absent-image conditioning and does not need SAM3D Body. Offline uses
Body features and can apply contact refinement. These are distinct inference
modes; offline results are not a reference for causal live behavior. Current
exports contain a skeleton, not a body mesh.

See [live design](LIVE-CAMERA-DESIGN.md), [pipeline](LIVE-PIPELINING.md),
[parity evidence](LIVE-OFFLINE-PARITY.md), and
[detection cadence](LIVE-DETECTION-CADENCE.md). Some older notes describe
motion-bricks as mode-0-only. Its current native API implements the SMPL mode-2
encoder; streamed reference assembly and its physics connection remain separate.
This specification supersedes suggestions that LocalAI should publish SONIC's
ZMQ wire protocol directly.

## Ownership boundaries

| Owner | Responsibilities |
| --- | --- |
| gem-x.cpp | Resident inference pipeline, temporal state, person-selection policy, timing semantics, native skeleton metadata, SOMA-to-SMPL conversion, native examples and validation |
| LocalAI | Model installation/configuration, auth, quotas, media decoding/ingestion, session routing, backend RPC adapter, public Protobuf/WebSocket protocol, subscriber queues, recording jobs and UI |
| Consumer | Target-specific calibration, physical-base alignment, reference buffering/resampling/prediction, SONIC observation construction, simulation/control and stale-input response |

GEM-X must not depend on LocalAI's internal backend RPC schema. LocalAI owns a
separate public, versioned motion `.proto`, usable to generate Go, TypeScript
and C++ clients. A generic `smpl24` output profile is allowed; robot observation
vectors and controller state are not native pose output formats.

## 1. Resident live pipeline API

Add an additive C API around a stateful pipeline, reusing existing inference
implementations. Preserve the current low-level API and its numerical behavior.
Exact exported names are implementation choices; required operations are:

- Create/configure a pipeline with explicit model paths, backend/device, thread
  limit, precision, rolling-window size, detector cadence and temporal policy.
- Query capabilities, effective configuration and output definitions.
- Submit one RGB frame with sequence number, timestamp and optional subject box.
- Return either a pose result or an explicit warm-up/lost/reset status.
- Reset temporal/subject state without reloading immutable weights.
- Destroy the pipeline and release all owned resources.

The first API may be synchronous. No unbounded native frame queue is required.
A pipeline owns its observation window, detection/crop state and decoding state.
Calls on one pipeline are serialized; independent pipelines must not share
mutable state. Weight sharing is optional and must not be claimed unless
implemented. Report per-instance memory behavior.

Use opaque handles and fixed-width scalar/pointer/count arguments for new APIs.
Document ownership and lifetimes; provide explicit result destruction or
caller-owned copy-out buffers. Results must not be silently overwritten by the
next inference call. Validate capacities, dimensions and finite numeric inputs;
no C++ exception may cross the C boundary. Keep invalid arguments distinct from
normal warm-up and missing-subject outcomes.

Models remain loaded across frames. Do not spawn a CLI or reload GGUFs per frame.
Document cancellation granularity honestly: reset/destroy must not race with an
in-flight call. If native inference is not interruptible, the host can stop
accepting work and discard the eventual result; do not promise immediate abort.

The demo should adopt this API after equivalence checks, so demo and integration
paths exercise the same pipeline implementation.

## 2. Source timing and temporal semantics

Each accepted frame carries a caller-assigned sequence number and an int64
source timestamp in microseconds relative to a declared stream origin. Return
both unchanged on the corresponding estimate. These are capture/media times,
not inference completion times or a fabricated constant frame rate.

Within an epoch, require strictly increasing timestamps and sequence numbers;
sequence gaps are allowed. Reject duplicate/out-of-order input without changing
pipeline state. A host seeking backward in a video starts a new epoch via reset.
Expose the epoch on results so a consumer cannot interpolate across resets.

Document the model's frame-index-based temporal assumptions. The baseline mode
processes accepted observations as supplied and reports their actual cadence;
it must not imply time-aware inference or relabel irregular observations as
30 FPS. A future fixed-cadence mode must explicitly specify selection,
interpolation/duplication and maximum tolerated gaps, and be validated separately.

Configure a maximum source-time gap. Crossing it clears temporal state before
processing the new frame and reports a reset. Changed image dimensions,
subject changes and explicit restarts also invalidate the relevant crop and
observation state. Camera/calibration changes require an explicit reset.
Report the minimum warm-up requirement and whether each submission emitted a pose.

Timing metrics may include detector, pose-model and GEM durations and accepted
frame cadence. Use a local monotonic clock for processing durations. Do not
subtract clocks from different machines or call inference duration end-to-end
latency. Clock synchronization and transport age measurement belong to the host.

Do not silently add smoothing. Any optional filter declares its parameters,
time basis, reset behavior and measured delay. Never apply offline contact
refinement requiring future frames to the live path.

## 3. Person selection and continuity

Initial scope is one selected subject, not multi-person motion capture.
Expose selection policy, bounding box, detection freshness, and track epoch.
Distinguish detector confidence/2D keypoint confidence from 3D pose confidence;
do not fabricate the latter from the former.

Support an explicit caller-supplied subject box and a documented automatic
selection policy. The initial automatic policy may use conservative box
association. It must report loss or ambiguous association rather than silently
asserting identity continuity. Do not claim re-identification support.

Detector cadence and reusing the previous crop are not tracking. Force fresh
detection after loss/reset. Report when a pose used a reused crop. A detected
subject change clears the rolling window before emitting that subject's poses.

Preserve an explicit upstream-parity mode (largest detection/full-image fallback
where required by the reference). Keep it separate from continuity-oriented
mode. Full-image fallback must not be advertised as a confidently tracked person.
Test and document the behavior and quality differences between policies.

## 4. Skeleton metadata and pose results

Expose a versioned native skeleton definition through the library:

- Stable schema identifier/version and ordered joint names.
- Joint count, parent indices and root index.
- Reference/rest local transforms, with their identity/scale policy specified.
- Metre units, handedness, axis basis and quaternion component order.
- Exact coordinate space for each output field and available output channels.

Topology and conventions remain fixed for a configured stream. Shape-dependent
translations may vary per frame: a static rest pose must not be represented as
a guarantee of constant bone lengths. Preserve the accepted live policy using
the emitted frame's identity/scale, rather than averaging across the window.

Native pose output must provide sequence, source timestamp, epoch, tracking
state, root orientation/translation, SOMA joint positions and local rotations.
Expose local translations where needed to reconstruct the same skeleton as the
native renderer/exporter. Confidence and contacts are optional capabilities,
not fields filled with invented values.

Specify whether positions are camera-space, gravity-aligned, or world-space,
and whether root removal subtracts translation only or also removes rotation.
Specify quaternion normalization and sign continuity. Never hide display-axis
conversions or recentering in the API.

Rolling windows currently restart global translation rollout. Do not label that
translation as a continuous world trajectory. Either mark it window-relative
with its origin, or implement and independently validate persistent rollout.
Continuous locomotion recovery is not required for the initial root-local stream.

## 5. Generic SOMA-to-SMPL conversion

Implement a callable, transport-independent conversion from a decoded SOMA pose
to a documented `smpl24` profile. Reuse it for live and recorded poses. Expose:

- 24 joint positions in canonical SMPL order, in metres.
- Root-local positions and a separate gravity-aligned anchor orientation.
- Exact source/target bases, pelvis removal, base-rotation removal, and the
  mathematical transform reconstructing positions from local joints and anchor.
- A versioned output definition, including quaternion order and joint mapping.

Port the relevant upstream mapping rather than guessing from joint names. Pin
the upstream revision and fixture provenance; preserve attribution and applicable
license notices. The adapter must operate on the same per-frame identity/scale
as the validated live path.

This profile does not fit SMPL mesh parameters and must not imply that it does.
It does not produce a robot target, physical-base-relative orientation, G1 wrist
angles, policy tokens or SONIC's padded observation vector.

Validate positions and anchor orientation against independent upstream fixtures,
including nonzero heading, asymmetric limb poses and changes in scale. Compare
rotations modulo quaternion sign. An identity pose alone cannot validate basis,
left/right or quaternion-order conversions.

## 6. LocalAI integration contract (external implementation)

LocalAI-specific endpoints belong under `/api/motion/...`, not the
OpenAI-compatible `/v1/...` namespace. HTTP JSON configures sessions. WebSocket
binary messages carry Protobuf envelopes with a definition, pose, or lifecycle
event payload. Camera ingestion may use a separate connection from subscribers.

Send the selected skeleton definition once upon subscription and again upon
reconnection, before any poses. Do not repeat joint names, topology or coordinate
conventions per frame. A session's output schema is immutable; changing model or
skeleton format requires a new session. Reference-shape changes, if exposed,
need an explicit event rather than silent reinterpretation of existing data.

Pose messages use packed Protobuf `float` arrays (float32), integer source times,
sequence numbers, epoch and subject identity. Define array lengths and flattening
order in the public schema. Each pose is self-contained; do not delta-encode the
initial protocol. Negotiate optional channels at subscription, not per frame.
Publish units and quaternion ordering once in the definition. No body mesh is
required or advertised by the GEM-X stream.

Bound subscriber queues; replace unsent old live poses with newer ones without
blocking inference. Preserve control/definition/reset event ordering. TCP can
retain already-sent stale data: disconnect stalled subscribers rather than
claiming queue replacement removes all network backlog. Complete offline
recordings use a separate reliable path, not a lossy live subscription.

External implementations generate their own bindings from the public `.proto`.
The GEM-X native library itself acquires no Protobuf or WebSocket dependency.

## 7. Examples and consumer responsibilities

Provide a native example that creates the pipeline, feeds timestamped RGB
frames, reads the definition once, consumes poses, resets, and frees resources.
Include an SMPL conversion example without requiring SONIC or a network server.

A separate example in motion-bricks.cpp can subscribe to LocalAI, convert native
poses or request `smpl24`, and drive its simulation when its streamed-human
reference entry point is available. This is an interoperability test, not a
requirement to add physics to GEM-X or LocalAI.

For the original SONIC mode-2 encoder, the consumer constructs ten SMPL samples
at 20 ms intervals spanning reference time through +180 ms, applies heading and
physical-base alignment, and supplies wrist references and physical-state
history. Buffering, holding, prediction and stale-input behavior are consumer
policies. A lower-rate estimate interpolated at 50 Hz is not 50 Hz inference.
Other policy versions may have different contracts; none is baked into the
GEM-X output format.

## 8. Validation and acceptance

Implementation is complete when:

1. A native client runs a persistent live pipeline without the demo process,
   Python, SAM3D Body, LocalAI or network libraries.
2. The parity configuration reproduces existing live reference results within
   the existing thresholds, including warm-up and rolling-window eviction.
   No relaxed thresholds are used to hide orchestration regressions.
3. SOMA-to-SMPL fixtures match the pinned upstream converter, with independent
   expected values and documented tolerances.
4. Skeleton metadata reconstructs the emitted native pose consistently with
   the existing skeleton/export path; varying identity/scale is covered.
5. Tests cover ordering errors, timestamp gaps, explicit reset, resolution
   changes, detector loss, ambiguous subject association and reacquisition.
6. Two pipeline instances do not leak temporal or subject state into each other;
   ownership, invalid buffers and repeated creation/destruction are checked.
7. Long-running replay demonstrates bounded memory and no accumulating frame
   queue. Report throughput and latency percentiles separately from model load.
8. A host adapter round-trips Protobuf poses with a separately generated client,
   covering float arrays, 64-bit timestamp handling, reconnection definitions,
   resets and slow subscribers. These transport tests belong in LocalAI.
9. Live and offline remain distinguishable: no claim of long-sequence upstream
   parity beyond the existing validated scope, no hidden offline smoothing,
   and no accidental Body dependency for live mode.

Use deterministic fixtures where possible; gate weight-dependent tests on local
assets. Preserve existing CPU/Vulkan precision policies. Performance evidence
must identify hardware, precision, sampling/detection policy and measurement
scope. Follow existing eight-core benchmark limits; do not promise a target FPS
without measurement. Document each supported OS/backend combination rather than
assuming all GGML backends work.

## Suggested delivery sequence

1. Metadata/result contract and reference-tested SOMA-to-SMPL conversion.
2. Stateful native live pipeline with source timing, reset and ownership tests.
3. Explicit continuity policy and loss events, separate from parity mode.
4. Refactor the demo to exercise the reusable API and record regression evidence.
5. LocalAI adapter/public Protobuf stream and independent client interoperability.
6. Optional motion-bricks sample consumer and separately scoped offline API
   improvements. SAM3D Body integration remains a separate backend task.

## References

- [Public native API](../include/gemx.h)
- [Live/offline validation](LIVE-OFFLINE-PARITY.md)
- [Upstream SOMA-to-SMPL converter](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/examples/live_camera_teleop/soma_to_smpl.py)
- [Upstream live bridge](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/examples/live_camera_teleop/README.md)

The upstream links are discovery references. Implementation must pin the exact
revision used for conversion and validation rather than depending on mutable main.
