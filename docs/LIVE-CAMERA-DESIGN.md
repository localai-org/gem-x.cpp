# Live GEM-X camera investigation

Reviewed 2026-09-18 against the [upstream camera example][camera],
[SOMA-to-SMPL publisher][publisher] and [teleoperation guide][guide].
Native live inference and rolling replay are implemented and checked against
the actual upstream example. See [the parity report](LIVE-OFFLINE-PARITY.md).
The demo now has live camera transport and controls. Browser validation uses
a simulated webcam; no physical camera or robot session was run.

The orchestration is now available through the [native streaming API](MOTION-STREAMING.md),
including source timing, reset/selection semantics and the pinned SOMA-to-SMPL adapter.

## What upstream actually runs

The camera example estimates a person box and ViTPose observations for each
frame, buffers a rolling sequence and emits only the newest GEM prediction.
Its recommended example uses a 30-frame window. This is endpoint inference
with past context, not the offline estimate for that frame with future context.

Crucially, the inspected example always selects `no_imgfeat`:
`no_imgfeat=args.no_imgfeat or True`. It supplies zero Body tokens to a model
whose **image condition is marked absent**. It does not run SAM3D Body.
The standard and absent-image models are different inference configurations;
a zero token passed to the standard model is not equivalent.

The example assumes a static camera and uses identity camera rotation deltas.
It starts producing 3D output after two observations. It selects the last
decoded pose and gravity-aligned root orientation, then publishes root-local
SMPL-ordered positions and orientation through SONIC protocol v3. It does not
stream a commanded world-space root trajectory. Its optional smoothing is an
EMA on positions and a sign-aligned normalized quaternion blend.

## Implemented inference and remaining integration

`gemx_session_create_live` selects the converted absent-image constant, keeps
CLIFF conditioning present, and decodes scale without the offline clamp.
Body features may be NULL. There is no Body-enabled live mode.
`scripts/replay_live_native.py` keeps detector, ViTPose and GEM resident,
buffers observations in a 30-frame window and returns the newest prediction
from each window after two-frame warm-up.

Upstream invokes `detect_and_track` separately for each frame, recreating its
tracker each time. Parity therefore selects the largest detected person each
frame, or the full image when there is no detection. Persistent tracking and
causal smoothing would change that behavior; neither is part of this parity
implementation. Offline still uses whole-clip dominant tracking and smoothing.

The demo now exposes Live webcam/Offline video modes, camera selection,
start/stop controls, actual rate/frame-age feedback and synchronized overlays.
Its resident `--live-worker` buffers only observations; the browser captures
into two bounded buffers, preparing the next frame near inference completion. HTTP frame requests are
processed in order, with one following frame prepared/uploaded concurrently.
See [the current pipeline](LIVE-PIPELINING.md) for bounded scheduling and measurements. Stop, hidden tabs, idle expiry and disconnects
release the worker; restarting clears context. Frames/results are temporary.
See [demo instructions](../demo/README.md) for usage and lifecycle limits.
The SONIC publisher described below remains integration work.

## Temporal behavior that needs an explicit choice

- A 30-frame window covers three seconds at 10 Hz and one second at 30 Hz.
  Preserve capture timestamps and report actual observation cadence; do not
  relabel an irregular stream as 30 FPS. GEM's local displacements are indexed
  by frames, while SONIC runs on its own control clock.
- Sliding-window global rollout restarts its translation origin. For the
  initial root-local SONIC path, do not use that restarted translation as a
  locomotion command. A world-motion viewer needs a persistent rollout; the
  upstream source also provides streaming rollout helpers in `motion_utils.py`.
- Native skeleton construction averages identity and scale across the supplied
  sequence. The camera example converts the last frame's identity/scale. These
  policies can yield different bone lengths. The live parity audit uses each emitted frame
  separately to match upstream; live integration should preserve that policy.
- Existing global rollout includes symmetric smoothing of camera-derived yaw.
  A rolling-window result can revise its past estimates; this is not a trained
  causal model. Test rotations and motion near the newest frame explicitly.
- Do not copy smoothing `0.8` without measuring delay. At 10 Hz an EMA with
  previous-state weight 0.8 has roughly 400 ms low-frequency delay. Lighter or
  time-aware filtering may be preferable. This excludes inference/network delay.
- Clear temporal state on identity changes, time discontinuities and restarts.
  Expire stale estimates; a still-open socket is not evidence of fresh motion.

## SONIC handoff

Port the upstream mapping from the 77 SOMA joints to the 24 SMPL joints, with
metre units, pelvis centering, Y-up to Z-up conversion, SMPL base-rotation
removal, and root-local positions. `body_quat` is WXYZ. Protocol v3 also carries
`smpl_pose [1,21,3]`, G1 joint position/velocity arrays and a frame index; the
example leaves the pose and wrist commands zero. Use its actual packer as the
wire-format oracle rather than inventing a JSON pose protocol.

The sibling motion-bricks.cpp native SONIC conversion supports G1 mode 0 only.
Human SMPL input selects mode 2. Start with upstream SONIC in MuJoCo; an
all-native deployment also needs that encoder branch and observation assembly
ported and checked. GEM's optional offline retargeter is not required for this
learned-human-encoder route.

## Experiments before enabling the live robot path

1. Rolling replay is covered on the pinned clip, including warm-up and buffer
   eviction, with Body omitted. Expand coverage to other movements and people.
2. Compare 15/30/60-frame windows against full-sequence output. Report quality
   and latency separately; offline disagreement is not itself ground-truth error.
3. Benchmark actual capture-to-pose age and drop rate on the target hardware.
   Strict F32 resident replay measured about 134 ms per accepted frame on the
   RTX 5070 Ti, including graph warm-up but excluding model loading, capture
   and transport. This is not a measured camera-to-robot rate.
4. Verify person loss, occluded feet, turns, restart and stale-stream behavior
   in MuJoCo. A smooth human skeleton does not establish stable robot tracking.

All benchmark processes should retain the user's eight-core limit: CPU affinity
`0-7`, at most eight compilation jobs, explicit Torch/ORT/BLAS thread limits,
and `GOMAXPROCS=8` for the demo. Do not queue camera frames for batching at the
expense of pose age.

[camera]: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/examples/live_camera_teleop/webcam_stream.py
[publisher]: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/examples/live_camera_teleop/soma_to_smpl.py
[guide]: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/live_camera_teleop.html
