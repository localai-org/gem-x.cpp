#ifndef GEMX_STREAM_H
#define GEMX_STREAM_H
#include "gemx.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Additive streaming ABI v1. All counts are elements, except JSON byte counts.
 * Handles own resources; returned results survive later submissions and pipeline
 * destruction. Calls on a pipeline serialize. Destruction must not race a call.
 * Inference is synchronous and not interruptible. No network/Body dependency. */
typedef struct gemx_live gemx_live;
typedef struct gemx_live_result gemx_live_result;
#define GEMX_LIVE_FRAME_INDEX 0u
#define GEMX_LIVE_PARITY 0u
#define GEMX_LIVE_CONTINUITY 1u
#define GEMX_LIVE_STRICT_F32 0u
#define GEMX_LIVE_BACKEND_DEFAULT 1u
#define GEMX_LIVE_WARMUP 0u
#define GEMX_LIVE_POSE 1u
#define GEMX_LIVE_LOST 2u
#define GEMX_LIVE_AMBIGUOUS 3u
#define GEMX_LIVE_RESET 1u
#define GEMX_LIVE_CROP_REUSED 2u
#define GEMX_LIVE_FULL_IMAGE 4u
#define GEMX_LIVE_DETECTOR_RAN 8u
#define GEMX_LIVE_CALLER_BOX 16u
/* Result float channels. Copy-out with NULL,0 queries required count.
 * Empty channels on warmup/loss are explicit, not fabricated zero poses. */
#define GEMX_LIVE_POSITIONS 0u /* 77*3, Y-up gravity-aligned, root translation zero */
#define GEMX_LIVE_ROTATIONS 1u /* 77*4 local XYZW, canonical w>=0, no temporal filter */
#define GEMX_LIVE_TRANSLATIONS 2u /* 77*3 local, frame-specific shape */
#define GEMX_LIVE_SMPL_JOINTS 3u /* 24*3 root-local, see definition */
#define GEMX_LIVE_SMPL_ANCHOR 4u /* 4 WXYZ, unit length, w>=0 */
#define GEMX_LIVE_ROOT_AXIS_ANGLE 5u /* 3 gravity-aligned SOMA root rotation */
#define GEMX_LIVE_ROOT_TRANSLATION 6u /* 3 zero: no continuous world trajectory */
#define GEMX_LIVE_CAMERA_POSITIONS 7u /* 77*3 camera-oriented, root at zero */
#define GEMX_LIVE_CAMERA_TRANSLATION 8u /* 3 camera translation for projection */
#define GEMX_LIVE_KEYPOINTS 9u /* 77*3 pixel x/y/2D confidence */
#define GEMX_LIVE_BOX 10u /* 4 xyxy source pixels */
#define GEMX_LIVE_METRICS 11u /* detector/vitpose/GEM/world/camera ms, source dt us,
 detector count, context size, selected detector confidence (-1 unavailable) */
#define GEMX_LIVE_IDENTITY 12u /* 45 frame-specific SOMA identity coefficients */
#define GEMX_LIVE_SCALES 13u /* 69 frame-specific SOMA scales */
/* threads 1..8, window 2..120, cadence 1..30, max_gap_us>0. Frame-index
 * temporal policy only. F32 weights required. Strict Vulkan requires host-set
 * GGML_VK_DISABLE_F16/COOPMAT/COOPMAT2=1 before any backend initialization;
 * the API checks the environment and never changes process-global settings. */
GEMX_API gemx_status gemx_live_create(const char *gem_model,const char *pose_model,
 const char *detector_model,const char *module,const char *backend,uint32_t device,
 const char *expected_device,uint32_t threads,uint32_t window,uint32_t cadence,
 uint32_t selection,uint32_t precision,uint32_t temporal_policy,int64_t max_gap_us,gemx_live **out,
 char *error,uint64_t error_capacity);
GEMX_API void gemx_live_destroy(gemx_live *pipeline);
/* Returns effective config, capabilities, and soma77/smpl24 definitions in JSON.
 * Required byte count includes NUL. No borrowed internal pointers. */
GEMX_API gemx_status gemx_live_definition(gemx_live *pipeline,char *json,
 uint64_t capacity,uint64_t *required,char *error,uint64_t error_capacity);
GEMX_API gemx_status gemx_live_topology(gemx_live *pipeline,int32_t *parents,
 uint64_t count,char *error,uint64_t error_capacity);
/* New epoch; weights stay loaded. The next accepted result has RESET set. */
GEMX_API gemx_status gemx_live_reset(gemx_live *pipeline,char *error,uint64_t error_capacity);
/* RGB is borrowed for the call. timestamps are nonnegative source microseconds
 * relative to the caller's declared origin; seq/time strictly increase within an
 * epoch. Validation rejection does not mutate state. Gaps and size changes reset
 * context automatically. Intrinsics are f=max(width,height), principal point at
 * image centre; calibration changes require reset. subject_box is NULL/0 or four
 * finite in-image xyxy values; subject_id identifies caller-selected subjects.
 * Changing between automatic and caller selection resets the context.
 * Runtime inference failure clears temporal state; next accepted result resets. */
GEMX_API gemx_status gemx_live_submit(gemx_live *pipeline,const uint8_t *rgb,
 uint64_t capacity,uint32_t width,uint32_t height,uint64_t stride,
 uint64_t sequence,int64_t source_time_us,const float *subject_box,
 uint64_t box_count,uint64_t subject_id,gemx_live_result **out,
 char *error,uint64_t error_capacity);
GEMX_API void gemx_live_result_destroy(gemx_live_result *result);
GEMX_API gemx_status gemx_live_result_info(const gemx_live_result *result,
 uint64_t *sequence,int64_t *source_time_us,uint64_t *epoch,uint64_t *track_epoch,
 uint32_t *outcome,uint32_t *flags,char *error,uint64_t error_capacity);
GEMX_API gemx_status gemx_live_result_copy(const gemx_live_result *result,
 uint32_t channel,float *output,uint64_t capacity,uint64_t *required,
 char *error,uint64_t error_capacity);
/* Transport-independent conversion of decoded SOMA-v1 joint positions in Y-up
 * metres with global rotation already applied, and that SOMA root's axis-angle.
 * No mesh fit/robot calibration. Input/output counts exactly 231/3/72/4.
 * Output: q = Rx(pi/2)*q_root*conjugate([.5,.5,.5,.5]);
 * local[j] = inverse(q) * Rx(pi/2)*(soma[mapping[j]]-soma[pelvis]).
 * Canonical unit WXYZ q; rotations equivalent modulo sign. */
GEMX_API gemx_status gemx_soma_to_smpl(const float *joints,uint64_t joint_count,
 const float *root_axis_angle,uint64_t rotation_count,float *local,uint64_t local_count,
 float *anchor_wxyz,uint64_t anchor_count,char *error,uint64_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
