#!/usr/bin/env python3
"""Profile a resident strict-F32 live worker and hash every complete pose.

Run under taskset -c 0-7. CSV stages include GPU synchronization. Warm-up and
steady state are reported separately; no capture, compression or HTTP cost.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def summarize(rows):
    if not rows:
        return {}
    result = {}
    for name in rows[0]:
        if not name.endswith('_ms'):
            continue
        values = sorted(float(r[name]) for r in rows)
        result[name] = dict(mean=statistics.mean(values), median=statistics.median(values),
                            p95=values[min(len(values)-1, int(.95*len(values)))],
                            p99=values[min(len(values)-1, int(.99*len(values)))])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--binary', type=Path)
    p.add_argument('--library-dir', type=Path)
    p.add_argument('--module', type=Path, help='Isolated backend module for kernel experiments')
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--detect-interval',type=int,default=1)
    p.add_argument('--source-step-us',type=int,help='Synthetic replay source cadence; independent of shader/model execution time')
    p.add_argument('--save-poses',action='store_true',help='Keep outputs for quality comparisons')
    p.add_argument('--compare', type=Path, help='Previous report whose per-frame pose hashes must match exactly')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--device-name',default='',help='Optional exact device description; otherwise select by index')
    a = p.parse_args()
    if len(os.sched_getaffinity(0))>8:p.error('restrict CPU affinity to at most eight cores')
    if a.source_step_us is not None and a.source_step_us<=0:p.error('source-step-us must be positive')
    if not 1<=a.detect_interval<=30:p.error('detection interval must be 1..30')
    if not 1 <= a.repeats <= 20:
        p.error('repeats must be 1..20')
    a.root = a.root.resolve();a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = [f.read_bytes() for f in sorted(a.frames.glob('*.input'))]
    if len(inputs) < 32:
        p.error('need at least 32 frames to cover window eviction')
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('GGML_VK_', 'SAM3D_')):
            env.pop(key)
    env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1',
               OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',GEMX_LIVE_PROFILE=str(a.output/'stages.csv'))
    if a.library_dir:
        env['LD_LIBRARY_PATH'] = str(a.library_dir.resolve())
    cmd = [str((a.binary or a.root/'build/vulkan/gemx-pipeline').resolve()), '--live-worker']
    cmd += [str(a.root/'generated/reference'/name) for name in
            ('gem-x-contact-f32.gguf','vitpose-f32.gguf','yolox-f32.gguf')]
    cmd += [str((a.module or a.root/'build/vulkan/bin/libggml-vulkan.so').resolve()),'Vulkan',str(a.device),a.device_name or '-','8','30',str(a.output)]
    cmd += [str(a.detect_interval)]
    started = time.monotonic()
    hashes = [];rss = []
    with (a.output/'worker.log').open('w') as log:
        worker = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
        try:
            assert worker.stdout.readline().strip() == 'READY', 'worker startup failed; see worker.log'
            startup = time.monotonic()-started
            replay_started = time.monotonic()
            for index, data in enumerate(inputs*a.repeats):
                (a.output/'frame.input').write_bytes(data)
                worker.stdin.write(f'FRAME {index} {index*a.source_step_us}\n' if a.source_step_us else 'FRAME\n');worker.stdin.flush()
                reply = worker.stdout.readline().strip()
                assert reply.startswith('POSE ') or index == 0 and reply.startswith('WARMUP '), reply
                status = Path(f'/proc/{worker.pid}/status').read_text()
                rss.append(int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:'))))
                if reply.startswith('POSE '):
                    data=(a.output/'pose.gpose').read_bytes()
                    hashes.append(hashlib.sha256(data).hexdigest())
                    if a.save_poses:(a.output/f'{index:06d}.gpose').write_bytes(data)
            replay_seconds = time.monotonic()-replay_started
            worker.stdin.close();assert worker.wait(timeout=30)==0
        finally:
            if worker.poll() is None:
                worker.kill();worker.wait()
    rows = list(csv.DictReader((a.output/'stages.csv').open()))
    report = dict(startup_seconds=startup,replay_seconds=replay_seconds,frames_per_second=len(rows)/replay_seconds,
                  source_step_us=a.source_step_us,rss_kib=rss,frames=len(rows),pose_hashes=hashes,detect_interval=a.detect_interval,
                  detector_calls=sum(int(r['detector_ran']) for r in rows),
                  warmup=summarize([r for r in rows if int(r['context'])<30]),
                  steady=summarize([r for r in rows if int(r['context'])==30]))
    if a.compare:
        expected = json.loads(a.compare.read_text())['pose_hashes']
        assert hashes == expected, f'pose bytes changed in {sum(x!=y for x,y in zip(hashes,expected))} frames'
        report['byte_identical_to'] = str(a.compare)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pose_hashes'},indent=2))

if __name__ == '__main__':
    main()
