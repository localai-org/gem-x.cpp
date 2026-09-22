#!/usr/bin/env python3
"""Regenerate the small adapter fixture with pinned upstream functions (CPU PyTorch).

No native code is used as an oracle. SOMA FK is replaced with supplied decoded
joints: this tests the conversion, not the learned body model. The separate
native lifecycle test checks FK with per-frame inferred identity and scale.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch

REV = '7f151314d4d1606544bf249d2a7a1cb754c64582'
PATHS = ['gear_sonic/examples/live_camera_teleop/soma_to_smpl.py',
         'gear_sonic/isaac_utils/rotations.py',
         'gear_sonic/trl/utils/torch_transform.py',
         'gear_sonic/trl/utils/kornia_transform.py']


def selected(path, names, env):
    # Execute the original function bodies. Drop JIT decorators only, to avoid
    # importing the unrelated simulation dependency tree into this CPU oracle.
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if hasattr(node, 'decorator_list'):
                node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sources', type=Path)
    p.add_argument('rig', type=Path)
    p.add_argument('--output', type=Path, default=Path('reference/streaming_reference.txt'))
    a = p.parse_args()
    torch.set_num_threads(8)
    hashes = {path: hashlib.sha256((a.sources / Path(path).name).read_bytes()).hexdigest() for path in PATHS}
    provenance = Path('reference/streaming-sources.json')
    if provenance.exists():
        recorded = json.loads(provenance.read_text())
        assert hashes == recorded['sha256'], 'upstream source hash mismatch'
        assert hashlib.sha256(a.rig.read_bytes()).hexdigest() == recorded.get('rig_sha256', hashlib.sha256(a.rig.read_bytes()).hexdigest()), 'SOMA rig hash mismatch'
    else:
        provenance.write_text(json.dumps(dict(repository='https://github.com/NVlabs/GR00T-WholeBodyControl', revision=REV,
            sha256=hashes, fixture='streaming_reference.txt', generator='scripts/generate_streaming_reference.py',
            license='Apache-2.0', scope='Decoded synthetic SOMA joints; actual upstream adapter, smoothing disabled'), indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('upstream_kornia', a.sources/'kornia_transform.py')
    kornia = importlib.util.module_from_spec(spec); spec.loader.exec_module(kornia)
    common = dict(torch=torch, np=np, Tensor=torch.Tensor, angle_axis_to_quaternion=kornia.angle_axis_to_quaternion)
    tt = dict(common)
    selected(a.sources/'torch_transform.py', {'normalize', 'quat_conjugate', 'quat_inv', 'quat_apply'}, tt)
    rot = dict(common)
    selected(a.sources/'rotations.py', {'quat_conjugate','quat_mul','remove_smpl_base_rot','smpl_root_ytoz_up'}, rot)
    for name, values in [('gear_sonic.isaac_utils.rotations',rot),('gear_sonic.trl.utils.torch_transform',tt)]:
        mod = types.ModuleType(name); mod.__dict__.update(values); sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location('upstream_bridge', a.sources/'soma_to_smpl.py')
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    class DecodedSoma:
        joint_names = list(np.load(a.rig)['joint_names'])
        def __call__(self, **params):
            return {'joints': self.joints[None]}
    soma = DecodedSoma(); converter = bridge.SomaToSmpl(soma, device='cpu', smooth=0)
    rng = np.random.default_rng(78201)
    cases = []
    for aa in [[0,0,0],[0,1.2,0],[.3,-.8,.2],[-.7,2.9,.4],[1e-8,0,0],[.2,0,-1.5]]:
        for scale in [.65,1.,1.7]:
            # Asymmetric and nonzero pelvis; deliberately distinct joint coords.
            soma.joints = torch.tensor(rng.normal(size=(77,3))*scale + [2,-1,.3], dtype=torch.float32)
            root = np.array(aa,dtype=np.float32)
            out = converter.convert(dict(body_pose=np.zeros(228),global_orient=root,
                identity_coeffs=np.zeros(45),scale_params=np.r_[scale,np.zeros(68)]))
            cases.append(np.concatenate([soma.joints.numpy().ravel(),root,out['smpl_joints'].ravel(),out['body_quat'].ravel()]))
    a.output.write_text(str(len(cases))+'\n'+'\n'.join(' '.join(format(float(x),'.9g') for x in row) for row in cases)+'\n')
    print(f'{len(cases)} independent upstream cases; torch {torch.__version__}')

if __name__ == '__main__':
    main()
