import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import pypose as pp


torch.set_num_threads(1)
base_path = Path(sys.argv[1])
namespace = {'__name__': 'pypose.module._validation_baseline_pnp',
             '__package__': 'pypose.module'}
exec(compile(base_path.read_text(), str(base_path), 'exec'), namespace)
Baseline = namespace['EPnP']
candidate_path = Path(inspect.getfile(pp.module.EPnP))
assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == \
    '55fbaaf38a4a8063329c99cd80b7d2f736b49c4ab8ca28ab55af668f21128874'


def fixture(batch, count, dtype):
    generator = torch.Generator().manual_seed(237)
    points = torch.randn(batch, count, 3, dtype=dtype, generator=generator)
    points *= torch.tensor([1., .7, .5], dtype=dtype)
    rotation = pp.so3(torch.tensor([.05, -.08, .02], dtype=dtype)).Exp().tensor()
    pose = pp.SE3(torch.cat((torch.tensor([.2, -.1, 5.], dtype=dtype), rotation)))
    intrinsics = torch.tensor([[450., 0., 320.], [0., 440., 240.],
                               [0., 0., 1.]], dtype=dtype)
    pixels = pp.point2pixel(points, intrinsics, pose)
    pixels += .01 * torch.randn(batch, count, 2, dtype=dtype, generator=generator)
    return points, pixels, intrinsics


def retained_storage(output):
    storages, seen, queue = {}, set(), [output.grad_fn]
    while queue:
        fn = queue.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        for name in dir(fn):
            if not name.startswith('_saved_'):
                continue
            value = getattr(fn, name)
            for tensor in value if isinstance(value, (list, tuple)) else (value,):
                if isinstance(tensor, torch.Tensor):
                    storage = tensor.untyped_storage()
                    storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        queue.extend(link[0] for link in fn.next_functions)
    return {'bytes': sum(storages.values()), 'unique_storages': len(storages),
            'autograd_nodes': len(seen)}


records = []
for batch, count in [(1, 32), (4, 128), (2, 4096), (1, 65536)]:
    for dtype in [torch.float32, torch.float64]:
        for refine in [False, True]:
            inputs = fixture(batch, count, dtype)
            outputs, subspaces, gradients, storages, subtractions = {}, {}, {}, {}, {}
            for name, implementation in [('baseline', Baseline), ('candidate', pp.module.EPnP)]:
                points, pixels, intrinsics = [value.detach().clone().requires_grad_()
                                              for value in inputs]
                model = implementation(intrinsics, refine=refine)
                # Compare the unchanged solver contract with stable SVD beta solutions.
                # The default QR driver has independent near-zero beta variation.
                model.solver = pp.optim.solver.LSTSQ(driver='gelsd')
                with torch.no_grad():
                    bases = model._svd_basis(points)
                    alpha = model._compute_alpha(points, bases)
                    subspaces[name] = model._compute_nullv(pixels, alpha, intrinsics)
                pose = model(points, pixels)
                outputs[name] = pose.tensor().detach()
                storages[name] = retained_storage(pose)
                gradients[name] = torch.autograd.grad(
                    pose.tensor().square().sum(), (points, pixels, intrinsics))
                with torch.no_grad(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    model._compute_nullv(pixels, alpha, intrinsics)
                subtractions[name] = sum(event.count for event in profile.key_averages()
                                         if event.key == 'aten::sub')
            assert torch.equal(subspaces['candidate'], subspaces['baseline'])
            tolerance = 3e-4 if dtype == torch.float32 else 1e-9
            torch.testing.assert_close(outputs['candidate'], outputs['baseline'],
                                       rtol=tolerance, atol=tolerance)
            for actual, expected in zip(gradients['candidate'], gradients['baseline']):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            expected_saving = 6 * batch * count * inputs[0].element_size()
            actual_saving = storages['baseline']['bytes'] - storages['candidate']['bytes']
            assert actual_saving == expected_saving, (actual_saving, expected_saving)
            assert subtractions['baseline'] - subtractions['candidate'] == 6
            record = {
                'shape': [batch, count], 'dtype': str(dtype), 'refine': refine,
                'beta_solver': 'gelsd', 'null_basis_bitwise_equal': True,
                'poses_bitwise_equal': torch.equal(outputs['candidate'], outputs['baseline']),
                'poses_and_point_pixel_intrinsics_gradients_close': True,
                'max_pose_abs_difference': (outputs['candidate'] - outputs['baseline']).abs().max().item(),
                'max_gradient_abs_difference': [
                    (actual - expected).abs().max().item()
                    for actual, expected in zip(gradients['candidate'], gradients['baseline'])],
                'retained_storage': storages, 'retained_bytes_reduced': actual_saving,
                'projective_subtraction_counts': subtractions,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
report = {
    'torch': torch.__version__, 'threads': torch.get_num_threads(),
    'baseline_source_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
    'candidate_source_sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    'metric': 'Distinct storage referenced by saved tensors in the live output autograd graph; not process peak memory.',
    'solver_note': 'Public comparisons configure the existing gelsd beta solver to isolate projection construction from previously observed default gelsy variation. Production solver selection is unchanged; refinement still uses its existing internal solver.',
    'records': records,
}
Path('epnp-offset-results.json').write_text(json.dumps(report, indent=2) + '\n')
