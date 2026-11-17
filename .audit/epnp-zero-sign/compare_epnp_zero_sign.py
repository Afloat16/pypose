import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import pypose as pp


baseline_path = Path(sys.argv[1])
namespace = {'__name__': 'pypose.module._epnp_baseline', '__package__': 'pypose.module'}
exec(compile(baseline_path.read_text(), str(baseline_path), 'exec'), namespace)
Baseline = namespace['EPnP']
Candidate = pp.module.EPnP
torch.set_num_threads(1)
records = []


def record(value):
    records.append(value)
    print(json.dumps(value), flush=True)


# These controls deliberately avoid zero coefficients. They check compatibility
# with the baseline, including its existing coefficient ordering, not an oracle
# that declares that ordering mathematically correct.
for dtype in (torch.float32, torch.float64):
    for batch in ((), (2, 3)):
        for seed in range(16):
            generator = torch.Generator().manual_seed(seed + 804)
            values = {}
            for width in (3, 6, 10):
                raw = torch.randn(batch + (width,), generator=generator, dtype=dtype)
                values[width] = raw.sign() * (raw.abs() + .25)
            weight = torch.randn((4,) + batch + (4,), generator=generator, dtype=dtype)
            outputs = []
            gradients = []
            for implementation in (Baseline, Candidate):
                model = implementation(torch.eye(3, dtype=dtype), refine=False)
                inputs = {width: value.clone().requires_grad_()
                          for width, value in values.items()}
                model.solver.forward = lambda matrix, rhs: inputs[matrix.shape[-1]]
                output = model._compute_betas(torch.zeros(batch + (6, 10), dtype=dtype),
                                               torch.ones(batch + (6,), dtype=dtype))
                gradients.append(torch.autograd.grad((output * weight).sum(), tuple(inputs.values())))
                outputs.append(output.detach())
            assert torch.equal(outputs[0], outputs[1])
            assert all(torch.equal(a, b) for a, b in zip(*gradients))
            record({'kind': 'nonzero_solver_output_compatibility', 'dtype': str(dtype),
                    'batch': batch, 'seed': seed, 'all_four_candidates_bitwise_equal': True,
                    'all_solver_coefficient_gradients_bitwise_equal': True})


def camera_problem(batch, count, dtype):
    generator = torch.Generator().manual_seed(613)
    points = torch.randn(batch + (count, 3), generator=generator, dtype=dtype)
    points = points * torch.tensor([1., .7, .5], dtype=dtype)
    rotation = pp.so3(torch.tensor([.05, -.08, .02], dtype=dtype)).Exp().matrix()
    translation = torch.tensor([.2, -.1, 5.], dtype=dtype)
    intrinsics = torch.tensor([[450., 0., 320.], [0., 440., 240.],
                               [0., 0., 1.]], dtype=dtype)
    camera = (rotation @ points.mT).mT + translation
    projected = (intrinsics @ camera.mT).mT
    pixels = projected[..., :2] / projected[..., 2:]
    return points, pixels, intrinsics, rotation, translation


for batch, count in [((1,), 32), ((2,), 128)]:
    for dtype in (torch.float32, torch.float64):
        for refine in (False, True):
            values = camera_problem(batch, count, dtype)
            outputs, gradients, coefficients = [], [], []
            for implementation in (Baseline, Candidate):
                points, pixels, intrinsics = (value.clone().requires_grad_() for value in values[:3])
                model = implementation(intrinsics, refine=refine)
                model.solver = pp.optim.solver.LSTSQ(driver='gelsd')
                solve = model.solver.forward
                captured = []

                def capture(matrix, rhs):
                    result = solve(matrix, rhs)
                    captured.append(result.detach())
                    return result

                model.solver.forward = capture
                result = model(points, pixels).matrix()
                weight = torch.arange(result.numel(), dtype=dtype).reshape_as(result) / result.numel()
                gradients.append(torch.autograd.grad((result * weight).sum(), (points, pixels, intrinsics)))
                outputs.append(result.detach())
                coefficients.append(captured)
            assert all(torch.equal(a, b) for a, b in zip(*coefficients))
            assert all(torch.count_nonzero(value == 0) == 0 for value in coefficients[0])
            assert torch.equal(*outputs)
            tolerance = 2e-5 if dtype == torch.float32 else 1e-11
            for before, after in zip(*gradients):
                torch.testing.assert_close(before, after, rtol=tolerance, atol=tolerance)
            record({'kind': 'public_call_nonzero_control', 'dtype': str(dtype), 'batch': batch,
                    'points': count, 'refine': refine, 'main_beta_solver': 'gelsd',
                    'all_beta_coefficients_nonzero': True, 'poses_bitwise_equal': True,
                    'point_pixel_intrinsics_gradients_close': True})


for dtype in (torch.float32, torch.float64):
    for refine in (False, True):
        for zero in (0., -0.):
            values = camera_problem((), 32, dtype)
            observations = []
            for implementation in (Baseline, Candidate):
                points, pixels, intrinsics, rotation, translation = values
                model = implementation(intrinsics, refine=refine)
                model.solver = pp.optim.solver.LSTSQ(driver='gelsd')
                solve = model.solver.forward

                def replay(matrix, rhs):
                    result = solve(matrix, rhs)
                    if matrix.shape[-1] == 3:
                        result = result.clone()
                        result[..., 0] = zero
                    return result

                model.solver.forward = replay
                try:
                    result = model(points, pixels).matrix()
                    tolerance = 2e-3 if dtype == torch.float32 else 2e-7
                    torch.testing.assert_close(result[:3, :3], rotation, rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(result[:3, 3], translation, rtol=tolerance, atol=tolerance)
                    observations.append({'success': True,
                                         'rotation_error': (result[:3, :3] - rotation).abs().max().item(),
                                         'translation_error': (result[:3, 3] - translation).abs().max().item()})
                except torch.linalg.LinAlgError as error:
                    observations.append({'success': False, 'error': str(error)})
            assert not observations[0]['success'] and observations[1]['success']
            record({'kind': 'public_call_signed_zero_replay', 'dtype': str(dtype), 'refine': refine,
                    'zero': zero, 'main_beta_solver': 'gelsd', 'injected_first_coefficient': True,
                    'baseline': observations[0], 'candidate': observations[1]})

payload = {'baseline_source_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
           'candidate_source_sha256': hashlib.sha256(Path(inspect.getfile(Candidate)).read_bytes()).hexdigest(),
           'records': records,
           'scope': 'Zero-sign reconstruction only. Nonzero controls preserve existing ordering; the historical full-four ordering restoration is excluded as previously implemented. Camera replay is controlled injection, not a natural failure-frequency claim. No finite-gradient claim at exact zero magnitudes.'}
Path('epnp-zero-sign-results.json').write_text(json.dumps(payload, indent=2) + '\n')
