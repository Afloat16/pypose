"""Compare complete native IMU updates with the exact unchanged source."""
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import pypose as pp


BASE = '298e9f0862cb228b0e8851b2616e682ccdca6a32'
assert Path(pp.__file__).resolve() == Path.cwd() / 'pypose/__init__.py'
reference_path = Path(__file__).with_name('pypose_imu_unchanged_reference.py')
reference_path.write_text(subprocess.check_output([
    'git', '-C', str(Path.cwd()), 'show',
    BASE + ':pypose/module/imu_preintegrator.py',
], text=True))
spec = importlib.util.spec_from_file_location('pypose.module.imu_reference', reference_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
classes = [reference.IMUPreintegrator, pp.module.IMUPreintegrator]
torch.set_num_threads(1)


def inputs(batch, frames, dtype):
    gen = torch.Generator().manual_seed(37)
    dt = .005 + .01 * torch.rand(batch, frames, 1, dtype=dtype, generator=gen)
    gyro = .3 * torch.randn(batch, frames, 3, dtype=dtype, generator=gen)
    acc = torch.randn(batch, frames, 3, dtype=dtype, generator=gen)
    acc[..., 2] += 9.81007
    return dt, gyro, acc


def equal_states(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if left[key] is None:
            assert right[key] is None
        else:
            # Reusing a conversion must preserve every forward output exactly.
            assert torch.equal(left[key].tensor() if isinstance(left[key], pp.LieTensor)
                               else left[key],
                               right[key].tensor() if isinstance(right[key], pp.LieTensor)
                               else right[key]), key


validation = []
for dtype in (torch.float32, torch.float64):
    for batch in (1, 3):
        for known_rotation in (False, True):
            dt, gyro, acc = inputs(batch, 7, dtype)
            rotation = pp.so3(gyro * .2).Exp() if known_rotation else None
            models = [cls(reset=False).to(dtype=dtype) for cls in classes]
            # Include nonzero covariance carried through repeated public calls.
            for model in models:
                model.cov = .01 * torch.eye(9, dtype=dtype).expand(batch, 9, 9).clone()
            for step in range(2):
                results, gradients = [], []
                for model in models:
                    cg = torch.full((batch, 7, 3), 1e-5, dtype=dtype, requires_grad=True)
                    ca = torch.full((batch, 7, 3), .0064, dtype=dtype, requires_grad=True)
                    result = model(dt, gyro, acc, rot=rotation, gyro_cov=cg, acc_cov=ca)
                    gradients.append(torch.autograd.grad(result['cov'].square().sum(),
                                                         (cg, ca), retain_graph=True))
                    results.append(result)
                equal_states(*results)
                for before, after in zip(*gradients):
                    torch.testing.assert_close(before, after)
                validation.append(dict(dtype=str(dtype), batch=batch, known_rotation=known_rotation,
                                       stateful_step=step, exact_forward=True, noise_gradients=True))

    # Direct covariance API also permits rotation gradients.
    dt, gyro, acc = inputs(2, 7, dtype)
    results, gradients = [], []
    for cls in classes:
        angle = gyro.clone().requires_grad_()
        state = {'dt': dt, 'Rij': pp.so3(angle).Exp(),
                 'Rk': pp.so3(gyro * dt).Exp(), 'Ha': pp.vec2skew(acc)}
        covariance = cls.propagate_cov(state, torch.eye(9, dtype=dtype).repeat(2, 1, 1),
                                      torch.full((2, 1, 3), 1e-5, dtype=dtype),
                                      torch.full((2, 1, 3), .0064, dtype=dtype))
        gradients.append(torch.autograd.grad(covariance['cov'].square().sum(), angle)[0])
        results.append(covariance)
    equal_states(*results)
    torch.testing.assert_close(*gradients)
    validation.append(dict(dtype=str(dtype), direct_covariance=True, exact_forward=True,
                           rotation_gradients=True))

records = []
with torch.no_grad():
    for dtype in (torch.float32, torch.float64):
        for batch, frames in ((1, 1), (1, 63), (8, 127), (16, 255)):
            data = inputs(batch, frames, dtype)
            models = [cls(reset=True).to(dtype=dtype) for cls in classes]
            for _ in range(4):
                for model in models:
                    model(*data)
            samples = [[], []]
            for repeat in range(21):
                results = [None, None]
                for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    start = time.perf_counter_ns()
                    results[index] = models[index](*data)
                    samples[index].append((time.perf_counter_ns() - start) / 1e6)
                equal_states(*results)
            medians = [statistics.median(s) for s in samples]
            row = dict(dtype=str(dtype), batch=batch, frames=frames,
                       baseline_ms=medians[0], candidate_ms=medians[1],
                       speedup=medians[0] / medians[1], exact_forward=True,
                       baseline_samples_ms=samples[0], candidate_samples_ms=samples[1])
            records.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith('_samples_ms')}), flush=True)
Path(sys.argv[1]).write_text(json.dumps(dict(base=BASE, validation=validation, benchmark=records),
                                      indent=2))
