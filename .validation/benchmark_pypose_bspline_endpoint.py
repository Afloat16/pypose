"""Compare complete SE3 interpolation with its unchanged upstream source."""
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
reference_path = Path(__file__).with_name('bspline_unchanged_reference.py')
reference_path.write_text(subprocess.check_output([
    'git', 'show', BASE + ':pypose/function/spline.py',
], text=True))
spec = importlib.util.spec_from_file_location('pypose.function.spline_reference',
                                             reference_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
functions = [reference.bspline, pp.bspline]
torch.set_num_threads(1)


def coordinates(batch, count, dtype):
    generator = torch.Generator().manual_seed(31)
    return .12 * torch.randn(batch + (count, 6), dtype=dtype, generator=generator)


validation = []
for dtype in (torch.float32, torch.float64):
    for batch in ((), (2,), (2, 3)):
        for count, extrapolate in ((1, True), (2, True), (4, False),
                                   (9, False), (9, True)):
            for strided in (False, True):
                seed = coordinates(batch, count * (2 if strided else 1), dtype)
                outputs, gradients = [], []
                for function in functions:
                    values = seed.clone().requires_grad_()
                    data = pp.se3(values).Exp()
                    if strided:
                        data = data[..., ::2, :]
                    output = function(data, interval=.3, extrapolate=extrapolate)
                    weights = torch.linspace(.5, 1.5, output.numel(), dtype=dtype)
                    loss = (output.tensor() * weights.reshape(output.shape)).sum()
                    gradients.append(torch.autograd.grad(loss, values)[0])
                    outputs.append(output.tensor())
                assert torch.equal(*outputs), (dtype, batch, count, extrapolate, strided)
                torch.testing.assert_close(*gradients)
                validation.append(dict(dtype=str(dtype), batch=batch, count=count,
                                       extrapolate=extrapolate, strided=strided,
                                       exact_output=True, gradient_parity=True,
                                       max_gradient_difference=(gradients[0] - gradients[1]).abs().max().item()))
print('Output and input-gradient comparisons:', len(validation), flush=True)

records = []
if '--parity-only' not in sys.argv:
    with torch.no_grad():
        for dtype in (torch.float32, torch.float64):
            for batch, count, interval in ((1, 64, .1), (1, 1024, .1),
                                           (1, 1024, .5), (8, 256, .1),
                                           (8, 256, .5), (1, 4096, .1),
                                           (1, 4096, .5)):
                data = pp.se3(coordinates((batch,), count, dtype)).Exp()
                for _ in range(4):
                    for function in functions:
                        function(data, interval=interval)
                samples = [[], []]
                for repeat in range(21):
                    outputs = [None, None]
                    for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                        start = time.perf_counter_ns()
                        outputs[index] = functions[index](data, interval=interval)
                        samples[index].append((time.perf_counter_ns() - start) / 1e6)
                    assert torch.equal(outputs[0].tensor(), outputs[1].tensor())
                medians = [statistics.median(s) for s in samples]
                row = dict(dtype=str(dtype), batch=batch, count=count, interval=interval,
                           baseline_ms=medians[0], candidate_ms=medians[1],
                           speedup=medians[0] / medians[1], exact_output=True,
                           baseline_samples_ms=samples[0], candidate_samples_ms=samples[1])
                records.append(row)
                print(json.dumps({k: v for k, v in row.items()
                                  if not k.endswith('_samples_ms')}), flush=True)
Path(sys.argv[1]).write_text(json.dumps(dict(base=BASE, validation=validation,
                                            benchmark=records), indent=2))
