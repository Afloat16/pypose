import ast
import hashlib
import inspect
import json
import math
import statistics
import sys
import textwrap
import types
from pathlib import Path

import torch
import pypose as pp
from torch.utils.benchmark import Timer


torch.set_num_threads(1)
base_path = Path(sys.argv[1])
base_source = base_path.read_text()
base_ast = ast.parse(base_source)
base_class = next(node for node in base_ast.body if isinstance(node, ast.ClassDef) and node.name == 'LQR')
base_method = next(node for node in base_class.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'lqr_backward')
base_function = textwrap.dedent(ast.get_source_segment(base_source, base_method))
namespace = {}
exec(compile(base_function, str(base_path), 'exec'),
     pp.module.LQR.lqr_backward.__globals__, namespace)
base_backward = namespace['lqr_backward']
candidate_path = Path(inspect.getfile(pp.module.LQR))
assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == \
    'ae3d1274551f2494c69f86542969873b0d33e54b2bdcb2f2a67f4ccbf8cf16e3'


def fixture(batch, horizon, ns, nc, dtype, gradients=False):
    generator = torch.Generator().manual_seed(192 + ns + nc)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)

    A = 0.85 * torch.eye(ns, dtype=dtype) + 0.04 * randn(batch, ns, ns) / math.sqrt(ns)
    B = randn(batch, ns, nc) / math.sqrt(nc)
    factor = randn(batch, horizon, ns + nc, ns + nc)
    Q = (factor.mT @ factor / (ns + nc) + 0.5 * torch.eye(ns + nc, dtype=dtype)).detach()
    p, x = randn(batch, horizon, ns + nc), randn(batch, ns)
    Q.requires_grad_(gradients)
    p.requires_grad_(gradients)
    x.requires_grad_(gradients)
    system = pp.module.LTI(A, B, torch.eye(ns, dtype=dtype),
                          torch.zeros(ns, nc, dtype=dtype))
    return pp.module.LQR(system, Q, p, horizon), Q, p, x


results = []
for batch, horizon, ns, nc in [(1, 1, 3, 2), (3, 2, 3, 2),
                              (8, 32, 12, 6), (4, 24, 64, 32)]:
    for dtype in [torch.float32, torch.float64]:
        models, states, outputs, gradients, profiles = {}, {}, {}, {}, {}
        for name in ['baseline', 'candidate']:
            model, Q, p, x = fixture(batch, horizon, ns, nc, dtype, gradients=True)
            if name == 'baseline':
                model.lqr_backward = types.MethodType(base_backward, model)
            models[name], states[name] = model, x
            output = model(x)
            outputs[name] = [value.detach() for value in output]
            gradients[name] = torch.autograd.grad(
                sum(value.square().sum() for value in output), (Q, p, x))
            with torch.no_grad(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
            ) as profile:
                model(x)
            profiles[name] = {
                'bmm_count': sum(event.count for event in profile.key_averages()
                                 if event.key == 'aten::bmm'),
                'cumulative_positive_self_allocated_bytes': sum(
                    max(event.self_cpu_memory_usage, 0) for event in profile.events()),
            }
        for actual, reference in zip(outputs['candidate'], outputs['baseline']):
            assert torch.equal(actual, reference)
        tolerance = 3e-5 if dtype == torch.float32 else 1e-10
        for actual, reference in zip(gradients['candidate'], gradients['baseline']):
            torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
        assert profiles['candidate']['bmm_count'] < profiles['baseline']['bmm_count']

        @torch.no_grad()
        def invoke(name):
            return models[name](states[name])

        times = {'baseline': [], 'candidate': []}
        for repeat in range(7):
            order = ['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']
            for name in order:
                timing = Timer(stmt='invoke(name)', globals={'invoke': invoke, 'name': name},
                               num_threads=1).blocked_autorange(min_run_time=0.2)
                times[name].append(timing.median)
        ratios = [candidate / baseline for baseline, candidate
                  in zip(times['baseline'], times['candidate'])]
        record = {
            'shape': [batch, horizon, ns, nc], 'dtype': str(dtype),
            'outputs_bitwise_equal': True, 'raw_Q_p_x_gradients_close': True,
            'profiles': profiles, 'paired_seconds': times,
            'paired_candidate_over_baseline': ratios,
            'paired_ratio_median': statistics.median(ratios),
        }
        results.append(record)
        print(json.dumps(record), flush=True)

report = {'torch': torch.__version__, 'threads': torch.get_num_threads(),
          'baseline_source_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
          'candidate_source_sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
          'memory_metric': 'sum of positive per-event self_cpu_memory_usage; cumulative, not peak',
          'results': results}
Path('lqr-backward-results.json').write_text(json.dumps(report, indent=2) + '\n')
