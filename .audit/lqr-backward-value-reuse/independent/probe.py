import hashlib
import json
import pathlib
import subprocess
import torch
import pypose as pp

repo = pathlib.Path('/workspace/scratch/4bed741d4966/worktrees/pypose-candidate')
output = pathlib.Path('/workspace/scratch/4bed741d4966/audit/pypose-lqr-value-independent-20261117')
source = repo / 'pypose/module/lqr.py'
expected_hash = 'ae3d1274551f2494c69f86542969873b0d33e54b2bdcb2f2a67f4ccbf8cf16e3'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
base_source = subprocess.check_output(['git', 'show', '298e9f0862:pypose/module/lqr.py'], cwd=repo, text=True)
namespace = {'__name__': 'pypose.module._independent_lqr_base', '__package__': 'pypose.module'}
exec(compile(base_source, '<exact-base-lqr>', 'exec'), namespace)
Baseline = namespace['LQR']
torch.set_num_threads(1)
rows = []
for dtype in (torch.float32, torch.float64):
    for horizon in (1, 2, 5, 9):
        for warm in (False, True):
            generator = torch.Generator().manual_seed(7300 + horizon)
            def rand(*shape):
                return torch.randn(*shape, generator=generator, dtype=dtype)
            batch, ns, nc = 2, 2, 3
            factor = rand(batch, horizon, ns + nc, ns + nc)
            Q = factor.mT @ factor + 0.7 * torch.eye(ns + nc, dtype=dtype)
            p = rand(batch, horizon, ns + nc)
            x = rand(batch, ns)
            A = 0.3 * torch.eye(ns, dtype=dtype) + 0.05 * rand(batch, ns, ns)
            B = 0.2 * rand(batch, ns, nc)
            c = 0.1 * rand(batch, ns)
            nominal = 0.15 * rand(batch, horizon, nc)
            originals = [Q, p, x, A, B, c] + ([nominal] if warm else [])
            directions = [rand(*item.shape) for item in originals]
            results = []
            for implementation in (Baseline, pp.module.LQR):
                values = [item.detach().clone().requires_grad_() for item in originals]
                qv, pv, xv, av, bv, cv = values[:6]
                C = torch.eye(ns, dtype=dtype).expand(batch, ns, ns)
                D = torch.zeros(batch, ns, nc, dtype=dtype)
                system = pp.module.LTI(av, bv, C, D, cv)
                result = implementation(system, qv, pv, horizon)(xv, u_traj=values[6] if warm else None)
                loss = sum((index + 1) * item.square().sum() for index, item in enumerate(result))
                gradients = torch.autograd.grad(loss, values, create_graph=True)
                directional = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
                hvp = torch.autograd.grad(directional, values)
                results.append((result, gradients, hvp))
            reference, candidate = results
            assert all(torch.equal(a, b) for a, b in zip(reference[0], candidate[0]))
            tolerance = 2e-4 if dtype == torch.float32 else 2e-10
            for order in (1, 2):
                for actual, expected in zip(candidate[order], reference[order]):
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            row = {'dtype': str(dtype), 'horizon': horizon, 'warm_start': warm,
                   'forward_bitwise_equal': True,
                   'raw_Q_p_x_A_B_c_and_warm_start_gradients_match': True,
                   'hessian_vector_products_match': True,
                   'max_gradient_abs_difference': max((a-b).abs().max().item() for a,b in zip(candidate[1], reference[1])),
                   'max_hvp_abs_difference': max((a-b).abs().max().item() for a,b in zip(candidate[2], reference[2]))}
            rows.append(row)
            print(json.dumps(row), flush=True)
report = {'base': subprocess.check_output(['git','rev-parse','298e9f0862'],cwd=repo,text=True).strip(),
          'source_sha256': expected_hash, 'torch': torch.__version__, 'rows': rows,
          'scope': 'Independent base/candidate compatibility probes; 16 CPU cases cover raw Q, dynamics A/B/c, state, linear cost, warm starts and second-order differentiation. Not GPU validation or a performance benchmark.'}
(output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
print('ALL_INDEPENDENT_CASES_PASSED', len(rows))
