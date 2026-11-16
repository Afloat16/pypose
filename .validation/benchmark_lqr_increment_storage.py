import collections
import gc
import hashlib
import importlib.util
import itertools
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import types

import numpy as np
import torch
import pypose as pp

BASE = "298e9f0862cb228b0e8851b2616e682ccdca6a32"
PRODUCTION = "23853bda897abc44180471cc050c5c9f50abd0d9"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
original = subprocess.check_output(
    ["git", "show", BASE + ":pypose/module/lqr.py"], text=True
)
module = types.ModuleType("pypose.module._baseline_increment_storage")
module.__package__ = "pypose.module"
exec(compile(original, "baseline_lqr.py", "exec"), module.__dict__)
classes = {"base_a": module.LQR, "base_b": module.LQR, "candidate": pp.module.LQR}


def data_for(batch, horizon, ns, nc, dtype, warm):
    generator = torch.Generator().manual_seed(540)
    def normal(*shape):
        return torch.randn(*shape, dtype=dtype, generator=generator)
    nsc = ns + nc
    A = 0.7 * torch.eye(ns, dtype=dtype).expand(batch, -1, -1)
    A = A + 0.02 * normal(batch, ns, ns)
    B = 0.15 * normal(batch, ns, nc)
    c = 0.05 * normal(batch, ns)
    factor = 0.03 * normal(batch, horizon, nsc, nsc)
    Q = factor.mT @ factor + torch.eye(nsc, dtype=dtype)
    p = 0.1 * normal(batch, horizon, nsc)
    x = normal(batch, ns)
    nominal = 0.2 * normal(batch, horizon, nc) if warm else None
    return A, B, c, Q, p, x, nominal


def construct(cls, data, training):
    A, B, c, Q, p_data, x_data, nominal = data
    batch, horizon, _ = p_data.shape
    ns, nc = A.shape[-1], B.shape[-1]
    x = x_data.clone().requires_grad_(training)
    p = p_data.clone().requires_grad_(training)
    system = pp.module.LTI(
        A, B, torch.eye(ns, dtype=p.dtype).expand(batch, -1, -1),
        p.new_zeros(batch, ns, nc), c, torch.zeros_like(c)
    )
    return cls(system, Q, p, horizon), x, p, nominal


def execute(cls, data, training):
    with torch.set_grad_enabled(training):
        controller, x, p, nominal = construct(cls, data, training)
        outputs = controller(x, u_traj=nominal)
        gradients = ()
        if training:
            loss = sum((i + 1) * value.square().sum() for i, value in enumerate(outputs))
            gradients = torch.autograd.grad(loss, (x, p))
        return tuple(t.detach() for t in outputs), tuple(t.detach() for t in gradients)


def digest(values):
    return hashlib.sha256(b"".join(
        t.contiguous().numpy().tobytes() for t in values
    )).hexdigest()


parity = []
for dtype, warm, (batch, horizon), training in itertools.product(
    (torch.float32, torch.float64), (False, True),
    ((1, 1), (3, 9)), (False, True)
):
    data = data_for(batch, horizon, 4, 2, dtype, warm)
    before = execute(classes["base_a"], data, training)
    after = execute(classes["candidate"], data, training)
    exact_outputs = all(torch.equal(a, b) for a, b in zip(before[0], after[0]))
    if not exact_outputs:
        raise AssertionError("Forward values changed")
    rtol, atol = (2e-5, 3e-6) if dtype == torch.float32 else (1e-10, 1e-11)
    for a, b in zip(before[1], after[1]):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    parity.append({
        "dtype": str(dtype), "warm": warm, "batch": batch, "horizon": horizon,
        "training": training, "outputs_exact": exact_outputs,
        "outputs_sha256": digest(before[0]),
        "gradients_exact": all(torch.equal(a, b) for a, b in zip(before[1], after[1])),
        "gradient_max_abs_difference": [
            float((a - b).abs().max()) for a, b in zip(before[1], after[1])
        ]
    })

benchmarks = []
configs = [
    (2, 32, 4, 2, torch.float64),
    (16, 128, 4, 2, torch.float32),
    (64, 256, 8, 4, torch.float64),
]
orders = list(itertools.permutations(classes))
for batch, horizon, ns, nc, dtype in configs:
    for training in (False, True):
        data = data_for(batch, horizon, ns, nc, dtype, True)
        for _ in range(2):
            for cls in classes.values():
                execute(cls, data, training)
        samples = {name: [] for name in classes}
        gc.collect()
        for iteration in range(12):
            for name in orders[iteration % len(orders)]:
                started = time.perf_counter()
                result = execute(classes[name], data, training)
                elapsed = (time.perf_counter() - started) * 1000
                samples[name].append(elapsed)
                del result
        medians = {name: statistics.median(values) for name, values in samples.items()}
        record = {
            "batch": batch, "horizon": horizon, "states": ns, "controls": nc,
            "dtype": str(dtype), "training": training, "warm": True,
            "samples_ms": samples, "median_ms": medians,
            "base_a_over_candidate": medians["base_a"] / medians["candidate"],
            "base_b_over_candidate": medians["base_b"] / medians["candidate"],
            "base_a_over_base_b": medians["base_a"] / medians["base_b"],
            "omitted_history_buffer_bytes": batch * horizon * nc * data[0].element_size()
        }
        benchmarks.append(record)
        print("BENCHMARK_CASE " + json.dumps(record), flush=True)

data = data_for(64, 256, 8, 4, torch.float64, True)
profiles = {}
graphs = {}
for name in ("base_a", "candidate"):
    gc.collect()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True
    ) as profile:
        execute(classes[name], data, True)
    events = profile.events()
    profiles[name] = {
        "positive_self_cpu_memory_bytes": sum(
            max(0, int(event.self_cpu_memory_usage)) for event in events
        ),
        "copy_related_events": [
            {"name": event.key, "count": event.count,
             "self_cpu_memory_usage": event.self_cpu_memory_usage}
            for event in profile.key_averages()
            if "CopySlices" in event.key or event.key in ("aten::copy_", "aten::clone", "aten::empty")
        ],
        "memory_note": "Sum of positive per-event self CPU allocation accounting, not process peak memory."
    }
    with torch.enable_grad():
        controller, x, p, nominal = construct(classes[name], data, True)
        outputs = controller(x, u_traj=nominal)
        pending = [t.grad_fn for t in outputs]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            pending.extend(parent for parent, _ in node.next_functions)
        graphs[name] = dict(collections.Counter(type(node).__name__ for node in seen))
        del outputs, pending, seen, controller, x, p

result = {
    "base": BASE, "production": PRODUCTION,
    "python": platform.python_version(), "torch": torch.__version__,
    "numpy": np.__version__, "threads": torch.get_num_threads(),
    "platform": platform.platform(), "parity": parity,
    "benchmarks": benchmarks, "profiles": profiles, "autograd_graph_nodes": graphs,
    "timing_note": "Twelve permuted measurements per implementation, two identical unchanged controls; no peak-memory claim.",
}
with open("lqr-increment-results.json", "w") as output:
    json.dump(result, output, indent=2)
print("LQR_INCREMENT_RESULTS " + json.dumps(result), flush=True)
