import json
import torch
import pypose as pp

torch.set_num_threads(1)
torch.manual_seed(17)

class NoObservation(pp.module.NLS):
    def state_transition(self, state, input, t=None):
        return state

    def observation(self, state, input, t=None):
        return torch.zeros_like(state[..., :1])

class Identity(pp.module.NLS):
    def state_transition(self, state, input, t=None):
        return state

    def observation(self, state, input, t=None):
        return state

x = torch.tensor([0.2, -0.1], dtype=torch.float64)
P = torch.tensor([[1., 0.3], [0.3, 0.5]], dtype=x.dtype)
Q = torch.zeros_like(P)
pf = pp.module.PF(NoObservation(), particles=200000)
xe, Pe = pf(x, torch.zeros(1, dtype=x.dtype), torch.zeros_like(x), P, Q,
            torch.ones(1, 1, dtype=x.dtype))
print("NO_OBSERVATION", json.dumps({"expected_cov": P.tolist(), "actual_cov": Pe.tolist(),
                                    "expected_mean": x.tolist(), "actual_mean": xe.tolist()}))
R = torch.eye(2, dtype=x.dtype) * 0.8
y = torch.tensor([1., -0.5], dtype=x.dtype)
gain = torch.linalg.solve(P + R, P).mT
mean = x + gain @ (y - x)
cov = P - gain @ P
pf = pp.module.PF(Identity(), particles=200000)
xe, Pe = pf(x, y, torch.zeros_like(x), P, Q, R)
print("LINEAR_POSTERIOR", json.dumps({"expected_cov": cov.tolist(), "actual_cov": Pe.tolist(),
                                      "expected_mean": mean.tolist(), "actual_mean": xe.tolist()}))
