import contextlib
import io
import json
import torch
import pypose as pp

torch.set_num_threads(1)
dtype = torch.float64

class ReorderedEPnP(pp.module.EPnP):
    def _compute_betas(self, l_mat, rho):
        betas = super()._compute_betas(l_mat, rho)
        betas[3] = betas[3].flip(-1)
        return betas

def data(seed, count):
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(count, 3, dtype=dtype, generator=generator)
    rotation = pp.so3(torch.randn(3, dtype=dtype, generator=generator) * 0.3).Exp()
    pose = pp.SE3(torch.cat((torch.tensor([0.2, -0.1, 5.], dtype=dtype), rotation.tensor())))
    K = torch.tensor([[400., 0., 320.], [0., 450., 240.], [0., 0., 1.]], dtype=dtype)
    return points, pp.point2pixel(points, K, pose), K, pose

records = []
for count in [4, 5, 6, 8]:
    for seed in range(150):
        points, pixels, K, expected = data(seed, count)
        try:
            errors = []
            for cls in [pp.module.EPnP, ReorderedEPnP]:
                solved = cls(refine=False)(points, pixels, K)
                errors.append(pp.reprojerr(points, pixels, K, solved, reduction="norm").mean().item())
            if errors[0] > 1e-5 and errors[1] < errors[0] * 0.8:
                records.append({"seed": seed, "count": count, "original": errors[0], "reordered": errors[1]})
        except Exception as e:
            print("PROBE_EXCEPTION", count, seed, type(e).__name__, str(e)[:100])
records.sort(key=lambda x: x["original"] - x["reordered"], reverse=True)
print("BETA_REORDER_CANDIDATES", json.dumps(records[:10]))
for record in records[:10]:
    points, pixels, K, expected = data(record["seed"], record["count"])
    errors = []
    with contextlib.redirect_stdout(io.StringIO()):
        for cls in [pp.module.EPnP, ReorderedEPnP]:
            solved = cls(refine=True)(points, pixels, K)
            errors.append(pp.reprojerr(points, pixels, K, solved, reduction="norm").mean().item())
    record["refined_original"], record["refined_reordered"] = errors
print("REFINED_BETA_REORDER", json.dumps(records[:10]))
