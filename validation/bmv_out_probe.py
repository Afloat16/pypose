import json
import torch
import pypose as pp

matrix = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
vector = torch.tensor([2., -1.], dtype=matrix.dtype)
buffer = torch.full((8,), -99., dtype=matrix.dtype)
out = buffer[1:7:2]
before = {"shape": list(out.shape), "stride": list(out.stride()), "offset": out.storage_offset()}
result = pp.bmv(matrix, vector, out=out)
expected = torch.full_like(buffer, -99.)
expected[1:7:2] = matrix @ vector
print("BMV_OUT", json.dumps({"before": before, "after_stride": list(out.stride()),
    "actual_buffer": buffer.tolist(), "expected_buffer": expected.tolist(),
    "returned_is_out": result is out, "result": result.tolist()}))
