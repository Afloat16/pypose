import warnings
import pytest
import torch
import pypose as pp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bmv_strided_output_preserves_unselected_storage(dtype):
    matrix = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype)
    vector = torch.tensor([2., -1.], dtype=dtype)
    buffer = torch.full((8,), -99., dtype=dtype)
    out = buffer[1:7:2]
    expected = buffer.clone()
    expected[1:7:2] = matrix @ vector
    stride, offset = out.stride(), out.storage_offset()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pp.bmv(matrix, vector, out=out)

    torch.testing.assert_close(buffer, expected)
    assert result is out
    assert out.shape == (3,) and out.stride() == stride and out.storage_offset() == offset
    assert not caught


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["strided", "transposed"])
def test_bmv_broadcast_output_preserves_layout(dtype, layout):
    matrix = torch.arange(12, dtype=dtype).reshape(2, 1, 3, 2)
    vector = torch.arange(8, dtype=dtype).reshape(1, 4, 2) - 1
    expected = torch.einsum("...ij,...j->...i", matrix, vector)
    if layout == "strided":
        buffer = torch.full((2, 4, 7), -99., dtype=dtype)
        out = buffer[..., 1::2]
        expected_buffer = buffer.clone()
        expected_buffer[..., 1::2] = expected
    else:
        buffer = torch.full((3, 2, 4), -99., dtype=dtype)
        out = buffer.permute(1, 2, 0)
        expected_buffer = expected.permute(2, 0, 1)
    stride, offset = out.stride(), out.storage_offset()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pp.bmv(matrix, vector, out=out)

    torch.testing.assert_close(buffer, expected_buffer)
    torch.testing.assert_close(out, expected)
    assert result is out
    assert out.stride() == stride and out.storage_offset() == offset
    assert not caught


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bmv_correctly_shaped_contiguous_output_is_not_resized(dtype):
    matrix = torch.eye(3, dtype=dtype).expand(2, 3, 3)
    vector = torch.arange(6, dtype=dtype).reshape(2, 3)
    out = torch.empty_like(vector)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pp.bmv(matrix, vector, out=out)
    assert result is out
    torch.testing.assert_close(result, vector)
    assert not caught


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [0, 1])
def test_bmv_resizes_output_when_needed(dtype, size):
    matrix = torch.arange(6, dtype=dtype).reshape(3, 2)
    vector = torch.tensor([2., -1.], dtype=dtype)
    out = torch.empty(size, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = pp.bmv(matrix, vector, out=out)
    assert result is out
    torch.testing.assert_close(out, matrix @ vector)


def test_bmv_without_output_retains_higher_order_gradients():
    generator = torch.Generator().manual_seed(0)
    matrix = torch.randn(2, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    vector = torch.randn(3, dtype=torch.float64, generator=generator, requires_grad=True)
    assert torch.autograd.gradcheck(pp.bmv, (matrix, vector))
    assert torch.autograd.gradgradcheck(pp.bmv, (matrix, vector))
