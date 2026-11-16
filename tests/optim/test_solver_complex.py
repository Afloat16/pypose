import pytest
import torch

from pypose.optim.solver import CG


def _layout(matrix, layout):
    if layout == "coo":
        return matrix.to_sparse_coo()
    if layout == "csr":
        return matrix.to_sparse_csr()
    return matrix


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
def test_cg_complex_isotropic_residual(dtype, layout):
    # Positive eigenvalues are 1 and 3, and A @ b == b. The transpose-only
    # product b.T @ b is zero even though the Hermitian norm is sqrt(2).
    dense = torch.tensor([[2, 1j], [-1j, 2]], dtype=dtype)
    rhs = torch.tensor([[1], [1j]], dtype=dtype)

    actual = CG(maxiter=2, tol=1e-6)(_layout(dense, layout), rhs)

    torch.testing.assert_close(actual, rhs)
    torch.testing.assert_close(dense @ actual, rhs)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
@pytest.mark.parametrize("initial_guess", [False, True])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_cg_complex_hermitian_system(dtype, layout, initial_guess, preconditioned):
    generator = torch.Generator().manual_seed(1729)
    n = 5
    factor = torch.randn(n, n, dtype=dtype, generator=generator) / n
    dense = factor.mH @ factor + 2 * torch.eye(n, dtype=dtype)
    rhs = torch.randn(n, 1, dtype=dtype, generator=generator)
    initial = torch.randn(n, 1, dtype=dtype, generator=generator) if initial_guess else None
    preconditioner = None
    if preconditioned:
        factor = torch.randn(n, n, dtype=dtype, generator=generator) / n
        preconditioner = factor.mH @ factor + torch.eye(n, dtype=dtype)
    tolerance = 1e-6 if dtype == torch.complex64 else 1e-12

    actual = CG(maxiter=4 * n, tol=tolerance)(
        _layout(dense, layout), rhs, x=initial, M=preconditioner
    )
    expected = torch.linalg.solve(dense, rhs)

    atol, rtol = (2e-6, 2e-5) if dtype == torch.complex64 else (1e-11, 1e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(dense @ actual, rhs, atol=atol, rtol=rtol)
