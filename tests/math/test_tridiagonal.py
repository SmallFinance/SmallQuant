"""Thomas algorithm on a known tridiagonal system."""

import torch

from smallquant.math.tridiagonal import solve


def test_solve_recovers_a_constant_vector():
    lower = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float64)
    diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
    upper = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float64)
    rhs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    solution = solve(lower, diag, upper, rhs)
    assert torch.allclose(solution, torch.ones(3, dtype=torch.float64))


def test_solve_is_batched():
    lower = torch.tensor([[0.0, -1.0], [0.0, -2.0]], dtype=torch.float64)
    diag = torch.tensor([[2.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    upper = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    rhs = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    solution = solve(lower, diag, upper, rhs)
    reconstructed = (
        torch.cat((torch.zeros_like(solution[:, :1]), lower[:, 1:] * solution[:, :-1]), dim=-1)
        + diag * solution
        + torch.cat((upper[:, :-1] * solution[:, 1:], torch.zeros_like(solution[:, :1])), dim=-1)
    )
    assert torch.allclose(reconstructed, rhs)
