import torch


def compute_info_gain(candidates: torch.Tensor, mu_p: torch.Tensor, mu_n: torch.Tensor) -> torch.Tensor:
    """Information gain via cosine similarity difference."""
    sim_p = torch.cosine_similarity(candidates, mu_p.unsqueeze(0), dim=1)
    sim_n = torch.cosine_similarity(candidates, mu_n.unsqueeze(0), dim=1)
    return sim_p - sim_n
