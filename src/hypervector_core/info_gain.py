import torch


def compute_info_gain(candidates: torch.Tensor, mu_plus: torch.Tensor, mu_minus: torch.Tensor) -> torch.Tensor:
    sim_p = torch.cosine_similarity(candidates, mu_plus.unsqueeze(0), dim=1)
    sim_n = torch.cosine_similarity(candidates, mu_minus.unsqueeze(0), dim=1)
    return sim_p - sim_n
