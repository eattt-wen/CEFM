import torch
import torch.nn as nn

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
    mask = torch.ones_like(sim)
    mask = mask.fill_diagonal_(0)
    negative_samples = sim[mask.bool()].reshape(2 * batch_size, -1)
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
