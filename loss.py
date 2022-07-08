import numpy as np
import torch


def custom_loss(output, target):
    scale_factor = 10

    p = torch.diff(output, dim=1)
    t = torch.diff(target, dim=1)

    dot_product = torch.einsum('ijk, ijk->ij', p, t)

    length_diff = torch.square(torch.linalg.norm(p, dim=-1) - torch.linalg.norm(t, dim=-1))

    relative_loss = scale_factor * torch.mean(-dot_product + length_diff)

    abs_loss = torch.mean((output - target)**2)

    loss = abs_loss + relative_loss

    return relative_loss


if __name__ == '__main__':

    true = np.random.randint(1, 10, (32, 10, 2))
    pred = true + np.random.normal(0, 10, (32, 10, 2))

    loss = custom_loss(torch.Tensor(pred), torch.Tensor(true))

    print(loss)
