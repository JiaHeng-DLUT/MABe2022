import torch
import torch.distributed as dist
import torch.nn.functional as F


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.

    From: https://github.com/spijkervet/SimCLR/blob/HEAD/simclr/modules/gather.py
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def info_nce_loss(x_list):
    x_list = [torch.cat(GatherLayer.apply(x), dim=0) for x in x_list]
    x = torch.cat(x_list)
    M = len(x_list)
    N = x_list[0].shape[0]

    # create logits
    logits = x @ x.T

    # create labels
    labels = torch.eye(M * N)
    idx = list(range(M * N))
    idx = idx[N:] + idx[:N]
    labels = labels[idx]

    # create masks
    a = torch.ones((M, M)).view(M, M, 1, 1)
    b = torch.eye(N).view(1, 1, N, N)
    masks = (a * b).transpose(1, 2).reshape(M * N, M * N) - labels
    masks = masks.to(logits).long()

    # mask logits
    logits = logits.masked_fill(masks == 1, float("-inf"))

    # calculate loss
    labels = labels.argmax(dim=1).to(logits).long()
    l_intra = F.cross_entropy(logits[:N], labels[:N])
    l_inter = F.cross_entropy(logits[N:], labels[N:])

    return l_intra, l_inter


def info_nce_loss2(x):
    """
    x: (B, T, D)
    """
    x = torch.cat(GatherLayer.apply(x), dim=0)
    B, T, D = x.shape
    x = x.view(-1, D)

    logits = x @ x.T

    masks = torch.eye(B)[:, :, None, None] * torch.ones((1, 1, T, T))
    masks = masks.transpose(1, 2).reshape(B * T, -1)

    labels = torch.eye(T)
    labels = torch.cat([labels[1:], labels[:1]])
    labels = torch.eye(B)[:, :, None, None] * labels[None, None, :, :]
    labels = labels.transpose(1, 2).reshape(B * T, -1)

    masks = masks - labels

    masks = masks.to(logits)
    labels = labels.to(logits)

    logits = logits.masked_fill(masks == 1, float("-inf"))

    labels = labels.argmax(dim=1).to(logits).long()

    loss = F.cross_entropy(logits, labels)
    return loss
