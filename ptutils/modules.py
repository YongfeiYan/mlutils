from os import path
import numpy as np
from torch import nn
import torch


def get_embedding(vocab_path=None,
                  embedding_np=None,
                  num_embeddings=0, embedding_dim=0, freeze=True, **kargs):
    """Create embedding from:
    1. saved numpy vocab array, vocab_path, freeze
    2. numpy embedding array, embedding_np, freeze
    3. raw embedding n_vocab, embedding_dim
    """
    if isinstance(vocab_path, str) and path.exists(vocab_path):
        embedding_np = np.load(vocab_path)
    if embedding_np is not None:
        return nn.Embedding.from_pretrained(torch.Tensor(embedding_np), freeze=freeze)
    return nn.Embedding(num_embeddings, embedding_dim, **kargs)


# extract last output in last time  step


def extract_last_timestep(output, lengths, batch_first):
    """Get the output of last time step.
    output: seq_len x batch_size x dim if not batch_first. Else batch_size x seq_len x dim
    length: one dimensional torch.LongTensor of lengths in a batch.
    """
    idx = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    idx = idx.unsqueeze(time_dimension)
    if output.is_cuda:
        idx = idx.cuda(output.data.get_device())
    return output.gather(time_dimension, idx).squeeze(time_dimension)


class Sequential(nn.Sequential):
    """Wrapper for torch.nn.Sequential."""
    def __init__(self, args):
        super(Sequential, self).__init__(args)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        return input


def get_mlp(features, activate):
    if isinstance(activate, str):
        activate = getattr(nn, activate)
    layers = []
    for in_f, out_f in zip(features[:-1], features[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activate())
    return nn.Sequential(*layers)


def kld_normal(mu, log_sigma):
    """KL divergence to standard normal distribution.
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)

