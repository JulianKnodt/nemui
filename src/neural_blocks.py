import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms.functional as TVF

from itertools import chain
from typing import Optional, Union
import math

from .utils import ( fourier, create_fourier_basis, smooth_min )

class PositionalEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    max_freq: float = 6.,
    N: int = 64,
    log_sampling: bool = False,
  ):
    super().__init__()
    if log_sampling:
      bands = 2**torch.linspace(1, max_freq, steps=N, requires_grad=False, dtype=torch.float)
    else:
      bands = torch.linspace(1, 2**max_freq, steps=N, requires_grad=False, dtype=torch.float)
    self.bands = nn.Parameter(bands, requires_grad=False)
    self.input_dims = input_dims
  def output_dims(self): return self.input_dims * 2 * len(self.bands)
  def forward(self, x):
    assert(x.shape[-1] == self.input_dims)
    raw_freqs = torch.tensordot(x, self.bands, dims=0)
    raw_freqs = raw_freqs.reshape(x.shape[:-1] + (-1,))
    return torch.cat([ raw_freqs.sin(), raw_freqs.cos() ], dim=-1)

class FourierEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    # TODO rename this num freqs to be more accurate.
    freqs: int = 128,
    sigma: int = 1 << 5,
    device="cpu",
  ):
    super().__init__()
    self.input_dims = input_dims
    self.freqs = freqs
    self.basis = create_fourier_basis(freqs, features=input_dims, freq=sigma, device=device)
    self.basis = nn.Parameter(self.basis, requires_grad=False)
    self.extra_scale = 1
  def output_dims(self): return self.freqs * 2
  def forward(self, x): return fourier(x, self.extra_scale * self.basis)
  def scale_freqs(self, amt: 1+1e-5, cap=2):
    self.extra_scale *= amt
    self.extra_scale = min(self.extra_scale, cap)

class LearnedFourierEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    num_freqs: int = 16,
    sigma: int = 1 << 5,
    device="cpu",
  ):
    super().__init__()
    self.input_dims = input_dims
    self.n_freqs = num_freqs
    self.basis = create_fourier_basis(num_freqs, features=input_dims, freq=sigma, device=device)
    self.basis = nn.Parameter(self.basis, requires_grad=False)
    self.extra_scale = nn.Parameter(torch.tensor(1, requires_grad=True), requires_grad=True)
  def output_dims(self): return self.n_freqs * 2
  def forward(self, x): return fourier(x, self.extra_scale * self.basis)

# It seems a cheap approximation to SIREN works just as well? Not entirely sure.
class NNEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    out: int = 32,
    device=None,
  ):
    super().__init__()
    self.fwd = nn.Linear(input_dims, out)
  def output_dims(self): return self.fwd.out_features
  def forward(self, x):
    assert(x.shape[-1] == self.fwd.in_features)
    return torch.sin(30 * self.fwd(x))

# how to initialize the MLP, otherwise defaults to torch
mlp_init_kinds = {
  None,
  "zero",
  "kaiming",
  "siren",
  "xavier",
}

class SkipConnMLP(nn.Module):
  "MLP with skip connections and fourier encoding"
  def __init__(
    self,

    num_layers = 5,
    hidden_size = 256,
    in_size=3, out=3,

    skip=3,
    activation = nn.LeakyReLU(inplace=True),
    latent_size=0,

    enc: Optional[Union[FourierEncoder, PositionalEncoder, NNEncoder]] = None,

    # Record the last layers activation
    last_layer_act = False,

    linear=nn.Linear,

    init=None,
  ):
    assert(init in mlp_init_kinds), "Must use init kind"
    super(SkipConnMLP, self).__init__()
    self.in_size = in_size
    map_size = 0

    self.enc = enc
    if enc is not None: map_size += enc.output_dims()

    self.dim_p = in_size + map_size + latent_size
    self.skip = skip
    self.latent_size = latent_size
    skip_size = hidden_size + self.dim_p

    # with skip
    hidden_layers = [
      linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size, hidden_size,
      ) for i in range(num_layers)
    ]


    self.init =  nn.Linear(self.dim_p, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = nn.Linear(hidden_size, out)
    weights = [
      self.init.weight, self.out.weight,
      *[l.weight for l in self.layers],
    ]
    biases = [
      self.init.bias, self.out.bias,
      *[l.bias for l in self.layers],
    ]
    if init is None:
      ...
    elif init == "zero":
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    elif init == "xavier":
      for t in weights: nn.init.xavier_uniform_(t)
      for t in biases: nn.init.zeros_(t)
    elif init == "siren":
      for t in weights:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(t)
        a = math.sqrt(6 / fan_in)
        nn.init._no_grad_uniform_(t, -a, a)
      for t in biases: nn.init.zeros_(t)
    elif init == "kaiming":
      for t in weights: nn.init.kaiming_normal_(t, mode='fan_out')
      for t in biases: nn.init.zeros_(t)

    self.activation = activation
    self.last_layer_act = last_layer_act

  def forward(self, p, latent: Optional[torch.Tensor]=None):
    batches = p.shape[:-1]
    init = p.reshape(-1, p.shape[-1])

    if self.enc is not None: init = torch.cat([init, self.enc(init)], dim=-1)
    if self.latent_size != 0:
      assert(latent is not None), "Did not pass latent vector when some was expected"
      init = torch.cat([init, latent.reshape(-1, self.latent_size)], dim=-1)
    else: assert((latent is None) or (latent.shape[-1] == 0)), "Passed latent vector when none was expected"

    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x))
    if self.last_layer_act: setattr(self, "last_layer_out", x.reshape(batches + (-1,)))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))
  # smoothness of this sample along a given dimension for the last axis of a tensor
  def l2_smoothness(self, sample, values=None, noise=1e-1, dim=-1):
    if values is None: values = self(sample)
    adjusted = sample + noise * torch.rand_like(sample)
    adjusted = self(adjusted)
    return (values-adjusted).square().mean()
  def zero_last_layer(self):
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)
  def uniform_last_layer(self, a=1e-4):
    nn.init.uniform_(self.out.weight, -a, a)
    nn.init.uniform_(self.out.bias, -a, a)
  # add an additional method for capt
  def variance(self, shape=None):
    return torch.stack([l.var(shape) for l in self.layers], dim=0)



