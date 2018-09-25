import functools
import operator

import numpy as np
from chainer.functions.connection import linear
from chainer import initializers, initializer
from chainer import link
from chainer import variable


class FanConstant(initializer.Initializer):
    """initialize constants `scale` * (1 / sqrt{fan_in})"""
    def __init__(self, scale, dtype=None):
        self.scale = scale
        super().__init__(dtype=dtype)

    def __call__(self, array):
        if len(array.shape) == 1:
            fan_in = array.shape[0]
        else:
            fan_in, fan_out = initializer.get_fans(array.shape)
        fill_value = self.scale / np.sqrt(fan_in)
        initializers.Constant(fill_value)(array)


class NoisyLinear(link.Link):
    """NoisyLinear layer as described in NoisyNet[1].

    For unfactorised (independent) case, initial_mu should be LeCunUniform with scale 1 and
    initial_sigma a constant 0.017.
    For factorized case, initial_mu should be LeCunUniform with scale sqrt{1/3} and
    initial_sigma a constant 0.5/sqrt{in_size}.

    [1]: Fortunato et. al. (2017). Noisy Networks for Exploration, http://arxiv.org/abs/1706.10295
    """
    def __init__(self, in_size, out_size=None, nobias=False, initial_mu=None, initial_sigma=None, factorized=False):
        super().__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size
        self.factorized = factorized

        if initial_mu is None:
            if self.factorized:
                initial_mu = initializers.LeCunUniform(scale=np.sqrt(1 / 3))
            else:
                initial_mu = initializers.LeCunUniform(scale=1.0)
        if initial_sigma is None:
            if self.factorized:
                initial_sigma = FanConstant(scale=0.5)
            else:
                initial_sigma = 0.017

        with self.init_scope():
            mu_initializer = initializers._get_initializer(initial_mu)
            sigma_initializer = initializers._get_initializer(initial_sigma)
            self.mu_w = variable.Parameter(mu_initializer)
            self.sigma_w = variable.Parameter(sigma_initializer)
            if nobias:
                self.mu_b = None
                self.sigma_b = None
            else:
                # `LeCunUniform` does not allow one-dim size initialization
                self.mu_b = variable.Parameter(mu_initializer, (1, out_size))
                self.sigma_b = variable.Parameter(sigma_initializer, out_size)
            if in_size is not None:
                self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.mu_w.initialize((self.out_size, in_size))
        self.sigma_w.initialize((self.out_size, in_size))

    def _eps(self, shape, dtype):
        xp = self.xp
        r = xp.random.standard_normal(shape).astype(dtype)
        return xp.copysign(xp.sqrt(xp.abs(r)), r)

    def __call__(self, x):
        if self.mu_w.data is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)
        dtype = self.mu_w.dtype
        out_size, in_size = self.mu_w.shape
        if self.factorized:
            eps_in = self._eps(in_size, dtype)
            eps_b = self._eps(out_size, dtype)
            eps_w = self.xp.outer(eps_b, eps_in)
        else:
            eps_w = self._eps((out_size, in_size), dtype)
            eps_b = self._eps(out_size, dtype)
        W = self.mu_w + self.sigma_w * eps_w
        if self.mu_b is None:
            b = None
        else:
            b = self.mu_b.reshape((out_size,)) + self.sigma_b * eps_b
        return linear.linear(x, W, b)
