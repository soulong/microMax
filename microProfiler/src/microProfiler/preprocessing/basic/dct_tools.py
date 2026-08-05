"""2D and 3D discrete cosine transform tools."""

from abc import ABC, abstractmethod #, abstractproperty
from functools import partial
import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import scipy.fft as osp_fft
from jax import lax
# from jax._src.numpy.util import _wraps
from functools import wraps
from jax._src.util import canonicalize_axis

__all__ = [
    "dct2d",
    "idct2d",
    "dct3d",
    "idct3d",
]

# `_W4` copied from jax._src.scipy.fft
def _W4(N, k):
    return jnp.exp(-0.5j * jnp.pi * k / N)


def _slice_ref_in_dim(x, start, stop, stride, axis):
    return tuple(
        slice(start, stop, stride) if dim == axis else slice(None, None)
        for dim in range(x.ndim)
    )


def _dct_interleave_inverse(x, axis):
    inverse = True
    if inverse:
        N = x.shape[axis]
        v0 = lax.slice_in_dim(x, None, (N + 1) // 2, 1, axis)
        v1 = lax.rev(lax.slice_in_dim(x, (N + 1) // 2, None, 1, axis), (axis,))
        ref0 = _slice_ref_in_dim(x, None, None, 2, axis)
        ref1 = _slice_ref_in_dim(x, 1, None, 2, axis)
        out = jnp.zeros(x.shape, dtype=x.dtype)
        out = out.at[ref0].set(v0)
        out = out.at[ref1].set(v1)
        return out


# REFACTOR I'm sure there is a much cleaner way of doing this
# @_wraps(osp_fft.idct)
@wraps(osp_fft.idct)
def idct(x, norm=None, axis=-1):
    axis = canonicalize_axis(axis, x.ndim)
    N = x.shape[axis]
    k = lax.expand_dims(jnp.arange(N), [a for a in range(x.ndim) if a != axis])
    V = _W4(N, -k) * x

    x0 = lax.slice_in_dim(x, None, 1, 1, axis) / 2
    V = V.at[_slice_ref_in_dim(V, None, 1, 1, axis)].set(x0)

    if norm == "ortho":
        factor = lax.concatenate(
            [
                lax.full((1,), 2 * jnp.sqrt(N), V.dtype),
                lax.full((N - 1,), jnp.sqrt(2 * N), V.dtype),
            ],
            0,
        )
        factor = lax.expand_dims(factor, [a for a in range(V.ndim) if a != axis])

        V = V * factor

    v = jnp.fft.ifft(V, axis=axis)

    xrev = lax.slice_in_dim(x, 1, None, 1, axis)
    xrev = lax.rev(xrev, (axis,))

    out = _dct_interleave_inverse(v, axis)
    return out.real


class DCT(ABC):
    # @abstractproperty
    @property
    @abstractmethod
    def _backend(self) -> str:
        ...

    @staticmethod
    @abstractmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def dct3d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def idct3d(arr: np.ndarray) -> np.ndarray:
        ...

class JaxDCT(DCT):
    _backend = "JAX"

    @staticmethod
    def dctnd(arr: np.ndarray) -> np.ndarray:
        return jax.scipy.fft.dctn(arr, norm="ortho", type=2)

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.dctnd(arr)

    @staticmethod
    def dct3d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.dctnd(arr)

    # custom idct since JAX only implements dct type 2 (not idct, dct type 3)
    @staticmethod
    @partial(jit, static_argnames=["ndims"])
    def idctnd(arr: np.ndarray, ndims: int) -> np.ndarray:
        for i in range(ndims):
            arr = idct(arr, norm="ortho", axis=i)
        return arr

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.idctnd(arr, 2)

    @staticmethod
    def idct3d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.idctnd(arr, 3)
