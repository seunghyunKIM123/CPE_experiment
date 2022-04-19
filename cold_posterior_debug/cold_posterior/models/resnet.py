

import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
import numpy as np

ModuleDef = Any
dtypedef = Any


class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int

    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        x = x + residual

        return nn.relu(x)


class DownSampleResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int

    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=(2, 2),
            features=self.out_channels,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        x = x + self.pad_identity(residual)

        return nn.relu(x)

    @nn.nowrap
    def pad_identity(self, x):
        # Pad identity connection when downsampling
        return jnp.pad(
            x[:, ::2, ::2, ::],
            ((0, 0), (0, 0), (0, 0), (self.out_channels // 4, self.out_channels // 4)),
            "constant",
        )


class ResNet(nn.Module):
    # Define collection of datafields here
    filter_list: Sequence[int]
    N: int
    num_classes: int

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv and linear layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    # For train/test differences, want to pass “mode switches” to __call__
    @nn.compact
    def __call__(self, x, train):

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.1,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.filter_list[0],
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)

        x = norm()(x)
        x = nn.relu(x)

        # First stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[0], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[0],
            out_channels=self.filter_list[1],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Second stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[1], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[1],
            out_channels=self.filter_list[2],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Third stage
        for _ in range(0, self.N):
            x = ResidualBlock(
                in_channels=self.filter_list[2], norm=norm, dtype=self.dtype
            )(x)

        # Global pooling
        x = jnp.mean(x, axis=(1, 2))

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(
            features=self.num_classes, kernel_init=self.kernel_init, dtype=self.dtype
        )(x)

        return x



'''

from functools import partial
from typing import Callable, Optional, Sequence, Tuple
import jax.numpy as jnp
from flax import linen as nn
import jax
from flax.linen.initializers import zeros, ones

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py # for resnet18 cite this link
# https://github.com/cs-giung/giung2-jax/tree/main/projects/Baseline-CIFAR 



class BatchNorm2d(nn.Module):
    momentum: float = 0.9
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x (Array): An input array with shape [N, H, W, C,].
        
        Returns:
            y (Array): An output array with shape [N, H, W, C,].
        """
        mean = jnp.mean(x, (0, 1, 2,))
        sq_mean = jnp.mean(jax.lax.square(x), (0, 1, 2,))
        var = jnp.maximum(0., sq_mean - jax.lax.square(mean))
        y = x - jnp.reshape(mean, (1, 1, 1, -1,))
        y = jnp.multiply(y, jnp.reshape(jax.lax.rsqrt(var + self.epsilon), (1, 1, 1, -1,)))

        w = self.param('w',lambda rng, shape: jnp.ones(shape), x.shape[-1]) # 이게 임의로 파라미터 벡터에 넣어주는 코드
        b = self.param('b', lambda rng, shape: jnp.zeros(shape), x.shape[-1])
        y = jnp.multiply(y, jnp.reshape(w, (1, 1, 1, -1,)))
        y = jnp.add(y, jnp.reshape(b, (1, 1, 1, -1,)))
        return y


class resnetblock(nn.Module):
    channels: int # output의 channel 의미한다. input 채널은 mlp같이 알아서 모양대로 계산해줌
    stride: Tuple[int,int]
    expansion= 1
    norm : nn.Module
    kernel_init: Callable = nn.initializers.kaiming_normal()
    @nn.compact
    def __call__(self,x):
        y = nn.Conv(self.channels, kernel_size =(3,3), strides= self.stride, padding=((1,1),(1,1)),
        kernel_init = self.kernel_init)(x)
        y= self.norm()(y)
        y= nn.relu(y)
        y= nn.Conv(self.channels, kernel_size =(3,3), strides=(1,1), padding=((1,1),(1,1)),
        kernel_init = self.kernel_init)(y)
        y= self.norm()(y)

        if self.stride != (1,1) or x.shape[-1] != self.channels: # block을 반복해서 channel 수 늘릴 때 발생함
            short_cut=nn.Conv(self.channels,kernel_size=(1,1), strides=self.stride, use_bias=False,kernel_init = self.kernel_init)(x) # 이부분 64가 좀 문제있을지도
            short_cut=self.norm()(short_cut)
            y=y+short_cut
            # 채우기
        else:
            y = y+x

        y = nn.relu(y)

        return y

# padding = 1 을 줘야하는지 말아야하는지 결정해야함

# https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/jax_models.py


## nan 떠서 relu로 바꿈

class ResNet(nn.Module): # for resnet 20,32,44 line  -- cifar
    num_classes: int
    depth: int  
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self,x,train):
        num_filters = 16
        num_res_blocks = (self.depth - 2) // 6
        if (self.depth - 2) % 6 != 0:
            raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

        norm = partial(nn.BatchNorm,use_running_average=not train,momentum=0.1,epsilon=1e-5,)
        
        x = nn.Conv(num_filters,kernel_size=(3,3),strides=(1,1), padding='SAME',kernel_init = self.kernel_init)(x)
        x = norm()(x)
        x = nn.relu(x)
        
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = (2,2) if stack>0 and res_block==0 else (1,1)

                x= resnetblock(num_filters,stride=strides,norm=norm)(x)

            num_filters = num_filters *2

        x = jnp.mean(x, axis=(1, 2))
        x = x.reshape(x.shape[0], -1)      
        y = nn.Dense(self.num_classes)(x)

        return y
'''