from functools import partial
from typing import Callable, Optional, Sequence, Tuple
import jax.numpy as jnp
from flax import linen as nn
import jax

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py # for resnet18 cite this link
# https://github.com/cs-giung/giung2-jax/tree/main/projects/Baseline-CIFAR 
#
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
    @nn.compact
    def __call__(self,x):
        y = nn.Conv(self.channels, kernel_size =(3,3), strides= self.stride, padding=((1,1),(1,1)))(x)
        y= self.norm()(y)
        y= nn.relu(y)
        y= nn.Conv(self.channels, kernel_size =(3,3), strides=(1,1), padding=((1,1),(1,1)))(y)
        y= self.norm()(y)

        if self.stride != (1,1) or x.shape[-1] != self.channels: # block을 반복해서 channel 수 늘릴 때 발생함
            short_cut=nn.Conv(self.channels,kernel_size=(1,1), strides=self.stride, use_bias=False)(x) # 이부분 64가 좀 문제있을지도
            short_cut=self.norm()(short_cut)
            y=y+short_cut
            # 채우기
        else:
            y = y+x

        y = nn.relu(y)

        return y

# padding = 1 을 줘야하는지 말아야하는지 결정해야함

class ResNet(nn.Module): # for resnet 18,34,50,101,152  imagenet
    stage_sizes: Sequence[int] # resnet18 => [2,2,2,2]
    out_channels = 64 # for cifar 10,100
    num_classes: int # 이거 config로 바꿔야 dd될듯
    norm: nn.Module
    @nn.compact
    def __call__(self,x):
        block_idx = 0
        y = nn.Conv(self.out_channels,kernel_size=(3,3),strides=(1,1), padding=((1,1),(1,1)), use_bias=False)(x)
        y=  self.norm()(y)  # first block
        block_idx = block_idx + 1

        for i,block_size in enumerate(self.stage_sizes): # (1,1) (1,1) (1,1)
            for j in range(block_size):
                strides = (2,2) if i>0 and j==0 else (1,1)
                y = resnetblock(self.out_channels*(2**i),stride=strides,norm=self.norm)(y)
        y = nn.avg_pool(y,window_shape=(4,4))
        y = nn.Dense(self.num_classes)(y)
        y = jnp.squeeze(y)
        return y


# https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/jax_models.py




class ResNet20(nn.Module): # for resnet 20,32,44 line  -- cifar
    num_classes: int
    norm : nn.Module
    depth: int

    def _resnet_layer(self,x,num_filters,strides,kernel_size=(3,3)):

        x = nn.Conv(num_filters,kernel_size=kernel_size,strides=strides, padding='SAME')(x)
        x = self.norm()(x)
        x = nn.relu(x)

        return x

    @nn.compact
    def __call__(self,x):
        num_filters = 16
        num_res_blocks = (self.depth - 2) // 6
        if (self.depth - 2) % 6 != 0:
            raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')
        
        x = self._resnet_layer(x,num_filters=num_filters,strides=(1,1))

        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = (1,1)

                if stack > 0 and res_block ==0:
                    strides = (2,2)

                y = self._resnet_layer(x,num_filters = num_filters,strides=strides)
                y = self._resnet_layer(y,num_filters = num_filters,strides=(1,1))

                if stack >0 and res_block ==0:
           
                    x = self._resnet_layer(x,num_filters = num_filters,strides=strides,kernel_size=(1,1))
                
                x = nn.relu(x+y)

            num_filters = num_filters *2
        x = nn.avg_pool(x,window_shape = (8,8))
        x=jnp.squeeze(x)
        x = nn.Dense(self.num_classes)(x)

        return x
