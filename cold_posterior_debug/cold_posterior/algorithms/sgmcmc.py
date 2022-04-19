from jax._src.api import T
from utils.util_func import *
from utils.loss import *
import jax
import jax.numpy as jnp
from models.resnet import *
from flax.serialization import to_state_dict
from utils.data import *
from functools import partial
from jax.scipy.special import logsumexp
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import torchvision
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.utils.data import DataLoader
# One step update function for each algorithm. using jax.lax.scan to train on whole batch
# https://github.com/timgaripov/swa/blob/411b2fcad59bec60c6c9eb1eb19ab906540e5ea2/train.py#L94
# weight decay 전부

def augmentation(image):
    ''' input: N,H,W,C -> output: N,H,W,C '''

    image = jnp.pad(array=image,pad_width=((2,2),(2,2),(0,0)))

    crop_size = (32,32,3)
    h, w = 36,36
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]

    if np.random.rand() < 0.5:
      image = image[:, ::-1, :]
    
    return image

aug_map = jax.vmap(augmentation,in_axes=0)



class SGHMC:
    def __init__(self, config):
        self.config = config
        self.metadata = get_metadata(config)
        self.net = ResNet(filter_list=[16,32,64], N=3, dtype=jnp.float32, num_classes=self.metadata['num_classes'])
         # get_model 함수에 config 넣어서 선택할 수 있게해야함
    def init_model(self, key):
        net = self.net
        variables = net.init(key, jnp.ones((1,)+self.metadata["shape"]),train=True)
        theta, batch_stats = variables["params"],variables["batch_stats"]
        return theta, batch_stats

    def init_state(self):
        key = jax.random.PRNGKey(self.config.seed)
        key, init_key, normal_key = jax.random.split(key, 3)
        theta,batch_stats = self.init_model(init_key)

        momentum = jax.tree_map(jnp.zeros_like,theta) 
        t_curr =0
        state = theta,batch_stats, momentum, t_curr, normal_key
        return key, state

    def is_ensembling_epoch(self,epoch):
        if epoch <= self.config.burnin_epochs:
            return False
        if ((epoch - 1) % self.config.cycle_epochs + 1) / self.config.cycle_epochs <= self.config.expl_ratio:
            return False
        return (epoch % self.config.thin == 0)
        
    def make_train_epoch(self, train_dataset):

        if self.config.likelihood=='dirichlet':

            def loss_fn(theta,batch_stats,batch,noise=1e-2):
                def gauss_log_lik(mean,var,x):
                    log_lik = (0.5*(x-mean)**2/var).sum(-1)
                    return log_lik

                x, y = batch
                labels = jax.nn.one_hot(y,self.metadata['num_classes'])
                alpha = labels + noise
                gamma_var = jnp.log((1 / alpha + 1))
                gamma_mean = jnp.log(alpha) - gamma_var / 2
                logits,new_batch_stat = self.net.apply({'params':theta,'batch_stats':batch_stats},x
                    ,mutable=["batch_stats"],
                train=True
                )

                noisy_nll = gauss_log_lik(gamma_mean,gamma_var,logits).mean() # (batch,num_classes) 에서 num_classes 에 대한 sum
                l2 = (1/self.metadata["num_train"])*self.config.weight_decay* l2_reg(theta)

                loss = (noisy_nll + l2)
                nll=noisy_nll
                acc = (logits.argmax(-1) == y).mean()
                return loss,(new_batch_stat['batch_stats'],nll,acc) 

        elif self.config.likelihood=='softmax':

            def loss_fn(theta,batch_stats,batch):
                x, y = batch
                logits,new_batch_stat = self.net.apply({'params':theta,'batch_stats':batch_stats},x
                ,mutable=["batch_stats"],
                train=True
                )

                nll = cross_entropy(logits, y).mean()
                l2 = (1/self.metadata["num_train"])*self.config.weight_decay* l2_reg(theta)
                loss =  (nll + l2)
                acc = (logits.argmax(-1) == y).mean()
                return loss,(new_batch_stat['batch_stats'],nll,acc) 

        def update_fn(state, batch):   # 이거 코드를 좀 바꿔야될거같어 scalable하지가 않어

            cycle_steps = (self.metadata["num_train"]//self.config.batch_size) * self.config.cycle_epochs # batch 개수가 되야되는거아냐?
            init_lr = self.config.sghmc_init_lr
            burnin_step = self.config.burnin_epochs*(self.metadata["num_train"]//self.config.batch_size)

            def schedule(step):  #C(t)
                t = step
                t = (t % cycle_steps) / cycle_steps
                return 0.5 * (1 + jnp.cos(t * jnp.pi))

            theta,batch_stats, momentum, t_curr, normal_key = state

            eps = self.config.sghmc_init_lr / self.metadata["num_train"]*schedule(t_curr) # debugged - lr 자체를 바꾸는게 아니고 뒤에 C(t) 곱해주는거
            

            if self.config.alg== "sgd":

                eps = self.config.sghmc_init_lr / self.metadata["num_train"]*schedule(t_curr)


            grad_fn = jax.value_and_grad(loss_fn, argnums=0,has_aux=True)
            (loss,(batch_stats,nll,acc)), grad_U = grad_fn(theta, batch_stats,batch)
            noise, normal_key = normal_like_tree(theta, normal_key)
            sample = ((t_curr % cycle_steps) / cycle_steps > self.config.expl_ratio) & (t_curr > burnin_step)

            a = jnp.sqrt(2 * self.config.alpha*self.config.temperature*eps) # jnp.sqrt(2 * self.config.alpha * eps)

            def mult(b):  # 상수곱
                return a * b

            noise = jax.lax.cond(sample, lambda noise: jax.tree_map(mult, noise),
                                 lambda noise: jax.tree_map(jnp.zeros_like, noise), noise)
            if self.config.alg== "sgd":
                noise = jax.tree_map(jnp.zeros_like,noise)

            momentum = jax.tree_multimap(
                lambda m, g, n: (1 - self.config.alpha) * m - eps * self.metadata["num_train"] * g + n, momentum,
                grad_U, noise)
            update = jax.tree_map(lambda x: x*eps,momentum) # 이거 때문인가? 수렴속도 느린게
            theta = tree_add(theta, momentum) # update로 할건지 momentum으로 할건지 정해야함
            t_curr = t_curr + 1
            state = theta, batch_stats, momentum, t_curr, normal_key
            return state, (nll,acc)

        @jax.jit
        def train_epoch(key, state):

            def train_step(state, batch_indices):
                batch = jax.tree_map(lambda x: x[batch_indices], train_dataset)
                image,label = batch
                for idx,img in enumerate(image):
                    image = image.at[idx].set(augmentation(img))
                #image = aug_map(image)
                state, (nll,acc) = update_fn(state, (image,label))
                return state, (nll,acc)

            key, perm_key = jax.random.split(key, 2)
            train_indices = batch_divider(perm_key, self.config.batch_size, num_data=self.metadata["num_train"])
            state, (nll,acc) = jax.lax.scan(train_step, state, train_indices)
            return key, state,(jnp.mean(nll),jnp.mean(acc))
               
        return train_epoch

    def sghmc_test_step(self,samples,batch_indices,test_dataset): # 여기서는 batch norm에 대한 mutable에 대해서 false를 줘야함
        K=len(samples)
        batch = jax.tree_map(lambda x: x[batch_indices], test_dataset)
        x, y = batch
        logits = []
        nlls = []
        for params in samples:
            logits_ = self.net.apply(params,x # 여기서 문제가 나는거같긴한데..
            ,mutable=False,
            train=False
            )  
            logits.append(logits_)
            nlls.append(cross_entropy(logits_, y))
        nll = jnp.mean(-logsumexp(-jnp.stack(nlls, 0), 0) + jnp.log(K))
        avg_logits = logsumexp(jnp.stack(logits, 0), 0) - jnp.log(K)
        acc = (avg_logits.argmax(-1) == y).mean()
        return samples, (nll, acc)

    def sgd_test_step(self,theta,batch_indices,batch_stats,test_dataset):
        batch = jax.tree_map(lambda x: x[batch_indices], test_dataset)
        x, y = batch
        logit = self.net.apply({'params':theta,'batch_stats':batch_stats},x,train=False)

        nll = cross_entropy(logit,y)
        acc = (logit.argmax(-1) == y).mean()
        return theta, (nll,acc)

    def train(self):
  
        train_dataset = get_cifar(name =self.config.dataset, split ='train',data_augmentation =self.config.aug) # 여기서 1을 준다는거죠.
        test_dataset = get_cifar(name =self.config.dataset, split='test',data_augmentation=0)
        train_epoch = self.make_train_epoch(train_dataset)
        key, state = self.init_state()
        samples = []
        sghmc_test_step = partial(self.sghmc_test_step,test_dataset=test_dataset)


        for epoch in range(1, self.config.num_epoch + 1):

            key,state,(nll,acc)= train_epoch(key,state)
            print("train nll:",nll,'\n',"train_acc:",acc)

            if self.is_ensembling_epoch(epoch) & (self.config.alg== 'sghmc'):
                samples.append({'params':state[0],'batch_stats':state[1]})  # 딕셔너리를 주는걸로 바꿔야된다.

            if (self.config.alg== "sgd") & (self.config.num_epoch +1 == epoch):
                samples.append({'params':state[0],'batch_stats':state[1]})

            if (self.config.alg=="sghmc") & (epoch > self.config.burnin_epochs) & (epoch % self.config.test_freq == 0) & (len(samples) > 0):
                sghmc_test_step = partial(self.sghmc_test_step,test_dataset=test_dataset)
                key,perm_key = jax.random.split(key, 2)
                test_indices = batch_divider(jax.random.PRNGKey(1), self.config.batch_size, num_data=self.metadata['num_test'])
                samples, (nlls, accs) = jax.lax.scan(sghmc_test_step,samples, xs= test_indices)
                print(epoch, jnp.mean(nlls), jnp.mean(accs),len(samples))
            
            if (self.config.alg=="sgd") & (epoch % self.config.test_freq == 0):
                sgd_test_step = partial(self.sgd_test_step,batch_stats=state[1],test_dataset=test_dataset) 
                theta = to_state_dict(state[0])
                key,perm_key = jax.random.split(key, 2)
                test_indices = batch_divider(jax.random.PRNGKey(1), self.config.batch_size, num_data=self.metadata['num_test'])
                theta, (nlls, accs) = jax.lax.scan(sgd_test_step,theta, xs= test_indices)
                print(epoch, jnp.mean(nlls), jnp.mean(accs))

        return samples



# def Estimate_M(self,grad,momeuntum):


# TODO - Izmailov랑 wenzel 코드 성능 확인해보고 맞춰보기

