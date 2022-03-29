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


'''One step update function for each algorithm. using jax.lax.scan to train on whole batch '''
# 아 이건 이렇게 한스텝한스텝 짜기가 좀 애매하네요 garipov pytorch 코드 보고 그냥 새로짜는게 낫겠어요
# 여러모로 걍 클래스로 짜는게 좋긴하네..


# https://github.com/timgaripov/swa/blob/411b2fcad59bec60c6c9eb1eb19ab906540e5ea2/train.py#L94

# weight decay 전부 

class SGHMC:
    def __init__(self, config):
        self.config = config
        self.metadata = get_metadata(config)
        self.net = ResNet20(num_classes=10,depth=20,norm=BatchNorm2d) # get_model 함수에 config 넣어서 선택할 수 있게해야함

    def init_model(self, key):
        net = ResNet20(num_classes=10,depth=20,norm=BatchNorm2d)
        theta = net.init(key, jnp.ones((1,)+self.metadata['shape']))
        return net, theta

    def init_state(self):
        key = jax.random.PRNGKey(self.config.seed)
        key, init_key, normal_key = jax.random.split(key, 3)
        net, theta = self.init_model(init_key)
        theta, _ = normal_like_tree(theta, normal_key, std=self.config.init_param_sd)
        momentum = jax.tree_map(jnp.zeros_like,theta)
        t_curr =0
        state = theta, momentum, t_curr, normal_key
        return key, state

    def is_ensembling_epoch(self,epoch):
        if epoch <= self.config.burnin_epochs:
            return False

        if ((epoch - 1) % self.config.cycle_epochs + 1) / self.config.cycle_epochs <= self.config.expl_ratio:
            return False

        return (epoch % self.config.thin == 0)

    def make_train_epoch(self, train_dataset):

        def loss_fn(params, batch):
            x, y = batch
            logits = self.net.apply(params,x)
            nll = cross_entropy(logits, y).mean()
            l2 = (1/self.metadata['num_train'])*self.config.weight_decay* l2_reg(params)
            loss = nll + l2
            return (1/self.config.temperature)*loss

        def update_fn(state, batch):

            cycle_steps = (self.metadata['num_train']//self.config.batch_size) * self.config.cycle_epochs # batch 개수가 되야되는거아냐?
            init_lr = self.config.sghmc_init_lr

            def schedule(step):
                t = step
                t = (t % cycle_steps) / cycle_steps
                return 0.5 * init_lr * (1 + jnp.cos(t * jnp.pi))

            theta, momentum, t_curr, normal_key = state
            eps = jnp.sqrt(schedule(t_curr) / self.metadata['num_train'])
            grad_fn = jax.value_and_grad(loss_fn, argnums=0)
            loss, grad_U = grad_fn(theta, batch)
            noise, normal_key = normal_like_tree(theta, normal_key)
            sample = (t_curr % cycle_steps) / cycle_steps > self.config.expl_ratio   
            # expl_ratio = 1 로 잡으면 sgd with momentum으로 바꿀 수 있음
            a = jnp.sqrt(2 * self.config.alpha * eps)

            def mult(b):  # 상수곱
                return a * b

            noise = jax.lax.cond(sample, lambda noise: jax.tree_map(mult, noise),
                                 lambda noise: jax.tree_map(jnp.zeros_like, noise), noise)

            momentum = jax.tree_multimap(
                lambda m, g, n: (1 - self.config.alpha) * m - eps * self.metadata['num_train'] * g + n, momentum,
                grad_U, noise)
            update = jax.tree_map(lambda x: x*eps,momentum)
            theta = tree_add(theta, update)

            t_curr = t_curr + 1
            state = theta, momentum, t_curr, normal_key
            return state, loss

        @jax.jit
        def train_epoch(key, state):

            def train_step(state, batch_indices):
                batch = jax.tree_map(lambda x: x[batch_indices], train_dataset)
                state, loss = update_fn(state, batch)
                return state, loss

            key, perm_key = jax.random.split(key, 2)
            train_indices = batch_divider(perm_key, self.config.batch_size, num_data=self.metadata['num_train'])
            state, loss = jax.lax.scan(train_step, state, train_indices)

            return key, state

        return train_epoch

    def sghmc_test_step(self,samples,batch_indices,test_dataset):
        K=len(samples)
        batch = jax.tree_map(lambda x: x[batch_indices], test_dataset)
        x, y = batch
        logits = []
        nlls = []
        for theta in samples:
            logits_ = self.net.apply(theta,x)
            logits.append(logits_)
            nlls.append(cross_entropy(logits_, y))
        nll = jnp.mean(-logsumexp(-jnp.stack(nlls, 0), 0) + jnp.log(K))
        avg_logits = logsumexp(jnp.stack(logits, 0), 0) - jnp.log(K)
        acc = (avg_logits.argmax(-1) == y).mean()
        return samples, (nll, acc)

    def train(self):
        train_dataset = get_cifar(name=self.config.dataset, split='train',data_augmentation=self.config.DA)
        test_dataset = get_cifar(name=self.config.dataset, split='test',data_augmentation=False)
        train_epoch = self.make_train_epoch(train_dataset)

        key, state = self.init_state()
        samples = []
        sghmc_test_step = partial(self.sghmc_test_step,test_dataset=test_dataset) # 이거부터 좀 문제임
        for epoch in range(1, self.config.num_epoch + 1):
            key,state= train_epoch(key,state)

            if self.is_ensembling_epoch(epoch):
                samples.append(to_state_dict(state[0]))  # state[0] = theta

            if (epoch > self.config.burnin_epochs) & (epoch % self.config.test_freq == 0):
                key,perm_key = key, perm_key = jax.random.split(key, 2)
                test_indices = batch_divider(perm_key, self.config.batch_size, num_data=self.metadata['num_test'])
                samples, (nlls, accs) = jax.lax.scan(sghmc_test_step,samples, xs= test_indices)
                print(epoch, jnp.mean(nlls), jnp.mean(accs))
                
        return samples

