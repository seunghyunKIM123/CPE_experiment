from jax._src.api import T
from utils.util_func import *
from utils.loss import *
import jax
import jax.numpy as jnp
jnp.ones(1)
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
# swa는 아예 한 에폭마다 full batch 돌리는 코드를 여기에 짜주는게 나은듯


# https://github.com/timgaripov/swa/blob/411b2fcad59bec60c6c9eb1eb19ab906540e5ea2/train.py#L94

# weight decay 전부 

class SGHMC:
    def __init__(self, config):
        self.config = config
        self.metadata = get_metadata(config)
        self.net = ResNet(stage_sizes = [2,2,2,2],num_classes = self.metadata['num_classes'],norm = BatchNorm2d)

    def init_model(self, key):
        net = ResNet(stage_sizes = [2,2,2,2],num_classes = self.metadata['num_classes'],norm = BatchNorm2d)
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
            l2 = self.config.weight_decay* l2_reg(params)
            loss = nll + l2
            return loss

        def update_fn(state, batch):

            cycle_steps = self.config.batch_size * self.config.cycle_epochs
            init_lr = self.config.sghmc_init_lr

            def schedule(step):
                t = step
                t = (t % cycle_steps) / cycle_steps
                return 0.5 * init_lr * (1 + jnp.cos(t * jnp.pi))

            theta, momentum, t_curr, normal_key = state
            eps = schedule(t_curr) / self.metadata['num_train']
            grad_fn = jax.value_and_grad(loss_fn, argnums=0)
            loss, grad_U = grad_fn(theta, batch)
            noise, normal_key = normal_like_tree(theta, normal_key)
            sample = (t_curr % cycle_steps) / cycle_steps > self.config.expl_ratio
            a = jnp.sqrt(2 * self.config.alpha * eps)

            def mult(b):  # 상수곱
                return a * b

            noise = jax.lax.cond(sample, lambda noise: jax.tree_map(mult, noise),
                                 lambda noise: jax.tree_map(jnp.zeros_like, noise), noise)

            momentum = jax.tree_multimap(
                lambda m, g, n: (1 - self.config.alpha) * m - eps * self.metadata['num_train'] * g + n, momentum,
                grad_U, noise)
            theta = tree_add(theta, momentum)

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
        train_dataset = get_cifar(name=self.config.dataset, split='train')
        test_dataset = get_cifar(name=self.config.dataset, split='test')
        print(';')
        train_epoch = self.make_train_epoch(train_dataset)

        key, state = self.init_state()
        samples = []
        sghmc_test_step = partial(self.sghmc_test_step,test_dataset=test_dataset) # 이거부터 좀 문제임
        for epoch in range(1, self.config.num_epoch + 1):
            print(epoch)
            key,state= train_epoch(key,state)

            if self.is_ensembling_epoch(epoch):
                samples.append(to_state_dict(state[0]))  # state[0] = theta

            if (epoch > self.config.burnin_epochs) & (epoch % self.config.test_freq == 0):
                key,perm_key = key, perm_key = jax.random.split(key, 2)
                test_indices = batch_divider(perm_key, self.config.batch_size, num_data=self.metadata['num_test'])
                samples, (nlls, accs) = jax.lax.scan(sghmc_test_step,samples, xs= test_indices)
                print(epoch, jnp.mean(nlls), jnp.mean(accs))
                
        return samples


class DeepEnsemble:
    def __init__(self, config):
        self.config = config
        self.metadata = get_metadata(config)
        self.net = ResNet(stage_sizes = [2,2,2,2],num_classes = self.metadata['num_classes'],norm = BatchNorm2d)

    def init_state(self):
        key = jax.random.PRNGKey(self.config.seed)
        key,normal_key = jax.random.split(key,2)
        multi_key = jax.random.split(key, num=self.config.ensemble_num)
        multi_params = jax.vmap(self.net.init, in_axes=(0, None)) # 이거 좀 고쳐야할듯? in axes에서 (0,1) -> (0,None)으로 바꾸도 될듯 뒤에꺼가 필요한가?
        theta = multi_params(multi_key, jnp.ones(self.metadata['shape']))
        theta, _ = normal_like_tree(theta, normal_key, std=self.config.init_param_sd)
        return key, theta

    def make_train_epoch(self, train_dataset):

        cycle_steps = self.config.batch_size * self.config.cycle_epochs
        init_lr = self.config.sgd_lr

        def schedule(step):
            t = step
            t = (t % cycle_steps) / cycle_steps
            return 0.5 * init_lr * (1 + jnp.cos(t * jnp.pi))

        def loss_fn(params, batch):
            x, y = batch
            multi_apply = jax.jit(jax.vmap(self.net.apply, in_axes=(0, None))) # jit으로 감싸봄
            logits = multi_apply(params, x)  # num_ensemble * batch_size * num_class
            nll = cross_entropy(logits, y).mean(-1)  # num_ensemble*1 --->  앞에 num_data 곱하는 이유는 estimator 기 때문 N/n으로
            l2 = self.config.weight_decay*jax.vmap(l2_reg, in_axes=0)(params)  # num_ensemble*1  # 문제생기면 weight_decay 디버깅. 지금 에너지 스케일 떄문에 학습안되는건지 체크중
            loss_for_grad = nll + l2  # num_ensemble*1 , 미분하려면 1차원으로 바꿔줘야함
            return loss_for_grad.sum()  # , loss  # debug


        def update_fn(state, batch):
            theta,t_curr = state
            grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
            grad_U = grad_fn(theta, batch)
            lr = schedule(t_curr)
            update = jax.tree_map(lambda u: -lr * u, grad_U)
            theta = tree_add(theta, update)
            t_curr= t_curr +1
            state = theta,t_curr
            return state, 1  # debugged - > output으로 두개 주긴 해야함

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
        
    def test_step(self,theta,batch_indices,test_dataset):

        batch = jax.tree_map(lambda x: x[batch_indices], test_dataset)
        x, y = batch
        multi_apply = jax.vmap(self.net.apply, in_axes=(0, None))
        logit = multi_apply(theta, x)
        nll = jnp.mean(-logsumexp(-cross_entropy(logit, y), 0) + jnp.log(self.config.ensemble_num))
        avg_logits = logsumexp(logit, 0) - jnp.log(self.config.ensemble_num)
        acc = (avg_logits.argmax(-1) == y).mean()
        return theta, (nll, acc)  # carry = theta


    def train(self):
        train_dataset = get_cifar(name=self.config.dataset, split='train')
        test_dataset = get_cifar(name=self.config.dataset, split='test')
        train_epoch = self.make_train_epoch(train_dataset)
        key, theta = self.init_state()
        t_curr = 0
        state = theta,t_curr
        for epoch in range(1, self.config.num_epoch + 1):
            print("epoch:",epoch)
            key,state = train_epoch(key,state)

            if (epoch == 3) or (epoch % self.config.test_freq ==0):
                theta,_ = state
                key, perm_key = jax.random.split(key, 2)
                test_step = partial(self.test_step,test_dataset=test_dataset)
                test_indices = batch_divider(perm_key, self.config.batch_size, num_data=self.metadata['num_test'])
                _,(nlls,accs) = jax.lax.scan(test_step,theta,test_indices)

                print(epoch, jnp.mean(nlls), jnp.mean(accs))

            theta,_ = state
        return theta 

class SWA:

    def __init__(self,config): # 이런식으로 다른 상위 클래스를 상속해서 받으려면 super 머시기를 밑에 적어줘야함
        self.config = config
        self.metadata = get_metadata(config)
        self.net = ResNet(stage_sizes = [2,2,2,2],num_classes = self.metadata['num_classes'],norm = BatchNorm2d)

    def init_state(self):
        key = jax.random.PRNGKey(self.config.seed)
        key, normal_key = jax.random.split(key, 2)
        theta = self.net.init(key, jnp.ones((1,) + self.metadata['shape']))
        theta,_ = normal_like_tree(theta, normal_key, std=self.config.init_param_sd)

        return key,theta

    def make_train_epoch(self, train_dataset,lr):

        def loss_fn(params, batch):
            x, y = batch
            logits = self.net.apply(params,x)
            nll = cross_entropy(logits, y).mean() 
            l2 =  self.config.weight_decay*l2_reg(params)
            loss = nll + l2
            return loss

        def update_fn(theta, batch):
            grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0)) # 고침 좀 빨리하라고
            loss, grad_U = grad_fn(theta, batch)
            update = jax.tree_map(lambda u: -lr* u, grad_U)  # swa에선 여길 바꿔줘야함
            theta = tree_add(theta, update)

            return theta, loss  # debugged - > output으로 두개 주긴 해야함

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

    def schedule(self, epoch):   # swa도 일종의 burnin 이 있다.
        t = epoch / self.config.burnin_epochs
        lr_ratio = 0.5
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.config.sgd_lr * factor

    def test_step(self,theta,batch_indices,test_dataset):
        batch = jax.tree_map(lambda x: x[batch_indices], test_dataset)
        x, y = batch
        logits = self.net.apply(theta,x)
        nll = cross_entropy(logits, y).mean()
        acc = (logits.argmax(-1) == y).mean()
        return theta, (nll, acc)

    def train(self):
        train_dataset = get_cifar(name=self.config.dataset, split='train')
        test_dataset = get_cifar(name=self.config.dataset, split='test')
        key,theta = self.init_state()
        n_models = 0
        train_epoch = self.make_train_epoch(train_dataset,lr=self.config.sgd_lr)
        for epoch in range(1,self.config.num_epoch+1):
            print('epoch:',epoch)
            key, theta = train_epoch(key, theta)

            if epoch == self.config.burnin_epochs:
              theta_swa = theta # burnin 끝내고 init

            if (epoch > self.config.burnin_epochs) & (epoch % self.config.thin == 0):
                n_models = n_models + 1
                theta_swa = jax.tree_multimap(lambda t1,t2: (t1*n_models+t2)/(n_models+1), theta_swa, theta) # theta swa 와 theta의 비례배분 (N/(N+1)과 1/N+1)

            if (epoch > self.config.burnin_epochs) & (epoch % self.config.test_freq == 0): # 이렇게하면 10 epoch마다 test해버리면 theta init으로만하니까 학습이 안되는거처럼 보일 수 있다.
                key, perm_key = jax.random.split(key, 2)
                test_step = partial(self.test_step, test_dataset=test_dataset)
                test_indices = batch_divider(perm_key, self.config.batch_size, num_data=self.metadata['num_test'])
                _,(nlls, accs) = jax.lax.scan(test_step, theta_swa, test_indices)
                print(epoch, jnp.mean(nlls), jnp.mean(accs))

        return theta_swa