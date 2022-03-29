import argparse
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax.serialization import to_state_dict
from utils.data import *
from utils.util_func import *
from utils.loss import *
from models.resnet import *
from algorithms.sgmcmc import *
import tensorflow as tf
# argparser에 그냥 3개 알고리즘에 필요한 hyperparam 다때려박고 돌려보자 일단

parser = argparse.ArgumentParser(description='SWA/SGHMC/Deep Ens for cifar-10 & 100 using resnet18')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_epoch', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--test_freq', type=int, default=10, metavar='N', help='test frequency (default: 10)')
parser.add_argument('--alg', type=str, metavar='N', help='optimizer, one of sg-mcmc methods')
parser.add_argument('--burnin_epochs', type=int, default=160, metavar='N', help='burnin epoch for sghmc (default: 160)')
parser.add_argument('--cycle_epochs', type=int, default=20, metavar='N', help='cycle epoch for sghmc (default: )')
parser.add_argument('--thin', type=int, default=2, metavar='N', help='thinning interval for sghmc(default:2 )')
parser.add_argument('--init_param_sd', type=float, default=0.01, metavar='N', help='sd of initial theta(default: 0.01 )')
parser.add_argument('--seed', type=int,default =42, metavar='N',help="random seed = jax.PRNGKey(seed)")
parser.add_argument('--sghmc_init_lr', type=float, default = 0.01 , help = "sghmc initial lr (if not cyclical,just lr)")
parser.add_argument('--alpha', type=float, default=0.01, help="sghmc momentum_decay and friction term(default:0.01)")
parser.add_argument('--expl_ratio', type=float, default=0.8, help="sghmc exploration ratio(default:0.8)")
parser.add_argument('--weight_decay', type=float, default=5e-04, help="weight decay on prior(L2 norm)")
parser.add_argument('--DA', type=bool, default=True, help="option to turn on/off data_augmentation")
parser.add_argument('--temperature',type=float,default=1., help= 'give temperature to energy function(default = 1)')
args = parser.parse_args()

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
#    tf.config.experimental.set_memory_growth(gpus[0], True)
#  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
#    print(e)

if args.alg=='sghmc':

    alg = SGHMC(config = args)
    alg.train()

if args.alg == "deep ensemble":
    alg = DeepEnsemble(config=args)
    alg.train()