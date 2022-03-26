import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def cr_loss(theta,x,y): # 명시적으로 x= batch[0], y=batch[1] 이렇게 input에 넣어줘라
    num_classes= logits.shape[-1]
    logits = multi_apply(theta,x)
    labels = jax.nn.one_hot(y,num_classes)
    return -jnp.sum(labels * jax.nn.log_softmax(logits), -1)

def cross_entropy(logits, y):
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(y, num_classes)
    return -jnp.sum(labels * jax.nn.log_softmax(logits), -1)

def l2_reg(params, filter_vectors=True):
    return 0.5 * sum([jnp.sum(p**2) \
            for p in jax.tree_leaves(params) \
            if not filter_vectors or p.ndim > 1])


def energy(theta,batch,coef=1): # for multi chain
  x,y = batch
  logits = multi_apply(theta,x) # 20 * 500 * 10
  nll = num_data*cross_entropy(logits, y).mean(-1) # 20*1 --->  여기에 data N을 곱하냐? 이게 좀 헷갈리네
  l2 = coef*jax.vmap(l2_reg,in_axes=0)(theta)#*weight_decay # 20*1  # 문제생기면 weight_decay 디버깅. 지금 에너지 스케일 떄문에 학습안되는건지 체크중
  loss = -nll - l2 # 이부분 그냥 논문 코드랑 맞춰줄게. loss는 posterior고 grad는 energy를 쓰네. 이건 좀 문제있는거아닌가? -> 근데 training 할때도 이세팅을 써서 애초에 Q,D파라미터자체가 바뀌겠네 이대로안하면
  loss_for_grad= nll+l2
  return loss_for_grad.sum(),loss # debug