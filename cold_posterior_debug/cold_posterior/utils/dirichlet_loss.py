

def loss_fn(theta,batch_stats,batch,noise=1e-2):

    def gauss_log_lik(mean,var,x):
        log_lik = jnp.log(0.5*(x-mean)**2/var).sum(-1)
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
    
    return loss,new_batch_stat['batch_stats']
