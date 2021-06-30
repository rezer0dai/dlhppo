import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np

def kl(p, log_p, log_q):
    return (p * (log_p - log_q)).sum(-1, keepdim=True)

def kl_from_logits(p_logits, q_logits):
    p = F.softmax(p_logits, dim=1)
    log_p = F.log_softmax(p_logits, dim=1)
    log_q = F.log_softmax(q_logits, dim=1)
    return kl(p, log_p, log_q)

def self_kl_from_logits(logits):
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    log_q = log_p.detach()
    log_q.requires_grads = False
    return kl(p, log_p, log_q)

# https://github.com/ilyasu123/trpo/blob/master/utils.py#31
def self_kl_from_gauss(dist):
    mu2 = dist.dist.loc
    sig2 = dist.dist.scale#torch.stack([d.b for d in dist])
    mu1, sig1 = mu2.detach(), sig2.detach()
    mu1.requires_grads = False
    sig1.requires_grads = False

    var1 = (2*sig1).exp()
    var2 = (2*sig2).exp()
    kl = (sig2 - sig1 + (var1 + (mu1 - mu2)**2)/(2 * var2) - .5).sum(-1, keepdim=True)
    return kl

def mp_pinverse(matrix, treshold=1e-6):
    u, s, v = torch.svd(matrix)

    s = s[s > s.max() * treshold]
    s = torch.cat([1. / s, torch.zeros(u.shape[0] - s.shape[0])])

    return v @ (s.diag() @ u.t())

def assign_gradients(parameters, grads):
    #  return torch.nn.utils.convert_parameters.vector_to_parameters(grads, parameters)
    ind = 0
    for i, p in enumerate(parameters):
        for j, g in enumerate(p.grad):
            #  assert not p.grad[j].sum()
            p.grad[j] = grads[ind:ind+len(p.grad[j])]
            ind += len(p.grad[j])
    assert ind == len(grads)

def assign_data(parameters, data):
    #  return torch.nn.utils.convert_parameters.vector_to_parameters(grads, parameters)
    ind = 0
    for i, p in enumerate(parameters):
        for j, g in enumerate(p.data):
            #  assert not p.grad[j].sum()
            p.data[j] = data[ind:ind+len(p.data[j])]
            ind += len(p.data[j])
    assert ind == len(data)

# https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py
# http://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/
def hessian_vector_product(loss_wrt_kl, parameters, vector):
    """
    Returns the product of the Hessian of the KL divergence and the given vector
    - could be boosted by calc derivs of logits wrt parameters instead of KL
    - though i like here KL stuff as it is more undestandable
    """
    g_zero = lambda p: Variable(torch.zeros(p.size()).to(p.device)).view(-1)

    p_grad = lambda grad, p: grad.flatten() if grad is not None else g_zero(p)
    kl_grad = torch.autograd.grad(loss_wrt_kl, parameters, create_graph=True, allow_unused=True)#, retain_graph=True)
    #  kl_grad_vector = torch.cat([grad.view(-1) if grad is not None else g_zero(p) for grad, p in zip(kl_grad, parameters)])
    kl_grad_vector = torch.cat([p_grad(g, p) for g, p in  zip(kl_grad, parameters)])

    grad_vector_product = torch.sum(kl_grad_vector * vector)
    grad_grad = torch.autograd.grad(grad_vector_product, parameters, retain_graph=True, allow_unused=True)

    cg_damping = 1e-3
    #  fisher_vector_product = torch.cat([grad.flatten() if grad is not None else g_zero(p) for grad, p in zip(grad_grad, parameters)]).data
    fisher_vector_product = torch.cat([p_grad(g, p) for g, p in zip(grad_grad, parameters)])
    return fisher_vector_product + (cg_damping * vector.data)

import time
#from hessian import hessian

class NGDOptim(torch.optim.SGD):#Adam):#
    def __init__(self, parameters, lr, momentum=.7, nesterov=True):
        super().__init__(parameters, lr=lr)#, momentum=momentum, nesterov=nesterov)
        [self.parameters] = [ g['params'] for g in self.param_groups ]

        for p in self.parameters:
            p.grad = Variable(torch.zeros(p.size()).to(p.device))

    def _conjugate_gradient(self, dist, grads):
        """
        Returns F^(-1)b where F is the Hessian of the KL divergence
        """
        p = grads.clone().data
        r = grads.clone().data
        x = torch.zeros_like(grads, requires_grad=False)
        rdotr = r.dot(r)
        cg_iters = 10
        for _ in range(cg_iters):

            self.zero_grad()
            loss_wrt_kl = self_kl_from_gauss(dist).mean()

            z = hessian_vector_product(loss_wrt_kl, self.parameters, Variable(p)).squeeze(0)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            residual_tol = 1e-10
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self, x, fullstep, expected_improve_rate, accept_ratio=1e-3):
        """
        Returns the parameter vector given by a linesearch
        """
        max_backtracks = 10
        fval = self.surrogate_loss(x)
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            #  print("Search number {} {}...".format(stepfrac, _n_backtracks + 1))
            #  stepfrac = 1e-1
            xnew = x + stepfrac * fullstep
            newfval = self.surrogate_loss(xnew)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            #  return xnew
            if ratio > accept_ratio and actual_improve > 0:
#                print("\nTRPO HIT", _n_backtracks, ratio, expected_improve, actual_improve, expected_improve_rate)
                return xnew
#            print(":FAIL:", ratio, expected_improve, actual_improve, expected_improve_rate)
        return None

    def step_clocked(self, callback):
        start = time.time()
        self.step(callback)
        end = time.time()
        return end - start

    def naive_ngd_step(self, logits):
        grads = torch.cat([p.grad.view(-1) for p in self.parameters])

        loss_wrt_kl = self_kl_from_logits(logits).mean()
        fish = hessian(loss_wrt_kl, self.parameters)
        ngd = mp_pinverse(fish) @ grads

        assign_gradients(self.parameters, ngd)

    def surrogate_loss(self, data):
        assign_data(self.parameters, data)
        with torch.no_grad():
            return self._eval()

    def conjugate_ngd_step(self, dist, _eval):
        self._eval = _eval
        data = torch.cat([p.data.view(-1) for p in self.parameters])
        grads = torch.cat([p.grad.view(-1) for p in self.parameters])

#        ngd = self._conjugate_gradient(dist, grads)
#        assign_gradients(self.parameters, ngd)
#        return

        ngd = self._conjugate_gradient(dist, -grads)
#        ngd = self._conjugate_gradient(dist, grads)

        # Do line search to determine the stepsize of theta in the direction of step_direction
        max_kl = 1e-3
        self.zero_grad()
        loss_wrt_kl = self_kl_from_gauss(dist).mean()
        shs = .5 * ngd @ hessian_vector_product(loss_wrt_kl, self.parameters, ngd)#.t()
        lm = torch.sqrt(shs / max_kl)
        fullstep = ngd / lm
        gdotstepdir = -(grads @ ngd).item()

        #  accept_ratio = 1e-6#1e-5
#        while accept_ratio > shs.abs():
#            accept_ratio /= 10.
#        print("\nSHS:", shs, gdotstepdir, lm, gdotstepdir / lm, "-->", accept_ratio)

        theta = self.linesearch(
                parameters_to_vector(self.parameters), fullstep, gdotstepdir / lm,
                )#accept_ratio=accept_ratio)

        if theta is not None:
            assign_data(self.parameters, theta)
            assign_gradients(self.parameters, torch.zeros_like(ngd))
        else:
            assign_data(self.parameters, data)
            assign_gradients(self.parameters, torch.zeros_like(ngd))#ngd)

def callback_with_logits(optim, dist, _eval):
    #  return lambda: () # vanilla first order gradient descent
    #  return lambda: optim.naive_ngd_step(logits)
    return lambda: optim.conjugate_ngd_step(dist, _eval)
