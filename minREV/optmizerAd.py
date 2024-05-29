
from typing import List
from torch import Tensor
from torch.optim import Optimizer
import math
import torch

def Tema(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avgs_2: List[Tensor],
    exp_avgs_3: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_sqs_2: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    epoch: int,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
    lr: float,
    weight_decay: float,
    eps: float,
):

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_2 = exp_avgs_2[i]
        exp_avg_3 = exp_avgs_3[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_sq_2 = exp_avg_sqs_2[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha= 1 - (beta1))
        exp_avg_2.mul_(beta3).add_(exp_avg, alpha=1 - (beta3))

        exp_avg_3.mul_(beta4).add_(exp_avg_2,alpha=1- (beta4))
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value= 1 - beta2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        tema = (3 * exp_avg) - (3 * exp_avg_2) + exp_avg_3
        #tema_norm = tema /(1 - 0.9**step)
        param.addcdiv_(tema, denom, value=-step_size)

class FAME(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        beta3 = 0.5,
        beta4 = 0.5,
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        self.epochs = 0
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, beta3=beta3, beta4=beta4, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(FAME, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FAME, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []  # EMA1
            exp_avgs_2 = [] # EMA2
            exp_avgs_3 = [] # EMA3
            exp_avg_sqs = []
            exp_avg_sqs_2 = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            beta3 = group["beta3"]
            beta4 = group["beta4"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Dema
                        state["exp_avg_2"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq_2"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_3"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avgs_2.append(state["exp_avg_2"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    exp_avg_sqs_2.append(state["exp_avg_sq_2"])
                    exp_avgs_3.append(state["exp_avg_3"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])
            Tema(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avgs_2,
                exp_avgs_3,
                exp_avg_sqs,
                exp_avg_sqs_2,
                max_exp_avg_sqs,
                state_steps,
                self.epochs,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                beta4=beta4,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )
        return loss
