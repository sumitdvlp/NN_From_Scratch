from torch import nn
import torch
class CustomAdam:
    def __init__(self, params: nn.Parameter,  stepsize = 0.001, bias_m1 = 0.9, bias_m2 = 0.999, epsilon = 10e-8, bias_correction = True):
        self.params = list(params)
        self.stepsize = stepsize
        self.betas = (bias_m1, bias_m2)
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.state = {p: {'step': 0, 'first_momentum': torch.zeros_like(p.data), 'sec_raw_momentum': torch.zeros_like(p.data)} for p in self.params}
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            state = self.state[p]
            state['step'] += 1
            t = state['step']
            last_first_momentum = state['first_momentum']
            last_sec_raw_momentum = state['sec_raw_momentum']
            grad = p.grad.data
            first_momentum = last_first_momentum.mul_(self.betas[0]).add_(grad * (1 - self.betas[0]))
            sec_raw_momentum = last_sec_raw_momentum.mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])
            if self.bias_correction:    
                first_momentum = first_momentum / (1 - self.betas[0] ** t)
                sec_raw_momentum = sec_raw_momentum / (1 - self.betas[1] ** t)
            # p.data.addcdiv_(first_momentum, sec_raw_momentum.sqrt().add_(self.epsilon), value=-self.stepsize)
            # Update the state
            state['first_momentum'] = first_momentum
            state['sec_raw_momentum'] = sec_raw_momentum
            # Update the parameter
            p.data = p.data - (self.stepsize * first_momentum / (sec_raw_momentum.sqrt() + self.epsilon))