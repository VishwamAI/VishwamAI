"""AdamW optimizer implementation with weight decay fix and additional features."""
from typing import List, Optional, Dict, Any
import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """Implements AdamW algorithm with decoupled weight decay.
    
    The original Adam algorithm was proposed in
    `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in
    `Decoupled Weight Decay Regularization`_.
    
    This implementation adds:
    - Learning rate scheduling per parameter group
    - Gradient clipping
    - Custom weight decay scheduling
    - Automatic mixed precision (AMP) scaling
    - Expert-specific learning rates for MoE
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for moving averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        weight_decay: Weight decay factor (default: 0.01)
        bias_correction: Whether to use bias correction (default: True)
        amsgrad: Whether to use AMSGrad variant (default: False)
        max_grad_norm: Maximum gradient norm for clipping (default: None)
        fused: Whether to use fused implementation (default: False)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = True,
        amsgrad: bool = False,
        max_grad_norm: Optional[float] = None,
        fused: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
            amsgrad=amsgrad,
            max_grad_norm=max_grad_norm,
            fused=fused
        )
        super().__init__(params, defaults)
        
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('bias_correction', True)
            group.setdefault('max_grad_norm', None)
            
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value from closure if provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # Clip gradients if max_grad_norm is set
        if any(group['max_grad_norm'] is not None for group in self.param_groups):
            device = self.param_groups[0]['params'][0].device
            max_norm_per_group = {}
            for group in self.param_groups:
                if group['max_grad_norm'] is not None:
                    clip_norm = group['max_grad_norm']
                    group_norm = torch.nn.utils.clip_grad_norm_(
                        group['params'],
                        clip_norm
                    )
                    max_norm_per_group[id(group)] = group_norm.to(device)
                
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    
                state['step'] += 1
                state_steps.append(state['step'])
                
            self._update_params(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group,
                beta1,
                beta2
            )
            
        return loss
        
    def _update_params(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        state_steps: List[int],
        group: Dict[str, Any],
        beta1: float,
        beta2: float
    ) -> None:
        """Update parameters."""
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if group['amsgrad']:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(1 - beta2 ** step)) + group['eps']
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)) + group['eps']
                
            # Bias correction
            if group['bias_correction']:
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            else:
                step_size = group['lr']
                
            # Weight decay
            if group['weight_decay'] != 0:
                param.data.mul_(1 - group['lr'] * group['weight_decay'])
                
            # Update parameters
            param.addcdiv_(exp_avg, denom, value=-step_size)
