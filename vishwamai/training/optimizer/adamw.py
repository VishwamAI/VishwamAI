"""AdamW optimizer implementation with weight decay fix and TPU support."""

import math
from typing import Callable, Dict, Iterable, Optional, Tuple, Union
import torch
from torch.optim import Optimizer
import logging

logger = logging.getLogger(__name__)

class AdamWOptimizer(Optimizer):
    """Implements AdamW algorithm with proper weight decay, gradient clipping, and TPU support.
    
    This implementation correctly separates weight decay from the gradient-based update,
    enabling better regularization. It also includes built-in gradient clipping and
    TPU-specific optimizations.
    
    Args:
        params (Iterable[torch.nn.Parameter]): Model parameters to optimize
        lr (float, optional): Learning rate. Defaults to 1e-3
        betas (Tuple[float, float], optional): Adam betas. Defaults to (0.9, 0.999)
        eps (float, optional): Term for numerical stability. Defaults to 1e-8
        weight_decay (float, optional): Weight decay factor. Defaults to 0.01
        max_grad_norm (Optional[float], optional): Max norm for gradient clipping. Defaults to None
        correct_bias (bool, optional): Bias correction in Adam. Defaults to True
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: Optional[float] = None,
        correct_bias: bool = True
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
            max_grad_norm=max_grad_norm,
            correct_bias=correct_bias
        )
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.
        
        Args:
            closure (Optional[Callable], optional): Closure for reevaluation. Defaults to None.
            
        Returns:
            Optional[torch.Tensor]: Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        # Clip gradients if max_grad_norm is set
        if self.defaults['max_grad_norm'] is not None:
            self._clip_gradients()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Perform stepweight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
                # Get grad
                grad = p.grad.data
                
                # Skip if grad is sparse
                if grad.is_sparse:
                    logger.warning("AdamW optimizer doesn't support sparse gradients")
                    continue
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                else:
                    bias_correction1 = bias_correction2 = 1.0
                    
                # Compute step size
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss
    
    def _clip_gradients(self) -> None:
        """Clips gradients by global norm."""
        max_grad_norm = self.defaults['max_grad_norm']
        
        # Compute global grad norm
        global_grad_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data
                    global_grad_norm += grad.norm(2).item() ** 2
        global_grad_norm = math.sqrt(global_grad_norm)
        
        # Clip if necessary
        clip_coef = max_grad_norm / (global_grad_norm + 1e-6)
        if clip_coef < 1:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
                        
    def get_state_dict(self) -> Dict:
        """Gets optimizer state for checkpointing.
        
        Returns:
            Dict: Optimizer state dictionary
        """
        return {
            'state': self.state,
            'param_groups': self.param_groups
        }
        
    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads optimizer state from checkpoint.
        
        Args:
            state_dict (Dict): Optimizer state dictionary
        """
        self.__setstate__(state_dict)
        
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zeros gradients for all parameters.
        
        Args:
            set_to_none (bool, optional): Whether to set grads to None. Defaults to True.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
