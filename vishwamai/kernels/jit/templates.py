"""Kernel code generation templates."""

from typing import Dict, Any, Optional, List, Union
import string
from dataclasses import dataclass

from ..core.kernel import HardwareType, KernelConfig

@dataclass
class KernelTemplate:
    """Template for generating kernel code."""
    name: str
    hardware: HardwareType
    code_template: str
    default_config: Dict[str, Any]

class TemplateManager:
    """Manage and render kernel templates."""
    
    def __init__(self):
        self.templates: Dict[str, KernelTemplate] = {}
        self._initialize_templates()
        
    def _initialize_templates(self):
        """Initialize built-in templates."""
        # TPU MatMul template
        self.register_template(
            KernelTemplate(
                name="tpu_matmul",
                hardware=HardwareType.TPU,
                code_template="""
                def ${kernel_name}(x, w, *, precision=${precision}):
                    # Reshape for TPU optimization
                    x = x.reshape((-1, ${block_size}, x.shape[-1]))
                    w = w.reshape((-1, ${block_size}, w.shape[-1]))
                    
                    # Define MatMul computation
                    def matmul_computation(x, w):
                        return jax.lax.dot_general(
                            x, w,
                            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                            precision=${precision}
                        )
                    
                    # Compile for TPU
                    return jax.jit(matmul_computation, backend="tpu")(x, w)
                """,
                default_config={
                    "block_size": 128,
                    "precision": "highest"
                }
            )
        )
        
        # GPU MatMul template
        self.register_template(
            KernelTemplate(
                name="gpu_matmul",
                hardware=HardwareType.GPU,
                code_template="""
                @torch.jit.script
                def ${kernel_name}(x, w):
                    # Enable tensor cores
                    with torch.cuda.amp.autocast():
                        # Reshape for efficient CUDA execution
                        x = x.view(-1, ${block_size}, x.size(-1))
                        w = w.view(-1, ${block_size}, w.size(-1))
                        
                        # Use torch.matmul for tensor core acceleration
                        return torch.matmul(x, w)
                """,
                default_config={
                    "block_size": 64,
                    "use_tensor_cores": True
                }
            )
        )
        
        # TPU Attention template
        self.register_template(
            KernelTemplate(
                name="tpu_attention",
                hardware=HardwareType.TPU,
                code_template="""
                def ${kernel_name}(q, k, v, mask=None, *, scale=${scale}):
                    # Fused attention computation
                    @partial(jax.jit, static_argnums=(4,))
                    def attention_fn(q, k, v, mask, scale):
                        # Compute attention scores
                        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
                        
                        if mask is not None:
                            scores = jnp.where(mask, scores, -1e9)
                            
                        # Apply softmax and dropout
                        attn = jax.nn.softmax(scores, axis=-1)
                        if ${use_dropout}:
                            attn = jax.random.dropout(
                                jax.random.PRNGKey(0),
                                ${dropout_rate},
                                attn
                            )
                            
                        # Compute attention output
                        return jnp.einsum('bhqk,bhkd->bhqd', attn, v)
                        
                    return attention_fn(q, k, v, mask, ${scale})
                """,
                default_config={
                    "scale": 1.0 / 8.0,
                    "use_dropout": True,
                    "dropout_rate": 0.1
                }
            )
        )
        
        # GPU Attention template
        self.register_template(
            KernelTemplate(
                name="gpu_attention",
                hardware=HardwareType.GPU,
                code_template="""
                @torch.jit.script
                def ${kernel_name}(q, k, v, mask: Optional[torch.Tensor] = None):
                    # Use flash attention for long sequences
                    if q.size(2) >= ${flash_attn_threshold}:
                        return flash_attention(q, k, v, mask)
                        
                    # Standard scaled dot-product attention
                    scale = ${scale}
                    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                    
                    if mask is not None:
                        scores = scores + mask
                        
                    attn = torch.nn.functional.softmax(scores, dim=-1)
                    if ${use_dropout}:
                        attn = torch.nn.functional.dropout(
                            attn,
                            p=${dropout_rate},
                            training=True
                        )
                        
                    return torch.matmul(attn, v)
                """,
                default_config={
                    "scale": 1.0 / 8.0,
                    "use_dropout": True,
                    "dropout_rate": 0.1,
                    "flash_attn_threshold": 1024
                }
            )
        )
        
        # TPU LayerNorm template
        self.register_template(
            KernelTemplate(
                name="tpu_layer_norm",
                hardware=HardwareType.TPU,
                code_template="""
                def ${kernel_name}(x, scale, bias, *, epsilon=${epsilon}):
                    @jax.jit
                    def layer_norm_fn(x, scale, bias):
                        mean = jnp.mean(x, axis=-1, keepdims=True)
                        variance = jnp.var(x, axis=-1, keepdims=True)
                        
                        x_norm = (x - mean) * jax.lax.rsqrt(variance + epsilon)
                        return x_norm * scale + bias
                        
                    return layer_norm_fn(x, scale, bias)
                """,
                default_config={
                    "epsilon": 1e-6
                }
            )
        )
        
        # GPU LayerNorm template
        self.register_template(
            KernelTemplate(
                name="gpu_layer_norm",
                hardware=HardwareType.GPU,
                code_template="""
                @torch.jit.script
                def ${kernel_name}(x, weight, bias):
                    with torch.cuda.amp.autocast():
                        return torch.nn.functional.layer_norm(
                            x,
                            normalized_shape=(x.size(-1),),
                            weight=weight,
                            bias=bias,
                            eps=${epsilon}
                        )
                """,
                default_config={
                    "epsilon": 1e-6
                }
            )
        )
        
    def register_template(self, template: KernelTemplate):
        """Register a new template."""
        self.templates[template.name] = template
        
    def get_template(self, name: str) -> Optional[KernelTemplate]:
        """Get template by name."""
        return self.templates.get(name)
        
    def render_template(self,
                       template_name: str,
                       config: Optional[Dict[str, Any]] = None,
                       **kwargs) -> str:
        """Render template with given configuration."""
        template = self.get_template(template_name)
        if template is None:
            raise ValueError(f"Template not found: {template_name}")
            
        # Merge configs
        final_config = template.default_config.copy()
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        # Replace placeholders
        return string.Template(template.code_template).substitute(final_config)

# Global template manager instance
_template_manager = TemplateManager()

def get_template_manager() -> TemplateManager:
    """Get global template manager instance."""
    return _template_manager