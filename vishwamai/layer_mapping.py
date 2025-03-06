"""Layer mapping configurations for knowledge distillation."""
from typing import Dict

def create_qwen_layer_mapping(student_layers: int, teacher_layers: int) -> Dict[int, int]:
    """Create layer mapping between student and teacher models for Qwen architecture.
    
    This implements a uniform layer mapping strategy optimized for Qwen's architecture,
    where we select teacher layers that best correspond to student layers based on
    relative position and architectural characteristics.
    
    Args:
        student_layers: Number of layers in student model
        teacher_layers: Number of layers in teacher model
        
    Returns:
        Dictionary mapping student layer indices to teacher layer indices
    """
    # Early layers are mapped more densely as they capture important low-level features
    mapping = {}
    
    # For each student layer, find the corresponding teacher layer
    for student_idx in range(student_layers):
        # Use proportional mapping with slight bias towards earlier layers
        teacher_idx = min(
            int(round((student_idx * teacher_layers) / student_layers)),
            teacher_layers - 1
        )
        mapping[student_idx] = teacher_idx
    
    return mapping

def create_feature_mapping_config(student_config: dict, teacher_config: dict) -> Dict[str, Dict]:
    """Create comprehensive feature mapping configuration for Qwen distillation.
    
    Args:
        student_config: Student model configuration dictionary
        teacher_config: Teacher model configuration dictionary
        
    Returns:
        Dictionary containing various feature mapping configurations
    """
    # Layer mapping
    layer_mapping = create_qwen_layer_mapping(
        student_layers=student_config['num_layers'],
        teacher_layers=teacher_config['num_layers']
    )
    
    # Attention head mapping
    student_heads = student_config['num_attention_heads']
    teacher_heads = teacher_config['num_attention_heads']
    
    # Map student heads to closest corresponding teacher heads
    head_mapping = {}
    step = teacher_heads / student_heads
    for i in range(student_heads):
        head_mapping[i] = int(i * step)
    
    # Hidden state dimension mapping
    student_hidden = student_config['hidden_size']
    teacher_hidden = teacher_config['hidden_size']
    
    # Create projection matrices if needed
    needs_projection = student_hidden != teacher_hidden
    
    return {
        'layer_mapping': layer_mapping,
        'head_mapping': head_mapping,
        'hidden_mapping': {
            'needs_projection': needs_projection,
            'student_dim': student_hidden,
            'teacher_dim': teacher_hidden
        },
        # Qwen-specific attention mechanism mapping
        'attention_mapping': {
            'num_key_value_heads': {
                'student': student_config['num_key_value_heads'],
                'teacher': teacher_config['num_key_value_heads']
            },
            'use_gqa': student_config.get('use_gqa', False) and teacher_config.get('use_gqa', False),
            'gqa_mapping': create_gqa_mapping(
                student_config['num_key_value_heads'],
                teacher_config['num_key_value_heads']
            )
        }
    }

def create_gqa_mapping(student_kv_heads: int, teacher_kv_heads: int) -> Dict[int, int]:
    """Create mapping for Grouped Query Attention (GQA) heads."""
    mapping = {}
    step = teacher_kv_heads / student_kv_heads
    
    for i in range(student_kv_heads):
        mapping[i] = int(i * step)
    
    return mapping
