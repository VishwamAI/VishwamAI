# /home/kasinadhsarma/VishwamAI/vishwamai/inference/inference_normal.py
"""
Inference interface for the Normal model in VishwamAI.
Provides fast predictions with confidence scores and explanations.
"""

import torch
import logging
from typing import Tuple, Dict, Any, Optional
from .optimized_inference import OptimizedInference
from vishwamai.optimisation.memory_optimization import MemoryOptimizer
from vishwamai.optimisation.performance_tuning import PerformanceTuner

try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit
    import flax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NormalInference:
    """
    Interface for running inference on the Normal model.
    Optimized for quick responses with confidence explanations.
    """
    def __init__(self, model, device: str = 'auto', precision: str = 'auto'):
        """
        Initialize the NormalInference with automatic device and precision detection.

        Args:
            model: The model to run inference with.
            device (str): 'auto', 'tpu', 'gpu', or 'cpu'.
            precision (str): 'auto', 'fp16', 'bf16', or 'fp32'.
        """
        self.model = model
        self.optimizer = OptimizedInference()
        self._setup_device_and_precision(device, precision)
        self.optimizer.optimize_model(self.model)
        self.device = self.optimizer.device
        self.memory_manager = MemoryOptimizer(model, device)
        self.performance_tuner = PerformanceTuner(model, device)
        logger.info(f"Initialized NormalInference on {self.device} with {precision} precision")

    def _setup_device_and_precision(self, device: str, precision: str):
        """Configure optimal device and precision settings."""
        if device == 'auto':
            if HAS_JAX:
                try:
                    jax.devices("tpu")
                    device = 'tpu'
                except:
                    device = 'gpu' if torch.cuda.is_available() else 'cpu'
            else:
                device = 'gpu' if torch.cuda.is_available() else 'cpu'

        if precision == 'auto':
            if device == 'tpu':
                precision = 'bf16'
            elif device == 'gpu':
                precision = 'fp16'
            else:
                precision = 'fp32'

        self.optimizer.set_device(device)
        self.optimizer.set_precision(precision)

    def predict(self, input_data: str, temperature: float = 1.0) -> Tuple[str, float, Dict[str, Any]]:
        """
        Generate a direct prediction with confidence score and performance metrics.

        Args:
            input_data (str): The input query or problem.
            temperature (float): Sampling temperature for generation.

        Returns:
            Tuple[str, float, Dict[str, Any]]: Prediction, confidence score, and metrics.
        """
        # Monitor performance and memory
        perf_stats = {}
        self.memory_manager.clear_device_memory()
        initial_memory = self.memory_manager.get_memory_usage()

        # Process input
        processed_input = self._preprocess_input(input_data)
        
        # Device-specific tensor creation
        if self.optimizer.device_type == 'tpu':
            input_tensor = jnp.array([ord(c) for c in processed_input])
            
            @jit
            def run_prediction(x):
                return self.model(x)
                
            with jax.default_device(self.device):
                output = run_prediction(input_tensor)
        else:
            input_tensor = torch.tensor([ord(c) for c in processed_input], 
                                      device=self.device).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.optimizer.precision != 'fp32'):
                output = self.optimizer.run_model(self.model, input_tensor, temperature=temperature)

        # Process output
        prediction = self._decode_output(output)
        confidence = self._get_confidence(output)

        # Collect performance metrics
        final_memory = self.memory_manager.get_memory_usage()
        perf_stats.update({
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'device_type': self.optimizer.device_type,
            'precision': self.optimizer.precision
        })

        return prediction, confidence, perf_stats

    def _preprocess_input(self, input_data: str) -> str:
        """Preprocess the input data with device-specific optimizations."""
        processed = f"Query: {input_data}"
        if len(processed) > 512:  # Truncate long inputs
            processed = processed[:512]
        return processed

    def _decode_output(self, output) -> str:
        """Decode model output with device-specific handling."""
        if isinstance(output, jnp.ndarray):
            return ''.join(chr(int(x)) for x in output)
        return ''.join(chr(int(x)) for x in output.cpu().numpy())

    def _get_confidence(self, output) -> float:
        """Calculate confidence score with device-specific implementations."""
        if isinstance(output, jnp.ndarray):
            logits = output.astype(jnp.float32)
            probs = jax.nn.softmax(logits, axis=-1)
            confidence = float(jnp.max(probs))
        else:
            logits = output.float()
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs.max().item())

        # Apply confidence calibration
        confidence = min(0.95, confidence)  # Cap maximum confidence
        if confidence < 0.3:
            confidence *= 0.8  # Penalize very low confidence predictions
        
        return confidence

    def explain_confidence(self, confidence: float, verbose: bool = False) -> Dict[str, Any]:
        """
        Provide a detailed explanation of the confidence score.

        Args:
            confidence (float): Confidence score between 0 and 1.
            verbose (bool): Whether to include detailed metrics.

        Returns:
            Dict[str, Any]: Confidence explanation and metrics.
        """
        explanation = {
            'confidence_score': confidence,
            'confidence_level': 'high' if confidence > 0.9 else 'moderate' if confidence > 0.7 else 'low',
            'explanation': self._get_confidence_explanation(confidence)
        }
        
        if verbose:
            explanation.update({
                'device_info': {
                    'type': self.optimizer.device_type,
                    'precision': self.optimizer.precision
                },
                'memory_usage': self.memory_manager.get_memory_usage(),
                'performance_metrics': self.performance_tuner.measure_latency(
                    torch.zeros(1, 1, device=self.device)
                )
            })
            
        return explanation

    def _get_confidence_explanation(self, confidence: float) -> str:
        """Generate a human-readable confidence explanation."""
        if confidence > 0.9:
            return "High confidence: The model is very certain about this answer."
        elif confidence > 0.7:
            return "Moderate confidence: The model is fairly certain but there might be some ambiguity."
        else:
            return "Low confidence: The model is uncertain and the answer may not be reliable."

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance and memory statistics."""
        return {
            'memory_usage': self.memory_manager.get_memory_usage(),
            'device_info': {
                'type': self.optimizer.device_type,
                'precision': self.optimizer.precision
            },
            'model_stats': self.performance_tuner.measure_latency(
                torch.zeros(1, 1, device=self.device)
            )
        }

if __name__ == "__main__":
    # Example usage with device detection and performance monitoring
    class DummyNormalModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([ord('P'), ord('a'), ord('r'), ord('i'), ord('s')], 
                              device=x.device)

    model = DummyNormalModel()
    normal_inf = NormalInference(model)  # Auto-detect device and precision
    
    input_data = "What is the capital of France?"
    prediction, confidence, perf_stats = normal_inf.predict(input_data)
    
    print("Prediction:", prediction)
    print("Confidence:", confidence)
    print("Performance Stats:", perf_stats)
    print("Explanation:", normal_inf.explain_confidence(confidence, verbose=True))
    print("Current Performance:", normal_inf.get_performance_stats())