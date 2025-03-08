# /home/kasinadhsarma/VishwamAI/vishwamai/inference/optimized_inference.py
"""
Optimized inference interface for VishwamAI models.
Handles device and precision settings for efficient inference, particularly on TPUs.
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

class OptimizedInference:
    """
    Interface for optimizing model inference across different devices and precisions.
    Used internally by other inference interfaces for TPU efficiency.
    """
    def __init__(self):
        self.device = None
        self.precision = None

    def set_device(self, device_type: str):
        """
        Set the device for inference.

        Args:
            device_type (str): 'tpu', 'gpu', or 'cpu'.
        """
        if device_type == 'tpu':
            self.device = xm.xla_device()
        elif device_type == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Device set to: {self.device}")

    def set_precision(self, precision: str):
        """
        Set the precision for computations.

        Args:
            precision (str): 'fp16', 'bf16', or 'fp32'.
        """
        if precision not in ['fp16', 'bf16', 'fp32']:
            raise ValueError(f"Unsupported precision: {precision}")
        self.precision = precision
        print(f"Precision set to: {precision}")

    def optimize_model(self, model):
        """
        Optimize the model for the set device and precision.

        Args:
            model: PyTorch model to optimize.
        """
        if self.device is None:
            raise ValueError("Device not set. Call set_device first.")
        if self.precision is None:
            raise ValueError("Precision not set. Call set_precision first.")

        model.to(self.device)
        if self.precision == 'fp16':
            model.half()
        elif self.precision == 'bf16':
            model.bfloat16()
        print("Model optimized for inference")

    def run_model(self, model, input_data, temperature=1.0):
        """
        Run the optimized model with the given input data.

        Args:
            model: PyTorch model to run.
            input_data: Input tensor or data.
            temperature (float): Temperature for softmax (default: 1.0).

        Returns:
            torch.Tensor: Model output.
        """
        if self.device is None or self.precision is None:
            raise ValueError("Device or precision not set. Configure OptimizedInference first.")
        
        input_data = input_data.to(self.device)
        with torch.no_grad():
            output = model(input_data)
            if temperature != 1.0:
                output = torch.softmax(output / temperature, dim=-1)
        xm.mark_step()  # Required for TPU synchronization
        return output

if __name__ == "__main__":
    # Dummy model for testing
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([ord('t'), ord('e'), ord('s'), ord('t')], device=x.device)

    model = DummyModel()
    opt_inf = OptimizedInference()
    opt_inf.set_device('tpu')
    opt_inf.set_precision('bf16')
    opt_inf.optimize_model(model)
    input_tensor = torch.tensor([1, 2, 3], device=opt_inf.device).unsqueeze(0)
    output = opt_inf.run_model(model, input_tensor)
    print("Output:", ''.join(chr(int(x)) for x in output))