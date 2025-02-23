#!/usr/bin/env python3
"""Export model for deployment in various formats."""

import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.onnx
from typing import Dict, Optional, Tuple
import numpy as np

from vishwamai.data.tokenization import SPTokenizer
from vishwamai.model.transformer.model import VishwamaiModel
from vishwamai.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def export_torchscript(
    model: torch.nn.Module,
    dummy_input: Tuple[torch.Tensor, ...],
    output_path: str,
    optimize: bool = True
) -> None:
    """Export model to TorchScript format.
    
    Args:
        model (Module): Model to export
        dummy_input (Tuple[Tensor, ...]): Example inputs
        output_path (str): Path to save exported model
        optimize (bool, optional): Whether to optimize model. Defaults to True.
    """
    logger.info("Exporting model to TorchScript...")
    
    # Put model in eval mode
    model.eval()
    
    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        
    # Optimize if requested
    if optimize:
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
    # Save model
    torch.jit.save(traced_model, output_path)
    logger.info(f"TorchScript model saved to {output_path}")
    
def export_onnx(
    model: torch.nn.Module,
    dummy_input: Tuple[torch.Tensor, ...],
    output_path: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict] = None
) -> None:
    """Export model to ONNX format.
    
    Args:
        model (Module): Model to export
        dummy_input (Tuple[Tensor, ...]): Example inputs
        output_path (str): Path to save exported model
        input_names (Optional[list], optional): Input names. Defaults to None.
        output_names (Optional[list], optional): Output names. Defaults to None.
        dynamic_axes (Optional[Dict], optional): Dynamic axes config. Defaults to None.
    """
    logger.info("Exporting model to ONNX...")
    
    # Set default names if not provided
    if input_names is None:
        input_names = ["input_ids", "attention_mask"]
    if output_names is None:
        output_names = ["logits"]
        
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
        
    # Export model
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
            do_constant_folding=True,
            export_params=True
        )
        
    logger.info(f"ONNX model saved to {output_path}")
    
def quantize_model(
    model: torch.nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    dtype: str = "qint8"
) -> torch.nn.Module:
    """Quantize model for reduced precision.
    
    Args:
        model (Module): Model to quantize
        calibration_loader (DataLoader): DataLoader for calibration
        dtype (str, optional): Quantization dtype. Defaults to "qint8".
        
    Returns:
        Module: Quantized model
    """
    logger.info(f"Quantizing model to {dtype}...")
    
    # Setup quantization configuration
    if dtype == "qint8":
        qconfig = torch.quantization.get_default_qconfig("fbgemm")
    else:
        qconfig = torch.quantization.get_default_qconfig("qnnpack")
        
    # Prepare model for quantization
    model.eval()
    model.qconfig = qconfig
    
    # Fuse modules where possible
    model = torch.quantization.fuse_modules(model, [["attention", "mlp"]])
    
    # Insert observers
    model = torch.quantization.prepare(model)
    
    # Calibrate using calibration data
    with torch.no_grad():
        for batch in calibration_loader:
            model(**batch)
            
    # Convert to quantized model
    model = torch.quantization.convert(model)
    
    return model
    
def main():
    """Main export script."""
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save exported models"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["torchscript", "onnx", "all"],
        default="all",
        help="Export format"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to quantize model"
    )
    parser.add_argument(
        "--quantization-dtype",
        type=str,
        default="qint8",
        choices=["qint8", "float16"],
        help="Quantization data type"
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        help="Path to calibration data for quantization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for exported model"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length for exported model"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    # Load model configuration
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
        
    # Create model and load weights
    model = VishwamaiModel(model_config)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Create dummy inputs
    dummy_input = (
        torch.zeros(args.batch_size, args.sequence_length, dtype=torch.long),
        torch.ones(args.batch_size, args.sequence_length, dtype=torch.long)
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quantize if requested
    if args.quantize:
        if not args.calibration_data:
            raise ValueError("Calibration data required for quantization")
            
        # Load calibration data
        calibration_data = torch.load(args.calibration_data)
        calibration_loader = torch.utils.data.DataLoader(
            calibration_data,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        model = quantize_model(
            model,
            calibration_loader,
            dtype=args.quantization_dtype
        )
        
    # Export in requested formats
    if args.format in ["torchscript", "all"]:
        torchscript_path = output_dir / "model.pt"
        export_torchscript(model, dummy_input, str(torchscript_path))
        
    if args.format in ["onnx", "all"]:
        onnx_path = output_dir / "model.onnx"
        export_onnx(model, dummy_input, str(onnx_path))
        
    # Save export configuration
    export_config = {
        "model_config": model_config,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "quantization": {
            "enabled": args.quantize,
            "dtype": args.quantization_dtype if args.quantize else None
        },
        "formats": args.format
    }
    
    config_path = output_dir / "export_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(export_config, f)
        
    logger.info("Model export completed successfully")
    
if __name__ == "__main__":
    main()
