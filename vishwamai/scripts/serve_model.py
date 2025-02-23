#!/usr/bin/env python3
"""Model serving script with FastAPI endpoint."""

import argparse
import logging
from pathlib import Path
import yaml
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from vishwamai.data.tokenization import SPTokenizer
from vishwamai.model.transformer.model import VishwamaiModel
from vishwamai.utils.logging import setup_logging

logger = logging.getLogger(__name__)
app = FastAPI(title="Vishwamai Model Server")

class InferenceRequest(BaseModel):
    """Request model for inference."""
    text: str
    max_length: Optional[int] = 2048
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.7
    num_return_sequences: Optional[int] = 1

class BatchInferenceRequest(BaseModel):
    """Request model for batch inference."""
    texts: List[str]
    max_length: Optional[int] = 2048
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.7
    num_return_sequences: Optional[int] = 1

class ModelServer:
    """Model server for handling inference requests."""
    
    def __init__(
        self,
        model_path: str,
        model_config_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        max_batch_size: int = 32
    ):
        # Load configurations
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
            
        # Initialize model and tokenizer
        self.tokenizer = SPTokenizer.from_pretrained(tokenizer_path)
        self.model = VishwamaiModel(model_config)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model = self.model.to(device)
        self.model.eval()
        
        self.device = device
        self.max_batch_size = max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def generate(
        self,
        text: str,
        max_length: int = 2048,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate text asynchronously.
        
        Args:
            text (str): Input text
            max_length (int, optional): Maximum sequence length. Defaults to 2048.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            num_return_sequences (int, optional): Number of sequences. Defaults to 1.
            
        Returns:
            List[str]: Generated sequences
        """
        # Tokenize input
        inputs = self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            )
            
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts
        
    async def generate_batch(
        self,
        texts: List[str],
        max_length: int = 2048,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[List[str]]:
        """Generate text for a batch of inputs asynchronously.
        
        Args:
            texts (List[str]): Input texts
            max_length (int, optional): Maximum sequence length. Defaults to 2048.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            num_return_sequences (int, optional): Number of sequences. Defaults to 1.
            
        Returns:
            List[List[str]]: Generated sequences for each input
        """
        # Split into batches
        batches = [
            texts[i:i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]
        
        all_generated = []
        for batch in batches:
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        do_sample=True,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        num_return_sequences=num_return_sequences,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                )
                
            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs.reshape(-1, num_return_sequences, outputs.size(-1)),
                skip_special_tokens=True
            )
            all_generated.extend(generated_texts)
            
        return all_generated

# Global model server instance
model_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize model server on startup."""
    global model_server
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    setup_logging()
    
    model_server = ModelServer(
        model_path=args.model_path,
        model_config_path=args.model_config,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        max_batch_size=args.max_batch_size
    )
    
@app.post("/generate")
async def generate(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Generate text endpoint."""
    try:
        generated_texts = await model_server.generate(
            text=request.text,
            max_length=request.max_length,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            num_return_sequences=request.num_return_sequences
        )
        return JSONResponse(content={"generated_texts": generated_texts})
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
@app.post("/generate_batch")
async def generate_batch(request: BatchInferenceRequest, background_tasks: BackgroundTasks):
    """Generate text for batch endpoint."""
    try:
        generated_texts = await model_server.generate_batch(
            texts=request.texts,
            max_length=request.max_length,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            num_return_sequences=request.num_return_sequences
        )
        return JSONResponse(content={"generated_texts": generated_texts})
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    setup_logging()
    
    logger.info(f"Starting model server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
