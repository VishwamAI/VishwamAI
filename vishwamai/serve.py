import os
import torch
import logging
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .inference_engine import InferenceEngine, InferenceMetrics
from .model import Transformer, ModelArgs
from prometheus_client import start_http_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    text: str
    max_length: int = Field(default=2048, le=4096)
    temperature: float = Field(default=0.7, gt=0, le=1)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, gt=0)
    stream: bool = Field(default=False)
    use_cache: bool = Field(default=True)
    secure_compute: bool = Field(default=False)

class InferenceResponse(BaseModel):
    generated_text: str
    metrics: InferenceMetrics
    secure_enclave_verified: Optional[bool] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    # Start Prometheus metrics server
    start_http_server(9090)
    
    # Initialize inference engine
    app.state.engine = InferenceEngine()
    
    # Load model and optimize for inference
    model_path = os.environ.get("MODEL_PATH", "./checkpoints/model")
    app.state.model = load_and_optimize_model(model_path, app.state.engine)
    
    yield
    
    # Cleanup
    if hasattr(app.state, "engine"):
        await app.state.engine.shutdown()

app = FastAPI(
    title="VishwamAI Inference Server",
    description="High-performance inference server with advanced features",
    version="1.0.0",
    lifespan=lifespan
)

def load_and_optimize_model(model_path: str, engine: InferenceEngine) -> torch.nn.Module:
    """Load and optimize model for inference."""
    try:
        # Load base model
        config_path = os.path.join(model_path, "config.json")
        model_config = ModelArgs.from_json(config_path)
        model = Transformer(model_config)
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
        
        # Optimize model using inference engine
        input_shape = [1, model_config.max_seq_len]
        optimized_model = engine.optimize_model(model, input_shape)
        
        return optimized_model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Model initialization failed")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB used"
    }

@app.get("/config")
async def get_config():
    """Get model and server configuration."""
    return {
        "model_type": "VishwamAI",
        "model_size": f"{sum(p.numel() for p in app.state.model.parameters())/1e9:.1f}B parameters",
        "version": "1.0.0",
        "features": {
            "secure_compute": app.state.engine.config["inference"]["security"]["confidential_computing"]["enabled"],
            "streaming": True,
            "quantization": app.state.engine.config["inference"]["engine"]["quantization"]["enabled"]
        }
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Generate predictions with advanced features."""
    try:
        # Tokenize input
        tokens = app.state.model.tokenize(request.text)
        
        # Configure inference options
        inference_kwargs = {
            "stream": request.stream,
            "secure_compute": request.secure_compute
        }
        
        # Run inference
        outputs = await app.state.engine.run(
            model=app.state.model,
            inputs=tokens,
            **inference_kwargs
        )
        
        # Get metrics
        metrics = app.state.engine.get_latest_metrics()
        
        # Schedule background cleanup
        background_tasks.add_task(app.state.engine.cleanup)
        
        return InferenceResponse(
            generated_text=app.state.model.decode(outputs),
            metrics=metrics,
            secure_enclave_verified=outputs.secure_verified if request.secure_compute else None
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream")
async def stream_predictions(websocket):
    """Streaming inference endpoint."""
    try:
        await websocket.accept()
        
        while True:
            # Receive request
            request_data = await websocket.receive_json()
            request = InferenceRequest(**request_data)
            
            # Setup streaming inference
            async for output in app.state.engine.stream(
                model=app.state.model,
                request=request
            ):
                await websocket.send_json({
                    "text": app.state.model.decode(output),
                    "finished": False
                })
            
            # Send completion message
            await websocket.send_json({"finished": True})
            
    except Exception as e:
        logger.error(f"Streaming failed: {str(e)}")
        await websocket.close(code=1001)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        reload=False
    )
