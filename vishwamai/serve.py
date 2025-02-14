import os
import torch
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, start_http_server

from vishwamai.model import Transformer, ModelArgs
from vishwamai.cache_augmentation import CacheConfig, DifferentiableCacheAugmentation
from vishwamai.neural_memory import ReasoningMemoryTransformer
from vishwamai.tree_of_thoughts import TreeOfThoughts
from vishwamai.reward_function import RewardConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics
INFERENCE_REQUESTS = Counter("inference_requests_total", "Total number of inference requests")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference request latency in seconds")

class PredictionRequest(BaseModel):
    text: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_cache: bool = True

class PredictionResponse(BaseModel):
    generated_text: str
    performance_metrics: Dict[str, float]

app = FastAPI(
    title="VishwamAI Model Server",
    description="API server for VishwamAI language model inference",
    version="1.0.0"
)

def load_model():
    """Load model and components."""
    try:
        model_path = os.environ.get("MODEL_PATH", "./checkpoints/model")
        config_path = os.path.join(model_path, "config.json")
        
        # Load model configuration
        model_config = ModelArgs.from_json(config_path)
        model = Transformer(model_config)
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
        model.eval().cuda()
        
        # Load auxiliary components
        cache_module = DifferentiableCacheAugmentation.from_pretrained(model_path).cuda()
        memory_module = ReasoningMemoryTransformer.from_pretrained(model_path).cuda()
        tree_module = TreeOfThoughts.from_pretrained(model_path).cuda()
        
        return model, cache_module, memory_module, tree_module
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Model initialization failed")

@app.on_event("startup")
async def startup_event():
    """Initialize model and metrics server on startup."""
    global model, cache_module, memory_module, tree_module
    model, cache_module, memory_module, tree_module = load_model()
    
    # Start Prometheus metrics server
    start_http_server(9090)
    logger.info("Model server initialized successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate predictions from input text."""
    INFERENCE_REQUESTS.inc()
    
    with INFERENCE_LATENCY.time():
        try:
            # Tokenize input
            tokens = model.tokenize(request.text)
            tokens = tokens.cuda()
            
            with torch.inference_mode():
                # Generate base output
                outputs = model(tokens)
                
                # Apply enhancements if enabled
                if request.use_cache:
                    enhanced = cache_module(outputs)
                    memory_enhanced = memory_module(enhanced)
                    final_output = tree_module(memory_enhanced)
                else:
                    final_output = outputs
                
                # Decode output
                generated_text = model.decode(final_output)
                
                # Calculate performance metrics
                metrics = {
                    "total_tokens": len(tokens[0]),
                    "generation_time": INFERENCE_LATENCY.observe(),
                    "memory_usage": torch.cuda.max_memory_allocated() / 1024**3
                }
                
                return PredictionResponse(
                    generated_text=generated_text,
                    performance_metrics=metrics
                )
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_model_config():
    """Get model configuration."""
    try:
        return {
            "model_type": "VishwamAI",
            "model_size": f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters",
            "version": "1.0.0",
            "components": {
                "cache": type(cache_module).__name__,
                "memory": type(memory_module).__name__,
                "tree": type(tree_module).__name__
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
