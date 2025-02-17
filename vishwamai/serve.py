import os
import torch
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, Request, Response,
    Depends, status, WebSocket
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from prometheus_client import (
    start_http_server, Counter, Histogram, Gauge,
    CollectorRegistry, multiprocess
)

from .inference_engine import InferenceEngine, InferenceMetrics
from .model import Transformer, ModelArgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure metrics
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    'request_count_total',
    'Total number of requests processed',
    registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    registry=REGISTRY
)
GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'Current GPU memory usage in bytes',
    registry=REGISTRY
)
MODEL_CACHE_SIZE = Gauge(
    'model_cache_size_bytes',
    'Current model cache size in bytes',
    registry=REGISTRY
)

# Security
security = HTTPBearer()

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.rate = config.requests_per_minute / 60  # tokens per second
        self.burst_size = config.burst_size
        self.tokens = self.burst_size
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Attempt to acquire a rate limit token.
        
        Returns:
            bool: Whether the token was acquired
        """
        async with self.lock:
            now = time.monotonic()
            # Add new tokens based on elapsed time
            new_tokens = (now - self.last_update) * self.rate
            self.tokens = min(self.burst_size, self.tokens + new_tokens)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

class InferenceRequest(BaseModel):
    """Request model for inference."""
    text: str | List[str]  # Support batch inference
    max_length: int = Field(default=2048, le=4096)
    temperature: float = Field(default=0.7, gt=0, le=1)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, gt=0)
    stream: bool = Field(default=False)
    use_cache: bool = Field(default=True)
    secure_compute: bool = Field(default=False)
    return_progress: bool = Field(default=False)  # Return generation progress
    
    @validator('text')
    def validate_text(cls, v):
        """Validate input text."""
        if not v.strip():
            raise ValueError("Input text cannot be empty")
        if len(v) > 32000:
            raise ValueError("Input text too long (max 32000 chars)")
        return v.strip()

class InferenceResponse(BaseModel):
    """Response model for inference."""
    generated_text: str | List[str]  # Support batch inference
    metrics: InferenceMetrics
    secure_enclave_verified: Optional[bool] = None
    request_id: str
    cached: bool = False
    progress: Optional[float] = None  # Generation progress (0-1)
    performance_stats: Optional[Dict[str, float]] = None  # Model performance stats

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    
    Handles initialization and cleanup of server resources.
    """
    # Start Prometheus metrics server
    metrics_port = int(os.getenv("METRICS_PORT", "9090"))
    start_http_server(metrics_port)
    logger.info(f"Started metrics server on port {metrics_port}")
    
    try:
        # Initialize inference engine
        app.state.engine = InferenceEngine()
        
        # Load model and optimize for inference
        model_path = os.getenv("MODEL_PATH", "./checkpoints/model")
        app.state.model = load_and_optimize_model(model_path, app.state.engine)
        
        # Initialize rate limiter
        app.state.rate_limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
                burst_size=int(os.getenv("RATE_LIMIT_BURST", "10"))
            )
        )
        
        # Start metrics collection
        asyncio.create_task(collect_metrics(app))
        
        logger.info("Server initialization complete")
        yield
        
    except Exception as e:
        logger.error(f"Server initialization failed: {str(e)}")
        raise
        
    finally:
        # Cleanup
        if hasattr(app.state, "engine"):
            await app.state.engine.shutdown()
        logger.info("Server shutdown complete")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="VishwamAI Inference Server",
        description="High-performance inference server with advanced features",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    return app

app = create_app()

async def collect_metrics(app: FastAPI):
    """Periodically collect and update metrics."""
    while True:
        try:
            if torch.cuda.is_available():
                GPU_MEMORY_USAGE.set(torch.cuda.memory_allocated())
            MODEL_CACHE_SIZE.set(app.state.engine.get_cache_size())
            await asyncio.sleep(15)
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            await asyncio.sleep(60)

def load_and_optimize_model(model_path: str, engine: InferenceEngine) -> torch.nn.Module:
    """
    Load and optimize model for inference.
    
    Args:
        model_path: Path to model files
        engine: Inference engine instance
        
    Returns:
        Optimized model
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Verify model files exist
        model_dir = Path(model_path)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "pytorch_model.bin"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        # Load base model
        model_config = ModelArgs.from_json(str(config_path))
        model = Transformer(model_config)
        model.load_state_dict(torch.load(str(weights_path)))
        
        # Optimize model using inference engine
        input_shape = [1, model_config.max_seq_len]
        optimized_model = engine.optimize_model(model, input_shape)
        
        logger.info("Model loaded and optimized successfully")
        return optimized_model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Model initialization failed") from e

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    if not await app.state.rate_limiter.acquire():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return await call_next(request)

@app.get("/health")
async def health_check():
    """
    Health check endpoint providing system status.
    
    Returns:
        Dict containing system health metrics
    """
    try:
        gpu_info = {
            "available": torch.cuda.is_available(),
            "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
            "memory_cached": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB"
        } if torch.cuda.is_available() else {"available": False}
        
        return {
            "status": "healthy",
            "gpu": gpu_info,
            "engine": {
                "cache_size": app.state.engine.get_cache_size(),
                "requests_processed": REQUEST_COUNT._value.get(),
                "avg_latency": REQUEST_LATENCY._sum.get() / max(REQUEST_LATENCY._count.get(), 1)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@app.get("/model/config")
async def get_model_config():
    """
    Get model and server configuration.
    
    Returns:
        Dict containing configuration details
    """
    # Get model stats
    model_stats = {
        "type": "VishwamAI",
        "size": f"{sum(p.numel() for p in app.state.model.parameters())/1e9:.1f}B parameters",
        "version": "1.0.0",
        "performance": {
            "throughput": REQUEST_COUNT._value.get() / max(time.time() - app.state.start_time, 1),
            "avg_latency": REQUEST_LATENCY._sum.get() / max(REQUEST_LATENCY._count.get(), 1),
            "cache_hit_rate": app.state.engine.get_cache_hit_rate()
        }
    }
    
    return {
        "model": model_stats,
        "features": {
            "secure_compute": app.state.engine.config["inference"]["security"]["confidential_computing"]["enabled"],
            "streaming": True,
            "quantization": app.state.engine.config["inference"]["engine"]["quantization"]["enabled"]
        },
        "limits": {
            "max_length": 4096,
            "max_batch_size": app.state.engine.config["inference"]["engine"]["max_batch_size"],
            "rate_limit": {
                "requests_per_minute": app.state.rate_limiter.rate * 60,
                "burst_size": app.state.rate_limiter.burst_size
            }
        }
    }

@app.post(
    "/predict",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    response_description="Generated text and inference metrics"
)
async def predict(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """
    Generate predictions with advanced features.
    
    Args:
        request: Inference request parameters
        background_tasks: FastAPI background tasks
        token: Security token
        
    Returns:
        InferenceResponse containing generated text and metrics
        
    Raises:
        HTTPException: If prediction fails
    """
    request_id = f"req_{time.time_ns()}"
    try:
        with REQUEST_LATENCY.time():
            # Tokenize input
            tokens = app.state.model.tokenize(request.text)
            
            # Check cache if enabled
            if request.use_cache:
                cached = await app.state.engine.get_cached(tokens, request_id)
                if cached is not None:
                    return InferenceResponse(
                        generated_text=app.state.model.decode(cached),
                        metrics=app.state.engine.get_latest_metrics(),
                        request_id=request_id,
                        cached=True
                    )
            
            # Configure inference options
            inference_kwargs = {
                "stream": request.stream,
                "secure_compute": request.secure_compute,
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k
            }
            
            # Run inference
            outputs = await app.state.engine.run(
                model=app.state.model,
                inputs=tokens,
                **inference_kwargs
            )
            
            # Update metrics
            REQUEST_COUNT.inc()
            
            # Schedule background cleanup
            background_tasks.add_task(app.state.engine.cleanup)
            
            return InferenceResponse(
                generated_text=app.state.model.decode(outputs),
                metrics=app.state.engine.get_latest_metrics(),
                secure_enclave_verified=outputs.secure_verified if request.secure_compute else None,
                request_id=request_id
            )
            
    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.websocket("/stream")
async def stream_predictions(websocket: WebSocket):
    """
    Streaming inference endpoint.
    
    Args:
        websocket: WebSocket connection
    """
    try:
        await websocket.accept()
        logger.info("Client connected to streaming endpoint")
        
        while True:
            # Receive and validate request
            try:
                request_data = await websocket.receive_json()
                request = InferenceRequest(**request_data)
            except Exception as e:
                await websocket.send_json({
                    "error": f"Invalid request: {str(e)}",
                    "finished": True
                })
                continue
            
            # Stream predictions
            try:
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
                await websocket.send_json({
                    "error": str(e),
                    "finished": True
                })
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close(code=1001)

if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    # Run server
    uvicorn.run(
        "serve:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
