import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Testing direct import of ErrorCorrectionTrainer")

try:
    # Direct import from the module without going through __init__.py
    from vishwamai.error_correction_trainer import ErrorCorrectionTrainer
    logger.info("Successfully imported ErrorCorrectionTrainer directly from vishwamai.error_correction_trainer")
    print("SUCCESS: ErrorCorrectionTrainer imported correctly")
except ImportError as e:
    logger.error(f"Failed to import ErrorCorrectionTrainer: {str(e)}")
    print(f"ERROR: {str(e)}")
