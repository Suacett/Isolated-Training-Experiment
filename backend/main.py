import logging
import os
import torch
from fastapi import FastAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Log the detected GPU device on startup.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Running on GPU: {device_name}")
    else:
        logger.warning("Running on CPU. No GPU detected.")

@app.get("/")
async def root():
    return {"message": "Proxmox AI Stock Predictor Backend is running"}
