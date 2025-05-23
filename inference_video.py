# /root/autodl-tmp/Convert2TRT/inference_video.py
import argparse
import os
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tools.pipeline import RealESRGANPipeline, logger

def get_model_path(width: int = 512, height: int = 512, scale: int = 2) -> str:
    """
    Construct the model path based on dimensions and scale factor.
    
    Args:
        width: Input width (default: 512)
        height: Input height (default: 512)
        scale: Scale factor (2 or 4)
    
    Returns:
        str: Path to the appropriate model
    """
    model_name = f"t2trt_{width}_{height}_x{scale}.trt"
    model_path = Path("./src/models/tensorrt_model_output") / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model {model_name} not found. Please create the model first using convert_trt.py "
            f"with arguments: --width {width} --height {height} --scale {scale}"
        )
    
    return str(model_path)

def process_video(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    width: int = 512,
    height: int = 512,
    scale_factor: int = 2,
    max_frames: Optional[int] = None,
    ensure_compatibility: bool = True
) -> None:
    """
    Process a video with RealESRGAN.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video (if None, will use input_path with _upscaled suffix)
        width: Input width (default: 512)
        height: Input height (default: 512)
        scale_factor: 2 for 2X model, 4 for 4X model
        max_frames: Maximum number of frames to process (None for all frames)
        ensure_compatibility: Whether to ensure cloud compatibility of the output video
    """
    input_path = Path(input_path)
    
    # Set default output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"
    else:
        output_path = Path(output_path)
        
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get appropriate model path
    model_path = get_model_path(width, height, scale_factor)
    
    # Initialize pipeline
    pipeline = RealESRGANPipeline(model_path, scale_factor)
    
    # Process video
    start_time = time.time()
    
    try:
        pipeline.process_video(input_path, output_path, max_frames, ensure_compatibility)
        
        total_elapsed = time.time() - start_time
        logger.info(f"Video processing completed in {total_elapsed:.2f} seconds")
        logger.info(f"Result saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description="RealESRGAN Video Upscaling")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--width", type=int, default=512, help="Input width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Input height (default: 512)")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4],
                        help="Upscaling factor (2 or 4)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process")
    parser.add_argument("--no-compatibility", action="store_true",
                        help="Skip ensuring cloud compatibility of output video")
    
    args = parser.parse_args()
    
    # Process video
    process_video(
        args.input,
        args.output,
        args.width,
        args.height,
        args.scale,
        args.max_frames,
        not args.no_compatibility
    )

if __name__ == "__main__":
    main()
