import argparse
import os
import torch
from torch2trt import torch2trt
from src.common import prepare, test

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Convert RealESRGAN model to TensorRT.")
    parser.add_argument("--height", type=int, default=512, help="Input height of the model")
    parser.add_argument("--width", type=int, default=512, help="Input width of the model")
    parser.add_argument("--pretrained_model", type=str, default="2", choices=["2", "4"],
                        help="Choose between pretrained RealESRGAN_x2plus or RealESRGAN_x4plus")
    parser.add_argument("--model_output_directory", type=str,
                        default="./src/models/tensorrt_model_output",
                        help="Output directory for saving TensorRT model")
    args = parser.parse_args()

    # Construct input size
    side = (args.width,args.height )

    # Check pretrained model path
    model_file = f"./src/models/pretrained_models/RealESRGAN_x{args.pretrained_model.lower()}plus.pth"
    if not os.path.isfile(model_file):
        print(f"[ERROR] Pretrained model not found: {model_file}")
        print("Please place the correct model file in './src/models/pretrained_models/'")
        return

    # Prepare model
    is_2x = False
    if args.pretrained_model.lower() == "2":
        is_2x = True
        
    model, x = prepare(side, model_file, is_2x)

    torch.cuda.empty_cache()
    with torch.no_grad():
        print("Converting to TensorRT...")
        model_trt = torch2trt(model, [x], fp16_mode=True)

    # Ensure output directory exists
    os.makedirs(args.model_output_directory, exist_ok=True)

    # Save model
    output_path = os.path.join(args.model_output_directory, f't2trt_{side[0]}_{side[1]}_x{args.pretrained_model}.trt')
    torch.save(model_trt.state_dict(), output_path)
    print(f"TensorRT model saved to {output_path}")

    # Test outputs
    y = test(x, model, "model")
    y_trt = test(x, model_trt, "model_trt")
    print("Error between original and TRT model:", torch.max(torch.abs(y - y_trt)))

if __name__ == "__main__":
    main()
