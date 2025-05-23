#!/bin/bash

# Create a log file with timestamp
LOG_FILE="./logs/test_$(date +%Y%m%d_%H%M%S).log"
echo "Starting tests at $(date)" | tee "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Function to run command and log output
run_command() {
    echo "Running: $1" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    eval "$1" 2>&1 | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Command completed with status: $?" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# # Convert models
# run_command "python convert_trt.py --pretrained_model 4"
# run_command "python convert_trt.py --pretrained_model 2"

# Process video samples
run_command "python inference_video.py --input /root/autodl-tmp/Convert2TRT/src/samples/sample.mp4 --output /root/autodl-tmp/Convert2TRT/src/samples/sample_x4.mp4 --scale 4"
run_command "python inference_video.py --input /root/autodl-tmp/Convert2TRT/src/samples/sample.mp4 --output /root/autodl-tmp/Convert2TRT/src/samples/sample_x2.mp4 --scale 2"

# Process image samples
run_command "python inference_image.py --input /root/autodl-tmp/Convert2TRT/src/samples/lora1_sample_512.jpeg --output /root/autodl-tmp/Convert2TRT/src/samples/lora1_sample_512_x2.png --scale 2"
run_command "python inference_image.py --input /root/autodl-tmp/Convert2TRT/src/samples/lora1_sample_512.jpeg --output /root/autodl-tmp/Convert2TRT/src/samples/lora1_sample_512_x4.png --scale 4"

echo "All tests completed at $(date)" | tee -a "$LOG_FILE"
echo "Log file saved as: $LOG_FILE" | tee -a "$LOG_FILE"
