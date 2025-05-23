# 🔼 RealESRGAN TensorRT Upscaler

This project provides a high-performance image and video upscaler using [RealESRGAN](https://github.com/xinntao/Real-ESRGAN), accelerated with NVIDIA TensorRT. It supports both 2x and 4x upscaling, and allows you to specify output height and width. Instructions are included for image and video upscaling using the converted models.

---

## ✅ Features

### 🚀 Performance
- ⚡ Up to **3–5× faster** inference using TensorRT vs standard PyTorch

### 🖼️ Image & Video Upscaling
- 🔧 Supports **2x** and **4x** RealESRGAN models
- 📐 Customizable **output dimensions** (height & width)
- 🖼️ Includes ready-to-use scripts for **image** and **video** processing

### 🛠️ Model Conversion
- 🔄 Simple conversion from PyTorch to TensorRT
- 📦 **Static TensorRT engine**: optimized for fixed input sizes  
  ↪️ *To change input size, re-export the engine*

### 🧪 In Development (Upcoming Features)
- 🔁 **Dynamic TensorRT**: handle varying input sizes without re-export
- 🔊 **Audio transfer** support: for full video enhancement pipelines

---

## 🧠 Environments Used

| Component  | Environment 1     
|------------|---------------
| GPU        | RTX 40 Series         
| CUDA       | 12.1               
| cuDNN      | 8.9.2             
| PyTorch    | 2.1.0+cu121         
| Python     | 3.10.8             
| OS         | Ubuntu 22.04      

---

## ⚙️ Installation

### 1. Install TensorRT

- Official guide: https://developer.nvidia.com/tensorrt/download
- Or via pip:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install wheel
python3 -m pip install --upgrade tensorrt
```

Verify:

```python
>>> import tensorrt
>>> print(tensorrt.__version__)
```

📚 Full guide: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html

---

### 2. Install `basicsr`

```bash
pip install basicsr==1.4.2 --index-url https://pypi.org/simple
```

---

### 3. Download Pretrained Models

- [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)  
- [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

Place in:

```
./src/models/pretrained_models/
```

---

### 4. Install `torch2trt`

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install --plugins
```

If you see `NvInfer.h: No such file or directory`, update `setup.py`:

```python
include_dirs=[
    trt_inc_dir(),
    '/usr/local/tensorrt/include'
],
library_dirs=[
    trt_lib_dir(),
    '/usr/local/tensorrt/targets/x86_64-linux-gnu/lib'
]
```

---

## 🔁 Model Conversion

### Default Conversion (2x, 512x512)

```bash
python convert_to_trt.py
```

### Custom Size

```bash
python convert_to_trt.py --height 480 --width 273
```

### Use 4x Model

```bash
python convert_to_trt.py --pretrained_model 4X
```

### Custom Output Directory

```bash
python convert_to_trt.py --model_output_directory ./my_outputs
```

### 🧪 Full Example

```bash
python convert_to_trt.py --height 480 --width 273 --pretrained_model 4X --model_output_directory ./my_outputs
```

---

## 🖼️ 2. Image Super-Resolution

Process images with different scaling factors:

```bash
python inference_image.py --input src/samples/input.jpeg --output src/samples/output_x2.png --scale 2
python inference_image.py --input src/samples/input.jpeg --output src/samples/output_x4.png --scale 4
```

---

## 🎥 3. Video Super-Resolution

Process videos with different scaling factors:

```bash
python inference_video.py --input src/samples/input.mp4 --output src/samples/output_x2.mp4 --scale 2
python inference_video.py --input src/samples/input.mp4 --output src/samples/output_x4.mp4 --scale 4
```

---

## 🖼️ Example Results
### 🖼️ Image Super-Resolution Results



<p align="center">
  <img src="src/samples/lora1_sample_512.jpeg" width="256">
  <img src="src/samples/lora1_sample_512_x2.png" width="256">
  <img src="src/samples/lora1_sample_512_x4.png" width="256">
</p>

<p align="center">
  <em>Left: Original | Middle: x2 Upscaled | Right: x4 Upscaled</em>
</p>

---

### 🎥 Video Super-Resolution Results

**Input:**  
`src/samples/sample.mp4`

**Outputs:**  
`src/samples/sample_x2.mp4`  
`src/samples/sample_x4.mp4`

To view the upscaled videos:

➡️ [Download and play the original video](src/samples/sample.mp4)  
➡️ [Download and play the 2x upscaled video](src/samples/sample_x2.mp4)  
➡️ [Download and play the 4x upscaled video](src/samples/sample_x4.mp4)


## 📁 Folder Structure

```
src/
├── models/
│   └── pretrained_models/
│       ├── RealESRGAN_x2plus.pth
│       └── RealESRGAN_x4plus.pth
├── samples/
│   ├── input.jpeg
│   ├── input.mp4
│   └── [upscaled output files]
├── convert_to_trt.py
├── inference_image.py
├── inference_video.py
```

---

## 📜 License

MIT License
