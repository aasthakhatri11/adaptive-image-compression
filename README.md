# CNN-based Adaptive Image Compression 

## Overview

This project implements an adaptive image compression pipeline that improves upon standard JPEG by allocating compression based on learned spatial importance.

A convolutional neural network predicts an importance map for each image, allowing the system to preserve high-detail regions while applying stronger compression to less important areas. This results in improved rate–distortion performance compared to uniform compression.

---

## Key Features

* CNN-based importance map prediction
* Adaptive JPEG quantization guided by learned features
* Block-wise DCT compression pipeline
* Rate–distortion evaluation using RMSE and Bits-Per-Pixel (BPP)

---

## Methodology

1. Input image is passed through a neural network to generate an importance map
2. Importance scores are computed for each 8×8 block
3. JPEG quantization is dynamically adjusted based on block importance
4. Image is reconstructed using inverse DCT
5. Performance is evaluated against standard JPEG compression

---

## Results

The adaptive compression approach demonstrates improved efficiency by reducing bitrate while maintaining comparable reconstruction quality.

Evaluation metrics include:

* RMSE (Reconstruction Error)
* BPP (Bits Per Pixel)
* Rate–Distortion Curves

---

## Project Structure

```bash
src/            # core implementation (models, compression pipeline)
notebooks/      # experiments and visualizations
experiments/    # evaluation outputs and plots
```

---

## Technologies Used

* PyTorch
* NumPy
* OpenCV
* Matplotlib

---

## Summary

This project explores how deep learning can enhance classical image compression techniques by introducing content-aware optimization, bridging traditional signal processing and modern neural methods.
