import numpy as np
import cv2 as cv

# DCT 
def dct_2d(block):
    return cv.dct(block.astype(np.float32))

#IDCT
def idct_2d(block):
    return cv.idct(block.astype(np.float32))

# Quantization Matrix
def get_quantization_matrix(quality):

    Q50 = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    return np.clip((Q50 * scale + 50) / 100, 1, None)

# Padding 
def pad_image_to_block_size(image, block_size=8):

    h, w = image.shape[:2]

    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if len(image.shape) == 2:
        padded = np.pad(image, ((0,pad_h),(0,pad_w)), mode='edge')
    else:
        padded = np.pad(image, ((0,pad_h),(0,pad_w),(0,0)), mode='edge')

    return padded, h, w

# Bit estimation
def estimate_bits(block):
    """
    Approximates Huffman-coded JPEG bit cost.
    DC coefficient uses magnitude bits.
    AC coefficients use run-length encoding — zero runs are cheap.
    """
    coeffs = block.flatten().astype(int)

    # DC coefficient
    dc = coeffs[0]
    bits = 2 + (int(np.log2(abs(dc) + 1)) + 1) if dc != 0 else 2

    # AC coefficients — JPEG zigzag order, run-length encoded
    ac = coeffs[1:]
    zero_run = 0

    for coeff in ac:
        if coeff == 0:
            zero_run += 1
        else:
            # Each non-zero: ~4 bits for run/size header + magnitude bits
            bits += 4 + int(np.log2(abs(coeff) + 1)) + 1
            zero_run = 0

    # EOB if trailing zeros
    if zero_run > 0:
        bits += 2  # EOB marker is cheap

    return bits

# Adaptive JPEG
def process_image(image, quality, importance_map, alpha=1.0):
    block_size = 8

    # ✅ Convert to YCbCr — process channels separately
    img_ycbcr = cv.cvtColor(image, cv.COLOR_RGB2YCrCb).astype(np.float32)
    
    padded, orig_h, orig_w = pad_image_to_block_size(img_ycbcr, block_size)
    result = np.zeros_like(padded)
    total_bits = 0

    Q_base = get_quantization_matrix(quality).astype(np.float32)
    
    # Pad importance map to match
    imp_padded = np.pad(importance_map, 
                        ((0, padded.shape[0]-importance_map.shape[0]),
                         (0, padded.shape[1]-importance_map.shape[1])), 
                        mode='edge')

    for c in range(3):
        channel = padded[:, :, c]
        
        # ✅ Chroma channels (Cb, Cr) get fixed 2x coarser quantization
        # Luma (Y) gets adaptive importance-based quantization
        is_luma = (c == 0)

        for i in range(0, padded.shape[0], block_size):
            for j in range(0, padded.shape[1], block_size):
                block = channel[i:i+block_size, j:j+block_size]
                if block.shape != (block_size, block_size):
                    continue

                dct_block = dct_2d(block)
                # Scale alpha down at high quality — less aggressive at high bitrates
                effective_alpha = alpha * (1.0 - quality / 200.0)

                if is_luma:
                    # Adaptive quantization on Y channel
                    imp_region = imp_padded[i:i+block_size, j:j+block_size]
                    importance_score = imp_region.mean()
                    scale = 1.0 + alpha * (1.0 - importance_score) ** 2
                    scale = np.clip(scale, 0.5, 3.0)
                    Q = Q_base * scale
                else:
                    # Fixed 2x coarser for chroma — human eye doesn't care
                    Q = Q_base * 2.0

                quantized = np.round(dct_block / Q)
                total_bits += estimate_bits(quantized)
                dequantized = quantized * Q
                reconstructed = np.clip(idct_2d(dequantized), 0, 255)
                result[i:i+block_size, j:j+block_size, c] = reconstructed

    result = result[:orig_h, :orig_w]
    
    # ✅ Convert back to RGB
    result_rgb = cv.cvtColor(result.astype(np.uint8), cv.COLOR_YCrCb2RGB)
    return result_rgb, total_bits

# Standard JPEG
def process_image_standard(image, quality):

    block_size = 8

    padded_image, orig_h, orig_w = pad_image_to_block_size(image, block_size)

    padded_image = padded_image.astype(np.float32)

    processed_image = np.zeros_like(padded_image)

    total_encoded_bits = 0

    Q = get_quantization_matrix(quality).astype(np.float32)

    h_padded, w_padded = padded_image.shape[:2]

    if len(padded_image.shape) == 2:
        channels = 1
    else:
        channels = padded_image.shape[2]

    for c in range(channels):

        channel_data = padded_image if channels == 1 else padded_image[:,:,c]

        for i in range(0, h_padded, block_size):
            for j in range(0, w_padded, block_size):

                block = channel_data[i:i+block_size, j:j+block_size]

                if block.shape != (block_size, block_size):
                    continue

                dct_block = dct_2d(block)

                quantized_block = np.round(dct_block / Q)

                total_encoded_bits += estimate_bits(quantized_block)

                dequantized = quantized_block * Q

                reconstructed = idct_2d(dequantized)

                reconstructed = np.clip(reconstructed, 0, 255)

                if channels == 1:
                    processed_image[i:i+block_size, j:j+block_size] = reconstructed
                else:
                    processed_image[i:i+block_size, j:j+block_size, c] = reconstructed

    processed_image = processed_image[:orig_h, :orig_w]

    return processed_image.astype(np.uint8), total_encoded_bits
def process_image_bitrate_neutral(image, quality, importance_map, alpha=1.0):
    """
    Redistribute bits from unimportant to important regions.
    Total bits should stay approximately equal to standard JPEG.
    """
    block_size = 8
    padded, orig_h, orig_w = pad_image_to_block_size(image, block_size)
    padded = padded.astype(np.float32)
    result = np.zeros_like(padded)
    total_bits = 0

    Q_base = get_quantization_matrix(quality).astype(np.float32)

    # Precompute importance score per block
    h, w = padded.shape[:2]
    imp_padded = np.pad(importance_map,
                        ((0, h - importance_map.shape[0]),
                         (0, w - importance_map.shape[1])), mode='edge')

    block_scores = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            score = imp_padded[i:i+block_size, j:j+block_size].mean()
            block_scores.append(score)

    scores = np.array(block_scores)

    # ✅ Key: normalize scales so they average to 1.0
    # This means total quantization is same as standard — just redistributed
    raw_scales = 1.0 + alpha * (1.0 - scores) ** 2  # unimportant → high scale
    # Invert: important blocks get FINER quantization (scale < 1)
    # Unimportant blocks get COARSER quantization (scale > 1)
    # Normalize so mean scale = 1.0 → bitrate neutral
    raw_scales = raw_scales / raw_scales.mean()
    raw_scales = raw_scales * 1.05  # ← ADD THIS: matches process_image_torch
    raw_scales = np.clip(raw_scales, 0.3, 4.0)

    channels = 1 if len(padded.shape) == 2 else padded.shape[2]
    block_idx = 0

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            scale = raw_scales[block_idx % len(raw_scales)]
            block_idx += 1

            Q = Q_base * scale

            for c in range(channels):
                block = padded[i:i+block_size, j:j+block_size] if channels == 1 \
                        else padded[i:i+block_size, j:j+block_size, c]

                if block.shape != (block_size, block_size):
                    continue

                dct_block = dct_2d(block)
                quantized = np.round(dct_block / Q)
                total_bits += estimate_bits(quantized)
                dequantized = quantized * Q
                reconstructed = np.clip(idct_2d(dequantized), 0, 255)

                if channels == 1:
                    result[i:i+block_size, j:j+block_size] = reconstructed
                else:
                    result[i:i+block_size, j:j+block_size, c] = reconstructed

    return result[:orig_h, :orig_w].astype(np.uint8), total_bits

# RMSE
def calculate_rmse(original, reconstructed):

    return np.sqrt(((original - reconstructed) ** 2).mean())