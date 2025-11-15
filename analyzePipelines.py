"""
Module for evaluating image compression methods.
Calculates metrics such as PSNR, SSIM, MSE, Compression Ratio, BPP, Space Saving, and Image Complexity.
Saves results to CSV for further analysis.
"""

import numpy as np
import cv2
import csv
import time
import json
import compressionPipeline
import copy
import skimage.measure  
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
from datetime import datetime
from PIL import Image

class CompressionEvaluator:
    """
    Class for evaluating compression methods on images
    Calculates the main metrics and saves results to CSV
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.csv_rows = []
        
    def load_image(self, image_path):
        """Load image in RGB format."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def compress_jpeg(self, original_array, config):
        """
        Runs the full JPEG encode/decode pipeline on an in-memory image array.
        
        Args:
            original_array (np.ndarray): The original image (H, W, 3)
            config (dict): Configuration for the pipeline
        """
        if original_array is None:
            raise ValueError("You must input an image array.")
        
        # Convert numpy array to PIL Image for the pipeline
        pil_img = Image.fromarray(original_array)

        # Timing the encoding process-----------
        encode_start = time.time()
        # Pass the PIL Image object directly
        meta = compressionPipeline.jpeg_encode_pipeline(pil_img, config) 
        encode_time = (time.time() - encode_start) * 1000
        #--------------------------------------
        
        # Calculate compressed size from num_bits for precision
        y_data_bits = meta['Y']['num_bits']
        cb_data_bits = meta['Cb']['num_bits']
        cr_data_bits = meta['Cr']['num_bits']
        
        # Compressed data size in bits
        compressed_data_bits = y_data_bits + cb_data_bits + cr_data_bits

        # Estimating the header size
        meta_headers = copy.deepcopy(meta)
        for channel in ['Y', 'Cb', 'Cr']:
            if channel in meta_headers:
                if 'packed_bitstream' in meta_headers[channel]:
                    del meta_headers[channel]['packed_bitstream'] 
                if 'num_bits' in meta_headers[channel]:
                    del meta_headers[channel]['num_bits']
        for key in ['Y_original', 'Cb_original', 'Cr_original']:
            if key in meta_headers:
                del meta_headers[key]
        header_size_estimate_bytes = len(json.dumps(meta_headers).encode('utf-8'))
            
        # Total compressed size in bytes
        compressed_size_bytes = (compressed_data_bits // 8) + header_size_estimate_bytes
        
        # Timing the decoding process ----------------------
        decode_start = time.time()
        decompressed_img = compressionPipeline.jpeg_decode_pipeline(meta)
        decode_time = (time.time() - decode_start) * 1000
        #---------------------------------------------------
        decompressed_array = np.array(decompressed_img)
        
        return decompressed_array, compressed_size_bytes, encode_time, decode_time
    
    def calculate_mse(self, original, compressed):
        return mean_squared_error(original, compressed)
    
    def calculate_psnr(self, original, compressed):
        return psnr(original, compressed, data_range=255)
    
    def calculate_ssim(self, original, compressed):
        return ssim(original, compressed, data_range=255, channel_axis=2)
    
    def calculate_compression_ratio(self, original_size, compressed_size):
        if compressed_size == 0:
            return np.inf
        return compressed_size / original_size
    
    def calculate_bpp(self, compressed_size, image_shape):
        num_pixels = image_shape[0] * image_shape[1]
        bits = compressed_size * 8
        return bits / num_pixels
    
    def calculate_space_saving(self, original_size, compressed_size):
        return ((original_size - compressed_size) / original_size) * 100
    
    def calculate_image_complexity(self, image):
        """
        Calculate image complexity metrics (edge density, entropy).
        Returns dict with edge_density and entropy
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Edge Density - percentage of pixels that are edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        
        # Calculate Shannon Entropy
        img_entropy = skimage.measure.shannon_entropy(gray)
        
        return {
            'edge_density': round(edge_density, 2),
            'entropy': round(img_entropy, 2),
        }
    
    def evaluate_image(self, config):        
        """
        Evaluate a single image with a given configuration.
        """
        image_path = config['image_path']
                
        original = self.load_image(image_path)

        # These are constant for this image
        h, w, c = original.shape
        original_size = h * w * c # Uncompressed size in bytes
        image_name = Path(image_path).name
        resolution = f"{w}x{h}"
        complexity = self.calculate_image_complexity(original)

        print(f"  Running: {image_name} (Q={config['quality']}, Quant={config['quantization_method']}, Chroma={config['chroma_method']})")

        compressed, comp_size, enc_time, dec_time = self.compress_jpeg(
            original_array=original,
            config=config
        )
        
        row = {
            'image_name': image_name,
            'resolution': resolution,
            'type': 'JPEG',
            'quantization_method': config['quantization_method'],
            'chroma_method': config['chroma_method'],
            'quality_setting': config['quality'],
            'psnr_db': round(self.calculate_psnr(original, compressed), 2),
            'ssim': round(self.calculate_ssim(original, compressed), 4),
            'mse': round(self.calculate_mse(original, compressed), 2),
            'compression_ratio': round(self.calculate_compression_ratio(original_size, comp_size), 2),
            'bpp': round(self.calculate_bpp(comp_size, original.shape), 2),
            'original_size_bytes': original_size,
            'compressed_size_bytes': comp_size,
            'space_saving_percent': round(self.calculate_space_saving(original_size, comp_size), 2),
            'encode_time_ms': round(enc_time, 2),
            'decode_time_ms': round(dec_time, 2),
            'edge_density': complexity['edge_density'],
            'entropy': complexity['entropy'],
        }
        self.csv_rows.append(row)


    def evaluate_dataset(self, image_folder, jpeg_qualities, quantization_methods, chroma_methods, collect_stats_for=False):
        """
        Evaluate all images in a folder
        """
        image_folder = Path(image_folder)
        image_paths = [p for p in image_folder.iterdir() if p.suffix in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']]
        
        if not image_paths:
            raise ValueError(f"No images found in {image_folder}")
        
        print(f"Found {len(image_paths)} images in {image_folder}")
        
        # Build the list of all configurations to run
        configs = []
        
        for path in image_paths:
            for jpeg_quality in jpeg_qualities:
                for quant_method in quantization_methods:
                    for chroma_method in chroma_methods:
                        configs.append({
                            'image_path': str(path),
                            'quality': jpeg_quality,
                            'quantization_method': quant_method,
                            'chroma_method': chroma_method,
                            'collect_stats_for': collect_stats_for
                    })
                
        
        total_runs = len(configs)
        print(f"Total experiment runs to perform: {total_runs}")
        
        for run_count, config in enumerate(configs, 1):
            print(f"\n--- Run {run_count}/{total_runs} ---")
            try:
                self.evaluate_image(config)
            except Exception as e:
                print(f"Error processing {config['image_path']} with config {config}: {e}")

        self.save_to_csv()
        return self.csv_rows
    
    def save_to_csv(self):
        """Save results to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"compression_results_{timestamp}.csv"
        
        if not self.csv_rows:
            print("No results to save")
            return

        fieldnames = list(self.csv_rows[0].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.csv_rows)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    evaluator = CompressionEvaluator(output_dir='compression_results')
    test_folder = 'images' 
    
    jpeg_qualities = [10, 30, 50, 70, 90] 
    
    quantization_methods = [
        'standard',
        'flat',
        'deadzone'
    ]

    chroma_methods = [
        'nearest',
        'average',
        '444'
    ]

    print(f"Evaluating images in folder: {test_folder}")

    results = evaluator.evaluate_dataset(
        test_folder,
        jpeg_qualities,
        quantization_methods,
        chroma_methods,
        collect_stats_for = True
    ) 
    print(f"Evaluation complete.")
