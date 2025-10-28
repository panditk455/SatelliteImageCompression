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
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

class CompressionEvaluator:
    """
    Class for evaluating compression methods on images
    Calculates the main metrics and saves results to CSV
    These metrics include:
    - PSNR
    - SSIM
    - MSE
    - Compression Ratio
    - Bits Per Pixel (BPP)
    - Space Saving Percentage
    - Encoding and Decoding Time (ms)
    - Image Complexity Metrics (Edge Density, Entropy, Spatial Complexity)
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.csv_rows = []
        
    def load_image(self, image_path):
        """Load image in RGB format. This is redundant with the compression pipeline but keeps things clear."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def compress_jpeg(self, quality, image_path=None):
        if image_path is None:
            raise ValueError("You must input an image path.")

        # Timing the encoding process-----------
        encode_start = time.time()
        meta = compressionPipeline.jpeg_encode_pipeline(
            str(image_path), 
            quality=quality, 
            chroma_method="nearest", 
            show_first_block=False
        )
        encode_time = (time.time() - encode_start) * 1000
        #--------------------------------------
        
        y_data_size = len(meta['Y']['packed_bitstream'])
        cb_data_size = len(meta['Cb']['packed_bitstream'])
        cr_data_size = len(meta['Cr']['packed_bitstream'])
        
        # Compressed data size of the packed bitstreams (minus the metadata headers)
        compressed_data_size = y_data_size + cb_data_size + cr_data_size
    
        # Estimating the header size
        # TODO: Might actually compress this to bytes (in the other compression file) to get a more accurate size/process
        meta_headers = copy.deepcopy(meta)
        for channel in ['Y', 'Cb', 'Cr']:
            if channel in meta_headers and 'packed_bitstream' in meta_headers[channel]:
                del meta_headers[channel]['packed_bitstream'] 
        header_size_estimate = len(json.dumps(meta_headers).encode('utf-8'))
            
        # Total compressed size of the ("jpeg"?) image is image data itself plus the header
        compressed_size = compressed_data_size + header_size_estimate
        
        # Timing the decoding process ----------------------
        decode_start = time.time()
        decompressed_img = compressionPipeline.jpeg_decode_pipeline(meta)
        decode_time = (time.time() - decode_start) * 1000
        #---------------------------------------------------
        decompressed_array = np.array(decompressed_img) # NOTE: Should I include conversion to np array during timing?

        return decompressed_array, compressed_size, encode_time, decode_time
    
    def compress_jpeg2000(self, image, quality_layers):
        """
        JPeg2000 compression goes here
        TODO: Implement this function
        """
        pass
    
    def calculate_mse(self, original, compressed):
        """Calculate Mean Squared Error"""
        return mean_squared_error(original, compressed)
    
    def calculate_psnr(self, original, compressed):
        """Calculate Peak Signal-to-Noise Ratio"""
        return psnr(original, compressed, data_range=255)
    
    def calculate_ssim(self, original, compressed):
        """Calculate Structural Similarity Index"""
        # Convert to grayscale for SSIM calculation
        # This is what I see online but it does lose color info
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original
            compressed_gray = compressed
        return ssim(original_gray, compressed_gray, data_range=255)
    
    def calculate_compression_ratio(self, original_size, compressed_size):
        """Calculate compression ratio"""
        if compressed_size == 0:
            return np.inf
        return original_size / compressed_size
    
    def calculate_bpp(self, compressed_size, image_shape):
        """Calculate bits per pixel"""
        num_pixels = image_shape[0] * image_shape[1]
        bits = compressed_size * 8
        return bits / num_pixels
    
    def calculate_space_saving(self, original_size, compressed_size):
        """Calculate space saving percentage"""
        return ((original_size - compressed_size) / original_size) * 100
    
    def calculate_image_complexity(self, image):
        """
        Calculate image complexity metrics
        Returns dict with edge_density, entropy, and spatial_complexity
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Edge Density - percentage of pixels that are edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        
        # Shannon Entropy
        img_entropy = skimage.measure.shannon_entropy(gray)
        
        # Spatial Complexity - standard deviation of gradients
        # Generated by Gemini
        # TODO: Verify correctness, but it is here as an option to consider
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        spatial_complexity = np.std(gradient_magnitude)
        
        return {
            'edge_density': round(edge_density, 2),
            'entropy': round(img_entropy, 2),
            'spatial_complexity': round(spatial_complexity, 2)
        }
    
    def evaluate_image(self, image_path, jpeg_qualities):
        """
        Evaluate a single image with both compression methods
        TODO: Add JPEG2000 Evaluation Here. To do that can just copy the JPEG loop and modify.
        Add parameter for jpeg2000 quality layers if taking that approach.
        """
        print(f"\nEvaluating: {Path(image_path).name}")
        
        # Load original image
        original = self.load_image(image_path)

        # These are constant in the output across quality levels
        h, w, c = original.shape
        original_size = h * w * c
        image_name = Path(image_path).name
        resolution = f"{w}x{h}"
        complexity = self.calculate_image_complexity(original)
        
        # Evaluate JPEG
        for quality in jpeg_qualities:
            compressed, comp_size, enc_time, dec_time = self.compress_jpeg(
                quality, 
                image_path=image_path
            )
            
            row = {
                'image_name': image_name,
                'resolution': resolution,
                'type': 'JPEG',
                'quality_setting': quality,
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
                'spatial_complexity': complexity['spatial_complexity']
            }
            
            self.csv_rows.append(row)
            
    def evaluate_dataset(self, image_folder, jpeg_qualities):
        """Evaluate all images in a folder"""
        image_folder = Path(image_folder)
        image_paths = [p for p in image_folder.iterdir()]
        
        if not image_paths:
            raise ValueError(f"No images found in {image_folder}")
        
        print(f"Evaluating {len(image_paths)} images")
        
        for img_path in image_paths:
            try:
                self.evaluate_image(img_path, jpeg_qualities)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        self.save_to_csv()
        return self.csv_rows
    
    def save_to_csv(self):
        """Save results to CSV file"""
        # Found interesting way to print tables to console: 
        # https://medium.com/@inzaniak/create-tables-in-your-terminal-with-python-6747d68d71a6
        from rich.console import Console
        from rich.table import Table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"compression_results_{timestamp}.csv"
        
        if not self.csv_rows:
            print("No results to save")
            return

        fieldnames = [
            'image_name',
            'resolution',
            'type',
            'quality_setting',
            'psnr_db',
            'ssim',
            'mse',
            'compression_ratio',
            'bpp',
            'original_size_bytes',
            'compressed_size_bytes',
            'space_saving_percent',
            'encode_time_ms',
            'decode_time_ms',
            'edge_density',
            'entropy',
            'spatial_complexity'
        ]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_rows)

        # Printing a preview of the results to console
        table = Table(title="Compression Metric Results Preview (First 10 Rows)")

        for column in fieldnames:
            table.add_column(column)
        for i, row in enumerate(self.csv_rows):
            if i == 10:
                break
            row_data = [str(row.get(col, 'N/A')) for col in fieldnames]
            table.add_row(*row_data, style='bright_green')

        console = Console()
        console.print(table)


if __name__ == "__main__":
    evaluator = CompressionEvaluator(output_dir='compression_results')
    
    # Arbitrary JPEG quality levels to test
    # This will be interesting to look into/change more for analysis
    jpeg_qualities = [30, 50, 70, 85, 95]

    results = evaluator.evaluate_dataset(
        'images',
        jpeg_qualities
    )
