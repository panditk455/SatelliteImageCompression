# Documentation on the files in this folder

## There are three files in this folder: quantization.py, partition.py, and entropy_parallel.py.
## These python files are created for three steps in the JPEG 2000 pipeline.

**1. Quantization:** After DWT, the image get divided into several layers with three parts: HH, HL, and LH and one LL part. We apply quantization to the output. We choose differet step value for each parts in the numpy array based on their importance. Quantization reduces the number of different integers and therefore make the data easier to store.

**2. Partition:** After quantization, we apply partition function, which divide the image further into smaller chuncks (64x64 by default if user doesn't specify). This step helps make the final compressed file size better (entropy coding works better with smaller blocks) and save some entropy coding time by parallely applying entropy coding to the blocks.

**3. Entropy coding:** After partitioning the data array into different blocks, we apply entropy coding (a combination of Arithmetic coding and DEFLATE based on the nature of the coefficients in the blocks) to each of the blocks in parallel.

**Note:** 
We made some decision and didn't strictly follow the JPEG2000 pipeline because of the time limitation and resource we have access to. <br>
(a)For quantization, we implemented a very basic version of linear quantization without optimization specifically designed for JPEG2000.<br>
(b)For tier 1 (EBCOT: entropy coding along with some format conversion for optimization), we decided to ignore most optimiaztion and format conversion for now but we kept the two very main steps, which are partitioning and arithmetic coder.  <br>
(c)We decide to convert the results after tier 1 into binary file instead of following the traditional tier 2 + jp2 format conversion from code stream.<br>
(d)More work (Example: optimization/tier 2/...) could be done for future work.<br>

## How to use:
### Quantization
**Encoding:**
```
Import quantization.py and call quantize_all(dwt_result: List[Dict], quality: float)
Parameters: dwt_result: the list of dictionaries (one dictionary for each in Y, Cb, Cr)
            quality: the quality value the user inputs (0 - 100, 0 means worst value and 100 means best value)
Return: a list of dictionaries (same format as result from DWT)
Return format: {   'level': [
                                {'HL': np.ndarray, 'LH': np.ndarray, 'HH': np.ndarray} #level 1
                                {'HL': np.ndarray, 'LH': np.ndarray, 'HH': np.ndarray} #level 2
                                ... # More levels
                            ],
                    'LL': np.ndarrray # There should be only one LL band in the entire image no matter how many levels there are
               }
```
**Decoding:**
```
Import quantization.py and call dequantize_all(result: List[Dict], quality: float)
Parameters: result: the list of dictionaries (one dictionary for each in Y, Cb, Cr) after we reverse partitioned blocks back to arrays
            quality: the quality value the user inputs (0 - 100, 0 means worst value and 100 means best value)
Return: a list of dictionaries (same format as result from DWT)
```

### Partition
**Encoding:**
```
Import partition.py and call partition_all(quantized_result: List[Dict], block_size: int = 64)
Parameters: quantized_result: the output list of dictionaries (one dictionary for each in Y, Cb, Cr) from quantization
            block_size: the size of the blocks we want to divide each bands (HH,HL,...) in each component into (user can define their specific size value based on their preference or natire of the images, but deault is 64x64)
Return: a list of dictionaries
Return format: {
                   'component': int,             (Y:0, Cb:1, Cr:2)
                   'level': int,                 (DWT level)
                   'band': str,                  (LL, HL, LH, or HH)
                   'position': (i, j),           (The left top coordinate of the block)
                   'shape': (h, w),              (Size of the block)
                   'data': np.ndarray            (Actual data)
               }
```
**Decoding:**
```
Import partition.py and call reverse_partition(blocks: List[Dict])
Parameters: quantized_result: the output list of dictionaries (one dictionary for each in Y, Cb, Cr) from entropy decoding
Return: a list of dictionaries (the same format as the output from quantization)
```

### Entropy Coding:
**Encoding:**
```
Import entropy_parallel.py and call entropy_encode_all(blocks: List[Dict], output_path: str)
Parameters: blocks: the return list of dictionaries from partitioning
            output_path: 'xxx.bin' the output path to the binary file
Return: No return value. Entropy coding part write the encoded data (in tuple format) into a binary file
```

**Decoding:**
```
Import entropy_parallel.py and call entropy_decode_all(input_path: str)
Parameters: input_path: Binary file path
Return: List of dictionaries ready to be fed into decoding in partitioning
```