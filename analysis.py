from jpeg2000 import full_encoding, full_decoding
from time import time
import os
from skimage import metrics, io
import csv

def main():
    inputs = open('inputs.txt', 'r') # Get list of images
    output = []

    try:
        os.mkdir("outputs")
    except FileExistsError:
        print("Directory already exists.")

    for line in inputs:
        file = line.strip()
        directory = f"outputs/{file}"

        try:
            os.mkdir(directory) # Make folder for outputs
        except FileExistsError:
            print("Directory already exists.")

        fname = f"images/{file}.bmp" # Original filename
        orig = io.imread(fname) # Read in original file
        io.imsave(f'{directory}/original.png', orig) # Save it

        for i in range(8):
            outfile = f"{directory}/output{i+1}.bin"
            reconfile = f"{directory}/recon{i+1}.png"
            dict = {"file": file, "decomp": i+1}

            curr = time()
            full_encoding(fname, outfile, n=i+1)
            dict.update({"time": time() - curr, "size": os.path.getsize(outfile)})

            full_decoding(outfile, reconfile)
            recon = io.imread(reconfile)

            dict.update({"psnr": metrics.peak_signal_noise_ratio(orig, recon)})
            output.append(dict)
            print(f"Completed level {i+1} on image {file}.")

    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['file', 'decomp', 'time', 'size', 'psnr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()