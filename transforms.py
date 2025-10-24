import numpy as np
from scipy.signal import convolve
from math import ceil

'''
Takes image as input and returns arrays containing Y, Cb, and Cr color channel information.
'''
def rgb_to_ycc(image):
    w = image.shape[0]
    h = image.shape[1]

    y, cb, cr = np.zeros((w, h)), np.zeros((w, h)), np.zeros((w, h))
    conv_arr = np.array([[0.256, 0.504, 0.098], [-0.148, -0.292, 0.441], [0.441, -0.369, -0.071]])

    for i in range(w):
        for j in range(h):
            dot = np.dot(conv_arr, image[i,j])
            y[i,j], cb[i,j], cr[i,j] = dot[0]+16, dot[1]+128, dot[2]+128

    return y, cb, cr

'''
Helper function which downsamples and transposes array during DWT processing.
'''
def downsample(array):
    new_len = ceil(len(array)/2) # First half of inputs rounded up
    array = array[1:new_len] # Cuts off second half of array
    return np.transpose(array) # Transposes and returns

'''
Performs 2-dimensional, multi-level DWT on an array of color data from an image.
Return dictionary containing four sets of coefficients of approximation and detail information.
'''
def DWT2D(data, n):
    # Forward filterbank
    lowpass = [0.02674875741080976, -0.01686411844287495, -0.07822326652898785, 0.2668641184428723, 0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495, 0.02674875741080976]
    highpass = [0.09127176311424948, -0.05754352622849957, -0.5912717631142470, 1.115087052456994, -0.5912717631142470, -0.05754352622849957, 0.09127176311424948]

    coeff_dict = {}
    
    for i in range(n):

        # Convolve array with filterbank
        # First pass - convolution on rows
        low = []
        high = []

        for row in data:
            low.append(convolve(row, lowpass)) # Low-pass filter convolution
            high.append(convolve(row, highpass)) # High-pass filter convolution

        # Downsample and transpose (columns are now rows)
        low = downsample(low)
        high = downsample(high)

        # Second pass - convolution on columns
        approx = []
        horiz = []
        vert = []
        diag = []

        for row in low:
            approx.append(convolve(row, lowpass)) # Approximation coefficients (LL)
            horiz.append(convolve(row, highpass)) # Horizontal residuals (LH)
        for row in high:
            vert.append(convolve(row, lowpass)) # Vertical residuals (HL)
            diag.append(convolve(row, highpass)) # Diagonal residuals (HH)

        # Downsample and re-transpose
        cA = downsample(approx)
        cH = downsample(horiz)
        cV = downsample(vert)
        cD = downsample(diag)

        coeff_dict[i] = {"LH": cH, "HL": cV, "HH": cD}
        data = cA

    coeff_dict["data"] = data

    return coeff_dict

'''
Performs inverse 2-dimensional DWT on dequantized set of coefficients
Returns array of color channel information.
'''
def InverveDWT2D(coeffs):
    # Inverse filterbank
    gL = [-0.09127176311424948, -0.05754352622849957, 0.5912717631142470, 1.115087052456994, 0.5912717631142470, -0.05754352622849957, -0.09127176311424948]
    gH = [0.02674875741080976, 0.01686411844287495, -0.07822326652898785, -0.2668641184428723, 0.6029490182363579, -0.2668641184428723, -0.07822326652898785, 0.01686411844287495, 0.02674875741080976]
    
