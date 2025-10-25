import skimage as ski
import pywt

'''
Returns Y, Cb and Cr coefficients with Haar wavelet transform applied
Coefficients stored in form (cA, (cH, cV, cD))
'''
def getDWTCoeffs(file):
    # Read image and convert to YCbCr colorspace
    img = ski.io.imread(file)
    ycc = ski.color.rgb2ycbcr(img)

    # Create 2D arrays for Y, Cb, Cr coeffs
    y_arr, cb_arr, cr_arr = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]

    # Run 2D Haar wavelet transform on each array
    y_coeffs = pywt.dwt2(y_arr, 'haar')
    cb_coeffs = pywt.dwt2(cb_arr, 'haar')
    cr_coeffs = pywt.dwt2(cr_arr, 'haar')

    return y_coeffs, cb_coeffs, cr_coeffs

def decodeFromCoeffs(coeffs):
    return pywt.idwt2(coeffs, 'haar')