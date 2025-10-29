import skimage as ski
import pywt
import numpy as np

def createCDFWavelet():
    dec_lo = [0.02674875741080976, -0.01686411844287495, -0.07822326652898785, 0.2668641184428723, 0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495, 0.02674875741080976]
    dec_hi = [0.09127176311424948, -0.05754352622849957, -0.5912717631142470, 1.115087052456994, -0.5912717631142470, -0.05754352622849957, 0.09127176311424948]
    rec_lo = [-0.09127176311424948, -0.05754352622849957, 0.5912717631142470, 1.115087052456994, 0.5912717631142470, -0.05754352622849957, -0.09127176311424948]
    rec_hi = [0.02674875741080976, 0.01686411844287495, -0.07822326652898785, -0.2668641184428723, 0.6029490182363579, -0.2668641184428723, -0.07822326652898785, 0.01686411844287495, 0.02674875741080976]

    filterbank = [dec_lo, dec_hi, rec_lo, rec_hi]
    wavelet = pywt.Wavelet("cdf", filter_bank=filterbank)

    return wavelet

def getYCbCrArrays(file):
    # Read image and convert to YCbCr colorspace
    img = ski.io.imread(file)
    ycc = ski.color.rgb2ycbcr(img)

    # Create 2D arrays for Y, Cb, Cr coeffs
    y, cb, cr = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]

    # Return as list of colorspace coefficients
    return [y, cb, cr]

def DWTAll(list):
    return [GetDWTCoeffs(array) for array in list]

def GetDWTCoeffs(array, n = 3):
    # cdf = createCDFWavelet()
    coeff_dict = {"levels": []}

    # Run 2D CDF wavelet transform
    coeffs = pywt.wavedec2(array, "haar", level=n)
    coeff_dict["LL"] = coeffs[0]

    for i in range(n):
        tuple = coeffs[i+1]
        coeff_dict["levels"].append({"LH": tuple[0], "HL": tuple[1], "HH": tuple[2]})

    return coeff_dict

def DecodeAll(list):
    return [DecodeCoeffs(coeffs) for coeffs in list]

def DecodeCoeffs(dict):
    n = len(dict["levels"])

    coeff_list = [dict["LL"]]
    for i in range(n):
        level = dict["levels"][i]
        coeff_list.append((level["LH"], level["HL"], level["HH"]))

    coeffs = pywt.waverec2(coeff_list, "haar")

    return coeffs

def reconstructRGB(arrays):
    image = np.stack(arrays, axis=2)
    rgb_img = ski.color.ycbcr2rgb(image)
    return (rgb_img*255).astype(np.uint8)