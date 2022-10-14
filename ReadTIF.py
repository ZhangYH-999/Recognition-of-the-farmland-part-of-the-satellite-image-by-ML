import cv2
import gdal
import imageio
import numpy as np


def readTIF(fileName):
    data = gdal.Open(fileName)
    if data is None:
        print(fileName + "cannot open!")
        return None
    im_width = data.RasterXSize
    im_height = data.RasterYSize
    im_data = data.ReadAsArray(0, 0, im_width, im_height)
    im_bands = data.RasterCount
    if im_bands == 1:
        return im_data
    im_redBand = im_data[0, :, :]
    im_greenBand = im_data[1, :, :]
    im_blueBand = im_data[2, :, :]
    image = np.array([im_blueBand, im_greenBand, im_redBand], dtype=np.uint8)
    image = np.transpose(image, (1, 2, 0))
    return image


def TIF2PNG(filename, save_path):
    image = readTIF(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageio.imwrite(save_path, image)
