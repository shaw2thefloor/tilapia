from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters import try_all_threshold

p = Path('C:/Users/fshaw/Documents/PycharmProjects/tilapia/data/raw/original_images/001sparrmanii.jpg')
img = rgb2gray(io.imread(p))
fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()

