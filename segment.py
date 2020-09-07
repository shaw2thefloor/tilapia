from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
import skimage.segmentation as seg
import skimage.color as color
import skimage.draw as draw
import math


def snakes():
    plt.close("all")
    p = Path('C:/Users/fshaw/Documents/PycharmProjects/tilapia/data/raw/padded_images/689nilo.jpg')
    image = io.imread(p)
    image_gray = color.rgb2gray(image)
    points = circle_points(40, [250, 250], 160)
    snake = seg.active_contour(image_gray, points, alpha=0.009, beta=0.0005)
    fig, ax = image_show(image)
    ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);
    plt.show()


def random_walker():
    plt.close("all")
    p = Path('C:/Users/fshaw/Documents/PycharmProjects/tilapia/data/raw/padded_images/2015_00T2A6_A.JPG')
    image = io.imread(p)
    image_gray = color.rgb2gray(image)
    image_labels = np.zeros(image_gray.shape, dtype=np.uint8)
    points = circle_points(resolution=40, center=[250, 250], radius=160)[:-1]

    # indices = draw.circle_perimeter(130, 250, 100)
    indices = draw.ellipse_perimeter(250, 250, r_radius=30, c_radius=200)
    image_labels[indices] = 1
    image_labels[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2
    image_show(image_labels);

    image_segmented = seg.random_walker(image_gray, image_labels, mode='bf', beta=200, tol=1.e-3)  # Check our results
    fig, ax = image_show(image_gray)
    ax.imshow(image_segmented == 1, alpha=0.3);

    plt.show()


def circle_points(resolution, center, radius):
    # Generate points which define a circle on an image.Centre refers to the centre of the circle
    radians = np.linspace(0, 2 * np.pi, resolution)
    c = center[1] + 1.5 * radius * np.cos(radians)  # polar co-ordinates
    r = center[0] + radius * np.sin(radians)

    return np.array([c, r]).T  # Exclude last point because a closed path should not have duplicate points


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


if __name__ == "__main__":
    random_walker()
