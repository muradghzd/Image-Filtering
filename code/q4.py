import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from math import sqrt

def time_convolution(filter_sizes, images):
    """
    Calculates the time passed doing 2D convolution 
    for each pair of given filter sizes and images 
    """
    times = []    
    for f_size in filter_sizes:
        kernel = np.ones(f_size)
        for image in images:
            start = time.time()
            result = cv2.filter2D(image, -1, kernel)
            end = time.time()
            times.append(end-start)
    times = np.array(times).reshape(len(filter_sizes), -1) 
    return times

def get_images(image):
    """
    Computes factors to make image size between 0.25 and 8 MPix
    resizes using new values
    """
    x, y = image.shape[:2]
    mpix = x*y 
    sizes_mpix = np.linspace(0.25,8, 7)
    factors = [sqrt(i*(10**6)/mpix) for i in sizes_mpix]

    result_images = []
    for f in factors:
        new_image = cv2.resize(image,dsize=None,fx=f, fy = f,)
        result_images.append(new_image)

    return result_images

def plot(filter_sizes, image_sizes, times):
    """
    Plots 3D contour of filter_sizes, image_sizes, and corresponding time values
    """
    Y, X = np.meshgrid(filter_sizes, image_sizes)
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, times, 50, cmap='binary')
    ax.set_xlabel('image size')
    ax.set_ylabel('filter size')
    ax.set_zlabel('time')
    ax.set_title('3D contour')
    plt.savefig('3dplot.jpg')

if __name__=='__main__':
    image = cv2.imread('../questions/RISDance.jpg')
    filter_sizes = [2*i+1 for i in range(1, 8)]
    images = get_images(image)
    times = time_convolution(filter_sizes, images)
    plot(filter_sizes, np.linspace(0.25,8,7), times)