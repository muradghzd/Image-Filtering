import cv2
import numpy as np

def my_filter2D(image, kernel, pad_reflected=False):
    # This function computes convolution given an image and kernel.
    # While "correlation" and "convolution" are both called filtering, here is a difference;
    # 2-D correlation is related to 2-D convolution by a 180 degree rotation of the filter matrix.
    #
    # Your function should meet the requirements laid out on the project webpage.
    #
    # Boundary handling can be tricky as the filter can't be centered on pixels at the image boundary without parts
    # of the filter being out of bounds. If we look at BorderTypes enumeration defined in cv2, we see that there are
    # several options to deal with boundaries such as cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.:
    # https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    #
    # Your my_filter2D() computes convolution with the following behaviors:
    # - to pad the input image with zeros,
    # - and return a filtered image which matches the input image resolution.
    # - A better approach is to mirror or reflect the image content in the padding (borderType=cv2.BORDER_REFLECT_101).
    #
    # You may refer cv2.filter2D() as an exemplar behavior except that it computes "correlation" instead.
    # https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)   # for extra credit
    # Your final implementation should not contain cv2.filter2D().
    # Keep in mind that my_filter2D() is supposed to compute "convolution", not "correlation".
    #
    # Feel free to add your own parameters to this function to get extra credits written in the webpage:
    # - pad with reflected image content
    # - FFT-based convolution

    ################
    a, b = kernel.shape

    if a%2==0 or b%2==0:
        raise Exception(f"Filter is of even-dimension")
 
    # Horizontal and Vertical pad
    h_pad, v_pad = (a-1)//2, (b-1)//2
    x, y = image.shape[:2]

    mode = 'reflect' if pad_reflected else 'constant'

    if len(image.shape) == 3: # RGB images
        new_image = np.pad(image, ((h_pad,h_pad), (v_pad,v_pad), (0,0)), mode)
        flipped_kernel = np.rot90(kernel, 2)
        # Broadcasted kernel for multiplication
        broadcasted_kernel = np.expand_dims(flipped_kernel, axis=-1)
        result_image = np.zeros_like(image)
        
        for i in range(x):
            for j in range(y):
                # Take the respective part of image for convolution
                subset_image = new_image[i:i+a,j:j+b]
                result_image[i,j] = np.sum(subset_image*broadcasted_kernel, axis=(0,1)) 
    else: # Grayscale images
        new_image = np.pad(image, ((h_pad,h_pad), (v_pad,v_pad)), mode)
        flipped_kernel = np.rot90(kernel, 2)
        result_image = np.zeros_like(image)
        
        for i in range(x):
            for j in range(y):
                subset_image = new_image[i:i+a,j:j+b]
                result_image[i,j] = np.sum(subset_image*flipped_kernel)

    return result_image

if __name__=='__main__':
    image = cv2.imread("/home/murad/Desktop/KAIST/Fall22/CS484/HW2/data/cat.bmp")
    print(f"Image size: {image.shape}")