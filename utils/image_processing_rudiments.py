import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
im = cv2.imread('utils/boat.png', cv2.IMREAD_UNCHANGED)
im_height, im_width = im.shape[:2]

# Convert the image to grayscale if it's not already
if len(im.shape) == 3 and im.shape[2] == 3:
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image to the range [0, 1]
im_normalized = im_gray / 255.0

############################################################################
##### S&V - gray level windowing
# Create a histogram
plt.figure()
plt.hist(im_gray.flatten(), bins=256)
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show(block=False)

# Gray level windowing
plt.figure()
plt.imshow(im_gray, cmap='gray', vmin=40, vmax=110)
plt.title("gray Level Windowing (vmin=40, vmax=110)")
plt.colorbar()
plt.show()

############################################################################
##### Gray scale transformation

# Soft thresholding
f0 = 0.5
w = 0.2
img_soft_thresholding = 1 / (1 + np.exp(-(im_normalized - f0) / w))
img_soft_thresholding = cv2.normalize(img_soft_thresholding, None, 0, 1, cv2.NORM_MINMAX)
plt.figure()
plt.imshow(img_soft_thresholding, cmap='gray')
plt.title("Soft Thresholding")
plt.colorbar()
plt.show()

# Logarithmic transformation
alpha = 0.02
img_logarithmic = np.log(alpha + im_normalized)
img_logarithmic = cv2.normalize(img_logarithmic, None, 0, 1, cv2.NORM_MINMAX)
plt.figure()
plt.imshow(img_logarithmic, cmap='gray')
plt.title("Logarithmic Transformation")
plt.colorbar()
plt.show()

############################################################################
##### Colour space operations

# Extract color channels
imr = im[:, :, 2]  # Red channel
img = im[:, :, 1]  # Green channel
imb = im[:, :, 0]  # Blue channel

# Apply weights to the channels
# Adjust these weights to maximize contrast between the boat, sea, and sky
weight_r = 0.2  # Weight for the red channel
weight_g = 0.3  # Weight for the green channel
weight_b = 1.0  # Weight for the blue channel
weight_g = 0.5  # Grey level

# Combine the weighted channels
imf = weight_r * imr + weight_g * img + weight_b * imb + weight_g

# Normalize the result to the range [0, 255]
imf = cv2.normalize(imf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display the result
plt.figure()
plt.imshow(imf, cmap='gray')
plt.title("Colour Space Operations with Enhanced Contrast")
plt.colorbar()
plt.show()

############################################################################
##### Colour space operations

# downsample with factor 8 by array slicing:
imdown = im[::8,::8,:]
plt.figure()
plt.imshow(imdown)
plt.show()
# up sample with factor 8
nr,nc = np.shape(imdown)[:2]
# generate indices of upsampled image:
rup = np.arange(0,8*nr)
cup = np.arange(0,8*nc)
# generate associated indices of input image:
rd = np.int32(np.floor(rup/8)) # MUST BE int
cd = np.int32(np.floor(cup/8)) # MUST BE int
rd = np.reshape(rd,[im_width,1]) # reshape to wx1
cd = np.reshape(cd,[1,im_height]) # reshape to 1xh
# upsample:
imup = imdown[rd,cd,:]
plt.figure()
plt.imshow(imup)
plt.show()

############################################################################
##### image rotation, cropping and resizing

# rotate over 30 degrees
nr,nc = np.shape(im)[:2] # image size
c = (nc/2,nr/2) # rotation point
Rmat = cv2.getRotationMatrix2D(center=c, angle=30, scale=1)
imrot = cv2.warpAffine(src=im, M=Rmat, dsize=(nc,nr))
plt.figure()
plt.imshow(imrot)
plt.show()

# image cropping
csize = (51,41) # width and height of crop area
ccenter = (349,193) # (x,y) center of crop area
imc = cv2.getRectSubPix(im, csize, ccenter)
plt.figure()
plt.imshow(imc)
plt.show()

# image resizing
newsize = (250,200) # Ncol,Nrow
imres1 = cv2.resize(im, newsize, 0, 0, interpolation=cv2.INTER_NEAREST)
imres2 = cv2.resize(im, newsize, 0, 0, interpolation=cv2.INTER_LINEAR)
plt.figure()
plt.imshow(imres1)
plt.show()
plt.figure()
plt.imshow(imres2)
plt.show()
