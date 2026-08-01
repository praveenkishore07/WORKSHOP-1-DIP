## WORKSHOP-1 

### Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## Program & output : 
**NAME** : A PRAVEEN KISHORE

**REG. NO.** : 212225220074

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
```
# Load the Face Image
faceImage = cv2.imread('myimage.png')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```
<img width="631" height="582" alt="image" src="https://github.com/user-attachments/assets/34127329-7462-4d1d-a339-61c6bf9ceddb" />

```
faceImage.shape
```
<img width="187" height="45" alt="image" src="https://github.com/user-attachments/assets/a0bada83-b5ca-4760-a19a-a422f7d18d29" />

```
#resized_faceImage.shape
faceImage.shape
```
<img width="182" height="43" alt="image" src="https://github.com/user-attachments/assets/1f684bd4-fcd2-4c97-bbd8-3de4b974710e" />

```
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```
<img width="757" height="376" alt="image" src="https://github.com/user-attachments/assets/ffc1a043-df96-493b-98f5-3f6d05761a52" />

```
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))
```
<img width="357" height="47" alt="image" src="https://github.com/user-attachments/assets/f90ddb71-8f27-4c1e-bbd7-64f6aa5a348c" />

```
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
```
```
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```
<img width="1488" height="236" alt="image" src="https://github.com/user-attachments/assets/4820a8d8-3a68-4a41-847b-98491a00b936" />

```


# 1. Slightly widen the glasses to fit your face width (width=280, height=90)
glassBGR = cv2.resize(glassBGR, (280, 90))

# Make a copy
faceWithGlassesNaive = faceImage.copy()

# 2. Perfect target coordinates for the eye level
faceWithGlassesNaive[380:470, 380:660] = glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
```
<img width="602" height="578" alt="image" src="https://github.com/user-attachments/assets/a346c8ed-1359-4aeb-81a2-e1ccc086eff3" />


```
# --- 1. RESIZE ARRAYS TO FIT THE EYE REGION (Width: 280, Height: 90) ---
glassBGR = cv2.resize(glassBGR, (280, 90))
glassMask1 = cv2.resize(glassMask1, (280, 90))

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask / 255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# --- 2. UPDATE THE COORDINATES HERE TO MATCH THE EYE REGION ---
eyeROI = faceWithGlassesArithmetic[380:470, 380:660]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# --- 3. PLACE THE FINAL EYE ROI BACK INTO THE ORIGINAL IMAGE ---
faceWithGlassesArithmetic[380:470, 380:660] = eyeRoiFinal

# Display the intermediate results
plt.figure(figsize=[20, 20])
plt.subplot(131); plt.imshow(maskedEye[..., ::-1]); plt.title("Masked Eye Region")
plt.subplot(132); plt.imshow(maskedGlass[..., ::-1]); plt.title("Masked Sunglass Region")
plt.subplot(133); plt.imshow(faceWithGlassesArithmetic[..., ::-1]); plt.title("Final Augmented Image")
plt.show()
```
<img width="1490" height="476" alt="image" src="https://github.com/user-attachments/assets/010bd6f5-f75a-45d8-8f71-a1ce4d9cbae1" />

```
# Replace the eye ROI with the output from the previous section
# Updated coordinates to perfectly match your eye region location
faceWithGlassesArithmetic[380:470, 380:660] = eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```
<img width="1512" height="710" alt="image" src="https://github.com/user-attachments/assets/d169089e-ee03-4fbb-b183-e938dd3d92d0" />


## RESULT : 
    Thus , the workshop has been completed sucessfully
