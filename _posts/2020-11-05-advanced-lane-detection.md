---
layout: post
title: Advanced Lane Detection
date: 2020-11-05
author: Awais Khan
summary: Identifying lanes with curvature and different lighting conditions.
feature-img: "assets/img/pexels/autonomous-driving.jpeg"
tags: [OpenCV, Computer Vision, Python, Self Driving Cars]
---

Taken from the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013), this project looks at defining a pipeline to identify lane boundaries from video footage taken by a forward-facing camera mounted on top of a vehicle - it also takes into account road curvature and lighting conditions.

**Objectives:**

- Detect and track lane boundaries
- Identify the location of the vehicle relative to the center of the road
- Determine the radius of curvature of the road

**Pipeline:**
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Apply a perspective transform to rectify binary image (birds-eye view).
4. Apply color thresholds to create a binary image which isolates the pixels representing lane boundaries.
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image and output visual display.
8. Output visual display of the numerical estimation of lane curvature and vehicle position.
9. Implement video processing pipeline.

**Dependencies:**
- [OpenCV](http://opencv.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [MoviePy](http://zulko.github.io/moviepy/)

**Helper Function:**

Here is a helper function used to display a set of images:
```python
def show_images(imgs, titles=['Original', 'Processed'], figsize=(10,5), to_RGB=[1], returnfig=False):
    """
    imgs:
     - array of images to display
    titles:
     - img titles
    figsize:
     - figure size
    to_RGB:
     - whether image should be converted to RGB
     - example: [1,0,0] means first image should be converted, rest should not
     - by default, all images will be converted to RGB
    return_fig:
     - Boolean to return fig or not
    """

    if to_RGB == [1]:
        to_RGB *= len(imgs)

    if len(imgs) > len(titles):
        titles_to_add = len(imgs) - len(titles)
        titles += [''] * titles_to_add
    elif len(imgs) < len(titles):
        titles = titles[:len(imgs)]

    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)

    for img_index in range(len(imgs)):
        if to_RGB[img_index]:
            axes[img_index].imshow(cv2.cvtColor(imgs[img_index], cv2.COLOR_BGR2RGB))
        else:
            axes[img_index].imshow(imgs[img_index])

        axes[img_index].set_title(titles[img_index])

    if returnfig:
        return fig, axes
```

---
### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
---

Images produced by a camera will almost always suffer from distortion due to refraction of light through the lens of the camera - see [here](https://en.wikipedia.org/wiki/Distortion_(optics)) for an in-depth explanation.

If the video feed is distorted, the pipeline (and by extension the vehicle) would inaccurately identify the lane boundaries which will mean motor insurance goes up. As such, the camera needs to be calibrated so that any distortion can be removed.

To calibrate the camera, multiple images of a calibration target at different angles needs to be captured. A common calibration target is a chessboard - any calibration target can be used but a chessboard tends to produce slightly more accurate results.

```python
## Load calibration images
calibration_images = glob.glob('camera_cal/calibration*.jpg')
```

With the calibration images collected, the camera is calibrated by collecting the following set of points:

- 3D real world points i.e. (*X*,*Y*,*Z*) (by keeping the board on an XY plane, the points are simplified to (*X*,*Y*) as *Z=0*)
- 2D image ponts i.e. (*X*,*Y*) which is just the image co-orindates for which the 3D real world points correspond to

See [camera calibration](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) for more.

Using the OpenCV functions `findChessboardCorners` and `drawChessboardCorners`, the corners are easily identified.

```python
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

for idx, fname in enumerate([calibration_images[11]]):
    original = cv2.imread(fname)
    cv2.imwrite('test_images_output/checkerboard_original.png', original)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)

        # Display images
        show_images([original, img], ['Original', 'With Corners'])
```

{% include aligner.html images="advanced_lane_detector/Chessboard.svg,advanced_lane_detector/With-Corners.svg" %}

With the checkerboard corners located, the OpenCV function `calibrateCamera` can be used to compute the camera calibration matrix and distortion coefficients.

```python
## Determine calibration parameters
def calibration(img):
    img = cv2.imread(img)
    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return ret, mtx, dist, rvecs, tvecs


## Set mtx & dist parameters
_, mtx, dist, _, _ = calibration(calibration_images[8])
```

---
### 2. Apply a distortion correction to raw images
---

Now that `mtx` and `dist` have been computed, the OpenCV function `undistort` can be used to remove any distortion from out test images. To make things simpler, a new function `undistort` is defined - this uses `mtx` and `dist` parameters with the OpenCV function `undistort` by default.

```python
# Remove distortion from images using mtx & dist parameters determined above
def undistort(img, mtx=mtx, dist=dist, read=False):  
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```

With the `undistort` function defined, the test images can now be undistorted.

```python
## Load test images
test_images = [cv2.imread(test_image) for test_image in glob.glob('test_images/*.jpg')]

## Example of undistorted image compared with original
show_images([(test_images[2]), undistort(test_images[2])], figsize=(15,7.5))
```

{% include aligner.html images="advanced_lane_detector/Original.svg,advanced_lane_detector/Undistorted.svg" %}

At first glance it is difficult to see any change but a lot of the distortion happens at the edges - have a look at the red car when compared to the left edge of the image.

---
### 3. Apply a perspective transform (birds-eye view)
---

For lane detection to work accurately, the pipeline needs to be able to predict the how the lanes bend as the vehicle drives i.e. the curvature of the lane. From the camera's point of view, this is relatively difficult to do.

A simpler way to do this would be to perform a [perspective transform](https://en.wikipedia.org/wiki/3D_projection#Perspective_projection) - this will return a bird's eye point of view of the lane which makes it a lot easier to fit a polynomial to the lane boundaries and hence predict it's curvature.

The transformation can be accomplished using the OpenCV function `getPerspectiveTransform` which takes in `src` (source) and `dst` (destination) points as inputs and outputs a matrix that maps `src` to `dst` and an inverse matrix to map `dst` to `src` as the transformation will need to be reversed towards the end of the pipeline.

`src` defines a set of points on the image before it has transformed and `dst` defines a set of points on the image after it has been transformed. Both are manually set - `src` should identify a portion of the lane and `dst` should identify lane boundaries that are parallel. Selecting the right points will simply be a matter of tweaking the values and trial and error.

The transformation only needs to be performed on one test image and it is easiest to do this with a test image containing straight lines as it is easier to tell if the lane boundaries are parallel in the transformation. The points selected are:

| Source        | Destination   |
|:-------------:|:-------------:|
| 203.3, 720      | 200, 720    |
| 577.5, 460    | 200, 0      |
| 710, 460      | 1000, 0      |
| 1126.7, 720     | 1000, 720    |

```python
# Perform perspective transform
def perspective_transform(img, display=False):
    ## Undistort image
    undist = undistort(img)

    ## Grayscale the image
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    ## Store image shape attributes
    img_size = (gray.shape[1], gray.shape[0])

    ## Not necessary but the offset parameter defines a smaller region on the transformed image
    offset = 200

    ## Source and destination points
    ## Selecting points manually to fit the lane boundaries of an image
    ## In this case, test_image[2] is used which is a straight line image
    src = np.float32([
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] / 2) - 62.5, img_size[1] / 2 + 100],
        [(img_size[0] / 2 + 70), img_size[1] / 2 + 100],
        [(img_size[0] * 5 / 6) + 60, img_size[1]]
    ])

    dst = np.float32([
        [offset, img_size[1]],
        [offset, 0],
        [(img_size[0] - offset), 0],
        [(img_size[0] - offset), img_size[1]]
    ])

    ## Determine M & Minv
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    if display:
        copy = undist.copy()
        warped = cv2.warpPerspective(undist, M, img_size)

        color = [255, 0, 0]
        w = 5

        for idx in range(len(src)-1):
            cv2.line(copy, tuple(src[idx]), tuple(src[idx+1]), color, w)
            cv2.line(warped, tuple(dst[idx]), tuple(dst[idx+1]), color, w)

        show_images([copy, warped], ['Original', 'Warped'], figsize=(15,7.5))
    else:
        return M, Minv
```

{% include aligner.html images="advanced_lane_detector/Original-With-Outline.svg,advanced_lane_detector/Warped-With-Outline.svg" %}

---
### 4. Apply color thresholds to create a binary image which isolates the pixels representing lane boundaries
---

Applying color thresholds is done to highlight the lane boundaries and ignore everything else. There are many different ways and combinations of applying thresholds and it is a matter of experimenting and tweaking values until only the lane boundaries appear in the transformation.

After some trial and error with the LUV, Lab, and HLS color spaces, the following combination of thresholds and values were chosen:

| Color Space   | Channel   | Min. Threshold | Max. Threshold |
|:-------------:|:---------:|:--------------:|:--------------:|
| LUV           |     L     |       150      |      200       |
| Lab           |     B     |       220      |      255       |

- The L channel from LUV was effective at identifying white lane boundaries but did not pick up on yellow lane boundaries.
- The B channel from Lab was effective at identifying yellow lane boundaries but did not pick up on white lane boundaries.

It is much more ideal that a channel identifies one boundary type really well than identify multiple boundary types to an okay standard - it is straightforward to combine the channels and so multiple boundary types can be identified accurately. For example, the S channel from the HLS color space was very good at identifying both white and yellow lane boundaries but had difficulty identifying all the pixels - it would be wise to instead use the combination above.

```python
def abs_thresh(warped, display=False):    
    l_channel = cv2.cvtColor(warped, cv2.COLOR_BGR2LUV)[:,:,0]
    b_channel = cv2.cvtColor(warped, cv2.COLOR_BGR2Lab)[:,:,2]   

    # Threshold color channel    
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    ## Combine L and B thresholds
    combined_binary = np.zeros_like(b_channel)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    if display:
        show_images(
            [warped, l_binary, b_binary, combined_binary],
            ['Undistorted', 'Warped', 'L-Channel', 'B-Channel', 'Combined L&B'],
            figsize=(20,10),
            to_RGB=[1,0,0,0]
        )
    else:
        return combined_binary
```

{% include aligner.html images="advanced_lane_detector/Warped.svg,advanced_lane_detector/L-Channel.svg" %}
{% include aligner.html images="advanced_lane_detector/B-Channel.svg,advanced_lane_detector/Combined-L&B.svg" %}

---
### 5. Detect lane pixels and fit to find the lane boundary
---

Now that the lane boundaries have been identified, the next step is to detect the lane pixels present at the lane boundaries and fit a polynomial to it.

An interesting approach can be taken by generating a histogram of the lane boundary pixels of the warped, binary (with color threshold) image. Given that the image should only be showing two separate lane boundaries and no other pixels, the histogram should show two distinctive peaks representing each lane boundary - these two peaks define the starting position for which the pipeline begins searching for pixels belonging to the lane boundaries.

Beginning from the bottom, the next step is to iteratively search regions of the image up to the top of the image - this is known as a [sliding window search](https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/). In each search, the location of pixels found are added to a list - this is done for each lane boundary.

To be a bit clever, the stored pixel location values can be averaged and used as a starting position instead of starting over with a histogram. It is likely the next set of lane pixels will be close to the previous set and so the search can be narrowed down without having to perform a search of the full width of the image. Of course, this is only sensible if there are enough detected pixels in the previous search - if a sufficient amount have not been found, the histogram technique should used.

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def find_lane_pixels(binary_warped, nwindows=9, margin=110, minpix=50, display=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right boundaries
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, display=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    if display:
        fig, axes = show_images(
            [binary_warped, out_img],
            ['Binary Warped', 'Polynomial Fit'],
            figsize=(15,7.5),
            to_RGB=[0,0],
            return_fig=True
        )
        # Plots the left and right polynomials on the lane boundaries
        axes[1].plot(left_fitx, ploty, color='yellow')
        axes[1].plot(right_fitx, ploty, color='yellow')
    else:
        return out_img, ploty, left_fit, right_fit, left_fit_cr, right_fit_cr
```

{% include aligner.html images="advanced_lane_detector/Binary-Warped.svg,advanced_lane_detector/Polynomial-Fit.svg" %}

---
### 6. Determine the curvature of the lane and vehicle position with respect to center
---

With the lane boundary pixel locations determined, the next step is less Computer Vision and more Mathematics. Using `np.polyfit`, a polynomial of order two can be fitted to the lane curvature.

```python
def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

    ### Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty
```

With the polynomials determined, the radius of the curvature can be determined. See [here](https://en.wikipedia.org/wiki/Radius_of_curvature) for more info on the mathematics behind calculating the radius of a curvature.

It's also a good idea to return the position of the vehicle with respect to the center - this is an easy metric to calculate and makes sense to include here as opposed to having it in a standalone function.

```python
def vehicle_metrics(img):
    '''
    Calculates
    - the curvature of polynomial functions in pixels
    - position of vehicle with respect to center
    '''    
    ploty = fit_polynomial(abs_thresh(adjust_perspective(img)))[1]
    left_fit_cr, right_fit_cr = fit_polynomial(abs_thresh(adjust_perspective(img)))[2:]

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate vehicle center
    x_max = img.shape[1]*xm_per_pix
    y_max = img.shape[0]*ym_per_pix
    vehicle_center = x_max / 2
    line_left = left_fit_cr[0]*y_max**2 + left_fit_cr[1]*y_max + left_fit_cr[2]
    line_right = right_fit_cr[0]*y_max**2 + right_fit_cr[1]*y_max + right_fit_cr[2]
    line_middle = (line_right + line_left)/2
    diff_from_vehicle = line_middle - vehicle_center

    return left_curverad, right_curverad, diff_from_vehicle
```

---
### 7. Warp the detected lane boundaries back onto the original image and output visual display
---

Back to Computer Vision - it's time to take all the hard work done so far and map it back to the original image. The key here will be using `Minv` - the inverse matrix that was determined earlier using `cv2.getPerspectiveTransform(dst, src)`.

```python
def fill_lane(img, Minv=Minv, display=False):
    left_fit, right_fit, left_fit_m, right_fit_m = fit_polynomial(abs_thresh(adjust_perspective(img)))[2:]

    y_range = img.shape[0]
    plot_y = np.linspace(0, y_range - 1, y_range)
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Calculate points.
    fit_x = []
    for fit in [left_fit, right_fit]:
        fit_x.append(fit[0]*plot_y**2 + fit[1]*plot_y + fit[2])

    # Recast the x and y points into usable format for cv2.fillPoly()
    points = [
        np.array([np.transpose(np.vstack([fit_x[0], plot_y]))]),
        np.array([np.flipud(np.transpose(np.vstack([fit_x[1], plot_y])))])
    ]
    points = np.hstack((points[0], points[1]))

    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int_([points]), isClosed=False, color=(255,155,0), thickness = 30)
    cv2.fillPoly(color_warp, np.int_([points]), (255,200, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    inversed_warp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    original_with_lane_overlay = cv2.addWeighted(img, 1, inversed_warp, 0.5, 0)

    if display:
        show_images([img, original_with_lane_overlay], ['Original', 'Lane Detected'], figsize=(15,7.5))
    else:
        return original_with_lane_overlay
```

{% include aligner.html images="advanced_lane_detector/Original-10.svg,advanced_lane_detector/Lane-Detected-10.svg" %}
{% include aligner.html images="advanced_lane_detector/Original-11.svg,advanced_lane_detector/Lane-Detected-11.svg" %}

---
### 8. Output visual display of numerical estimation of lane curvature and vehicle position
---

This is the final stage of the image pipeline. The only thing left to do is use the previous functions and parameters to calculate the lane curvature and vehicle position, and then overlay this on the image with the lane highlighted.

```python
def pipeline(img, fontScale=1, display=False):   
    left_fit, right_fit, left_fit_m, right_fit_m = fit_polynomial(abs_thresh(adjust_perspective(img)))[2:]
    output = fill_lane(img)

    # Calculate curvature
    curvature = measure_curvature_pixels(img)

    # Calculate vehicle center
    x_max = img.shape[1]*xm_per_pix
    y_max = img.shape[0]*ym_per_pix
    vehicle_center = x_max / 2
    line_left = left_fit_m[0]*y_max**2 + left_fit_m[1]*y_max + left_fit_m[2]
    line_right = right_fit_m[0]*y_max**2 + right_fit_m[1]*y_max + right_fit_m[2]
    line_middle = (line_right + line_left)/2
    diff_from_vehicle = line_middle - vehicle_center
    if diff_from_vehicle > 0:
        message = '{:.2f} m right'.format(diff_from_vehicle)
    else:
        message = '{:.2f} m left'.format(-diff_from_vehicle)

    # Draw info
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)
    cv2.putText(output, 'Curvature Radius: {:.0f} m'.format(np.mean(curvature)), (50, 80), font, fontScale, fontColor, 2)
    cv2.putText(output, 'Vehicle is {} of center'.format(message), (50, 130), font, fontScale, fontColor, 2)

    if display:
        fig, axes = show_images([img, output], ['Original', 'Lane Detection & Values'], figsize=(15,7.5), return_fig=True)
        fig.savefig('metrics_overlay.svg', transparent=True, bbox_inches='tight', pad_inches=0)
    else:    
        return output  
```

{% include aligner.html images="advanced_lane_detector/Original-20.svg,advanced_lane_detector/Lane-Detected-20.svg" %}
{% include aligner.html images="advanced_lane_detector/Original-21.svg,advanced_lane_detector/Lane-Detected-21.svg" %}

---
### 9. Implement video processing pipeline
---

There are two sub-steps in the final part of this project.

1. Create a `Class` to store parameters from previous frames
2. Combine all the steps above to create the video processing pipeline

Similar to storing pixel location values in the sliding window search (step 5), a class can be defined to store the parameters of previous frames which can then be averaged and used as a reference point for the next frame of the video.

```python
class Line:
    def __init__(self):
        # Was the line found in the previous frame?
        self.found = False

        # Store x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=10)
        self.top = deque(maxlen=10)

        # Store previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0

    def found_search(self, x, y):
        '''
        This function is applied when the lane lines have been detected in the previous frame.
        It uses a sliding window to search for lane pixels in close proximity (+/- 25 pixels in the x direction)
        around the previous detected polynomial.
        '''

        xvals, yvals = [], []

        if self.found == True:
            i = 720
            j = 630

            while j >= 0:
                yval = np.mean([i,j])
                xval = (np.mean(self.fit0))*yval**2 + (np.mean(self.fit1))*yval + (np.mean(self.fit2))

                x_idx = np.where((((xval - 25) < x)&(x < (xval + 25))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]

                if np.sum(x_window) != 0:
                    np.append(xvals, x_window)
                    np.append(yvals, y_window)

                i -= 90
                j -= 90

        if np.sum(xvals) == 0:
            self.found = False # If no lane pixels were detected then perform blind search

        return xvals, yvals, self.found

    def blind_search(self, x, y, image):
        '''
        This function is applied in the first few frames and/or if the lane was not successfully detected
        in the previous frame. It uses a slinding window approach to detect peaks in a histogram of the
        binary thresholded image. Pixels in close proimity to the detected peaks are considered to belong
        to the lane lines.
        '''

        xvals, yvals = [], []

        if self.found == False:
            i = 720
            j = 630

            while j >= 0:
                histogram = np.sum(image[j:i,:], axis=0)

                if self == Right:
                    peak = np.argmax(histogram[640:]) + 640
                else:
                    peak = np.argmax(histogram[:640])

                x_idx = np.where((((peak - 25) < x)&(x < (peak + 25))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]

                if np.sum(x_window) != 0:
                    xvals.extend(x_window)
                    yvals.extend(y_window)

                i -= 90
                j -= 90

        if np.sum(xvals) > 0:
            self.found = True
        else:
            yvals = self.Y
            xvals = self.X

        return xvals, yvals, self.found

    def radius_of_curvature(self, xvals, yvals):
        ym_per_pix = 30./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*np.max(yvals) + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])

        return curverad

    def sort_vals(self, xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]

        return sorted_xvals, sorted_yvals

    def get_intercepts(self, polynomial):
        bottom = polynomial[0]*720**2 + polynomial[1]*720 + polynomial[2]
        top = polynomial[0]*0**2 + polynomial[1]*0 + polynomial[2]

        return bottom, top
```

The next and final step is to combine all the different steps (it is more efficient for the code to be collated together into one function as the methods above were simply for demonstration purposes) and to add in the newly defined class `Line`.

```python
def process_vid(image):
    img_size = (image.shape[1], image.shape[0])

    # Calibrate camera and undistort image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Perform perspective transform
    offset = 200

    src = np.float32([
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] / 2) - 62.5, img_size[1] / 2 + 100],
        [(img_size[0] / 2 + 70), img_size[1] / 2 + 100],
        [(img_size[0] * 5 / 6) + 60, img_size[1]]
    ])

    dst = np.float32([
        [offset, img_size[1]],
        [offset, 0],
        [(img_size[0] - offset), 0],
        [(img_size[0] - offset), img_size[1]]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)

    # Generate binary thresholded images
    b_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab)[:,:,2]
    l_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2LUV)[:,:,0]  

    # Set the upper and lower thresholds for the b channel
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    # Set the upper and lower thresholds for the l channel
    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(b_channel)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    # Identify all non zero pixels in the image
    x, y = np.nonzero(np.transpose(combined_binary))

    if Left.found == True: # Search for left lane pixels around previous polynomial
        leftx, lefty, Left.found = Left.found_search(x, y)

    if Right.found == True: # Search for right lane pixels around previous polynomial
        rightx, righty, Right.found = Right.found_search(x, y)


    if Right.found == False: # Perform blind search for right lane lines
        rightx, righty, Right.found = Right.blind_search(x, y, combined_binary)

    if Left.found == False:# Perform blind search for left lane lines
        leftx, lefty, Left.found = Left.blind_search(x, y, combined_binary)

    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)

    # Calculate left polynomial fit based on detected pixels
    left_fit = np.polyfit(lefty, leftx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    leftx_int, left_top = Left.get_intercepts(left_fit)

    # Average intercepts across n frames
    Left.x_int.append(leftx_int)
    Left.top.append(left_top)
    leftx_int = np.mean(Left.x_int)
    left_top = np.mean(Left.top)
    Left.lastx_int = leftx_int
    Left.last_top = left_top

    # Add averaged intercepts to current x and y vals
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_top)
    lefty = np.append(lefty, 0)

    # Sort detected pixels based on the yvals
    leftx, lefty = Left.sort_vals(leftx, lefty)

    Left.X = leftx
    Left.Y = lefty

    # Recalculate polynomial with intercepts and average across n frames
    left_fit = np.polyfit(lefty, leftx, 2)
    Left.fit0.append(left_fit[0])
    Left.fit1.append(left_fit[1])
    Left.fit2.append(left_fit[2])
    left_fit = [np.mean(Left.fit0),
                np.mean(Left.fit1),
                np.mean(Left.fit2)]

    # Fit polynomial to detected pixels
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    Left.fitx = left_fitx

    # Calculate right polynomial fit based on detected pixels
    right_fit = np.polyfit(righty, rightx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    rightx_int, right_top = Right.get_intercepts(right_fit)

    # Average intercepts across 5 frames
    Right.x_int.append(rightx_int)
    rightx_int = np.mean(Right.x_int)
    Right.top.append(right_top)
    right_top = np.mean(Right.top)
    Right.lastx_int = rightx_int
    Right.last_top = right_top
    rightx = np.append(rightx, rightx_int)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_top)
    righty = np.append(righty, 0)

    # Sort right lane pixels
    rightx, righty = Right.sort_vals(rightx, righty)
    Right.X = rightx
    Right.Y = righty

    # Recalculate polynomial with intercepts and average across n frames
    right_fit = np.polyfit(righty, rightx, 2)
    Right.fit0.append(right_fit[0])
    Right.fit1.append(right_fit[1])
    Right.fit2.append(right_fit[2])
    right_fit = [np.mean(Right.fit0), np.mean(Right.fit1), np.mean(Right.fit2)]

    # Fit polynomial to detected pixels
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    Right.fitx = right_fitx

    # Compute radius of curvature for each lane in meters
    left_curverad = Left.radius_of_curvature(leftx, lefty)
    right_curverad = Right.radius_of_curvature(rightx, righty)

    # Only print the radius of curvature every 3 frames for improved readability
    if Left.count % 3 == 0:
        Left.radius = left_curverad
        Right.radius = right_curverad

    # Calculate the vehicle position relative to the center of the lane
    position = (rightx_int+leftx_int)/2
    distance_from_center = abs((640 - position)*3.7/700)

    Minv = cv2.getPerspectiveTransform(dst, src)

    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    points = [
        np.array([np.transpose(np.vstack([Left.fitx, Left.Y]))]),
        np.array([np.flipud(np.transpose(np.vstack([Right.fitx, Right.Y])))])
    ]
    points = np.hstack((points[0], points[1]))

    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int_([points]), isClosed=False, color=(0,155,255), thickness = 30)
    cv2.fillPoly(color_warp, np.int_([points]), (0,200,255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    inversed_warp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    original_with_lane_overlay = cv2.addWeighted(image, 1, inversed_warp, 0.5, 0)

    # Print distance from center on video
    if position > 640:
        message = 'Vehicle is {:.2f} m left of center'
    else:
        message = 'Vehicle is {:.2f} m right of center'

    cv2.putText(original_with_lane_overlay, message.format(distance_from_center), (50,80),fontFace = 16, fontScale = 1, color=(255,255,255), thickness = 2)

    # Print radius of curvature on video
    cv2.putText(original_with_lane_overlay, 'Radius of Curvature {} m'.format(int((Left.radius+Right.radius)/2)), (50,130),
             fontFace = 16, fontScale = 1, color=(255,255,255), thickness = 2)

    Left.count += 1

    return original_with_lane_overlay
```
And that's it, the pipeline is complete. Below is the code to process a video clip:

```python
# Define lane boundary for left and right side
Left = Line()
Right = Line()

# Read video file and provide a video output filename
video_output = 'project_video_result.mp4'
clip1 = VideoFileClip("project_video.mp4")

# Process the video through the pipeline
white_clip = clip1.fl_image(process_vid)
white_clip.write_videofile(video_output, audio=False)
```
And here is the final result:
<div style="width:100%;height:0px;position:relative;padding-bottom:56.250%;"><iframe src="https://streamable.com/e/b9klta" frameborder="0" width="100%" height="100%" allowfullscreen style="width:100%;height:100%;position:absolute;left:0px;top:0px;overflow:hidden;"></iframe></div>

---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
