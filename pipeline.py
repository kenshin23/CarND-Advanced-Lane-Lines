import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline


def get_chessboard_corners(pathname, chessboard_size=(9, 6)):
    """For a given path, seek through calibration images and return images detected
    successfully, along with their object points and image points.
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[1]*chessboard_size[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    paths = [] # Path of the successfully processed calibration image

    # Make a list of calibration images
    images = glob.glob(pathname)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            paths.append(fname)

    return paths, objpoints, imgpoints


def calibrate_camera(img, objpoints, imgpoints):
    """Calibrate a camera using the given object and image points.
    Return the camera matrix and distortion coefficients.
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
    return mtx, dist


def undistort_image(img, mtx, dist):
    """Return an undistorted version of the image, given the camera matrix and distortion coefficients."""
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Functions from the lessons:


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def rgb_binary_threshold_r(img, thresh=(200, 255)):
    """Returns a binary threshold image of the R channel in RGB color space."""
    R = img[:, :, 0]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary


def hls_binary_threshold_s(img, thresh=(90, 255)):
    """Returns a binary threshold image of the S channel in HLS color space."""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def combined_threshold(img, region=((None, None), (None, None)), args=None):
    """Find and return a binary image based on the combination of thresholds.
    Uses an optional region of interest polygon.
    """
    comb_img = np.copy(img)

    sx_binary = abs_sobel_thresh(comb_img, thresh=(20, 100))
    s_binary = hls_binary_threshold_s(comb_img, thresh=(170, 255))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1
    return combined_binary


def warper(img, src=None, dst=None, inverse=False):
    """Warps the image according to source and destination points."""
    img_size = (img.shape[1], img.shape[0])
    if src is None:
        src = np.float32(
            [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 70), img_size[1] / 2 + 100]])
    if dst is None:
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])

    # Given src and dst points, calculate the perspective transform matrix
    if not inverse:
        M = cv2.getPerspectiveTransform(src, dst)
    else:  # This is actually the calculation of the inverse matrix, Minv.
        M = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M

def get_lanes_window(img):
    """Retrieve lane lines based on a sliding-window-based search."""
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img)) * 255

    # Get histogram of lower half of the image:
    histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return (leftx, lefty), (rightx, righty), out_img


def get_lanes_predicted(img, left_fit, right_fit):
    """Retrieve lane lines based on existing best guess by get_lanes_window."""
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img)) * 255

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return (leftx, lefty), (rightx, righty), out_img


def fit_polynomial(leftpoints, rightpoints):
    """Fit a second order polynomial to each array of lane points."""
    left_fit = np.polyfit(leftpoints[1], leftpoints[0], 2)
    right_fit = np.polyfit(rightpoints[1], rightpoints[0], 2)
    return left_fit, right_fit


def display_lanes_window(src_img, left_points, right_points, left_fit, right_fit, out_img):
    # y_max = img.shape[0]
    # Generate x and y values for plotting
    fity = np.linspace(0, src_img.shape[0] - 1, src_img.shape[0])
    fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    out_img[left_points[1], left_points[0]] = [255, 0, 0]
    out_img[right_points[1], right_points[0]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(fit_leftx, fity, color='yellow')
    plt.plot(fit_rightx, fity, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def display_lanes_predicted(src_img, left_points, right_points, left_fit, right_fit, out_img):
    # Generate x and y values for plotting
    fity = np.linspace(0, src_img.shape[0] - 1, src_img.shape[0])
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

    # Create an image to show the selection window:
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[left_points[1], left_points[0]] = [255, 0, 0]
    out_img[right_points[1], right_points[0]] = [0, 0, 255]

    margin = 100

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx-margin, fity]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx+margin, fity])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx-margin, fity]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx+margin, fity])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(fit_leftx, fity, color='yellow')
    plt.plot(fit_rightx, fity, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def pipeline(img=None, mtx=None, dist=None, left_fit=None, right_fit=None):
    # Using a curved lane image to build the first part of the pipeline:
    if img is None:
        img_path = './test_images/test5.jpg'
        img = mpimg.imread(img_path)

    undistort = undistort_image(img, mtx, dist)
    thresholded_img = combined_threshold(undistort)
    pers_transf_img, _ = warper(thresholded_img)

    # if this is the first prediction or lines are not good enough, search
    # via windows:
    # question: what indicates a bad prediction? Parallelism between lane lines?
    if left_fit is None or right_fit is None:
        left_points, right_points, out_img = get_lanes_window(pers_transf_img)
    # otherwise, search via previously predicted:
    else:
        left_points, right_points, out_img = get_lanes_predicted(pers_transf_img, left_fit, right_fit)

    # Fit a polynomial to both point arrays:
    left_fit, right_fit = fit_polynomial(left_points, right_points)
    display_lanes_window(pers_transf_img.shape[0], left_points, right_points, left_fit, right_fit, out_img)


if __name__ == '__main__':
    cal_img_path = './camera_cal/calibration*.jpg'
    print('Retrieving chessboard corners for images in \'{}\''.format(cal_img_path))
    image_paths, objpoints, imgpoints = get_chessboard_corners(cal_img_path)
    print('Calibration images processed.')

    # Read in an image from the calibration set:
    img = cv2.imread('./camera_cal/calibration2.jpg')

    # Get camera matrix and distortion coefficients with the test image:
    mtx, dist = calibrate_camera(img, objpoints, imgpoints)

    # Now run the pipeline on a test image:
    pipeline(mtx=mtx, dist=dist)
