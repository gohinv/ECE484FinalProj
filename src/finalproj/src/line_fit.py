#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform
import PIL


def line_fit(binary_warped):
    """
    Find and fit a single lane region
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create output image for visualization
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    
    # Find the peak of the histogram - this will be the center of the lane
    lane_base = np.argmax(histogram)
    
    # Set up sliding window parameters
    nwindows = 9
    window_height = int(binary_warped.shape[0]/nwindows)
    margin = 40  # Increased margin to capture wider lane region
    minpix = 50
    
    # Find nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current position to be updated for each window
    lane_current = lane_base
    
    # Create empty list for lane pixel indices
    lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_x_low = max(lane_current - margin, 0)
        win_x_high = min(lane_current + margin, binary_warped.shape[1])
        
        # Draw the window
        cv2.rectangle(out_img, (win_x_low, win_y_low), 
                     (win_x_high, win_y_high), (0, 255, 0), 2)
        
        # Draw center point
        center_x = (win_x_low + win_x_high) // 2
        cv2.circle(out_img, (center_x, (win_y_low + win_y_high) // 2), 
                  5, (0, 0, 255), -1)
        
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                    (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        lane_inds.append(good_inds)
        
        # Recenter next window if enough pixels found
        if len(good_inds) > minpix:
            lane_current = int(np.mean(nonzerox[good_inds]))
    
    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)
    
    # Extract lane pixel positions
    lanex = nonzerox[lane_inds]
    laney = nonzeroy[lane_inds]
    
    # Calculate error from center
    image_center = binary_warped.shape[1] // 2
    lane_center = lane_current
    err = lane_center - image_center
    
    # Fit a second order polynomial
    try:
        lane_fit = np.polyfit(laney, lanex, 2)
    except TypeError:
        print("Unable to detect lane")
        return None
    
    return {
        'lane_fit': lane_fit,
        'nonzerox': nonzerox,
        'nonzeroy': nonzeroy,
        'out_img': out_img,
        'lane_inds': lane_inds,
        'err': err
    }


def tune_fit(binary_warped, prev_fit):
    """
    Given a previously fit line, quickly try to find the lane based on previous fit
    """
    # Find all nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set search margin around previous polynomial
    margin = 40  # Increased margin for wider lane region
    
    # Find pixels within margin of previous fit
    lane_inds = ((nonzerox > (prev_fit[0]*(nonzeroy**2) + 
                             prev_fit[1]*nonzeroy + 
                             prev_fit[2] - margin)) & 
                 (nonzerox < (prev_fit[0]*(nonzeroy**2) + 
                             prev_fit[1]*nonzeroy + 
                             prev_fit[2] + margin)))
    
    # Extract lane pixel positions
    lanex = nonzerox[lane_inds]
    laney = nonzeroy[lane_inds]
    
    # Check if we found enough points
    min_inds = 10
    if laney.shape[0] < min_inds:
        return None
    
    # Fit a second order polynomial
    lane_fit = np.polyfit(laney, lanex, 2)
    
    # Generate points for error calculation
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
    
    # Calculate error from center
    image_center = binary_warped.shape[1] // 2
    lane_center = lane_fitx[len(lane_fitx)//2]  # Use middle point of fit
    err = lane_center - image_center
    
    return {
        'lane_fit': lane_fit,
        'nonzerox': nonzerox,
        'nonzeroy': nonzeroy,
        'lane_inds': lane_inds,
        'err': err
    }



def viz1(binary_warped, ret, save_file=None):
    """
    Visualize sliding window location and predicted lane center
    """
    # Grab variables from ret dictionary
    lane_fit = ret['lane_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    out_img = ret['out_img']
    lane_inds = ret['lane_inds']

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]

    # Color detected lane pixels
    out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [0, 255, 0]
    
    plt.imshow(out_img)
    plt.plot(lane_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    plt.gcf().clear()

def bird_fit(binary_warped, ret, save_file=None):
    """
    Visualize the predicted lane center with margin
    """
    # Grab variables
    lane_fit = ret['lane_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    lane_inds = ret['lane_inds']

    # Create output images
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    window_img = np.zeros_like(out_img)
    
    # Color detected lane pixels
    out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [0, 255, 0]

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]

    # Generate polygon for search window
    margin = 40
    lane_window1 = np.array([np.transpose(np.vstack([lane_fitx-margin, ploty]))])
    lane_window2 = np.array([np.flipud(np.transpose(np.vstack([lane_fitx+margin, ploty])))])
    lane_pts = np.hstack((lane_window1, lane_window2))

    # Draw the lane
    cv2.fillPoly(window_img, np.int_([lane_pts]), (0,255, 0))
    
    # Draw center points and error line
    image_center = 320
    lane_center = int(lane_fitx[len(lane_fitx)//2])
    cv2.circle(out_img, (image_center, 240), 5, (0, 255, 0), thickness=-1)
    cv2.circle(out_img, (lane_center, 240), 5, (0, 0, 255), thickness=-1)
    cv2.line(out_img, (image_center, 240), (lane_center, 240), (255, 0, 0), thickness=2)
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.plot(lane_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    return result

def final_viz(undist, lane_fit, m_inv):
    """
    Final lane prediction visualized and overlayed on original image
    """
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]

    color_warp = np.zeros((720, 1280, 3), dtype='uint8')

    # Create polygon points for the lane region
    margin = 40
    pts_left = np.array([np.transpose(np.vstack([lane_fitx-margin, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lane_fitx+margin, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw lane region
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp back to original image space
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    
    # Combine images
    undist = np.array(undist, dtype=np.uint8)
    newwarp = np.array(newwarp, dtype=np.uint8)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result
