## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Perform lane detection and tracking for self-driving cars using Python and OpenCV.**

My software pipeline interprets raw front-facing camera footage and calculates lane awareness indicators for controlling an autonomous vehicle. The pipeline returns a video with lane perception markings as well as continuous measurements for radius of lane curvature and vehicle distance from center.

#### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

#### Files

* [output_project_video.mp4](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_project_video.mp4) in the [output_videos](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/tree/master/output_videos) folder is my project submission
* [main.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/main.py) takes in i/o filenames as arguments in terminal and returns a video i.e. `python main.py 'test_videos/project_video.mp4' 'output_videos/output_project_video.mp4'`
* [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) defines four classes to support detection and tracking
* [functions.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/functions.py) contains several functions called in both classes.py and main.py
* [camera_cal](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/tree/master/camera_cal) contains chessboard images for removing camera distortion
* [project_video.mp4](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4), [challenge_video.mp4](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/challenge_video.mp4), and [harder_challenge_video.mp4](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/harder_challenge_video.mp4) contain the raw driving footage

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[img1]: ./output_images/detect_corners.png
[img2]: ./output_images/undistort.png
[img3]: ./output_images/colorspaces.png
[img4]: ./output_images/sobels.png
[img5]: ./output_images/combined_binary.png
[img6]: ./output_images/plot_mask.png
[img7]: ./output_images/warped.png
[img8]: ./output_images/sliding_windows.png
[img9]: ./output_images/r_eq.png
[img10]: ./output_images/final_image.png

### Camera Calibration

Recognizing this is a relatively large project, I define a class `undistorter()` in [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) to perform initial calibration and process images throughout the project. In the constructor, I pass the number of chessboard corners to create an array `ObjPts` of desired (x,y,z) coordinates where z=0. I also initialize an array `ImgPts` to hold 2D coordinates of the corners as they appear in the image.

Iterating through an array of provided chessboard images, `cv2.findChessboardCorners()` identifies coordinates, which I append to `ImgPts`. Note, converting each image to grayscale is performed first as a prerequisite. Here is just how accurately that function performs corner recognition.

![img1]

Then, passing `ObjPts` and `ImgPts` to the function `cv2.calibrateCamera()` produces the two essential elements for undistorting images, the **camera matrix** and **distortion coefficients**. I store these as object attributes for recalling later in my method `undistort()`. This method uses the function `cv2.undistort` to remove the artificial *curviness* today's cameras induce.

Here's examples of removing distortion on chessboards as well as driving imagery. The difference is a bit harder to discern in the lower images. Nonetheless, this step is important for describing lanes with precise mathematical geometry used downstream to control the vehicle.

![img2]

### Binary Image

In this pivotal step, I create and tune a function to single out lane lines using color spaces and gradients. Many options for combining different color spaces and gradients exist. An intelligent approach would use machine learning to identify a good set of parameters that generalize well to multiple road conditions. However, I went for a more expedient method for now due to time constraints (full-time job, wedding planning, life, etc.).

`binarize()` in [functions.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/functions.py) first converts the image from RGB color space to HLS and HSV. The two channels that seem to accentuate lane lines the most are saturation in HLS and value in HSV, so I isolate those.

![img3]

The pipeline also takes the horizontal and vertical color gradients in a grayscale image to lend additional parameters for lane extraction. I take the absolute value of the gradient pixels and rescale them to span the range 0-255 as shown.

![img4]

I transform each of the above into binary images, where pixel values between my chosen thresholds are coded as *1* and all others are *0*. Here are the thresholds I chose.

|          | Saturation | Value | X-Gradient | Y-Gradient |
|:--------:|:----------:|:-----:|:----------:|:----------:|
|  **Low** |     60     |   40  |     20     |     35     |
| **High** |     255    |  255  |     255    |     255    |

Then, I combine the binary images using logical operators. Keeping all the activated gradient pixels (sobelx and sobely) while taking overlapping saturation and value pixels seems to yield the best result.

```python
(sobelx_binary == 1) OR (sobely_binary == 1) OR ((saturation_binary == 1) AND (value_binary == 1))
```

The two images below visualizes the effectiveness of the combined binary image (right) in isolating lane pixels, where white represents activated pixels. The next section takes care of eliminating most activated non-pavement scenery.

![img5]

### perspective transform

The code for my perspective transform is found in a class named `transformer()`, which begins at line 47 of [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) The constructor takes as inputs an image, as well as source points `pts`, horizontal offset `x_offset` and vertical offset `y_offset`. I wrote the utility method `plot_mask()` to make hard coding the source points easier by visualize them as follows.

![img6]

The method `create_transformation()` automatically calculates 4 corresponding destination points `dst` for the top-down transformation using the image size and hard coded offsets defined in the constructor. 

```python
dst = np.float32([[self.x, self.y],
                  [img_size[1] - self.x, self.y],
                  [img_size[1] - self.x, img_size[0] - self.y],
                  [self.x, img_size[0] - self.y]])
```

I pass the source and destination points to `cv2.getPerspectiveTransform()`, which returns a transformation matrix used to warp pipeline images in the method `warp()`. I also use the same two parameters to create an inverse transformation matrix for the method `unwarp()`.

I used `plot_mask()` and trial and error to pick source points and offsets that would produce parallel lines. My result is shown below.

![img7]

### Lane Detection

A sliding window technique is used to identify lane pixels in the rectified binary image. The code for this can be found in lines 132-199 of [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py). The method `find_lanes()` takes a histogram of the bottom half of each image to get the frequency of activated pixels with respect to the horizontal axis. The x-value of the maximum frequencies on the left and right hand sides gives me the lanes' beginning center points. From here, I grab all pixels within a 150x80 rectangular area. If more than 40 pixels exist in the area, the subsequent rectangle's horizontal position is adjusted to the mean x-value of those pixels. The hyperparameters mentioned are `window_width`, `n_windows`, and `minpix`, which are defined near the top of [main.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/main.py).

![img8]

Two unique aspects of my lane detection pipeline made significant contributions to overall performance. First, `find_lanes()` calls the `verify_window` method of the `lane()` class each sliding window. This method confirms the next window's horizontal position fell within a user-defined margin of the last n best fit lane lines, otherwise it overrides the proposed position to protect against misleading groups of activated pixels. The second safeguard is keeping a history of the last n detected lane pixels. Lines 282-283 of [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) store these pixel coordinates in class attributes `self.historical_x` and `self.historical_y`.

Finally, the method `fit_curve()` on line 278 of [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) calculates the best-fit second degree polynomial coefficients of the detected pixels over the last n video frames.

### Situational Awareness Measurements

The **radius of curvature** is measured in the method `get_radius()` of my `lane_finder()` class on line 204. It starts with simply taking an average of the left and right lane polynomial coefficients to get those of a hypothetical center lane. While I didn't work out the proof, I did test that the averaged coefficients exactly equaled those taken using `cv2.polyfit()` on a new set of line points `left_x + (right_x - left_x) / 2`.

Then, I plug the coefficients into the radius of curvature equation below ([derivation here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)). Finally, pixels are converted to meters based on the ratio of the pixel distance to the typical U.S. highway lane distance of 3.7 meters - see `generate_unit_conversion()` on line 120 of `classes.py`.

 ![img9]

The **distance from center** is measured in `get_center_distance()` on line 244 in [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) with a bit less rigor than above. I take the difference between the center of the lanes measured at max(y) and the center of the perspective transformed image, then convert pixels to meters. This is less rigorous because the horizontal center of the transformed image is not exactly equal to that of the original. This is one area of opportunity for improving my project.

### Final Output

The method `project_lines()` on line 215 of [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) unwarps the highlighted lane lines and filled area back onto the original image. This uses the inverse transformation matrix M_inv calculated earlier and the same `cv2.perspectivetransform` function. I do some iterating on the detected pixels in lines 229-234 to make them more visually apparent in the result.

Finally, I produce the following result after some text and picture-in-picture overlay in the `video_pipeline()` function.

![img10]

---

### Conclusion

Here's a [link to my video result](./output_videos/output_project_video.mp4).

Overall, the project successfully detects the lanes throughout the entire video with only slight aberrations in the shaded portion. It is the result of several iterations, ideas, and missteps. I learned a lot of lessons, from starting over in PyCharm after frustration writing object-oriented code in a jupyter notebook to brainstorming a more robust smoothing technique than averaging the last *n* polynomial coefficients.

Personally, the **most valuable takeaway** is understanding first-hand the challenges with generalizing algorithms to perform in unseen and highly variable driving environments. I am humbled by this pipeline's marginal detection in the two challenge videos. I look forward to learning techniques others applied in this project and revisiting it with new knowledge, as time permits. I see several ways for neural networks to enhance or replace these techniques given well-labeled training data. I could even see using genetic algorithms just to tune color thresholds. There are so many dials to turn with this project, and I learned plenty experimenting with just a handful of them.