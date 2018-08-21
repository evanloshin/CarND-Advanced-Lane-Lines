## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Perform lane detection and tracking for self-driving cars using Python and OpenCV.**

The software pipeline interprets raw front-facing camera footage and calculates situational awareness indicators for controlling an autonomous vehicle. The pipeline returns a video with lane indication markings as well as continuous measurements for radius of lane curvature and distance from center.

#### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
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

### Camera Calibration

Recognizing this is a relatively larger project, I begin by defining a class `undistorter()` in [classes.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/classes.py) to store calibration related parameters for recalling later. Here I pass in the number of chessboard corners to create an array `ObjPts` of desired (x,y,z) coordinates where z=0. I also create an array `ImgPts` for holding 2D coordinates of the corners as they appear in the image.

Iterating through an array of provided chessboard images, `cv2.findChessboardCorners()` identifies the coordinates for storing in `ImgPts`. Note, converting each image to grayscale is performed first as a prerequisite. Here is just how accurately that function performs corner recognition.

![img1]

Then, passing `ObjPts` and `ImgPts` to the function `cv2.calibrateCamera()` produces the two essential elements for undistorting images, the **camera matrix** and **distortion coefficients**. I store these as object attributes for recalling later in my method `undistort()`. This method uses the function `cv2.undistort` to remove the artificial *curviness* today's cameras induce.

Here's examples of removing distortion on chessboards as well as driving imagery. The difference is a bit harder to discern in the lower images. Nonetheless, this step is important for describing lanes with precise mathematical geometry used downstream to control the vehicle.

![img2]

### Binary Image

In this pivotal step, I create and tune a function to single out lane lines using color spaces and gradients. Many options for combining different color spaces and gradients exist. An intelligent approach would use machine learning to identify a good set of parameters that generalize well to multiple road conditions. However, I went for a more expedient method for now due to time constraints (full-time job, wedding planning, life, etc.).

`binarize()` in [functions.py](https://github.com/evanloshin/CarND-Advanced-Lane-Lines/blob/master/functions.py) first converts the image from RGB color space to HLS and HSV. The two channels that seemed to accentuate lane lines the most are saturation in HLS and value in HSV, so I isolate those.

![img3]

The pipeline also takes the horizontal and vertical color gradients in a grayscale image to lend additional parameters for lane extraction. I take the absolute value of gradient pixels and rescale them to span the range 0-255 as shown.

![img4]

Finally, I transform each of the above into binary images, where pixel values between my chosen thresholds are coded as *1* and all others are *0*. Then, I combine the binary images using logical operators. Keeping all the activated gradient pixels (sobelx and sobely) while taking overlapping saturation and value pixels seems to yield the best result.

`(sobelx_binary == 1) OR (sobely_binary == 1) OR ((saturation_binary == 1) AND (value_binary == 1))`

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

I used `plot_mask()` and trial and error to pick source points and offsets that produce parallel lane lines like those below.

![img7]

### Lane Detection

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

### Situational Awareness Measurements

I did this in lines # through # in my code in `my_other_file.py`

### Overlay Output

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Conclusion

Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
