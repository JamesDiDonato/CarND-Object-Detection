# **Vehicle Detection and Tracking** 

### Completed for Udacity Self Driving Car Engineer - 2018/04

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./ReportImages/1.png
[image2]: ./ReportImages/2.png
[image3]: ./ReportImages/3.png
[image4]: ./ReportImages/4.png
[image5]: ./ReportImages/5.png
[image6]: ./ReportImages/6.png
[image7]: ./ReportImages/7.png
[image8]: ./ReportImages/8.png
[image9]: ./ReportImages/9.png
[image10]: ./ReportImages/10.png
[image11]: ./ReportImages/11.png
[image12]: ./ReportImages/12.png
[image13]: ./ReportImages/13.png
[image14]: ./ReportImages/14.png
[image15]: ./ReportImages/15.png
[image16]: ./ReportImages/16.png
[image17]: ./ReportImages/17.png
[image18]: ./ReportImages/18.png
[image19]: ./ReportImages/19.png
[image20]: ./ReportImages/20.png
[image21]: ./ReportImages/21.png
[image22]: ./ReportImages/22.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started the project by loading the vehicle and non-vehicles images from my hard-drive in the variables `Vehicles` and `NotVehicles`:
```
9137 Vehicles Images Imported. 47.86%
9954 Non Vehicle Images Imported. 52.14%
```
I randomly displayed some images to explore the datasets:

![alt text][image1]

Next I began extracting features for a classifier, starting with HOG.

HOG features are extracted using the function `get_hog_features(def get_hog_features(img, orient, pix_per_cell, cell_per_block,vis, feature_vec)` found in the 4th jupyter notebook cell. The HOG function from `skimage.feature` is called with the parameters that are given as inputs. I started by visually exploring HOG features to get an idea of what they looked like for both vehicle and non - vehicle images. Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on random images:

![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters (and other training methods).

For HOG feature extraction, I tried various combinations of parameters and chose the combination that provided the largest test set accuracy with reasonable training time. I settled upon the below parameters after peforming the following test runs:

![alt text][image3]

Final HOG Parameters:
```
YCrCb Color Space
ALL HOG Channel
12  orientation bins
16 pixels per cell
2  cells per block
```

In addition to HOG, I also implemented color histogram  and spatial binning feature extraction techniques.  These can be found in jupyter notebook cells 5 and 6 respectively. 

For color histogram, I plotted the histogram for each color channel alongside the original images to observe any differentiation. Looking at the YUV color space, you can see differences in especially the Y color channel between vehicle and non-vehicle images:

![alt text][image4]

To determine optimal parameters for color features, I ran a number of experiments varying the  color spaces and number of histogram bins while observing test set accuracy. Optimal parameters were chosen (highlighted in yellow and indicated below) based on the results.

![alt text][image5]
```
YUV Color Space
Histogram Bins = 32
```

For spatial binning, I followed a similar process and came up with the following parameters:

![alt text][image6]
```
YUV Color Space
Image Size = 12x12
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Once I had optimized each of the feature extraction techniques individually, I combined them in the `extract_features()` function found in jupyter notebook cell 7. This function simply calls the indivdual functions from each extraction technique and concatenate's the feature vectors using `np.concatenate()`. The result is a single feature vector with length 1824 that contains HOG, spatial binning, and color histogram features combined. These features perform at a reasonable classification accuracy of around 99%.

![alt text][image7]

As for the classifier code itself, it is contained within jupyter notebook cell #8. First, the feature vectors for the roughly 19000 images are computed (which takes long enough). With my desktop, average generation times were roughly 70 seconds. These feature vectors are stacked into a single vector, labelled, and then randomly split into  training / test sets based on a split of 80/20 respectively. To keep things simple, a Linear SVM classifier is trained and the test accuracy is calculated. This classifier peformed satisfactory to my expectations for this project. Also, I should mention that a per-column scaler is generated from the training data and then applied to both the training and test set's to normalize the feature vectors with zero mean and variance. This scaler is used later on the notebook in the sliding window pipeline.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The `find_cars()` function, modified from the course notes, is at the core of my sliding window pipeline. This function, found in jupyter notebook cell #11, takes a single image as input and generates windows for a horizontal slice of the image given by the inputs `ystart` and `ystop`. It then generates feature vectors for each window and runs the classifier to determine the presence of a vehicle within each. The function returns a list of all windows where vehicles were found. As you will see below, this function was called multiple times to generate the search windows for numerous scaling factors, and is then used in the pipeline real time on images.


Sliding window search is completed by generating a number of search windows, or 'boxes', with different scale factors for each y starting point. 4 'Layers' of boxes are used, each with a different scale factor. Within each layer, there are two rows of boxes that are strategically placed at specific locations on the image to maximize the probability of a vehicle being present at that row with the chosen scale.  Here are the images of each layer and the windows that have been generated. As you can see, the smaller boxes are placed in the middle of the image because vehicles are further down the road in this location, whereas larger boxes are placed in the foreground. I chose to use a large overlap in the y direction to increase the probability of multiple true positive detections for each vehicle that is detected. This will be useful when filtering out false positives.

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

In total, there are 367 windows to search.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Initially, I discovered that there are too many false positives present on the test images to reliably filter out. To optimize the classifier performance, I applied hard image mining techniques by saving an image for each window in all test images and then manually sorted them based on whether a vehicle was present ( the code for this be seen commented out at the end of the find_cars() function ).  This significantly improved the false positive rate of my pipeline. In addition, instead of using a binary ouput from the classifier, the `svc.decision_function(test_features)` function is used to increase the threshold for positive classifications above 0, helping out with the number of false positives.

Here are some sample test images to demonstrate how my pipeline is working:


![alt text][image12]

![alt text][image13]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented a two tiered filtering strategy using 'heatmapping' to eliminate false positives  : Once for each frame and also between consequtive frames using a Tracker Class.

Starting with one frame at a time, I record the positions of positive detections from `find_cars()`. The resulting positives detection boxes are heatmapped:

![alt text][image14]

Then, the heatmap is thresholded to discard false positive detections:

![alt text][image15]

Finally, the thresholded heatmaps are grouped using `scipy.ndimage.measurements.label()` and the bounding boxes are overlaid on each detection:

![alt text][image16]

Once the bounding boxes have been generated for the current frame, they are stored in a `Vehicle_Tracker()` class. The purpose of the class is to keep track of bounding boxes that have been found in the last `n` frames to filter out false positives. The same 'heatmapping' strategy is applied to these bounding boxes, resulting in a time based filter that spans across video frames. Incresing the number of frames in the Vehicle Tracker would decrease the false positive detections but consequently allow the boxes to lag behind moving vehicles.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of this project was implementing the sliding window approach. I struggled with getting the 'scaling' factor to match up with appropriately sized boxes to search for vehicles.

Because I implemented hard positive and negative mining to improve false positives, this pipeline would likely fail if applied in the real world with many different vehicle colors, shapes, and sizes. The scope of my mining efforts would have to be expanded to accomodate real world scenarios.  Also, the pipeline would likely fail to recognize vehicles that are moving significantly faster due to the lag effect that the filtering introduces. To make it more robust, I would calculate how fast each vehicle is moving and then apply a scaling factor to the number of frames tracked in history.

In the real world, this pipeline would have to take into account the total time spent on each frame to ensure that it is operating in near real-time. There is a important trade-off between number of windows to search and how long it takes to scan the image. In the future, I would spend more time on the window generation to reduce the number of boxes to seach and improve performance. 

Finally, if I were to do this project again, I would consider using a neural network to classifiy vehicles instead of the sliding windows. I found the sliding window technique to be finicky and difficult to tune. 


