# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[left]: ./writeup/left.jpg "Left"
[center]: ./writeup/center.jpg "Center"
[right]: ./writeup/right.jpg "Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a convolution neural network consists of 4 convolutional layers, 3 fully connected layers and 1 output layer.

* conv1: 24x5x5 with max pooling (code line 45-46)
* conv2: 36x5x5 with max pooling (code line 48-49)
* conv3: 48x5x5 with max pooling (code line 51-52)
* conv4: 60x5x5 with max pooling (code line 54-55)
* fc1: 1024 with dropout (code line 59-60)
* fc2: 200 with dropout (code line 62-63)
* fc3: 20 with dropout (code line 65-66)
* output: 1 (code line 68)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 41). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers for the fully connected layers, and max pooling layers for the convolutional layers, in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 81). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving both clockwise and counterclockwise.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with known architectures, and tune the hyperparameters.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it has less parameters and resistent to overfitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had both a high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was underfitting. 

So I proceeded to use a more complex network similar to Nvidia's network for self-driving. It achieved low error on training data, but high error on validation data, so I added dropout layers to combat overfitting.

Then the network achieved similar loss on validation set as on training data. However, it didn't drive well on the test track.

Finally I incorprated image cropping to remove portions that are not relevant to driving (the upper and lower portions)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

* conv1: 24x5x5 with max pooling (code line 45-46)
* conv2: 36x5x5 with max pooling (code line 48-49)
* conv3: 48x5x5 with max pooling (code line 51-52)
* conv4: 60x5x5 with max pooling (code line 54-55)
* fc1: 1024 with dropout (code line 59-60)
* fc2: 200 with dropout (code line 62-63)
* fc3: 20 with dropout (code line 65-66)
* output: 1 (code line 68)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two clockwise laps and two counterclockwise laps on track one using center lane driving. Here is an example image of center lane driving:

![center driving][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to center if it gets too close to the left or right. These images show what a recovery looks like from left and right :

![recover from left][left]
![recover from right][right]

After the collection process, I had 52605 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as the loss wouldn't change much after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
