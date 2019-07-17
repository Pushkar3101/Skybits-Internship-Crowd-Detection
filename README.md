# Skybits-Internship-Crowd-Detection

## How To Train and Test Person Classifier(ultimately used for crowd detection in office spaces) Using TensorFlow (GPU) on Windows 10

### Brief Summary :
This  is a tutorial for how to use TensorFlow's Object Detection API to train a Person detection classifier for detecting small crowd in office spaces and departmental stores on Windows 10. I have used anaconda environment(version 4.6.14 ) with tensorflow version 1.13.1.

#### Steps Required :
1.Tensorflow GPU Installation
2.Set up TensorFlow Directory and Anaconda Virtual Environment
3.Labelling Pictures
4.Generate Training data
5.Creating label map and configuring training
6.Training the data
7.Exporting the Inference Graph
8.Testing
	      *** common errors and their solution

## 1. Tensorflow GPU Installation
### Conventional Approach -
- First find if the GPU is compatible with Tensorflow GPU or not! (From Here)
- Download and Install Cuda Toolkit from here.
- Download cuDNN by signing up on Nvidia Developer Website
- Install cuDNN by extracting the contents of cuDNN into the Toolkit path installed in Step 2. There will be files that you have to replace in CUDA Toolkit Directory.
- Is that it? No, then you need to check your path variables if CUDA_HOME is present or not. If not, please add it manually.
- Then check the path variables if your toolkit paths are available or not.
- Then finally install Anaconda or Miniconda
- Creating an Environment with Python and Pip packages installed.
- Then finally ‘pip install tensorflow-gpu’.
- Test your installation.

 ### Better Approach - (I have used this)
		       Install Anaconda and run the below commands :
               ...
			         conda create --name tf_gpu
               activate tf_gpu
               conda install tensorflow-gpu
               ...

This command will create an environment first named with ‘tf_gpu’ and    will install all the packages required by tensorflow-gpu including the cuda and cuDNN compatible versions.

 Visit TensorFlow's website for further installation details, including how to install it on other operating systems (like Linux). The object detection repository itself also has installation instructions.


** steps 2-7 are used for training purposes(making a model from scratch).
** Otherwise, download the tensorflow1 folder in C and  directly goto step 8 for testing purpose .
## 2. Set up TensorFlow Directory and Anaconda Virtual Environment
The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model.
This portion of the tutorial goes over the full set up required. It is fairly meticulous, but follow the instructions closely, because improper setup can cause unwieldy errors down the road.
    ### 2a. Download TensorFlow Object Detection API repository from GitHub
Create a folder directly in C: and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.
Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”.
Note: The TensorFlow models repository's code (which contains the object detection API) is continuously updated by the developers. Sometimes they make changes that break functionality with old versions of TensorFlow. It is always best to use the latest version of TensorFlow and download the latest models repository. 
### 2b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo. Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy. This tutorial will use the Faster-RCNN-Inception-V2 model. Download the model here. Open the downloaded faster_rcnn_inception_v2_coco_2018_01_28.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the C:\tensorflow1\models\research\object_detection folder.
### 2c. Download this  repository from GitHub
Download the full repository (click Clone or Download) and extract all the contents directly into the C:\tensorflow1\models\research\object_detection directory.This repository contains the images, annotation data, .csv files, and TFRecords needed to train a Person detector. You can use these images and data to practice making your own Person Detector. It also contains Python scripts that are used to generate the training data. It has scripts to test out the Person detection classifier on images, videos, or a webcam feed.
To train your own object detector, delete the following files (do not delete the folders):
- All files in \object_detection\images\train and \object_detection\images\test
- The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
- All files in \object_detection\training
- All files in \object_detection\inference_graph
 
### 2d. Set up new Anaconda virtual environment
Next, we'll work on setting up a virtual environment in Anaconda for tensorflow-gpu. From the Start menu in Windows, search for the Anaconda Prompt utility and open it.
In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command:
...
C:\>conda activate tf_gpu
...
Install the other necessary packages by issuing the following commands:
(tf_gpu) C:\> conda install -c anaconda protobuf
(tf_gpu) C:\> pip install pillow
(tf_gpu) C:\> pip install lxml
(tf_gpu) C:\> pip install Cython
(tf_gpu) C:\> pip install contextlib2
(tf_gpu) C:\> pip install jupyter
(tf_gpu) C:\> pip install matplotlib
(tf_gpu) C:\> pip install pandas
(tf_gpu) C:\> pip install opencv-python

### 2e. Configure PYTHONPATH environment variable
A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Do this by issuing the following commands (from any directory):
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
 
 
 
(***: Every time the "tf_gpu" virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again. You can use "echo %PYTHONPATH% to see if it has been set or not ***)
 
### 2f. Compile Protobufs and run setup.py
Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. Every .proto file in the \object_detection\protos directory must be called out individually by the command.
In the Anaconda Command Prompt, change directories to the \models\research directory:
...
(tf_gpu) C:\> cd C:\tensorflow1\models\research
...
 
Then copy and paste the following command into the command line and press Enter:
...
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto 
...
This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.
*** TensorFlow occasionally adds new .proto files to the \protos folder. If you get an error saying ImportError: cannot import name 'something_something_pb2' , you may need to update the protoc command to include the new .proto files. ***
Finally, run the following commands from the C:\tensorflow1\models\research directory:
...
(tf_gpu) C:\tensorflow1\models\research> python setup.py build
(tf_gpu) C:\tensorflow1\models\research> python setup.py install
...

## 3. Labelling Pictures
You can use your phone to take pictures of the objects or download images of the objects from Google Image Search.Make sure the images aren’t too large. They should be less than 200KB each, and their resolution shouldn’t be more than 720x1280. The larger the images are, the longer it will take to train the classifier. You can use the resizer.py script in this repository to reduce the size of the images.After you have all the pictures you need, move 20% of them to the \object_detection\images\test directory, and 80% of them to the \object_detection\images\train directory. Make sure there are a variety of pictures in both the \test and \train directories.
 LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it.
LabelImg GitHub link
LabelImg download link
Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory. This will take a while!

LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.
 
## 4. Training Data
With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
...
(tf_gpu) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
...
This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.
Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. 
# TO-DO replace this with label map
...
def class_text_to_int(row_label):
    if row_label == 'person':
        return 1
    else:
        None
 ...
 
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:
...
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
...
These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.
 
## 5. Create Label map and configure Training
### 5a. Label map
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. (Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map in the format below (the example below is the label map for my Person Detector):
...
item {
  id: 1
  name: 'person'
}
...
The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file.you can add different ID numbers for detecting multiple objects.
...
item {
  id: 1
  name: 'person'
}
 
item {
  id: 2
  name: 'dog'
}
 
item {
  id: 3
  name: 'cat'
}
...
 
 
### 5b. Configure training
Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_pets.config file into the \object_detection\training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the faster_rcnn_inception_v2_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

- Line 9. Change num_classes to the number of different objects you want the classifier to detect. For person detector, it would be num_classes : 1 .
- Line 106. Change fine_tune_checkpoint to:
  fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
- Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:
    input_path : "C:/tensorflow1/models/research/object_detection/train.record"
    label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
- Line 130. Change num_examples to the number of images you have in the \images\test directory.
- Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:
    input_path : "C:/tensorflow1/models/research/object_detection/test.record"
    label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
Save the file after the changes have been made. 
 
 
## 6. Run the Training
move train.py from /object_detection/legacy into the /object_detection folder and then continue following the steps below.
 
 From the \object_detection directory, issue the following command to begin training:
... 
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
...
If everything has been set up correctly, TensorFlow will initialize the training.
 
 

 
Each step of training reports the loss. It will start high and get lower and lower as training progresses.  It is  recommended  allowing your model to train until the loss consistently drops below 0.05.
## 7. Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

...
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
...

This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the object detection classifier.

## 8. Testing of Newly Trained Person Detection Model 

The object detection classifier is all ready to go!  Python scripts to test it out on an image, video, or webcam feed has been written.

* Object_detection_imapge.py (for person detection in image)
* Object_detection_video.py ( person detection in video)
* Object_detection_webcam.py (person detection in webcam)
* Object_detection_video_count.py (person detection and counting in each frame)

Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. (For Person Detector, there is only person  to detect, so NUM_CLASSES = 1.)
To test your object detector, move a picture of the object or objects into the \object_detection folder, and change the IMAGE_NAME variable in the Object_detection_image.py to match the file name of the picture. Alternatively, you can use a video of the objects (using Object_detection_video.py), or just plug in a USB webcam and point it at the objects (using Object_detection_webcam.py).
To run any of the scripts, type running command  in the Anaconda Command Prompt (with the “tf_gpu” virtual environment activated) and press ENTER. 
If everything is working properly, the object detector will initialize for about 10 -15  seconds and then display a window showing any objects it’s detected in the image!
 
 


 
 

