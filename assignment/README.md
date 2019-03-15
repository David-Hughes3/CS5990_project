# CS5990_project - Assignment

Contained in this README are assignment details for a project in CS5990 Deep Learning at Cal Poly Pomona; the specific project is for Object Detection: FasterRCNN and SSD for custom object detection. The assignment information concerns information on how to download the code/data and run the assignment.

## Getting Started

These instructions will get you a copy of the project running and provide information on how to customize your run.

### Prerequisites

In order to run this assignment, you will at least need a Google account and capabilities to run the Google Colaboratory jupyter notebook environment. Additionally, there is an option to use your own custom labeled dataset for training; this requires the use of  [LabelImg](https://github.com/tzutalin/labelImg).

```
- Google Colab
- [LabelImg](https://github.com/tzutalin/labelImg)
```

### Installing and Running

First, clone or download the repository as a zip.

```
$git clone https://github.com/David-Hughes3/CS5990_project.git

or 

https://github.com/David-Hughes3/CS5990_project/archive/master.zip
```

Next, move the Jupyter notebook file in this repository into your Google drive.

```
- Open your Google Drive in a browser
- Move the Assignment.ipynb into your Drive at your choice of directory
- Right-click Assignment.ipynb in your Drive > Open With > Google Colaboratory
```

Now you can use Google Colab to run the Object Detection API for both training and detection.

```
Menu bar > Runtime > Restart and Run all
```

Monitoring training can be done by opening the tensorboard link that will look something like "https://f1632e11.ngrok.io/" that is output above the training command step. Tensorboard with Colab has issues occasionally, so the display may or may not work well.


Additional Notes:
1. The unmodified colab file will run all the way through with a playing card dataset from EdjeElectronics.
2. As tensorflow is updating to 2.0.0 there is a lot of additional warnings that file output in the notebook.
3. Running the training command below fills up std output with a lot of text that tends to crash the Colab jupyter notebook environemnt. So, the output has been redirected to a /content/training_log.txt . You can download or view this file periodically to observe the training steps.
```
!python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config 2> /content/training_log.txt
```
4. You can stop the training at any time and simply use Menu Bar > Runtime > Run After to stop the training early. You can check how many checkpoints are available in /content/models/research/object_detection/training, so if you want to perform a manual early stopping you should at least let the notebook create one new checkpoint (typically at step 17XX).

### Code

The code is split into three parts: setting up the initial environment, training, and detecting.

*Two Options:*
1. Run the colab notebook as is and the default configuration will run of the tutorial data
2. Create your own images, bounding box xmls, change the settings of the faster_rcnn_inception_v2_pets.config, change the labelmap.pbtxt file, modify a function in generate_tfrecord.py, and comment out the first code block of the colab


## Modification

The modification tasks that need to be performed to use your own dataset are listed below. Make sure to store the files in the correct structure, because the notebook looks for files in that structure.

*Tasks:*
1. Gather and label images
2. Change the labelmap.pbtxt
3. Modify a function in generate_tfrecord.py
4. Modify faster_rcnn_inception_v2_pets.config
5. (Optional) Change a line in model_main.py
6. Zip up the custom folder and upload to the colab notebook
7. Comment out the first block of code and unblock comment the second block of code, so that the custom.zip can be unzipped and your dataset will be used

*Directory Structure:*
- custom
    - images
        - test
            - .JPG files of images
            - .xml files of bounding boxes labels from LabelImg
        - train
            - .JPG files of images
            - .xml files of bounding boxes labels from LabelImg
    - testing
        - .JPG files
    - training
        - faster_rcnn_inception_v2_pets.config
        - labelmap.pbtxt
    - generate_tfrecord.py
    - sizeChecker.py
    - xml_to_csv.py
    - model_main.py

### Gathering and Labeling Images

Training a deep learning model requires lots of good data. First, take pictures of the objects of different classes you desire to detect. Make sure to use a variety of backgrounds and overlapping objects. Limit the size of the pictures to <200KB each. After gathering images, move 80% to the directory custom/images/train and 20% to custom/images/test (to be validation).

Next, we need to label the images with classes and bounding boxes. Use [LabelImg](https://github.com/tzutalin/labelImg) for drawing a box around each image and assigning box labels. LabelImg saves an .xml file for each image in the directory that corresonds to whether the image is in custom/images/train or custom/images/test.

The image files and xml labels will be used to generate Tensorflow record format files, but that is handled by the colab notebook as long as the directory structure is followed.

### labelmap.pbtxt
Label map ID numbers match Function class_text_to_int(row_label) in generate_tfrecord.py. Create this file in custom/training/labelmap.pbtxt . An example structure of the file is below.
```
item {
  id: 1
  name: 'basketball'
}
item {
  id: 2
  name: 'shirt'
}
item {
  id: 3
  name: 'shoe'
}
```
### generate_tfrecord.py
Replace label map on line 31 of custom/generate_tfrecord.py . Match the returned values to the id values of labelmap.pbtxt . An example of how to setup the function is below.
```
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        return None
```
### faster_rcnn_inception_v2_pets.config
*Note: A BUG, sometimes Tensorflow changes the requirements of this file. If an exception is thrown concerning a setting in the file, download the original config from /object_detection/samples/config/faster_rcnn_inception_v2_pets.config. The lines numbers will probably be different, but the proto classes should be the same.*

Modify some settings in the config proto file located in custom/training/faster_rcnn_inception_v2_pets.config
```
 - Line 9: change num_classes to match the number of items in labelmap.pbtxt (the number of classes to detect)
 - Line 130: change num_examples to the number of images in custom/images/test
```
Another option is to change the number of steps, either increasing or decreasing depending on your patience and dataset. There are also a variety of other hyperparameters you can change in this file, but they are mostly left alone.
```
  - Line 113: change num_steps
``` 

If you had to follow the *note* and downloaded a new file from /object_detection/samples/config/faster_rcnn_inception_v2_pets.config : follow these additional steps to change the configs directory structure information to work for the colab notebook directory structure. 
```
 - Line 106:change fine_tune_checkpoint: "/content/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
 - Line 123 and Line 125 change:
    - input_path: "/content/models/research/object_detection/train.record"
    - label_map_path : "/content/models/research/object_detection/training/labelmap.pbtxt"
 - Line 135 and line 137 change:
    - Input_path: "/content/models/research/object_detection/test.record"
    - Label_map_path: "/content/models/research/object_detection/training/labelmap.pbtxt"
```    
### Optional Step

*This information is subject to change with new object detection api versions.*

The object detection api runs evaluation after each checkpoint is created, and each checkpoint is created after a certain time interval. This is the default behavior, but doesnt serve us well in this toy problem for visualizing on our validation dataset.

On line 27, add this line to make the steps print to the console
```
tf.logging.set_verbosity(tf.logging.INFO)
```

On line 62, in model_main.py (runs both train and eval):
```
config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
```
Change this line to be
```
config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=500, log_step_count_steps=100)
```
- save_checkpoints_steps makes it so that checkpoints will be created more often, and therefore eval will be ran more often.
- log_step_count changes how often output is printed and logs are created for tensorboard visualization

This modified file has been included in the default custom folder download that will be installed in the colab notebook.

TODO:
```
A common problem that does not have a good solution yet in the object detection api.
"INFO:tensorflow:Skip the current checkpoint eval due to throttle secs (600 secs)."
This throttle_secs parameter prevents the more frequent checkpointing from evaluation, causing the same problem of a sparse number of datapoints in the graph. So far modifiying the model_lib.py and model_main.py has been tried to include additonal flags and trying to pass a different argument to the EvalSpec constructor.

```

### Final Modification Steps
In order to move the images and modified scripts/configs, follow these steps.

```
- Zip up the custom folder in whatever manner you are familiar with.
- Upload the zip file to Google Colab notebook enviornment by 
    - using the notebook menu bar (Table of content, Code Snippets, Files)
    - navigating to Files 
    - click the Upload button.
```

This uploads the data/code into the /content/ directory of the colab environemnt. Make sure to comment out the block of code below as this downloads the default files from this repository.

```
#IF YOU WANT TO UPLOAD YOUR OWN CUSTOM DATASET THEN COMMENT THIS CODE OUT, otherwise this contains all of the needed files to run this colab notebook
!git clone https://github.com/David-Hughes3/CS5990_project.git
!mv /content/CS5990_project/assignment/custom /content/
```

Then uncomment this block to unzip the custom folder as long as it is uploaded in /content/custom.zip

```
# #uncomment this block if you want to upload your own custom folder (as a zip file)
# !unzip custom.zip
```

Note that any method can be used to upload the custom folder as long as the subfiles are stored in this directory path /content/custom. This is because later parts of the code move files from this directory to the tensorflow/models/research/object_detection directory as it is set up in the colab notebook. Examples of other methods include 1. storing the folder in your google drive and [mounting your gdrive](https://stackoverflow.com/a/53545162) 2. storing the zip in your google drive, [using the file id to download to colab](https://stackoverflow.com/a/51456666), and unzipping the zip file in colab

## Built With

* [Google colab](https://colab.research.google.com/) - Runtime environment that contains preinstalled libraries listed below
  - Protobuf 3.0.0
  - Python-tk
  - Pillow 1.0
  - lxml
  - tf Slim (which is included in the "tensorflow/models/research/" checkout)
  - Jupyter notebook
  - Matplotlib
  - Tensorflow (>=1.12.0)
  - Cython
  - contextlib2
  - cocoapi
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
  - Dependencies listed above, already installed in Colab
* Tensorflow-gpu >=1.12.0
  - Already installed on google colab
  - CUDA10
  - CUDNN



## Authors

* **David Hughes** - CS5990 Project 

## License

This project is licensed under the MIT License.

## Acknowledgments

* [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) for training code and data
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for the object_detection_tutorial.py code that was used for running detection
