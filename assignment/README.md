# Object Detection Assignment

The purpose of this assignment is to run various pretrained models for object detection. Note that you should only have to change one line in the notebook (the "MODEL_NAME" variable), but you will have to re-run certain sections to change variable names for a new model or to add more testing images.

## Getting Started

These instructions will get you a copy of the project up and running on the cloud for FREE.

### Prerequisites

In order to run this assignment, you will at least need a Google account and capabilities to run the Google Colaboratory jupyter notebook environment. More information about [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb).


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

Now you can use Google Colab to run the Object Detection API for detection.

```
Menu bar > Runtime > Restart and Run all
```

## Running the tests

The main point of this assignment is to run different models on different test images.

### Test Images

Upload images with objects that fit the mscoco label categories. The extensions tested were valid_images_ext = [".jpg",".gif",".png"].

```
- Upload the images to Google Colab notebook enviornment by 
    - using the notebook menu bar (Table of content, Code Snippets, Files)
    - navigating to Files 
    - click the Upload button.
```

The categories below come from the mscoco_label_map.pbtxt file:

```
"person"
"bicycle"
"car"
"motorcycle"
"airplane"
"bus"
"train"
"truck"
"boat"
"traffic light"
"fire hydrant"
"stop sign"
"parking meter"
"bench"
"bird"
"cat"
"dog"
"horse"
"sheep"
"cow"
"elephant"
"bear"
"zebra"
"giraffe"
"backpack"
"umbrella"
"handbag"
"tie"
"suitcase"
"frisbee"
"skis"
"snowboard"
"sports ball"
"kite"
"baseball bat"
"baseball glove"
"skateboard"
"surfboard"
"tennis racket"
"bottle"
"wine glass"
"cup"
"fork"
"knife"
"spoon"
"bowl"
"banana"
"apple"
"sandwich"
"orange"
"broccoli"
"carrot"
"hot dog"
"pizza"
"donut"
"cake"
"chair"
"couch"
"potted plant"
"bed"
"dining table"
"toilet"
"tv"
"laptop"
"mouse"
"remote"
"keyboard"
"cell phone"
"microwave"
"oven"
"toaster"
"sink"
"refrigerator"
"book"
"clock"
"vase"
"scissors"
"teddy bear"
"hair drier"
"toothbrush"
```

### Picking a Model ~ Model Zoo

Choose a model from Tensorflows' [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Specifically choose from the "COCO-trained models" with Outputs = "Boxes". For example if the link is:
```
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```
You should modify the variable named "MODEL_NAME" in the Model Download > Modify Download Variables section with the "ssd_mobilenet_v1_coco_2018_01_28" portion of the link. This is the only line of the Assingment you should have to modify unless you choose a model from elsewhere or choose a different label map.

### Finally

Once you have choosen a model and uploaded images you can run the file by Restarting and Running all.

```
Menu bar > Runtime > Restart and Run all
```

If you add images and all you want to do is run the same model with those images, just run the block labeled "Prepare test images" and the final block.

If you change the model, use the "Run After" option from the section of code titled "Model Download"
```
Menu bar > Runtime > Run After
```

After running, the image with boxes and confidence will be drawn in the output. Additionally, these images will be saved in a folder named 'output' located in /content/output/. Another output is the time it took for the that image to be ran through the model and drawn. 

### What to submit

The submission for this assignment should be a file containing runs of two different models with two different test images. Also, include the time it took to run the specific model for the test image. Therefore, the requirements are:

```
- Two model names
- 4 run times (two for each model)
- 4 images with boxes and confidence (two for each model)
```

**An example submission is located in the repository under the 'assignment' folder.**

The images with drawn boxes can be downloaded from the output folder in /content/output by right clicking on the file and choosing download.

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

* [PurpleBooth](https://github.com/PurpleBooth)
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for the object_detection_tutorial.py code that was used for running detection
