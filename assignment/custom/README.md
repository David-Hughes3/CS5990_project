Source of these files: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
Purpose: for download in assignment colab notebook

Make sure to follow the directory structure as the colab notebook looks for the files in this structure. 
Two Options:
    1. Create your own images, bounding box xmls, change the settings of the faster_rcnn_inception_v2_pets.config, change the labelmap.pbtxt file, and comment out the first code block of the colab
    2. Run the colab notebook as is and the default configuration will run of the tutorial data

Directory Structure:
custom
    -images
        -test
            -.JPG files of images
            -.xml files of bounding boxes
        -train
            -.JPG files of images
            -.xml files of bounding boxes from imageLbl
    -testing
        -.JPG files
    -training
        -faster_rcnn_inception_v2_pets.config
        -labelmap.pbtxt
    -generate_tfrecord.py
    -sizeChecker.py
    -xml_to_csv.py