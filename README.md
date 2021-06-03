# Transfer-learning

##Transfer learning for 2D Bounding box detection on new objects	
This is a training project done with the purpose of learning.

##Project Aim :
To collect 5-15 images of a rubber duck and then manually annotate them i.e. put bounding boxes around them to get the labels for (x,y,w,h).
Load a pretrained SSD with ResNet-50 and finetune it on these images.  	
Run the object detection algorithm on a new image to identify the rubber duck.

##Workflow

.tensorflow gpu model is installed and imported

utility modules(os,io,pathlib) of python are imported to change the path and directories of files.

the tensorflow model garden cloned from git and  the object detection API which provides the required framework is installed<br>
the following utility functions are imported :/
numpy  library for creating and handling multi-dimensional arrays of image data/
modules from IPython.display to display images in Jupyter notebook./
colab_utils for mannually annotating the images/
##visualization_utils for visualising the detections/
##object detection utilities/ 
##5.function 'image_into_numpy' is defined to load images from file into a numpy array./
##the image data is Obtained using the path in arguement and returned into a uint8 numpy array with shape(height, width, channels(RGB).
##plot_detections function is Defined which has the arguements: the numpy arrays image_np, boxes, classes,   scores
##a wrapper function is Defined  to visualize detections ie. bounding boxes and coorresponding text which takes the numpy array having the labels of the boxes annotated and saved  to numpy array 'annotations_np'.
##using matplotlib.pyplot 'annotations_np' array is saved as an image file and data from 'annotations_np'as an image is displayed 
##6. the path is set to the directory contaning the training images of 5 rubber ducks using os module
##data of the training images is loaded into a numpy array 'train_images_np'.
##using matplotlib.pyplot data  from 'train_images_np' is displayed as an image.
##the training images can be seen as the output.
##7.using colab_utils.annotate() function the images are manually annotated and the labels of the boxes are stored in the array 'gt_boxes'.
##8.data is prepared  for training by converting all classes into one hot labels and converting the images and bounding box labels into tensors.
##since we will be having just one class of rubber ducks ,value of 1 is assigned to class id.
##using a for loop one by one the data of every image and corresponding box labels are loaded into the train_image_np, gt_box_np array.
##now the data from numpy array array (train_image_np, gt_box_np array.) is converted to tensors( train_image_tensors , gt_classes_one_hot_tensors, respectively).
#the non-background classes start counting at 1 index.to start counting at the zeroth index we shift the classesby 1 index using label_id_offset variables and store it in tensor 'zero_indexed_groundtruth_classes'.
##all classes  are converted into one hot labels in the tensor 'gt_classes_one_hot_tensors'
##9.then the SSD ResNet50 checkpoint is downloaded.
##10.the num_classes is override as we need to detect just the one class(rubber duck)and its value is set to one.
##the checkpoint is saved inside the model garden.
##11.the checkpoint is saved inside the model garden.
##the pipeline configuartion file is stoted in  the 'pc' variable and is then loaded using cofigs var into the 'model_config' to build a detection model.
##12.then only the box regresssion head is restored but not the classification head./
##a model named 'new_model' is created which has the feature extraction layers of the ssd resnet model which is trained on coco architechture. but since we need it to detect only /
the rubber duck class, we create a model which has only the feature extraction layers in backend and the box regresssion head. the classification layer is initialised and then it learns the weights and parameters using the training set of rubber ducks given as np array./
13.now to train the the model on our data set, the weights of the model are restored.the model is run through a dummy image first to create the variables./
14. fine tuning of the model on the rubber duck images. is done by choosing some parameters and fine tuning them/.
since we are providing the input of just 5 images we set the batch_size to 4(0 to 4)./
a learning rate of 0.01 is chosen./
the variables in the top layer of resnet model are chosen and then stored in the array 'fine_tune_array'./
"model_training" function is defined for the forward and backward propoagation in a single training example. inside it we define another function 'training_step'. this takes as arguements the 

  


  
  
    

  
