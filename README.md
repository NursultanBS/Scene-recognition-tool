• Run_all_classifiers.py file runs our classification models on the sample image “university.jfif”. It outputs the predicted scene classification, type of environment, scene attributes and outputs the heat map for the image in “cam.jpg”. 

• The .ipynb file shows our work on training the multi-label attribute classifier and shows the output of other implemented models. The weights of the trained multi-label attribute classifier are saved in sun_attribute_classifier.pth. 

• Sun_dataloader.py file contains implementation of data loader for the SUN database. 

• Wideresnet.py is used in run_all_classifiers.py to load the ResNet model for the scene classifier, the pre-trained weights of which are loaded from wideresnet18_places365.pth.tar file

• License file is the file that allows to use the code from the authors codebase

• Categories_places365.txt file contains all of the scene categories for the scene classifier

• labels_SUNattribute.txt file contains all of the scene attributes used for multi-label attribute classifier

• IO_places365.txt file contains the type of environment (indoor is 1 and outdoor is 2) for all of the scene categories
