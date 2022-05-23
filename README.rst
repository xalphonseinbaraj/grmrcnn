This python package is used for object detection that combines Gradient-weighted Class Activation Mapping++ (Grad-CAM++) for localization and Mask Regional 
Convolution Neural Network (Mask R-CNN) for object detection. Hence this model is knowns as GradCAM-MLRCNN
========================================================================================
.. image:: https://user-images.githubusercontent.com/47241538/169694493-e8c5f961-4e26-44ed-bcc3-01192fc5b9e9.png
  :width: 800
  
                     This is architecture of the model

We are applied this model for various pretrained models like VGG16,VGG19,ResNet 101, ResNet 152 and ResNet 50.
This python package is exclusively for ResNet 50 only. 
For remaining pretrained models, still under development process

The main process of this models as follows

1) It will predict the Class Activation Map (CAM) on given input image
![image](https://user-images.githubusercontent.com/47241538/169694798-e1552f55-0e71-4a8f-87aa-ddec32d3bd4c.png)

2) For localization, we used Gradient-Weighted Class Activation Map (Grad-CAM++) and for object detection, we used Mask Regional Convolutinal Neural Network (Mask R-CNN)

![image](https://user-images.githubusercontent.com/47241538/169694957-ac0ac8a4-312f-4800-9a70-681463b0b221.png)

.. code:: python
                                       
                                        from tensorflow import keras
                                        import glob
                                        from grmrcnn import grmrcnn
                                        #need to mention and keep the model in the current directory (ResNet -Recommended)
                                        model = keras.models.load_model("ResNet50.h5") 
                                        path = glob.glob("...../Images/*.jpg") #Image Path
                                        GradCAMMRCNN=grmrcnn(model,path)

Installation
--------

Requirements
^^^^^^^^^^^^

- Python 3.7
            
Note:
     We used tensorflow 1.x version and keras 2.3.1.
    Then, install this module from pypi using ``pip``
    
.. code:: python 
                                         pip install grmrcnn
                                        
How this model(Grad-CAM++ with Mask RCNN) works:
-----------------

If you want to know more about GradCAM++ with Mask RCNN, read our article: https://www.mdpi.com/2075-1702/10/5/340/htm

Thanks
---------

Many, many thanks to our advisor professor Dr.Jyh-Horng Jeng and co-advisor Dr.Jer-Guang Hsieh for guiding and helping for this amazing model and our collegues Mr.Julio Jerison Macrohon and Ms.Charlyn Villavicencio to publish this article successfully.

This research was funded as a scholar of the Ministry of Science and Technology (MOST), Taiwan and I-Shou University, Kaohsiung City, Taiwan.

Thanks to everyone who works on all the awesome Python data science libraries like numpy, scipy, scikit-image, pillow, etc, etc that makes this kind of stuff so easy and fun in Python.
