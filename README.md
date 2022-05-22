This python package is used for object detection that combines Gradient-weighted Class Activation Mapping++ (Grad-CAM++) for localization and Mask Regional Convolution Neural Network (Mask R-CNN) for object detection. Hence this model is knowns as GradCAM-MLRCNN

![image](https://user-images.githubusercontent.com/47241538/169694493-e8c5f961-4e26-44ed-bcc3-01192fc5b9e9.png)
                      This is architecture of the model

We are applied this model for various pretrained models like VGG16,VGG19,ResNet 101, ResNet 152 and ResNet 50.
This python package is exclusively for ResNet 50 only. 
For remaining pretrained models, still under development process

The main process of this models as follows

1) It will predict the Class Activation Map (CAM) on given input image
![image](https://user-images.githubusercontent.com/47241538/169694798-e1552f55-0e71-4a8f-87aa-ddec32d3bd4c.png)

2) For localization, we used Gradient-Weighted Class Activation Map (Grad-CAM++) and for object detection, we used Mask Regional Convolutinal Neural Network (Mask R-CNN)

![image](https://user-images.githubusercontent.com/47241538/169694957-ac0ac8a4-312f-4800-9a70-681463b0b221.png)

Use the following code:
from tensorflow import keras
import glob
from grmrcnn import grmrcnn
**#need to mention and keep the model in the current directory (ResNet -Recommended)**
model = keras.models.load_model("ResNet50.h5") 
path = glob.glob("...../Images/*.jpg") 
