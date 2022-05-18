from scipy.ndimage.interpolation import zoom
import numpy as np
from tensorflow import keras
#from keras.backend import tensorflow_backend
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
import cv2
def grad_plus(input_model, img, layer_name,H=224,W=224):
    model=input_model
    img1=img
    cls = np.argmax(input_model.predict(img))
    y_c = input_model.output[0, cls]
    print(y_c)
   
    conv_output = input_model.get_layer(layer_name).output
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    #with tf.GradientTape() as tape:
    tape=tf.GradientTape()
    last_conv_layer_output, preds = grad_model(img)
    pred_index = tf.argmax(preds[0])
    class_channel = preds[:, pred_index]
    #grads = tape.gradient(class_channel, last_conv_layer_output)[0]
    grads=K.gradients(class_channel, last_conv_layer_output)[0]
    #grads = cv2.normalize(grads)
    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads *grads*grads
    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom
    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)
    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)
    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam,H/cam.shape[0])
    cam = cam / np.max(cam) # scale 0 to 1.0    
    #cam = resize(cam, (224,224))
    heatmap=cam
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 0.1
    heatmap = numer / denom
    from IPython.display import Image, display
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    import keras
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    #jet_heatmap = jet_heatmap.resize((224, 224))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    #superimposed_img = jet_heatmap* 0.4 + cam
    #superimposed_img = superimposed_img.resize((224, 224))
    
    #superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return cam,jet_heatmap
