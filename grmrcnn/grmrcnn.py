from mrcnn.visualize import display_instances
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
import numpy as np
from grad_plus import grad_plus
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def grmrcnn(model,path):
 plt.rcParams['figure.figsize'] = 8,8
 plt.show(block=True)
 self=model
 def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name
 ln=find_target_layer(model)
 labl=[]
 for imp in path:
    orig_img=cv2.imread(imp)
    orig_img= cv2.resize(orig_img, dsize=(224, 224))
    orig_img = orig_img.astype(np.uint8)
    img = np.expand_dims(orig_img,axis=0)
    img = preprocess_input(img)
    predictions = model.predict(img)
    i =np.argmax(predictions[0])
    top_n = 5
    top = decode_predictions(predictions, top=top_n)[0]
    cls = np.argsort(predictions[0])[-top_n:][::-1]
    [gradcamplus,je]=grad_plus(model,img,layer_name=ln)
    print(path)
    print("class activation map for:",top[0])
    fig, ax = plt.subplots(nrows=1,ncols=3)
    plt.subplot(121)
    plt.imshow(orig_img)
    plt.title("input image")
    s=je[:,:,1]
    s=s*0.8
    fimg=s+gradcamplus
    plt.subplot(122)
    plt.imshow(orig_img)
    plt.imshow(gradcamplus,alpha=0.8,cmap="jet")
    plt.title("Grad-CAM++")
    plt.show()
    ii=50
    ii=ii+1
    na=str(ii)+'.jpg'
    plt.imsave(na, gradcamplus,cmap="jet")
     # plt.savefig(na) # i changed  now worked well
    plt.close(fig)
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                   'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                   'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                   'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                   'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                   'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                   'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # MRCNN
    # draw an image with detected objects
    def draw_image_with_boxes(filename, boxes_list):
        # load the image
        data = pyplot.imread(filename)
        # plot the image
        pyplot.imshow(data)
        # get the context for drawing boxes
        ax = pyplot.gca()
        # plot each box
        for box in boxes_list:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
        # show the plot
        pyplot.show()
    # define the test configuration
    class TestConfig(Config):
        NAME = "test"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80
    # define the model
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
    # load coco model weights from "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
    rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
    # load photograph
    img1 = load_img('51.jpg') # this is result of Grad-CAM++
    img = img_to_array(img1)
    # make prediction
    results = rcnn.detect([orig_img], verbose=0)
    # visualize the results
    #draw_image_with_boxes(gradcamplus, results[0]['rois'])
    draw_image_with_boxes('51.jpg', results[0]['rois'])  #inba added (recent)
    r = results[0]
    # show photo with bounding boxes, masks, class labels and scores
    l1=display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])