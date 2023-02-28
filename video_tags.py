#importing libraries
import torch
import os
import cv2
import pandas as pd
import shutil, glob
from os import listdir
from os.path import isfile, join

from scenedetect import ContentDetector, ThresholdDetector, AdaptiveDetector
from scenedetect import SceneManager
from scenedetect import detect, split_video_ffmpeg, SceneManager, open_video

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
from PIL import Image

import wideresnet

execution_path = os.getcwd()


features_blobs = []

 # hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()

    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

def places_prediction(img):

    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # load the model
    model_p = load_model()

    # load the transformer
    tf = returnTF() # image transformer

    # get the softmax weight
    params = list(model_p.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    # result = pd.DataFrame()
    #np
    result = []
    input_img = V(tf(Image.open(img)).unsqueeze(0))

    # forward pass
    logit = model_p.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    for i in range(0, 5):
        # new_row = {'confidence':probs[i], 'class':idx[i], 'name':classes[idx[i]]}
        # result = result.append(new_row, ignore_index=True)
        new_row = [probs[i], idx[i], classes[idx[i]]]
        result = np.concatenate((result, new_row), axis=0)

    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)

    for i in range(-1,-10,-1):
        if(responses_attribute[idx_a[i]] > 0.5):
            # new_row = {'name':labels_attribute[idx_a[i]], 'confidence':responses_attribute[idx_a[i]], 'class':idx_a[i]}
            # result = result.append(new_row, ignore_index=True)
            new_row = [labels_attribute[idx_a[i]], responses_attribute[idx_a[i]], idx_a[i]]
            result = np.concatenate((result, new_row), axis=0)


    return result

def find_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector())
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    return scene_manager.get_scene_list()

def video2frames(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:

        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*5000)) 
        # Resize the image frames
        try:
            image = cv2.resize(image, (150, 130))
        except Exception as e:
            print(str(e))

        cv2.imwrite(join(pathOut, "frame%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def delete_files_in(folder):
    for filename in listdir(folder):
        file_path = join(folder, filename)
        try:
            if isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def object_detect_by_scence(video_path, model_y):

    df_video = pd.DataFrame()
    head, tail = os.path.split(video_path)
    filename = os.path.splitext(tail)[0]
    scene_list = find_scenes(video_path) #scene_list is now a list of FrameTimecode pairs representing the start/end of each scene
    split_video_ffmpeg(video_path, scene_list) #en mypath están ahora los vídeos de cada escena

    scenes = [f for f in listdir(execution_path) if f.startswith(filename)]
    for i, scene in enumerate(scenes):
        df_scene=pd.DataFrame()
        pathIn = join(execution_path, scene)
        pathOut = join(execution_path,"scenes\\")
        video2frames(pathIn, pathOut)

        frames = [f for f in listdir(pathOut) if isfile(join(pathOut, f))]
        for j, frame in enumerate(frames):            
            img = join(pathOut, frame)  # or file, Path, PIL, OpenCV, numpy, list
            df_places = places_prediction(img) # using Places365
            results = model_y(img) # using yolo model
            # df_frame = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc.
            df_frame = results.tolist()
            print("#### SHAPE ####", np.shape(df_frame))
            df_frame = df_frame.append(df_places)
            df_frame.insert(0, 'frame', j)
            df_scene = df_scene.append(df_frame)

        df_scene.insert(0, 'scene', i)
        df_video = df_video.append(df_scene, ignore_index=True)
        delete_files_in(pathOut)
    
    df_video.to_csv(filename + ".csv")
    return df_video

def excect(video_path):

    # Model
    model_y = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5n - yolov5x6, custom
    # mypath = join(execution_path, "C:\\Users\\Monica\\Documents\\Tyris\\Publi\\Notebooks\\youtube_trailers\\videos") 
    # video_filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Detection by scene
    # video_path = join(mypath, "56949.mp4") 
    df_out = object_detect_by_scence(video_path, model_y)