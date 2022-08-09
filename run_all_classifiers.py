# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms

features_blobs = []

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
    file_name_attribute = "labels_SUNattribute.txt"
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]

    return classes, labels_IO, labels_attribute

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

    import wideresnet
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

def format_image(x, tf):
    scale = np.random.rand() * 2 + 0.25
    w = int(x.size[0] * scale)
    h = int(x.size[1] * scale)
    if min(w, h) < 227:
        scale = 227 / min(w, h)
        w = int(x.size[0] * scale)
        h = int(x.size[1] * scale)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std= [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            normalize,
        ])
    x = test_transform(x)
    x = x.view(1, 3, 227, 227)
    return x

def get_top_10_attributes(x, classifier, labels_attribute):
    classifier.eval()
    classifier = classifier.cpu()
    logits = classifier(x)
    sorted_indices = torch.argsort(logits, descending=True)
    top_10_indices = sorted_indices.numpy()[0][:10]
    return [labels_attribute[i] for i in top_10_indices]

def load_attributes_classifier():
    load_network_path = './sun_attribute_classifier.pth'
    # use to load our trained network
    if load_network_path is not None:
        print('Loading saved network from {}'.format(load_network_path))
        classifier = torchvision.models.densenet201()
        classifier.classifier = nn.Linear(1920, 102)
        classifier.load_state_dict(torch.load(load_network_path, map_location=torch.device('cpu')))
    return classifier


def main():
    # load the labels
    classes, labels_IO, labels_attribute = load_labels()

    # load the model
    model = load_model()

    # load the transformer
    tf = returnTF() # image transformer

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    # load the test image
    # img_url = 'http://places.csail.mit.edu/demo/6.jpg'
    # os.system('wget %s -q -O test.jpg' % img_url)
    # # img = Image.open('test.jpg')
    img_name = 'university.jfif'
    img = Image.open(img_name)
    input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    print('RESULT ON ' + img_name)

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        print('--TYPE OF ENVIRONMENT: indoor')
    else:
        print('--TYPE OF ENVIRONMENT: outdoor')

    # output the prediction of scene category
    print('--SCENE CATEGORIES:')
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # output the scene attributes
    classifier = load_attributes_classifier()
    img = img.convert('RGB')
    img_formatted = format_image(img, tf)
    top_10_attributes = get_top_10_attributes(img_formatted, classifier, labels_attribute)

    print('--SCENE ATTRIBUTES:')
    print(', '.join(top_10_attributes))


    # generate class activation mapping
    print('Class activation map is saved as cam.jpg')
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    img = cv2.imread(img_name)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.5
    cv2.imwrite('cam.jpg', result)

if __name__ == "__main__":
    main()
