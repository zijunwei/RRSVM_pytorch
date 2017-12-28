#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import argparse

# import cv2
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms

from GradCam.gradcam import BackPropagation, GradCAM, GuidedBackPropagation


def main(args):

    # Load the synset words
    file_name = 'Miscs/synset_words.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ', 1)[
                           1].split(', ', 1)[0].replace(' ', '_'))

    print('Loading a model...')
    model = torchvision.models.vgg19(pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print('\nGrad-CAM')
    gcam = GradCAM(model=model, target_layer='features.36',
                   n_class=1000, cuda=args.cuda)
    gcam.load_image(args.image, transform)
    gcam.forward()

    for i in range(0, 5):
        print('\t{:.5f}\t{}'.format(gcam.prob[i], classes[gcam.idx[i]]))
        gcam.backward(idx=gcam.idx[i])
        cls_name = classes[gcam.idx[i]]
        output = gcam.generate()
        gcam.save('images/{}_gcam.png'.format(cls_name), output)

    print('\nBackpropagation')
    bp = BackPropagation(model=model, target_layer='features.0',
                         n_class=1000, cuda=args.cuda)
    bp.load_image(args.image, transform)
    bp.forward()

    for i in range(0, 5):
        print('\t{:.5f}\t{}'.format(bp.prob[i], classes[bp.idx[i]]))
        bp.backward(idx=bp.idx[i])
        cls_name = classes[bp.idx[i]]
        output = bp.generate()
        bp.save('images/{}_bp.png'.format(cls_name), output)

    print('\nGuided Backpropagation')
    gbp = GuidedBackPropagation(model=model, target_layer='features.0',
                                n_class=1000, cuda=args.cuda)
    gbp.load_image(args.image, transform)
    gbp.forward()

    for i in range(0, 5):
        cls_idx = gcam.idx[i]
        cls_name = classes[cls_idx]
        print('\t{:.5f}\t{}'.format(gbp.prob[i], cls_name))

        gcam.backward(idx=cls_idx)
        output_gcam = gcam.generate()

        gbp.backward(idx=cls_idx)
        output_gbp = gbp.generate()

        output_gcam -= output_gcam.min()
        output_gcam /= output_gcam.max()
        output_gcam_image = Image.fromarray(np.uint8(output_gcam*255))
        output_gcam_image = output_gcam_image.resize([224,224]).convert('RGB')
        # output_gcam = cv2.resize(output_gcam, (224, 224))
        # output_gcam = cv2.cvtColor(output_gcam, cv2.COLOR_GRAY2BGR)

        output = output_gbp * np.array(output_gcam_image).astype(np.float)/255

        gbp.save('images/{}_gbp.png'.format(cls_name), output_gbp)
        gbp.save('images/{}_ggcam.png'.format(cls_name), output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grad-CAM visualization')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--image', default='images/cat_dog.png',type=str)
    args = parser.parse_args()

    main(args)