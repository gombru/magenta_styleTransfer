from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

import numpy as np
import tensorflow as tf

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model
from magenta.models.image_stylization import ops

import cv2
import time


def _load_checkpoint(sess, checkpoint):
    """Loads a checkpoint file into the session."""
    model_saver = tf.train.Saver(tf.global_variables())
    checkpoint = os.path.expanduser(checkpoint)
    if tf.gfile.IsDirectory(checkpoint):
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
    model_saver.restore(sess, checkpoint)


def multiple_input_images(checkpoint, num_styles, input_images_dir, input_images, which_styles):
    """Added by Raul Gombru. Computes style transfer for a list of images"""

    result_images = {}

    with tf.Graph().as_default(), tf.Session() as sess:

        # Need to initialize the image var to load the model
        image_path = input_images_dir + input_images[0]
        image = np.expand_dims(image_utils.load_np_image(os.path.expanduser(image_path)), 0)
        stylized_images = model.transform(
            tf.concat([image for _ in range(len(which_styles))], 0),
            normalizer_params={
                'labels': tf.constant(which_styles),
                'num_categories': num_styles,
                'center': True,
                'scale': True}, reuse=tf.AUTO_REUSE)

        # Load the model
        _load_checkpoint(sess, checkpoint)

        # Stylize images
        for image_name in input_images:
            image_path = input_images_dir + image_name
            image = np.expand_dims(image_utils.load_np_image(os.path.expanduser(image_path)), 0)
            stylized_images = model.transform(
                tf.concat([image for _ in range(len(which_styles))], 0),
                normalizer_params={
                    'labels': tf.constant(which_styles),
                    'num_categories': num_styles,
                    'center': True,
                    'scale': True}, reuse=tf.AUTO_REUSE)
            stylized_images = stylized_images.eval()
            for which, stylized_image in zip(which_styles, stylized_images):
              result_images[image_name.split('.')[0] + '_' + str(which)] = stylized_image[None, ...]

    return result_images

def style_from_camera(checkpoint, num_styles, which_style, SaveVideo=False):
    """Added by Raul Gombru. Computes style transfer frame by frame"""

    # initialize video camera input
    cap = cv2.VideoCapture(0)

    if (SaveVideo):
        video = cv2.VideoWriter('output.avi', 0, 12.0, (640, 480))

    with tf.Graph().as_default(), tf.Session() as sess:

        # Need to initialize the image var to load the model
        frame_2_init = np.expand_dims(np.float32(np.zeros((640, 480, 3))), 0)
        stylized_images = model.transform(
            tf.concat([frame_2_init for _ in range(1)], 0),
            normalizer_params={
                'labels': tf.constant(which_style),
                'num_categories': num_styles,
                'center': True,
                'scale': True}, reuse=tf.AUTO_REUSE)

        # Load model
        _load_checkpoint(sess, checkpoint)

        # Read, style and show frames
        while (True):

            start_time = time.time()

            ret, frame = cap.read()
            original_frame = frame

            frame = np.expand_dims(np.float32(frame/255.0), 0)
            stylized_images = model.transform(
                tf.concat([frame for _ in range(1)], 0),
                normalizer_params={
                    'labels': tf.constant(which_style),
                    'num_categories': num_styles,
                    'center': True,
                    'scale': True}, reuse=tf.AUTO_REUSE)
            stylized_images = stylized_images.eval()
            for which, stylized_image in zip(which_style, stylized_images):
                out_frame = stylized_image[None, ...]

            out_frame = (out_frame[0,:,:,:]*255).astype('uint8')

            elapsed_time = time.time() - start_time

            print("Running at --> " + str(1 / elapsed_time) + " fps")

            # Show frames
            cv2.namedWindow("input")
            cv2.imshow('input', original_frame)

            cv2.namedWindow("output")
            cv2.imshow('output', out_frame)

            if (SaveVideo):
                video.write(out_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        if (SaveVideo):
            video.release()
        cv2.destroyAllWindows()