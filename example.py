import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import cv2
from PIL import Image

import tensorflow as tf

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)

classes = ['sidewalk']
# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


def train():
    labels = {1: 'sidewalk'}
    train_imgs_dir = "./VOC2007_train/JPEGImages"
    train_Anno_dir = "./VOC2007_train/Annotations"

    valide_imgs_dir = "./VOC2007_valide/JPEGImages"
    valide_Anno_dir = "./VOC2007_valide/Annotations"

    test_imgs_dir = "./VOC2007_test/JPEGImages"
    test_Anno_dir = "./VOC2007_test/Annotations"

    traindata = object_detector.DataLoader.from_pascal_voc(train_imgs_dir, train_Anno_dir, labels)
    validata = object_detector.DataLoader.from_pascal_voc(valide_imgs_dir, valide_Anno_dir, labels)
    testdata = object_detector.DataLoader.from_pascal_voc(test_imgs_dir, test_Anno_dir, labels)
    spec = model_spec.get('efficientdet_lite0')
    spec.uri = 'https://storage.googleapis.com/tfhub-modules/tensorflow/efficientdet/lite0/feature-vector/1.tar.gz'
    spec.input_image_shape = [512, 288]

    model = object_detector.create(traindata, model_spec=spec, batch_size=6, train_whole_model=True,
                                   validation_data=validata, epochs=120)
    model.summary()

    model.evaluate(testdata)
    model.export(export_dir='./detectMode')


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""

    signature_fn = interpreter.get_signature_runner()
    # Feed the input image to the model
    output = signature_fn(images=image)
    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj['class_id'])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
        cv2.putText(original_image_np, label, (xmin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


if __name__ == '__main__':
    # start to train
    train()

    # start to load the tflite to predict a image
    DETECTION_THRESHOLD = 0.3
    model_path = './detectMode/model.tflite'
    TEMP_FILE = './135.bmp'

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        TEMP_FILE,
        interpreter,
        threshold=DETECTION_THRESHOLD
    )

    # Show the detection result
    image = Image.fromarray(detection_result_image)
    image.show()


