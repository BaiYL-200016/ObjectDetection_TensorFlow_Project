import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import re

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    # print(output_details)
    # output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count

    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({'pos': positions[idx], '_id': classes[idx]})

    return result

def display_result(result, frame, labels):
    r"""Display Detected Objects"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1
    color = (0, 255, 0)  # Green color
    thickness = 2

    # position = [ymin, xmin, ymax, xmax]
    # x * CAMERA_WIDTH
    # y * CAMERA_HEIGHT
    for obj in result:
        pos = obj['pos']
        _id = obj['_id']

        x1 = int(pos[1] * CAMERA_WIDTH)
        x2 = int(pos[3] * CAMERA_WIDTH)
        y1 = int(pos[0] * CAMERA_HEIGHT)
        y2 = int(pos[2] * CAMERA_HEIGHT)

        cv2.putText(frame, labels[_id], (x1, y1), font, size, color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow('Object Detection', frame)


if __name__ == "__main__":

    model_path = 'model.tflite'
    label_path = 'labelmap.txt'
    # model_path = 'models/detect.tflite'
    # label_path = 'models/coco_labels.txt'


    cap = cv2.VideoCapture(0)
    cap.set(3, CAMERA_WIDTH)  # 帧宽度
    cap.set(4, CAMERA_HEIGHT)  # 帧高度
    cap.set(5, 30)  # fps

    interpreter = load_model(model_path)  # 加载模型
    labels = ['apple', 'banana', 'orange']  # 标签
    # labels = load_labels(label_path) # 加载标签

    input_details = interpreter.get_input_details() # 获取输入参数

    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']

    # Process Stream
    while True:
        ret, frame = cap.read()

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((width, height))

        top_result = process_image(interpreter, image, input_index)
        display_result(top_result, frame, labels)

        key = cv2.waitKey(1)
        if key == 27:  # esc
            break

    cap.release()
    cv2.destroyAllWindows()