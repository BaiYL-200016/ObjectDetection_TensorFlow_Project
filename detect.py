import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# # Load the labels into a list
# classes = ['???'] * model.model_spec.config.num_classes
# label_map = model.model_spec.config.label_map
# for label_id, label_name in label_map.as_dict().items():
#   classes[label_id-1] = label_name
classes = ['apple', 'banana', 'orange']

# 定义用于可视化的颜色列表
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


def preprocess_image(image_path, input_size):
    """预处理输入图像以馈送到 TFLite 模型"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def detect_objects(interpreter, image, threshold):
    """返回检测结果列表，每个结果都是对象信息的字典。"""

    signature_fn = interpreter.get_signature_runner()

    # 将输入图像输入模型
    output = signature_fn(images=image)

    # 获取模型的所有输出
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


def run_odt_and_draw_results(img, interpreter, threshold=0.5):
    """对输入图像进行物体检测并绘制检测结果"""
    # 加载模型所需的输入形状
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # 加载输入图像并对其进行预处理
    preprocessed_image, original_image = preprocess_image(
        img,
        (input_height, input_width)
    )

    # 在输入图像上运行对象检测
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # 在输入图像上绘制检测结果
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # 将对象边界框从相对坐标转换为绝对坐标
        # 基于原始图像分辨率的坐标
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # 查找当前对象的类索引
        class_id = int(obj['class_id'])

        # 在图像上绘制边界框和标签
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # 进行调整以使标签对所有对象可见
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
        cv2.putText(original_image_np, label, (xmin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 返回最终图像
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


if __name__ == '__main__':
    # 开始加载 tflite 来预测图像
    DETECTION_THRESHOLD = 0.5
    model_path = 'model.tflite'
    TEMP_FILE = '01.jpg'

    # 开始加载 tflite 来预测图像
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 在原始文件的本地副本上运行推理并绘制检测结果
    detection_result_image = run_odt_and_draw_results(
        TEMP_FILE,
        interpreter,
        threshold=DETECTION_THRESHOLD
    )

    # 显示检测结果
    image = Image.fromarray(detection_result_image)
    image.show()

    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     DETECTION_THRESHOLD = 0.5
    #     model_path = 'model.tflite'
    #
    #     interpreter = tf.lite.Interpreter(model_path=model_path)
    #     interpreter.allocate_tensors()
    #
    #     detection_result_image = run_odt_and_draw_results(frame, interpreter, threshold=DETECTION_THRESHOLD)
    #     image = Image.fromarray(detection_result_image)
    #     cv2.imshow('frame', detection_result_image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
