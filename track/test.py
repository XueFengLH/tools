import onnxruntime as ort
import numpy as np
import cv2

# 加载ONNX模型
model_path = 'your_model.onnx'
session = ort.InferenceSession(model_path)

# 加载图片
image_path = 'your_image.jpg'
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# 预处理图片（这里假设模型需要640x640的输入）
input_size = (640, 640)
input_image = cv2.resize(image, input_size)
input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
input_image = input_image.astype('float32') / 255.0
input_image = np.expand_dims(input_image, axis=0)

# 进行推理
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
detections = session.run([output_name], {input_name: input_image})[0]
detect = Detect()
output = detect.postprocess1(result[0],multi_label=False,max_det=1000)

# 解析输出（假设输出格式为 [batch_size, num_boxes, 5]，其中最后一个维度是 [x, y, w, h, score]）
detections = detections[0]  # 取出第一个batch的结果
for detection in detections:
    x, y, w, h, score = detection
    if score > 0.5:  # 过滤掉低置信度的检测
        left = int((x - w / 2) * image_width)
        top = int((y - h / 2) * image_height)
        right = int((x + w / 2) * image_width)
        bottom = int((y + h / 2) * image_height)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# 保存结果图片
output_image_path = 'result.jpg'
cv2.imwrite(output_image_path, image)
print(f'结果图片已保存到 {output_image_path}')
