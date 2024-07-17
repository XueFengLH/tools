from email.mime import image

import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
def box_iou(box1, box2):
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM

def numpy_nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()
    keep = []
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep
class Detect(object):
    def __init__(self):
        self.conf_thres = 0.4
        self.filter_classes = None
        self.classes = ['face']
        self.agnostic = False
        self.iou_thres = 0.45
    def postprocess1(
            self,
            prediction,
            multi_label=False,
            max_det=1000,
    ):
        """
        Post-process the network's output, to get the
        bounding boxes, key-points and their confidence scores.
        """

        """Runs Non-Maximum Suppression (NMS) on inference results.
        Args:
            prediction: (tensor), with shape [N, 15 + num_classes], N is the number of bboxes.
            multi_label: (bool), when it is set to True, one box can have multi labels, 
                                                otherwise, one box only huave one label.
            max_det:(int), max number of output bboxes.
        Returns:
            list of detections, echo item is one tensor with shape (num_boxes, 16), 
                                                16 is for [xyxy, ldmks, conf, cls].
        """

        num_classes = prediction.shape[2] - 15  # number of classes
        pred_candidates = np.logical_and(
            prediction[..., 14] > self.conf_thres,
            np.max(prediction[..., 15:], axis=-1) > self.conf_thres,
        )

        # Function settings.
        max_wh = 4096  # maximum box width and height
        max_nms = (
            30000  # maximum number of boxes put into torchvision.ops.nms()
        )
        multi_label &= num_classes > 1  # multiple labels per box

        output = [np.zeros((0, 16))] * prediction.shape[0]

        for img_idx, x in enumerate(
                prediction
        ):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 15:] *= x[:, 14:15]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,16), each row represents (xyxy, conf, cls, lmdks)
            if multi_label:
                box_idx, class_idx = np.nonzero(x[:, 15:] > self.conf_thres).T
                x = np.concatenate(
                    (
                        box[box_idx],
                        x[box_idx, class_idx + 15, None],
                        class_idx[:, None].astype(np.float32),
                        x[box_idx, 4:14],
                    ),
                    1,
                )
            else:
                conf = np.max(x[:, 15:], axis=1, keepdims=True)
                class_idx = np.argmax(x[:, 15:], axis=1, keepdims=True)
                x = np.concatenate(
                    (box, conf, class_idx.astype(np.float32), x[:, 4:14]), 1
                )[conf.ravel() > self.conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if self.filter_classes:
                fc = [
                    i
                    for i, item in enumerate(self.classes)
                    if item in self.filter_classes
                ]
                x = x[(x[:, 5:6] == np.array(fc)).any(1)]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[
                    x[:, 4].argsort(descending=True)[:max_nms]
                ]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (
                0 if self.agnostic else max_wh
            )  # classes
            boxes, scores = (
                x[:, :4] + class_offset,
                x[:, 4],
            )  # boxes (offset by class), scores

            keep_box_idx = numpy_nms(boxes, scores, self.iou_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]
        return output
# 定义图像预处理函数
def preprocess(image_path):
    # 打开图像
    image = Image.open(image_path).convert('RGB')

    # 定义图像变换（假设模型需要输入大小为224x224的图像）
    transform = transforms.Compose([
        transforms.Resize(320),  # 调整图像大小
        transforms.CenterCrop(320),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量  # 归一化
    ])

    # 应用变换
    image = transform(image)

    # 增加批次维度
    image = image.unsqueeze(0)
    output_image = transforms.ToPILImage()(image[0])

    # 保存图像
    output_image.save('output_image.jpg')

    return image.numpy()


def face_detect(image_path):
    # 读取并预处理图像
    model_path = '/home/spring/anylabeling_data/models/yolov6lite_l_face-r20230520/yolov6lite_l_face.onnx'
    session = ort.InferenceSession(model_path)

    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # 预处理图片（这里假设模型需要640x640的输入）
    input_size = (320, 320)
    input_image = cv2.resize(image, input_size)
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_image = input_image.astype('float32') / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # 进行推理
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    detections = session.run([output_name], {input_name: input_image})[0]
    detect = Detect()
    output = detect.postprocess1(detections,multi_label=False,max_det=1000)

    # 解析输出（假设输出格式为 [batch_size, num_boxes, 5]，其中最后一个维度是 [x, y, w, h, score]）
    detections = output[0]  # 取出第一个batch的结果
    detections = detections[:,:5]
    for detection in detections:
        x, y, w, h, score = detection
        x = x / 320
        y = y / 320
        w = w / 320
        h = h / 320
        if score > 0.5:  # 过滤掉低置信度的检测
            left = int(x * image_width)
            top = int(y * image_height)
            right = int(w * image_width)
            bottom = int(h * image_height)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)

    # 保存结果图片
    output_image_path = 'result.jpg'
    cv2.imwrite(output_image_path, image)
    print(f'结果图片已保存到 {output_image_path}')
if __name__ == '__main__':
    face_detect('/mnt/sda3/video/1/VID_20240709_160908.mp4_2327.jpg')
