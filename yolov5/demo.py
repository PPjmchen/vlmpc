import torch
from PIL import Image
import cv2
import numpy as np

# 加载 YOLOv5 模型
model = torch.hub.load('./', 'custom', path='./runs/train/adamW_yolov5s_newval/weights/epoch1500.pt', source="local")

model.conf = 0.5  # 设置置信度阈值

image_path = '/home/zwt/vlmpc/logs/det_error_yolo_zoom003_text_obstacle_bbxbottom_freq6_2024-07-08_11-20-57/frame_1359.png'
image = Image.open(image_path)
results = model(image)

detections = results.pred[0]
import ipdb;ipdb.set_trace()

image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

classes = model.names

for *box, conf, cls in detections:
    x1, y1, x2, y2 = map(int, box)  # 获取 bounding box 坐标并转换为整数
    confidence = conf.item()  # 获取置信度
    class_id = int(cls.item())  # 获取类别 ID
    print(f"class: {classes[class_id]}, bounding box: {(x1, y1, x2, y2)}")

    image_cv = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
    # 绘制 bounding box
    label = f'{classes[class_id]} {confidence:.2f}'
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制矩形框
    cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # 绘制标签

    output_image_path = f'./test/{classes[class_id]}.png'
    cv2.imwrite(output_image_path, image_cv)

# # 保存带有 bounding box 的图像
# output_image_path = '/home/cjm/Projects/vlmpc/yolov5/1_detect.png'
# cv2.imwrite(output_image_path, image_cv)

print(f'Detection results saved to {output_image_path}')

