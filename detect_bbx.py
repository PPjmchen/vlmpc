import torch
from PIL import Image
import cv2
import numpy as np

def det_bbox(image_path, model, obj_str):
    image = Image.open(image_path)
    results = model(image)
    detections = results.pred[0].cpu().numpy()
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    classes = model.names

    classes_dic = {"blue cube": 0,"blue moon": 1,"yellow pentagon": 2,"yellow star": 3,"red pentagon": 4,"red moon": 5,"green cube": 6,"green star": 7,"end effector": 8}
    class_id = classes_dic[obj_str]

    if not (detections[:,5]==class_id).any():
        return 0

    box_index = np.where(detections==class_id)[0][0]
    x1, y1, x2, y2 = map(int, detections[box_index][:4])
    confidence = detections[box_index][4]

    label = f'{obj_str} {confidence:.2f}'
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制矩形框
    cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # 绘制标签

    output_image_path = image_path[:-4]+f'_{obj_str}.jpg'
    cv2.imwrite(output_image_path, image_cv)

    return [[x1, y1],[x2, y2]]

if __name__ == "__main__":
    obj_str = "blue cube"
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/cjm/Projects/vlmpc/yolov5/runs/train/exp8/weights/best.pt')
    image_path = '/home/cjm/Projects/vlmpc/2.png'

    det_bbox(image_path=image_path, model=model, obj_str=obj_str)
