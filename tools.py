import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def bbox_convert_vert_to_xywh(bbox):
    # [[x0,y0], [x1,x1]] -> [x0,y0,w,h]
    return (bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        # torch.backends.cudnn.deterministic = True  #needed
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


def dict_to_numpy(d):
    return {
        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in d.items()
    }

def tf_collate_fn(batch):
        x, y = zip(*batch)
        x = torch.stack(x).permute(0, 3, 1, 2).type(torch.FloatTensor)
        y = torch.stack(y)
        return x, y

def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

def draw_bbx(bbx, image_path, text):

    bbx = (np.array(bbx)).astype(int).tolist()
    # b_box upper left
    ptLeftTop = np.array(bbx[0])
    # b_box lower right
    ptRightBottom =np.array(bbx[1])
    # bbox color
    point_color = (0, 255, 0)
    thickness = 2
    lineType = 4

    src = cv2.imread(image_path)
    
    src = np.array(src)
    cv2.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)


    t_size = cv2.getTextSize(text, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]

    textlbottom = ptLeftTop + np.array(list(t_size))

    cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color, -1)

    ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)

    cv2.putText(src, text , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    cv2.imwrite(image_path[:-4]+'_bbox.png', src)