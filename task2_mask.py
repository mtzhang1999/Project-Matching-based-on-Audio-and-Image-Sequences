import os
import cv2
import numpy as np

def premask(path):
    img = cv2.imread(path, 0)
    img = img / 255
    np.set_printoptions(threshold = np.inf)
    sum = 0
    S_x = 0
    S_y = 0
    for i in range(440):
        for j in range(440):
            sum = sum + img[i, j]
            S_x = S_x + i * img[i, j]
            S_y = S_y + j * img[i, j]

    center_x = int(S_x / sum)
    center_y = int(S_y / sum)
    return center_x, center_y



    
def mask_test(root_path):
    for parent, _, filenames in os.walk(root_path + '/mask'):  
        for filename in filenames:
            x_1, y_1 = premask(os.path.join(parent, filenames[1]))
            x_2, y_2 = premask(os.path.join(parent, filenames[-1]))
            move = [x_2-x_1 , y_2-y_1]
            return move




