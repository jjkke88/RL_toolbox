__author__ = 'Air'
import  cv2
import numpy as np
import math

def lookat(img, center_x, center_y, new_heigh, fov):
    img_width = img.shape[1]
    img_heigh = img.shape[0]

    warp_temp = []
    for i in range(0, new_heigh):
        warp_temp.append(i+1)
    warp_x = []
    for i in range(0, new_heigh):
        warp_x.append(warp_temp)
    warp_x = np.array(warp_x)
    warp_y = warp_x.T

    TX = warp_x.reshape((-1, 1))
    TY = warp_y.reshape((-1, 1))

    TX = (TX - 0.5 - new_heigh/2)
    TY = (TY - 0.5 - new_heigh/2)

    r = new_heigh/2.0 / math.tan(fov/2)

    R = (TY**2 + r*r)**0.5

    ANG_y = np.arctan(-TY/r)
    ANG_y += center_y

    X = np.sin(ANG_y)*R
    Y = -np.cos(ANG_y)*R
    Z = TX

    ANG_x = np.arctan(Z/ -Y)
    RZY = (Z**2 + Y**2)**0.5
    ANG_y = np.arctan(X / RZY)

    for i in range(ANG_y.shape[0]):
        for j in range(ANG_y.shape[1]):
            if abs(ANG_y[i, j]) > 3.141592653/2:
                ANG_x[i, j] += 3.141592653

    ANG_x += center_x

    for i in range(ANG_y.shape[0]):
        for j in range(ANG_y.shape[1]):
            if ANG_y[i, j] < -3.141592653/2:
                ANG_y[i, j] = -ANG_y[i, j] - 3.141592653
                ANG_x[i, j] += 3.141592653

    for i in range(ANG_x.shape[0]):
        for j in range(ANG_x.shape[1]):
            if ANG_x[i, j] <= 3.141592653:
                ANG_x[i, j] += 2*3.141592653

    for i in range(ANG_x.shape[0]):
        for j in range(ANG_x.shape[1]):
            if ANG_x[i, j] > 3.141592653:
                ANG_x[i, j] -= 2*3.141592653

    for i in range(ANG_x.shape[0]):
        for j in range(ANG_x.shape[1]):
            if ANG_x[i, j] > 3.141592653:
                ANG_x[i, j] -= 2*3.141592653

    for i in range(ANG_x.shape[0]):
        for j in range(ANG_x.shape[1]):
            if ANG_x[i, j] > 3.141592653:
                ANG_x[i, j] -= 2*3.141592653

    Px = (ANG_x + 3.141592653) / (2* 3.141592653) * img_width + 0.5
    Py = ((-ANG_y) + 3.141592653/2) / 3.141592653 * img_heigh + 0.5

    for i in range(Px.shape[0]):
        for j in range(Px.shape[1]):
            if Px[i, j] < 1:
                Px[i, j] += img_width

    Px.shape = [new_heigh, new_heigh]
    Py.shape = [new_heigh, new_heigh]

    img = np.hstack((img, img[:,0:2]))

    Px = Px.astype('float32')
    Py = Py.astype('float32')
    img = cv2.remap(img,Px,Py, cv2.INTER_CUBIC)

    return img

