import numpy as np
from matplotlib import pyplot as plt
import time


def proximity_matrix(norm_boxj, norm_boxi):
    P = np.sum(np.abs(np.array(norm_boxi) - np.array(norm_boxj)), axis=0)
    return P

def norm_matrix(dets, min_XT, max_XT, min_YT, max_YT, i):
    mask_X = (min_XT < min_XT[i]).astype(int)
    mask_reverse_X = 1 - mask_X
    min_X = mask_X * min_XT + mask_reverse_X * min_XT[i]

    mask_X = (max_XT > max_XT[i]).astype(int)
    mask_reverse_X = 1 - mask_X
    max_X = mask_X * max_XT + mask_reverse_X * max_XT[i]

    mask_Y = (min_YT < min_YT[i]).astype(int)
    mask_reverse_Y = 1 - mask_Y
    min_Y = mask_Y * min_YT + mask_reverse_Y * min_YT[i]

    mask_Y = (max_YT > max_YT[i]).astype(int)
    mask_reverse_Y = 1 - mask_Y
    max_Y = mask_Y * max_YT + mask_reverse_Y * max_YT[i]

    x11_norm = (dets[:,0] - min_X) / (max_X - min_X)
    x12_norm = (dets[:,2] - min_X) / (max_X - min_X)

    y11_norm = (dets[:,1] - min_Y) / (max_Y - min_Y)
    y12_norm = (dets[:,3] - min_Y) / (max_Y - min_Y)

    x21_norm = (dets[i][0] - min_X) / (max_X - min_X)
    x22_norm = (dets[i][2] - min_X) / (max_X - min_X)

    y21_norm = (dets[i][1] - min_Y) / (max_Y - min_Y)
    y22_norm = (dets[i][3] - min_Y) / (max_Y - min_Y)


    return [x11_norm, x12_norm, y11_norm, y12_norm], [x21_norm, x22_norm, y21_norm, y22_norm]


def confluence_fast(dets, iou_threshold = 0.5):

    min_conf_thres=0.05
    size_I = 1000000

    num = dets.shape[0]
    suppression = dets[:,-1] < min_conf_thres
    suppression = suppression.astype(int)
    index = np.array([i for i in range(num)])

    keep = []

    X = dets[:,[0,2]]
    Y = dets[:,[1,3]]

    min_XT = np.min(X, axis = 1)
    max_XT = np.max(X, axis = 1)
    min_YT = np.min(Y, axis = 1)
    max_YT = np.max(Y, axis = 1)

    t0 = time.time()
    for i in range(num):
        
        if suppression[i] == 1:
            continue

        optimalConfluence = size_I
        select_b = 0

        #Compute confluence
        norm_boxj, norm_boxi = norm_matrix(dets, min_XT, max_XT, min_YT, max_YT, i)
        P = proximity_matrix(norm_boxj, norm_boxi)
        c_mask = P < 2

        c_mask[suppression > 0] = False
        c_mask[i] = False

        if np.sum(c_mask) < 1:
            continue

        
 
        confluence = (P / dets[:,-1])[c_mask]
        cluster = index[c_mask]
        
        #Choose the best box in the blob
        select_b = cluster[np.argmin(confluence)]

        suppression[select_b] = 1
        keep.append(select_b)

        #Supression
        norm_boxj, norm_boxi = norm_matrix(dets, min_XT, max_XT, min_YT, max_YT, select_b)
        P = proximity_matrix(norm_boxj, norm_boxi)
        suppression[P < 1.0] = 1

    print("total time:", time.time() - t0)

    return keep

