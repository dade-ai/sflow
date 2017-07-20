# # -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd
# import cv2
# import os
#
#
# # read meta table
# df = pd.read_table('/data/face/celeba/Anno/list_landmarks_align_celeba.txt', header=1, delim_whitespace=True)
#
# # center of left eye
# pt1 = np.concatenate([np.expand_dims(df.lefteye_x.values, axis=1),
#                       np.expand_dims(df.lefteye_y.values, axis=1)], axis=1).astype(np.float32)
#
# # center of right eye
# pt2 = np.concatenate([np.expand_dims(df.righteye_x.values, axis=1),
#                       np.expand_dims(df.righteye_y.values, axis=1)], axis=1).astype(np.float32)
#
# # center of mouth
# pt3 = (np.concatenate([np.expand_dims(df.leftmouth_x.values, axis=1),
#                       np.expand_dims(df.leftmouth_y.values, axis=1)], axis=1).astype(np.float32) +
#        np.concatenate([np.expand_dims(df.rightmouth_x.values, axis=1),
#                       np.expand_dims(df.rightmouth_y.values, axis=1)], axis=1).astype(np.float32)) / 2
#
# # mean value
# ptm = np.asarray([np.mean(pt1, axis=0), np.mean(pt2, axis=0), np.mean(pt3, axis=0)])
#
# folder = '/data/face/celeba/process'
#
# # make directory for saving pre-processed images
# if not os.path.exists(os.path.join(folder, 'resized')):
#     os.makedirs(os.path.join(folder, 'resized'))
#     os.makedirs(os.path.join(folder, 'gray'))
#     os.makedirs(os.path.join(folder, 'sketch'))
#
# # do pre-processing
# for i in range(len(df)):
#
#     # load image
#     img = cv2.imread('/data/face/celeba/Img/img_align_celeba/' + df.index.values[i])
#
#     # get center point
#     pts = np.asarray([pt1[i], pt2[i], pt3[i]])
#
#     # check deviation from mean
#     if np.sum(np.square(pts - ptm)) < 5:
#
#         # do affine transform to align face image to the center
#         M = cv2.getAffineTransform(pts, ptm)
#         img_resize = cv2.warpAffine(img, M, (178, 218))  # [86:-36, 41:-41]
#
#         # make sketch image
#         img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
#         img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
#         img_sketch = cv2.divide(img_gray, img_blur, scale=256)
#
#         # save result
#         cv2.imwrite(os.path.join(folder, 'resized', df.index.values[i]), img_resize)
#         cv2.imwrite(os.path.join(folder, 'gray', df.index.values[i]), img_gray)
#         cv2.imwrite(os.path.join(folder, 'sketch', df.index.values[i]), img_sketch)
#
#         # logging
#         print '%s was processed.(%d/%d) ' % (df.index.values[i], i, len(df))
#
#
