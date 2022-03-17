from scipy.ndimage import map_coordinates as interp2
import numpy as np
from ReadCameraModel import ReadCameraModel
import os
import cv2
import matplotlib.pyplot as plt

# fx, fy, cx, cy, , LUT  = ReadCameraModel(‘./Oxford dataset reduced/model’)
# UndistortImage - undistort an image using a lookup table
# 
# INPUTS:
#   image: distorted image to be rectified
#   LUT: lookup table mapping pixels in the undistorted image to pixels in the
#     distorted image, as returned from ReadCameraModel
#
# OUTPUTS:
#   undistorted: image after undistortion



def UndistortImage(image,LUT):
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)

    
    return undistorted.astype(image.dtype)


fx, fy, cx, cy, g, LUT = ReadCameraModel('./Oxford_dataset_reduced/model')

k = np.zeros([3,3])
k[0][0] = fx
k[1][1] = fy
k[0][2] = cx
k[1][2] = cy
k[2][2] = 1
k_t = np.transpose(k)
dirs = os.listdir("./Oxford_dataset_reduced/images")
dirs.sort()
start = "./Oxford_dataset_reduced/images/"
imgs = []
for file in range(len(dirs)):
    filename = start + dirs[file]
    img = cv2.imread(filename, flags=-1)
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    imgs.append(color_image)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

fund_mats = []
essent_mats = []
rot_mats = []
t_mats = []

for i in range(len(imgs)-1):
    [f1, d1] = sift.detectAndCompute(imgs[i], None)
    [f2, d2] = sift.detectAndCompute(imgs[i+1], None)
    matches = bf.knnMatch(d1, d2, k=2)
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts2.append(f2[m.trainIdx].pt)
            pts1.append(f1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    fund_mats.append(F)
    # for a in range(len(pts1)):
    #     if mask[a] == 1:
    #         tmp_pr = np.ones([3])
    #         tmp_pr[0] = pts2[a][0]
    #         tmp_pr[1] = pts2[a][1]
    #         tmp = np.ones([3])
    #         tmp[0] = pts1[a][0]
    #         tmp[1] = pts1[a][1]
    #         g = np.matmul(tmp_pr,F)
    #         r = np.matmul(g, tmp)
    #         print(r)

    a = np.matmul(k_t,F)
    E = np.matmul(a,k)
    essent_mats.append(E)
    points, R, t, out_mask = cv2.recoverPose(E=E, points1=pts1, points2=pts2, cameraMatrix=k, mask = mask)
    rot_mats.append(R)
    t_mats.append(t)

x_init = np.zeros([4])
x_init[3] = 1
coords = x_init
cam_matrix = np.zeros([4,4])
cam_matrix[3][3] = 1

for row in range(3):
    for col in range(3):
        cam_matrix[row][col] = rot_mats[0][row][col]
    cam_matrix[row][3] = t_mats[0][row]
x_curr = np.linalg.solve(cam_matrix, x_init)
coords = np.vstack((coords, x_curr))

for i in range(1, len(rot_mats)):
    new_cam = np.zeros([4,4])
    for row in range(3):
        for col in range(3):
            new_cam[row][col] = rot_mats[i][row][col]
        new_cam[row][3] = t_mats[i][row]
    new_cam[3][3] = 1
    cam_matrix = np.matmul(new_cam, cam_matrix)
    x_curr = np.linalg.solve(cam_matrix, x_init)
    coords = np.vstack((coords, x_curr))


x_coords = coords[:, [0]]
y_coords = coords[:, [1]]
z_coords = coords[:, [2]]


plt.plot(x_coords, z_coords)
plt.show()

a = x_coords.flatten()
b = y_coords.flatten()
c = z_coords.flatten()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(a, b, c)
plt.show()



