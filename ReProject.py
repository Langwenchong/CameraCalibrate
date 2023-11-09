import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import random

# 计算重投影误差:


def CalcReprojError(data_path, output_path, num, world, cam, mtx, dist, rvecs, tvecs):
    errors = []
    file_list = os.listdir(data_path)
    if num == -1:
        num = len(file_list)

    for i in range(num):
        # 计算重投影坐标(dist=0不考虑畸变)
        # reproj, _ = cv2.projectPoints(world[i], rvecs[i], tvecs[i], mtx, dist)
        reproj = CalcProjectPoints(world[i], rvecs[i], tvecs[i], mtx, dist)
        # 原始坐标
        original_pts = cam[i].reshape(cam[i].shape[0], 2)
        # 重投影坐标
        reprojec_pts = reproj.reshape(reproj.shape[0], 2)
        # RMSE
        error = original_pts - reprojec_pts
        error = np.sum(error*error)**(1/2) / reproj.shape[0]
        # 等价:
        # error = cv2.norm(cam[i],reproj, cv2.NORM_L2) / reproj.shape[0]
        errors.append(error)
        # 重投影可视化
        img = cv2.imread(data_path+"/" + str(i+1)+'.jpg')
        # img,_,_ = utils.auto_reshape(img, 1920)
        drawReprojCor(img, original_pts, reprojec_pts, i, output_path)

    # 误差条形图
    # 可视化每张图像的误差(单位:像素)
    plt.bar(range(num), errors, width=0.8,
            label='reproject error', color='#87cefa')
    # 误差平均值(单位:像素)
    mean_error = sum(errors) / num
    plt.plot([-1, num], [mean_error, mean_error], color='r',
             linestyle='--', label='overall RMSE:%.3f' % (mean_error))
    plt.xticks(range(num), range(1, num+1))
    plt.ylabel('RMSE Error in Pixels')
    plt.xlabel('Images')
    plt.legend()
    plt.show()


# 重投影可视化


def drawReprojCor(img, original_pts, reprojec_pts, idx, output_path):
    r, g = (0, 0, 255), (0, 255, 0)

    for i in range(original_pts.shape[0]):
        # 原始角点
        x0, y0 = int(round(original_pts[i, 0])), int(round(original_pts[i, 1]))
        cv2.circle(img, (x0, y0), 3, g, 1, lineType=cv2.LINE_AA)
        # 重投影角点
        x1, y1 = int(round(reprojec_pts[i, 0])), int(round(reprojec_pts[i, 1]))
        cv2.circle(img, (x1, y1), 5, r, 1, lineType=cv2.LINE_AA)
    cv2.imwrite(output_path + "/"+str(idx+1)+'.jpg', img)

# 计算重投影坐标


def CalcProjectPoints(world, rvecs, tvecs, mtx, dist):
    # 旋转向量转旋转矩阵
    M = Rodriguez(rvecs)
    # c = Rw + t (世界坐标系转相机坐标系)
    R_t = (M @ world.T).T + tvecs
    # (相机坐标系到图像坐标系)
    # print(R_t)
    plain_pts = (mtx @ R_t.T)
    plain_pts = (plain_pts / plain_pts[2, :]).T[:, :2]

    # 去畸变
    c_xy = np.array([mtx[0, 2], mtx[1, 2]])
    f_xy = np.array([mtx[0, 0], mtx[1, 1]])

    k1, k2, p1, p2, k3 = dist[0]
    x_y = (plain_pts - c_xy) / f_xy
    r = np.sum(x_y * x_y, 1)

    x_distorted = x_y[:, 0] * (1 + k1*r + k2*r*r + k3*r*r*r) + \
        2*p1*x_y[:, 0]*x_y[:, 1] + p2*(r + 2*x_y[:, 0]*x_y[:, 0])
    y_distorted = x_y[:, 1] * (1 + k1*r + k2*r*r + k3*r*r*r) + \
        2*p2*x_y[:, 0]*x_y[:, 1] + p1*(r + 2*x_y[:, 1]*x_y[:, 1])
    u_distorted = f_xy[0]*x_distorted + c_xy[0]
    v_distorted = f_xy[1]*y_distorted + c_xy[1]
    plain_pts = np.array([u_distorted, v_distorted]).T
    return plain_pts

# 旋转向量转旋转矩阵


def Rodriguez(rvecs):
    # 旋转向量模长
    θ = (rvecs[0] * rvecs[0] + rvecs[1] *
         rvecs[1] + rvecs[2] * rvecs[2])**(1/2)
    # 旋转向量的单位向量
    r = rvecs / θ
    # 旋转向量单位向量的反对称矩阵
    anti_r = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])
    # 旋转向量转旋转矩阵(Rodriguez公式)     # np.outer(r, r) = r @ r.T 向量外积
    M = np.eye(3) * np.cos(θ) + (1 - np.cos(θ)) * \
        np.outer(r, r) + np.sin(θ) * anti_r
    return M

# 可视化标定过程中的相机位姿


def show_cam_pose(rvecs, tvecs):
    # 相机坐标系下基向量
    vec = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # 相机位姿模型
    cam = (2/3)*np.array([
        [1, 1, 2],
        [-1, 1, 2],
        [-1, -1, 2],
        [1, -1, 2],
        [1, 1, 2],
    ])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制每一个角度拍摄的相机
    for i in range(rvecs.shape[0]):
        # 旋转向量转旋转矩阵
        M = Rodriguez(rvecs[i, :])
        # 相机原点 w = R^(-1)(0 - t)
        x0, y0, z0 = M.T @ (-tvecs[i, :])
        c = ['r', 'g', 'b']
        # 随机颜色
        hex = '0123456789abcdef'
        rand_c = '#'+''.join([hex[random.randint(0, 15)] for _ in range(6)])
        # 绘制相机坐标系
        for j in range(3):
            # 相机位姿(相机坐标系转世界坐标系)
            # w = R^(-1)(c - t)
            x1, y1, z1 = M.T @ (vec[j, :] - tvecs[i, :])
            # 相机坐标系
            ax.plot([x0, x1], [y0, y1], [z0, z1], color=c[j])
        C = (M.T @ (cam - tvecs[i, :]).T).T
        # 绘制相机位姿
        for k in range(4):
            ax.plot([C[k, 0], C[k+1, 0]], [C[k, 1], C[k+1, 1]],
                    [C[k, 2], C[k+1, 2]], color=rand_c)
            ax.plot([x0, C[k+1, 0]], [y0, C[k+1, 1]],
                    [z0, C[k+1, 2]], color=rand_c)
        # 相机编号
        ax.text(x0, y0, z0, i+1)
    # 绘制棋盘格
    for i in range(9):
        ax.plot([0, 11], [-i, -i], [0, 0], color="black")
    for i in range(12):
        ax.plot([i, i], [-8, 0], [0, 0], color="black")
    # 绘制世界(棋盘格)坐标系
    for i in range(3):
        ax.plot([0, 3*vec[i, 0]], [0,  -3*vec[i, 1]],
                [0, 2*vec[i, 2]], color=c[i], linewidth=3)
    plt.xlim(-3, 14)
    plt.ylim(-13, 4)
    plt.show()


# 可视化标定过程中的棋盘位姿
def show_chessboard_pose(rvecs, tvecs):
    # 棋盘坐标系下基向量
    vec = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # 相机位姿模型
    C = np.array([
        [1, 1, 3],
        [-1, 1, 3],
        [-1, -1, 3],
        [1, -1, 3],
        [1, 1, 3],
    ])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = ['r', 'g', 'b']  # 坐标轴颜色
    # 绘制每一个角度拍摄的棋盘格
    for i in range(rvecs.shape[0]):
        # 旋转向量转旋转矩阵
        M = Rodriguez(rvecs[i, :])
        # 棋盘原点 c = Rw + t
        x0, y0, z0 = tvecs[i, :]
        # 随机颜色
        hex = '0123456789abcdef'
        rand_c = '#'+''.join([hex[random.randint(0, 15)] for _ in range(6)])
        # 绘制棋盘位姿
        for k in range(0, 9):
            b = np.array([[0, -k, 0], [11, -k, 0]])
            b = (M @ b.T + tvecs[i, :].reshape(-1, 1)).T
            ax.plot([b[0, 0], b[1, 0]], [b[0, 1], b[1, 1]],
                    [b[0, 2], b[1, 2]], color=rand_c)
        for k in range(0, 12):
            b = np.array([[k, -8, 0], [k, 0, 0]])
            b = (M @ b.T + tvecs[i, :].reshape(-1, 1)).T
            ax.plot([b[0, 0], b[1, 0]], [b[0, 1], b[1, 1]],
                    [b[0, 2], b[1, 2]], color=rand_c)
            k += 11
        # 绘制棋盘坐标系
        for j in range(3):
            # (世界坐标系转相机坐标系)
            # c = Rw + t
            x1, y1, z1 = M @ vec[j, :] + tvecs[i, :]
            ax.plot([x0, x1], [y0, y1], [z0, z1], color=c[j])
        # 棋盘编号
        ax.text(x0, y0, z0, i+1)
    # 绘制世界(相机)坐标系
    for i in range(3):
        ax.plot([0, vec[i, 0]], [0,  -vec[i, 1]],
                [0, 2*vec[i, 2]], color=c[i], linewidth=2)
    # 绘制相机位姿
    for i in range(4):
        ax.plot([C[i, 0], C[i+1, 0]], [C[i, 1], C[i+1, 1]],
                [C[i, 2], C[i+1, 2]], color="black")
        ax.plot([0, C[i+1, 0]], [0, C[i+1, 1]], [0, C[i+1, 2]], color="black")
    plt.show()


if __name__ == "__main__":
    # 原图片位置以及要输出的新图片位置
    data = "./camera02"
    root = data+"_output"
    param_path = root+'/param'
    output_path = root+"/reproject"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    world = np.load(param_path+"/world.npy")
    cam = np.load(param_path+"/cam.npy")
    mtx = np.load(param_path+"/mtx.npy")
    dist = np.load(param_path+"/dist.npy")
    rvecs = np.load(param_path+"/rvecs.npy")
    tvecs = np.load(param_path+"/tvecs.npy")

    CalcReprojError(data, output_path, -1, world, cam, mtx, dist, rvecs, tvecs)
    # 可视化棋盘的位姿(默认相机位置不变)
    # show_chessboard_pose(rvecs, tvecs)
    # 可视化当前的相机姿态(默认棋盘位置不变)
    show_cam_pose(rvecs, tvecs)
