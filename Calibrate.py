import numpy as np
import cv2
import os
import shutil

# 求解内外参数

# 想计算参数矩阵，需要至少三张图，这是因为H形成的vij与B(A^(-T)*A^-1)的求解需要至少3个方程组(每个方程组提供两个方程共需要求解6个B的未知数，B是对称矩阵)见https://zhuanlan.zhihu.com/p/94244568
# 而H单应性变换矩阵的求解至少需要4组角点，即棋盘格的最少格点，原因见https://zhuanlan.zhihu.com/p/74597564

# 这里的w和h是指棋盘格的规格


def CamCalibrate(w, h, num, root):
    # 图像缩放比例(如果你的图像进行了缩放，与实际拍摄的分辨率不一致，最终求得的参数需要乘上这个比例进行校正)
    ratio = 1  # 3648 / 1920
    world, cam = [], []
    # 读取root文件夹并将文件全部重新命名为1.jpg->xx.jpg
    # 获取文件夹中的所有文件列表
    file_list = os.listdir(root)

    # 遍历文件列表并重命名文件
    for idx, filename in enumerate(file_list):
        if os.path.isfile(os.path.join(root, filename)):
            # 获取文件的扩展名
            file_extension = os.path.splitext(filename)[1]
            # 新的文件名
            new_filename = str(idx+1) + file_extension
            # 构建新的文件路径
            new_filepath = os.path.join(root, new_filename)
            if os.path.exists(new_filepath):
                continue
            # 重命名文件
            os.rename(os.path.join(root, filename), new_filepath)
    if num == -1:
        num = len(file_list)
    # 多张图像进行标定，减小误差:
    for i in range(num):
        img = cv2.imread(root + "/"+str(i+1)+'.jpg')
        # img,_,_ = utils.auto_reshape(img, 1920)
        # 定位角点
        is_success, cam_coord = find_chessboard_cor(img)
        print('第'+str(i+1)+'张角点提取完毕, 角点数 =', cam_coord.shape[0])
        # 可视化角点
        draw_chessboard_cor(img, cam_coord, is_success)
        # 角点的世界坐标:
        # 注:相机参数的计算只要求角点之间的世界坐标比例一致,因此可以单位化 https://blog.csdn.net/SESESssss/article/details/124893508?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-124893508-blog-104789585.235^v38^pc_relevant_sort_base2&spm=1001.2101.3001.4242.1&utm_relevant_index=3
        # 这里为什么世界坐标系可以使用自定义的一个度量单位保持比例就可以而无需和像素坐标系使用相同的度量呢？
        # 因为最终计算的所有参数结果都是以像素这个虚拟单位度量的，即可以全程都是用虚拟的像素单位来衡量一切物理量
        # 这里世界坐标系的人为定义的坐标与像素坐标系的像素建立换算关系是通过u,U(X),V(Y)和v,U(X),V(Y)建立约束方程建立的见 https://zhuanlan.zhihu.com/p/94244568，因此最终所有的物理量都是由像素表示的当然反过来也可以都用人为定义的这个世界坐标系的度量单位衡量
        # 但是这也导致了求出来的焦距/平移等都是以像素为单位的，如果想求得真实的距离，其实还是需要建立1像素和真实物理距离的换算(换算关系见https://blog.csdn.net/baidu_38172402/article/details/81949447)
        # 因此虽然有的教程要求上来就要衡量棋盘格的物理距离和像素进行换算才能直接求得用真实物理量表示的参数
        # 其实这里本质上就是建立了 人为定义的虚拟度量单位->像素->真实物理量三者之间的换算关系，其中前两者是通过相机的成像原理建立的关系，后两者是直接通过计算1像素=xxxmm建立的关系的，当然其实人为定义的度量单位和物理量也是默认建立了关系，即一个网格的真实物理长度=1人为定定义的单位
        world_coord = np.zeros((w * h, 3), np.float32)
        world_coord[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        world_coord[:, 1] = -world_coord[:, 1]
        # world_coord[:,:2] = np.mgrid[0:w*len:len,0:h*len:len].T.reshape(-1,2)
        # 将世界坐标与像素坐标加入待求解系数矩阵，注意是一个数组，数组中的每一个元素对应一张照片的所有匹配坐标对
        world.append(world_coord)
        cam.append(cam_coord)

    # 求解摄像机的内在参数和外在参数
    # ret 非0表示标定成功 mtx 内参数矩阵，dist 畸变系数，rvecs 旋转向量，tvecs 平移向量
    # 注:求解的结果的单位为像素,若想化为度量单位还需乘上每个像素代表的实际尺寸(如:毫米/像素)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        world, cam, (img.shape[1], img.shape[0]), None, None)
    rvecs = np.array(rvecs).reshape(-1, 3)
    tvecs = np.array(tvecs).reshape(-1, 3)
    output_path = root+'_output'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    log_path = output_path+"/log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        # 单位:像素(1像素=??mm) 这里需要结合相机的传感器物理尺寸与当前相片的分辨率进行计算得到
        print("标定结果 ret:", ret, file=f)
        print("内参矩阵 mtx:\n", mtx, file=f)    # 内参数矩阵
        # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        print("畸变系数 dist:\n", dist, file=f)
        # 旋转向量  # 外参数(欧拉角)是基于Rodriguez的变换
        print("旋转向量(外参) rvecs:\n", rvecs, file=f)
        print("平移向量(外参) tvecs:\n", tvecs, file=f)  # 平移向量  # 外参数

    param_path = output_path+"/param"
    if not os.path.exists(param_path):
        os.mkdir(param_path)
    # 数据至少要保留5为有效数字
    # 相近内参矩阵，多张图片参考生成唯一一个矩阵
    np.save(param_path+'/mtx.npy', mtx)
    # 多张图片参考生成唯一一个畸变
    np.save(param_path+'/dist.npy', dist)
    # 每张图片对应一个，表示当前摄像机参考本张图片的棋盘格(左上角为世界坐标系原点也是像素坐标系的原点)的旋转
    np.save(param_path+'/rvecs.npy', rvecs)
    # 每张图片对应一个，表示当前摄像机参考本张图片的棋盘格(左上角为世界坐标系原点也是像素坐标系的原点)的平移
    np.save(param_path+'/tvecs.npy', tvecs)
    # 每张图片对应一个即比例正确的世界坐标(实际上都一样)
    np.save(param_path+'/world.npy', np.array(world))
    # 每张图片对应一个本次棋盘格再图像坐标系下的成像坐标
    np.save(param_path+'/cam.npy', np.array(cam))
    return ret, mtx, dist, rvecs, tvecs


# 定位角点


def find_chessboard_cor(img):
    # 转为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OpenCV内置函数提取棋盘格角点
    is_success, corner = cv2.findChessboardCorners(gray_img, (8, 5), None)
    # 计算亚像素时停止迭代的标准
    # 后者表示迭代次数达到了最大次数时停止，前者表示角点位置变化的最小值已经达到最小时停止迭代
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # 亚像素角点检测，提高角点定位精度, (7, 7)为考虑角点周围区域的大小
    corner = cv2.cornerSubPix(gray_img, corner, (7, 7), (-1, -1), criteria)
    return is_success, corner

# 可视化角点


def draw_chessboard_cor(img, cor, is_success):
    cv2.drawChessboardCorners(img, (8, 5), cor, is_success)
    cv2.imshow('cor', img)
    cv2.waitKey(50)


if __name__ == "__main__":
    # 参考 https://blog.csdn.net/SESESssss/article/details/124893508?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-124893508-blog-104789585.235^v38^pc_relevant_sort_base2&spm=1001.2101.3001.4242.1&utm_relevant_index=3
    # 需要确定修改棋盘格数, (8,5)为棋盘格尺寸-1 (9x6) 格子是从左到右为x,从上到下为进行排列
    # num==-1则会使用相加文件夹下的所有图片
    CamCalibrate(8, 5, -1, "./camera02")
