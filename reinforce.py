import numpy as np


# print(data.shape)
# 数据格式：    样本16432、通道（x,y,z）、帧数(300)、关节（0-16）17个关节、人数（统一为2）

def normalizez(data):  # -10~10
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val != min_val:
        return -10 + (data - min_val) * (20 / (max_val - min_val))
    else:
        return np.full(data.shape, 0)


def normalizex(data):  # -5到5
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val != min_val:
        return -5 + (data - min_val) * (10 / (max_val - min_val))
    else:
        return np.full(data.shape, 0)


def normalizey(data):  # -3到3
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val != min_val:
        return -3 + (data - min_val) * (6 / (max_val - min_val))
    else:
        return np.full(data.shape, 0)


# 数据格式：    样本16432、通道（x,y,z）、帧数(300)、关节（0-16）17个关节、人数（统一为2）
def aa():
    data = np.load(
        r"data/uav/test_joint_B1.npy")
    for i in range(len(data)):  # 样本
        for j in range(len(data[i])):  # 通道
            for k in range(len(data[i][j])):  # 帧数
                x1 = data[i][0][k][:, 0]
                y1 = data[i][1][k][:, 0]
                z1 = data[i][2][k][:, 0]

                # people2
                x2 = data[i][0][k][:, 1]
                y2 = data[i][1][k][:, 1]
                z2 = data[i][2][k][:, 1]

                data[i][0][k][:, 0] = normalizex(x1)
                data[i][1][k][:, 0] = normalizey(y1)
                data[i][2][k][:, 0] = normalizez(z1)

                data[i][0][k][:, 1] = normalizex(x2)
                data[i][1][k][:, 1] = normalizey(y2)
                data[i][2][k][:, 1] = normalizez(z2)
    np.save(r'data/uav/test_joint_B.npy', data)
if __name__ == '__main__':
    aa()