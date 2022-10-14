import os
import pandas as pd
from ReadTIF import *


def get_csv(image_path, label_path, pad):
    csv_path = './dataset/csv/' + str(pad) + 'x' + str(pad) + '.csv'

    print("getting CSV...")

    image_list = [(image_path + "/" + x) for x in os.listdir(image_path) if x.endswith(".tif")]
    label_list = [(label_path + "/" + x) for x in os.listdir(label_path) if x.endswith(".tif")]

    data = []

    for i in range(0, len(image_list)):
        image = readTIF(image_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        HSV = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
        label = readTIF(label_list[i])
        if label.shape == (500, 600, 3):
            label = label[:, :, 0]
        height, width, channels = image.shape
        for x in range(0, height - pad, pad):
            for y in range(0, width - pad, pad):
                tmp = image[x: x + pad, y: y + pad, :]
                R_mean = np.mean(tmp[:, :, 0])
                R_std = np.std(tmp[:, :, 0])
                G_mean = np.mean(tmp[:, :, 1])
                G_std = np.std(tmp[:, :, 1])
                B_mean = np.mean(tmp[:, :, 2])
                B_std = np.std(tmp[:, :, 2])
                H_mean = np.mean(HSV[x: x + pad, y: y + pad, 0])
                S_mean = np.mean(HSV[x: x + pad, y: y + pad, 1])
                V_mean = np.mean(HSV[x: x + pad, y: y + pad, 2])
                H_std = np.std(HSV[x: x + pad, y: y + pad, 0])
                S_std = np.std(HSV[x: x + pad, y: y + pad, 1])
                V_std = np.std(HSV[x: x + pad, y: y + pad, 2])
                lab = label[x: x + pad, y: y + pad]

                if np.sum(lab == 1) == lab.shape[0] * lab.shape[1]:
                    data.append([R_mean, R_std, G_mean, G_std, B_mean, B_std,
                                 H_mean, H_std, S_mean, S_std, V_mean, V_std,
                                 1])
                elif np.sum(lab == 0) == lab.shape[0] * lab.shape[1]:
                    data.append([R_mean, R_std, G_mean, G_std, B_mean, B_std,
                                 H_mean, H_std, S_mean, S_std, V_mean, V_std,
                                 0])

    data = pd.DataFrame(np.array(data), columns=['R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std',
                                                 'H_mean', 'H_std', 'S_mean', 'S_std', 'V_mean', 'V_std',
                                                 'label'])
    data.to_csv(csv_path)
    return csv_path
