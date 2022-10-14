import os
import pickle
from PIL import Image
from ReadTIF import *


def predict(image_path, model_path):

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    image_name = [x for x in os.listdir(image_path) if x.endswith(".tif")]
    image_list = [(image_path + "/" + x) for x in os.listdir(image_path) if x.endswith(".tif")]

    tmp_path = os.path.abspath(os.path.dirname(__file__)) + "/segment/bin/release/tmp"
    segment_path = os.path.abspath(os.path.dirname(__file__)) + "/segment/bin/release/segment"
    Data_path = tmp_path + "/Data.ppm"
    Canny_path = tmp_path + "/Canny.ppm"
    Cut_path = tmp_path + "/cut.ppm"
    argv = "0.3 200 25"
    command = segment_path + " " + argv + " " + Data_path + " " + Canny_path + " " + Cut_path

    for x in range(0, len(image_list)):
        tif = image_list[x]

        print("evaluating " + tif + "...")

        cut_path = "./results/cut/" + image_name[x] + "_cut.tif"
        canny_path = "./results/edge/" + image_name[x] + "_canny.tif"
        result = "./results/label/" + image_name[x] + "_out.tif"
        result_view = "./results/label_view/" + image_name[x] + "_out.tif"

        print("edge detection...")

        image = readTIF(tif)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(Data_path, image)
        canny = image.copy()
        canny = cv2.cvtColor(canny, cv2.COLOR_RGB2GRAY)
        canny = cv2.bilateralFilter(canny, 9, 100, 20)
        canny = cv2.Canny(canny, 0, 40)
        cv2.imwrite(canny_path, canny)
        canny = cv2.imread(canny_path)
        cv2.imwrite(canny_path, canny)
        canny = imageio.imread(canny_path)
        imageio.imwrite(Canny_path, canny)

        os.system(command)
        cut = Image.open(Cut_path)
        cut.save(cut_path)

        image = readTIF(tif)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
        HSV = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
        image_cut = imageio.imread(cut_path)
        data = {}
        index = {}
        for i in range(image_cut.shape[0]):
            for j in range(image_cut.shape[1]):
                temp = str(image_cut[i, j, :])
                if temp in data.keys():
                    index[temp].append([i, j])
                    data[temp].append([image[i, j, 0], image[i, j, 1], image[i, j, 2],
                                       HSV[i, j, 0], HSV[i, j, 1], HSV[i, j, 2]])
                else:
                    index[temp] = [[i, j]]
                    data[temp] = [[image[i, j, 0], image[i, j, 1], image[i, j, 2],
                                   HSV[i, j, 0], HSV[i, j, 1], HSV[i, j, 2]]]

        label = np.zeros_like(img)
        for key in data.keys():
            data[key] = np.array(data[key])
            index[key] = np.array(index[key])
            R_mean = np.mean(data[key][:, 0])
            G_mean = np.mean(data[key][:, 1])
            B_mean = np.mean(data[key][:, 2])
            H_mean = np.mean(data[key][:, 3])
            S_mean = np.mean(data[key][:, 4])
            V_mean = np.mean(data[key][:, 5])
            H_std = np.std(data[key][:, 3])
            S_std = np.std(data[key][:, 4])
            V_std = np.std(data[key][:, 5])
            lab = (model.predict(np.array([[R_mean, G_mean, B_mean,
                                            H_mean, H_std, S_mean, S_std, V_mean, V_std]])))[0]
            for i in range(len(index[key])):
                label[index[key][i][0], index[key][i][1]] = lab

        label[label == 1] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        label = cv2.morphologyEx(label, cv2.MORPH_OPEN, kernel)
        label = cv2.morphologyEx(label, cv2.MORPH_CLOSE, kernel)

        print("area:", np.sum(label == 255) / (label.shape[0] * label.shape[1]))

        imageio.imwrite(result_view, label)

        label[label == 255] = 1

        imageio.imwrite(result, label)

    print("done!")
