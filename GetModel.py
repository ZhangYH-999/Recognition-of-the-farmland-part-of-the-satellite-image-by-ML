import os
import pickle
import pandas as pd
from PIL import Image
from ReadTIF import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

seed = 13
columns = ['R_mean', 'G_mean', 'B_mean', 'H_mean', 'H_std', 'S_mean', 'S_std', 'V_mean', 'V_std']


def decisiontree(csv_path):

    print("loading " + csv_path + "...")

    X = pd.read_csv(csv_path, usecols=columns)
    Y = pd.read_csv(csv_path, usecols=['label'])

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=seed)

    min_samples = int(0.001 * len(X_train))

    print("training model...")

    DecisionTree = DecisionTreeClassifier(random_state=seed, min_samples_leaf=min_samples,
                                          min_samples_split=min_samples)

    DecisionTree.fit(X_train, y_train)

    print("saving model...")

    print('Train data score (average accuracy):', DecisionTree.score(X_train, y_train))
    print('Test data score (average accuracy): ', DecisionTree.score(X_test, y_test))

    with open("./model/DecisionTree.pkl", "wb") as f:
        pickle.dump(DecisionTree, f)

    print("done!")


def randomforest(csv_path):

    print("loading " + csv_path + "...")

    X = pd.read_csv(csv_path, usecols=columns)
    Y = pd.read_csv(csv_path, usecols=['label'])

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=seed)

    min_samples = int(0.001 * len(X_train))

    print("training model...")

    RandomForest = RandomForestClassifier(random_state=seed, min_samples_leaf=min_samples,
                                          min_samples_split=min_samples)

    RandomForest.fit(X_train, np.ravel(y_train))

    print('Train data score (average accuracy): ', RandomForest.score(X_train, y_train))
    print('Test data score (average accuracy):  ', RandomForest.score(X_test, y_test))

    print("saving model...")

    with open("./model/RandomForest.pkl", "wb") as f:
        pickle.dump(RandomForest, f)

    print("done!")


def evaluate(model_path):

    print("loading " + model_path + "...")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    image_path = "./dataset/image/data"
    label_path = "./dataset/image/label"
    image_list = [(image_path + "/" + x) for x in os.listdir(image_path) if x.endswith(".tif")]
    label_list = [(label_path + "/" + x) for x in os.listdir(label_path) if x.endswith(".tif")]
    average = 0

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

        reference = label_list[x]
        cut_path = "./results/cut/Data" + str(x + 1) + "_cut.tif"
        canny_path = "./results/edge/Data" + str(x + 1) + "_canny.tif"
        result = "./results/label/Data" + str(x + 1) + "_out.tif"
        result_view = "./results/label_view/Data" + str(x + 1) + "_out.tif"

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

        print("rate:", np.sum(label == 255) / (label.shape[0] * label.shape[1]))

        imageio.imwrite(result_view, label)

        label[label == 255] = 1

        imageio.imwrite(result, label)

        Real = readTIF(reference)

        T = 0
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                T += (label[i, j] == Real[i, j])
        average += T / (label.shape[0] * label.shape[1])
        print("accuracy:", T / (label.shape[0] * label.shape[1]))

    print('average accuracy:', average / (len(image_list)))
    print("done!")
