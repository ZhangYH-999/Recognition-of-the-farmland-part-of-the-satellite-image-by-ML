import sys
import getopt
import Predict
import GetModel
import GetDataSet


def help():
    print("You can use '--help' to get help.")
    print("--retrain [model_type] [pad]: training model after regenerating dataset with specified pad")
    print("--evaluate [model_path]: evaluate the model")
    print("--predict [model_path] [file_path]: using model to predict image")


def main(argv):
    try:
        if argv[0] == "--help":
            help()
        elif argv[0] == "--retrain":
            model_type = argv[1]
            pad = int(argv[2])
            csv_path = GetDataSet.get_csv("./dataset/image/data", "./dataset/image/label", pad)
            if model_type == "DecisionTree":
                GetModel.decisiontree(csv_path)
            elif model_type == "RandomForest":
                GetModel.randomforest(csv_path)
        elif argv[0] == "--evaluate":
            model_path = argv[1]
            GetModel.evaluate(model_path)
        elif argv[0] == "--predict":
            model_path = argv[1]
            image_path = argv[2]
            Predict.predict(image_path, model_path)
        else:
            help()
    except getopt.GetoptError:
        help()


if __name__ == "__main__":
   main(sys.argv[1:])
