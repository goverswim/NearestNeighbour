from functions import numpy_nn_get_neighbors
import os, sys
from sklearn.model_selection import train_test_split
import joblib

def load_dataset(dataset):
    try:
        data_path = os.path("dataset")
    except KeyError as e:
        print(f"Invalid dataset name {dataset}", file=sys.stderr)
        sys.exit(1)
    try:
        print(f"Loading dataset {dataset} from")
        print(f"    {data_path}")
        x, y = joblib.load(data_path)
    except IOError:
        print("Data not found", file=sys.stderr)
        print(data_path, file=sys.stderr)
        sys.exit(2)

    x, y = x.astype(np.float32), y.astype(int)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,
            random_state=0)

    return xtrain, xtest, ytrain, ytest
xtrain, xtest, ytrain, ytest = load_dataset("./data/spambase.gz")


numpy_nn_get_neighbors(xtrain, ytest, 5)