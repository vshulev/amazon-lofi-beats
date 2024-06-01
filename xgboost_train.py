#PSUEDOCODE UNTIL WE GET DATA
import pandas
import xgboost
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def train(args):
    # load data
    df = pd.read_csv(args['input_path'], sep='\t')
    # train_csv = train_csv['sample_id', 'nucraw', 'coord', 'country', ]
    # dataset = data.values

    # split data into X and y
    X = df.drop(columns=['genus'])
    Y = df['genus']

    # encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(Y)
    label_encoded_y = label_encoder.transform(Y)

    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)
    print(model)

    # make predictions for test data
    y_pred = model.predict(X_test)
    # y_pred_str = label_encoder.inverse_transform(y_pred)

    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)

    args = vars(parser.parse_args())

