#PSUEDOCODE UNTIL WE GET DATA
import pandas as pd
import xgboost
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset


# def trainDNAEnv(args):
# load data
# ecoDf = pd.read_csv(args['input_path'], sep='\t')
ecoDf = pd.read_csv('/p/home/jusers/zhuge3/jureca/shared/amazon-lofi-beats/data/geo_eDNA_data_clean.csv', sep='\t')
dnaEmbeds = load_dataset("LofiAmazon/BOLD-Embeddings", split='train')
# train_csv = train_csv['sample_id', 'nucraw', 'coord', 'country', ]

ecoDF = ecoDf[ecoDf['marker_code' == 'COI-5P']]
ecoDf = ecoDf[['processid','nucraw','coord','country','depth',
    'WorldClim2_BIO_Temperature_Seasonality',
    'WorldClim2_BIO_Precipitation_Seasonality','WorldClim2_BIO_Annual_Precipitation', 'EarthEnvTopoMed_Elevation',
    'EsaWorldCover_TreeCover', 'CHELSA_exBIO_GrowingSeasonLength',
    'WCS_Human_Footprint_2009', 'GHS_Population_Density',
    'CHELSA_BIO_Annual_Mean_Temperature']]

# grab DNA embeddings and merge them onto ecoDf by processid
X = pd.merge(ecoDf, dnaEmbeds, on='processid', how='left')

# split data into X and y
# X = df.drop(columns=['genus'])
Y = ecoDf['genus']

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



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_path', action='store', type=str)

#     args = vars(parser.parse_args())

