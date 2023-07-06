import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, hamming_loss
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1_l2
from sklearn.utils import class_weight
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Load the data
data_path = './datasets/group47/dataset/intermediate_datafiles/chapter5_group47_result.csv'
data = pd.read_csv(data_path, index_col=0)

# Define the target columns
target_columns = ['labelCycling', 'labelStairs', 'labelWalking', 'labelSitting', 'labelOther']

# Separate the features and target variables
X = data.drop(target_columns, axis=1)
y = data[target_columns]

# Fill missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Replace infinite values with the maximum finite float
X = X.replace([np.inf, -np.inf], np.finfo('float64').max)

selected_features = set()
clf = RandomForestClassifier()
for column in tqdm(y.columns, desc="Selecting features"):
    y_single_label = LabelEncoder().fit_transform(y[column])  # Convert to 1D array
    selector = RFE(clf, n_features_to_select=50, step=1)
    selector = selector.fit(X, y_single_label)
    selected_features.update(X.columns[selector.support_])

# Filter the features based on the selected features
X = pd.DataFrame(StandardScaler().fit_transform(X[list(selected_features)]), columns=list(selected_features))

# Use simple random sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)





# Define a function to create and compile a new model
def create_model(learning_rate=0.001, dropout_rate=0.2, neurons=100):
    model = Sequential()
    #model.add(LSTM(neurons, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(neurons, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), input_shape=(1, X_train.shape[1])))

    model.add(Dense(neurons, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(Dense(len(target_columns), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

# Define the random search parameters
param_dist = dict(learning_rate=[0.001, 0.01, 0.1], dropout_rate=[0.1, 0.2, 0.3], neurons=[50, 100, 150])

# Create Random Search
model = KerasClassifier(build_fn=create_model, verbose=0)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, n_jobs=-1, verbose=1)
#random_search_result = random_search.fit(X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1]), y_train) 
random_search_result = random_search.fit(X_train, y_train) 

# Compute class weights
class_weights_list = []
for column in y_train.columns:
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[column]), y_train[column])
    class_weights_dict = dict(enumerate(class_weights))
    class_weights_list.append(class_weights_dict)

# Fit the model with the training data
for i, column in enumerate(y_train.columns):
    model.fit(X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1]), y_train[column], validation_split=0.2, epochs=100, callbacks=[EarlyStopping(monitor='val_loss', patience=10)], class_weight=class_weights_list[i], verbose=1)

# Compute precision-recall curve
y_scores = model.predict(X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1]))
precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_scores.ravel())

# Select threshold that maximizes the F1 score
f1_scores = 2*recall*precision / (recall + precision)
best_threshold = thresholds[np.argmax(f1_scores)]

# Binarize the predictions
y_pred = np.where(y_scores > best_threshold, 1, 0)

# Compute metrics
def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    return precision, recall, f1

precision, recall, f1 = compute_metrics(y_test.values.ravel(), y_pred.ravel())
print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

# Compute Hamming loss
loss = hamming_loss(y_test.values.ravel(), y_pred.ravel())
print(f'Hamming Loss: {loss}')









































# import numpy as np
# import pandas as pd
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# from sklearn.multiclass import OneVsRestClassifier
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.regularizers import l1_l2
# from sklearn.model_selection import cross_val_score, KFold
# from keras.callbacks import EarlyStopping
# from tqdm import tqdm
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from sklearn.utils import class_weight
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import hamming_loss
# from sklearn.feature_selection import RFE

# # Load the data
# data_path = './datasets/group47/dataset/intermediate_datafiles/chapter5_group47_result.csv'
# data = pd.read_csv(data_path, index_col=0)

# # Define the target columns
# target_columns = ['labelCycling', 'labelStairs', 'labelWalking', 'labelSitting', 'labelOther']

# # Separate the features and target variables
# X = data.drop(target_columns, axis=1)
# y = data[target_columns]

# # Fill missing values with the mean of each column
# imputer = SimpleImputer(strategy='mean')
# X = imputer.fit_transform(X)

# # Replace infinite values with the maximum finite float
# X = np.where(np.isinf(X), np.finfo('float64').max, X)

# selected_features = set()
# clf = RandomForestClassifier()
# for column in tqdm(y.columns, desc="Selecting features"):
#     y_single_label = LabelEncoder().fit_transform(y[column])  # Convert to 1D array
#     selector = RFE(clf, n_features_to_select=50, step=1)
#     X_new = selector.fit_transform(X, y_single_label)
#     selected_features.update(X.columns[selector.get_support()])

# # Filter the features based on the selected features
# X = StandardScaler().fit_transform(X[list(selected_features)])

# # Use simple random sampling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Define a function to create and compile a new model
# def create_model(learning_rate=0.001, dropout_rate=0.2, neurons=100):
#     model = Sequential()
#     model.add(LSTM(neurons, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), input_shape=(X_train.shape[1], 1)))
#     model.add(Dense(neurons, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
#     model.add(Dense(len(target_columns), activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
#     return model

# # Define the random search parameters
# param_dist = dict(learning_rate=[0.001, 0.01, 0.1], dropout_rate=[0.1, 0.2, 0.3], neurons=[50, 100, 150])

# # Create Random Search
# model = KerasClassifier(build_fn=create_model, verbose=0)
# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, n_jobs=-1, verbose=1)
# random_search_result = random_search.fit(X_train, y_train) 

# # Compute class weights
# class_weights_list = []
# for column in y_train.columns:
#     class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[column]), y_train[column])
#     class_weights_dict = dict(enumerate(class_weights))
#     class_weights_list.append(class_weights_dict)

# # Fit the model with the training data
# for i, column in enumerate(y_train.columns):
#     model.fit(X_train, y_train[column], validation_split=0.2, epochs=100, callbacks=[early_stopping], class_weight=class_weights_list[i], verbose=1)

# # Reshape data for LSTM
# X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# # Create early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# # Fit the model with the training data
# model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

# # Compute precision-recall curve
# y_scores = model.predict(X_test)
# precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# # Select threshold that maximizes the F1 score
# f1_scores = 2*recall*precision / (recall + precision)
# best_threshold = thresholds[np.argmax(f1_scores)]

# # Binarize the predictions
# y_pred = np.where(y_scores > best_threshold, 1, 0)

# # Compute metrics
# def compute_metrics(y_true, y_pred):
#     precision = precision_score(y_true, y_pred, average='micro')
#     recall = recall_score(y_true, y_pred, average='micro')
#     f1 = f1_score(y_true, y_pred, average='micro')
#     return precision, recall, f1

# precision, recall, f1 = compute_metrics(y_test, y_pred)
# print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

# # Compute Hamming loss
# loss = hamming_loss(y_test, y_pred)
# print(f'Hamming Loss: {loss}')




























# # Define the grid search parameters
# learning_rate = [0.001, 0.01, 0.1]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# neurons = [50, 100, 150, 200]
# param_grid = dict(learning_rate=learning_rate, dropout_rate=dropout_rate, neurons=neurons)

# # Create Grid Search
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)
# grid_result = grid.fit(X_train, y_train)

# # Summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# # Define the cross validation iterator
# kfold = KFold(n_splits=10, shuffle=True)

# # Perform cross validation
# results = cross_val_score(model, X_train, y_train, cv=kfold)


# # Print the cross validation score
# print(f"Cross Validation Accuracy: {results.mean()} (+/- {results.std() * 2})")


# # Get the best parameters for LSTM model
# best_params_lstm = grid_result.best_params_

# # Define early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=50)

# # Create and compile the LSTM model with the best parameters
# model = create_model(learning_rate=best_params_lstm['learning_rate'], 
#                      dropout_rate=best_params_lstm['dropout_rate'], 
#                      neurons=best_params_lstm['neurons'])

# # Compute class weights
# class_weights_list = []
# for column in y_train.columns:
#     class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[column]), y_train[column])
#     class_weights_dict = dict(enumerate(class_weights))
#     class_weights_list.append(class_weights_dict)




# # Fit the model with the training data
# model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping], class_weight=class_weights_dict, verbose=1)

# # Make predictions on the testing data
# y_pred_bin = (model.predict(X_test) > 0.5).astype(int)

# # Print the classification report
# print(classification_report(y_test, y_pred_bin, target_names=target_columns))

# # Print precision, recall,and F1 score
# print(f"Precision: {precision_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")
# print(f"Recall: {recall_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")
# print(f"F1 Score: {f1_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")

# y_pred = (model.predict(X_test) > 0.5).astype(int)

# precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
# recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
# f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
# roc_auc = roc_auc_score(y_test, y_pred, average='weighted')


# print('Precision: ', precision)
# print('Recall: ', recall)
# print('F1 score: ', f1)
# print('ROC AUC: ', roc_auc)