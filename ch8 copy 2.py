import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1_l2
from sklearn.model_selection import cross_val_score, KFold
from keras.callbacks import EarlyStopping
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


# Load the data
data_path = './datasets/group47/dataset/intermediate_datafiles/chapter5_group47_result.csv'
data = pd.read_csv(data_path, index_col=0)

# Define the target columns
target_columns = ['labelCycling', 'labelStairs', 'labelWalking', 'labelSitting', 'labelOther']

# Separate the features and target variables
X = data.drop(target_columns, axis=1).fillna(0).replace([np.inf], np.finfo('float64').max).replace([-np.inf], np.finfo('float64').min)
y = data[target_columns]


selected_features = set()
for column in tqdm(y.columns, desc="Selecting features"):
    y_single_label = LabelEncoder().fit_transform(y[column])  # Convert to 1D array
    selector = SelectKBest(f_classif, k=50)
    X_new = selector.fit_transform(X, y_single_label)
    selected_features.update(X.columns[selector.get_support()])

# Filter the features based on the selected features
X = StandardScaler().fit_transform(X[list(selected_features)])

# Use simple random sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define a function to create and compile a new model
def create_model(learning_rate=0.001, dropout_rate=0.2, neurons=100):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(neurons, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(Dense(neurons, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(Dense(len(target_columns), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

# Create a new model
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the grid search parameters
learning_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
neurons = [50, 100, 150, 200]
param_grid = dict(learning_rate=learning_rate, dropout_rate=dropout_rate, neurons=neurons)

# Create Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)
grid_result = grid.fit(X_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Define the cross validation iterator
kfold = KFold(n_splits=10, shuffle=True)

# Perform cross validation
results = cross_val_score(model, X_train, y_train, cv=kfold)


# # Define the cross validation iterator
# kfold = KFold(n_splits=10, shuffle=True)

# # Initialize a list to store the scores for each fold
# scores = []

# # Loop over each fold
# for train_index, test_index in tqdm(kfold.split(X_train), total=10, desc="Cross Validation"):
#     # Split the data into training and validation sets for this fold
#     X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
#     y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

#     # Fit the model on the training data
#     model.fit(X_train_fold, y_train_fold)

#     # Make predictions on the validation data
#     y_pred = model.predict(X_val_fold)

#     # Calculate the accuracy of the predictions and add it to the scores list
#     scores.append(accuracy_score(y_val_fold, y_pred))

# # Convert the scores list to a numpy array
# results = np.array(scores)

# Print the cross validation score
print(f"Cross Validation Accuracy: {results.mean()} (+/- {results.std() * 2})")


# Get the best parameters for LSTM model
best_params_lstm = grid_result.best_params_

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

# Create and compile the LSTM model with the best parameters
model = create_model(learning_rate=best_params_lstm['learning_rate'], 
                     dropout_rate=best_params_lstm['dropout_rate'], 
                     neurons=best_params_lstm['neurons'])

# Fit the model with the training data
model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping], verbose=1)

# Make predictions on the testing data
y_pred_bin = (model.predict(X_test) > 0.5).astype(int)

# Print the classification report
print(classification_report(y_test, y_pred_bin, target_names=target_columns))

# Print precision, recall,and F1 score
print(f"Precision: {precision_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")
print(f"Recall: {recall_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")
print(f"F1 Score: {f1_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")
