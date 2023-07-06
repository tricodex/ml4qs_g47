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

# Load the data
data_path = './datasets/group47/dataset/intermediate_datafiles/chapter5_group47_result.csv'
try:
    data = pd.read_csv(data_path, index_col=0)
except FileNotFoundError:
    print(f"File not found at {data_path}")
    exit()

# Define the target columns
target_columns = ['labelCycling', 'labelStairs', 'labelWalking', 'labelSitting', 'labelOther']

# Separate the features and target variables
X = data.drop(target_columns, axis=1).fillna(0).replace([np.inf], np.finfo('float64').max).replace([-np.inf], np.finfo('float64').min)
y = data[target_columns]

selected_features = set()
for column in y.columns:
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
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Define your classifier
clf = OneVsRestClassifier(RandomForestClassifier(random_state=42, class_weight='balanced'))

# Define your parameter grid
param_grid = {
    'estimator__n_estimators': [50, 100, 200, 300, 400],
    'estimator__max_features': ['sqrt', 'log2'],
    'estimator__max_depth' : [4,5,6,7,8,9,10],
    'estimator__criterion' :['gini', 'entropy']
}

# Initialize GridSearchCV with F1 score as the scoring metric
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Define the cross validation iterator
kfold = KFold(n_splits=10, shuffle=True)

# Perform cross validation
results = cross_val_score(model, X_train, y_train, cv=kfold)

# Print the cross validation score
print(f"Cross Validation Accuracy: {results.mean()} (+/- {results.std() * 2})")

# Get the best parameters for LSTM model
best_params_lstm = grid_result.best_params_

# Get the best parameters for RandomForest model
best_params_rf = grid_search.best_params_

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


# Create and compile the LSTM model with the best parameters
model = create_model(learning_rate=best_params_lstm['learning_rate'], 
                     dropout_rate=best_params_lstm['dropout_rate'], 
                     neurons=best_params_lstm['neurons'])

# Fit the model with the training data
model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

# Make predictions on the testing data
y_pred_bin = (model.predict(X_test) > 0.5).astype(int)

# Print the classification report
print(classification_report(y_test, y_pred_bin, target_names=target_columns))

# Print precision, recall,and F1 score
print(f"Precision: {precision_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")
print(f"Recall: {recall_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")
print(f"F1 Score: {f1_score(y_test, y_pred_bin, average='weighted', zero_division=1)}")

# Create the RandomForest model with the best parameters
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=best_params_rf['estimator__n_estimators'],
                                                 max_features=best_params_rf['estimator__max_features'],
                                                 max_depth=best_params_rf['estimator__max_depth'],
                                                 criterion=best_params_rf['estimator__criterion'],
                                                 random_state=42, 
                                                 class_weight='balanced'))

# Fit the RandomForest model
clf.fit(X_train, y_train)

# Make predictions with the RandomForest model
y_pred_rf = clf.predict(X_test)

# Print the classification report for the RandomForest model
print(classification_report(y_test, y_pred_rf, target_names=target_columns))

# Print precision, recall,and F1 score for the RandomForest model
print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted', zero_division=1)}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted', zero_division=1)}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, average='weighted', zero_division=1)}")
