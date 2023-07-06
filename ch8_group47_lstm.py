import numpy as np
import pandas as pd


from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score





# Load the data
data_path = './datasets/group47/dataset/intermediate_datafiles/chapter5_group47_result.csv'
try:
    data = pd.read_csv(data_path, index_col=0)
except FileNotFoundError:
    print(f"File not found at {data_path}")
    exit()

# Check if data is empty
if data.empty:
    print("Data is empty")
    exit()

# Define the target columns
target_columns = ['labelCycling', 'labelStairs', 'labelWalking', 'labelSitting', 'labelOther']

# Separate the features and target variables
X = data.drop(target_columns, axis=1)
y = data[target_columns]

# Fill NaN values with 0 (or any other value you deem appropriate)
X = X.fillna(0)

# Replace positive infinity values with the maximum non-infinity number
X = X.replace([np.inf], np.finfo('float64').max)

# Replace negative infinity values with the minimum non-infinity number
X = X.replace([-np.inf], np.finfo('float64').min)


# # Perform feature selection for each label independently
# selected_features = set()
# for column in y.columns:
#     y_single_label = LabelEncoder().fit_transform(y[column])  # Convert to 1D array
#     selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=50)
#     X_new = selector.fit_transform(X, y_single_label)
    
#     # Get the selected features for this label
#     mask = selector.get_support()  # List of booleans for selected features
#     selected_features_single_label = X.columns[mask]
    
#     # Add the selected features for this label to the set of all selected features
#     selected_features.update(selected_features_single_label)

# # Filter the features based on the selected features
# X = X[list(selected_features)]

# Perform feature selection for each label independently
# SelectKBest with the f_classif scoring function to select the 50 best features for each label

selected_features = set()
for column in y.columns:
    y_single_label = LabelEncoder().fit_transform(y[column])  # Convert to 1D array
    selector = SelectKBest(f_classif, k=50)
    X_new = selector.fit_transform(X, y_single_label)
    
    # Get the selected features for this label
    mask = selector.get_support()  # List of booleans for selected features
    selected_features_single_label = X.columns[mask]
    
    # Add the selected features for this label to the set of all selected features
    selected_features.update(selected_features_single_label)

# Filter the features based on the selected features
X = X[list(selected_features)]


# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

 # Use simple random sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# Define the LSTM model
model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # Increase the number of units
model.add(Dropout(0.2))
model.add(LSTM(200, return_sequences=True))  # Add another LSTM layer
model.add(Dropout(0.2))
model.add(LSTM(100))  # Keep this LSTM layer
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(target_columns), activation='sigmoid'))  # Change activation function to softmax

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])  # Use learning_rate instead of lr


# Make predictions on the testing data
y_pred = model.predict(X_test)

# Convert the predictions to binary format
y_pred_bin = (y_pred > 0.5).astype(int)

# Print the classification report
print(classification_report(y_test, y_pred_bin, target_names=target_columns))
precision = precision_score(y_test, y_pred_bin, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_bin, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred_bin, average='weighted', zero_division=1)


# Print precision, recall, and F1 score
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# Define your classifier
clf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Wrap your classifier with OneVsRestClassifier
clf = OneVsRestClassifier(clf)

# Define your parameter grid
param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_features': ['sqrt', 'log2'],
    'estimator__max_depth' : [4,5,6,7,8],
    'estimator__criterion' :['gini', 'entropy']
}

# Initialize GridSearchCV with F1 score as the scoring metric
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro')  # Use F1 score

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print(f"Best parameters: {best_params}")
