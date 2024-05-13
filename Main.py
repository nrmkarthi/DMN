import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from maxout_layer import Maxout
from walrus_optimization import WalrusOptimization

# Load dataset
# Assuming the dataset is stored locally as "CIC-IDS2017.csv"
dataset = pd.read_csv("CIC-IDS2017.csv")

# Preprocess dataset
# Remove any NaN values
dataset.dropna(inplace=True)

# Encode categorical variables
categorical_cols = dataset.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    dataset[col] = label_encoder.fit_transform(dataset[col])

# Split dataset into features and labels
X = dataset.drop(columns=['label'])  # Features
y = dataset['label']  # Labels

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Instantiate the model
model = DMNWO()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Define objective function for Walrus Optimization
def objective_function(x):
    # Example objective function (minimization)
    # Replace with appropriate evaluation metric based on your dataset and task
    return np.sum(np.square(x))

# Instantiate WalrusOptimization
num_variables = X_train_scaled.shape[1]  # Number of features
walrus_optimizer = WalrusOptimization(objective_function, num_variables)

# Optimize the objective function
best_solution, best_fitness = walrus_optimizer.optimize()

# Print best solution and fitness
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
