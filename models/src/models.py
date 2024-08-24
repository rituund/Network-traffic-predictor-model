import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Load the preprocessed dataset
preprocessed_file_path = 'preprocessed_flowdata4.csv'
data = pd.read_csv(preprocessed_file_path)

# Extract features and target variable
X = data.drop(columns=['label'])
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a function to evaluate and save model results
def evaluate_and_save_model(model, model_name, X_train, X_test, y_train, y_test, results_dict):
    # Train the model
    model.fit(X_train, y_train)
   
    # Make predictions
    y_pred = model.predict(X_test)
   
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
   
    # Save the results in the dictionary
    results_dict[model_name] = {
        'Accuracy': accuracy,
        'Classification Report': report
    }

# Initialize a dictionary to store results
results_dict = {}

# Initialize and evaluate Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
evaluate_and_save_model(logistic_model, 'Logistic Regression', X_train, X_test, y_train, y_test, results_dict)

# Initialize and evaluate K-Nearest Neighbors
knn_model = KNeighborsClassifier()
evaluate_and_save_model(knn_model, 'K-Nearest Neighbors', X_train, X_test, y_train, y_test, results_dict)

# Initialize and evaluate Decision Tree
dt_model = DecisionTreeClassifier()
evaluate_and_save_model(dt_model, 'Decision Tree', X_train, X_test, y_train, y_test, results_dict)

# Initialize and evaluate Gaussian Naive Bayes
gnb_model = GaussianNB()
evaluate_and_save_model(gnb_model, 'Gaussian Naive Bayes', X_train, X_test, y_train, y_test, results_dict)

# Initialize and evaluate Random Forest
rf_model = RandomForestClassifier()
evaluate_and_save_model(rf_model, 'Random Forest', X_train, X_test, y_train, y_test, results_dict)

# Initialize and evaluate Support Vector Machine
svm_model = SVC()
evaluate_and_save_model(svm_model, 'Support Vector Machine', X_train, X_test, y_train, y_test, results_dict)

# Save the results to a file in tabular form
results_file_path = 'model_results.txt'
with open(results_file_path, 'w') as file:
    file.write('Model Results\n')
    file.write('-' * 60 + '\n')
    for model_name, results in results_dict.items():
        file.write(f'Model: {model_name}\n')
        file.write(f'Accuracy: {results["Accuracy"]:.4f}\n')
        file.write('Classification Report:\n')
        file.write(pd.DataFrame(results['Classification Report']).transpose().to_string())
        file.write('\n' + '-' * 60 + '\n')

print(f'All model results have been written to {results_file_path}')

# Path to the new data file
new_data_file_path = 'preprocessed_flowdata11.csv'

# Check if the file exists
if os.path.exists(new_data_file_path):
    new_data = pd.read_csv(new_data_file_path)
    
    # Extract features from the new data (no target variable)
    X_new = new_data.drop(columns=['label'], errors='ignore')  # Assuming 'label' might not be present

    # Standardize the new data using the same scaler
    X_new = scaler.transform(X_new)

    # Predict using the RF model
    rf_predictions = rf_model.predict(X_new)

    # Print the predictions
    print('Predictions for the new data:')
    print(rf_predictions)

    # Optionally, save the predictions to a file
    predictions_file_path = 'rf_predictions.csv'
    pd.DataFrame(rf_predictions, columns=['Predicted_Label']).to_csv(predictions_file_path, index=False)

    print(f'Predictions have been written to {predictions_file_path}')
else:
    print(f'Error: File {new_data_file_path} not found.')
