Network Flow Classification Using Machine Learning
Project Overview
This project focuses on classifying network flows using various machine learning models. The dataset used in this project consists of network flow data with features like protocol, traffic direction, connection state, and byte statistics. The objective is to predict the type of network flow based on these features, leveraging different machine learning models.

Dataset
The dataset contains the following columns:

dur: Duration of the network flow
proto: Protocol used (e.g., TCP, UDP)
dir: Direction of traffic (e.g., incoming, outgoing)
state: Connection state (e.g., established, closed)
stos: Source-to-destination bytes
dtos: Destination-to-source bytes
tot_pkts: Total number of packets
tot_bytes: Total number of bytes
src_bytes: Source bytes
label: Classification label indicating the type of network flow
Family: Classification family
Project Structure
data_processing.py: Script for preprocessing the dataset. This includes handling missing values, encoding categorical variables, splitting the dataset into training and testing sets, and standardizing the features. The preprocessed data is saved as preprocessed_flowdata11.csv.

models.py: Script for training and evaluating various machine learning models. The models evaluated in this project include:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Gaussian Naive Bayes
Random Forest
Support Vector Machine (SVM)
Multi-Layer Perceptron (MLP) Classifier
The results of each model, including accuracy and classification reports, are saved to model_results.txt. The script also includes functionality to predict on new data using the trained Random Forest model.

Installation
To run this project, you need to have Python installed, along with the following libraries:

bash
Copy code
pip install pandas scikit-learn
Usage
1. Preprocess the Dataset
Run the data_processing.py script to preprocess the dataset:

bash
Copy code
python data_processing.py
This will generate a preprocessed dataset preprocessed_flowdata11.csv.

2. Train and Evaluate Models
Run the models.py script to train and evaluate the models:

bash
Copy code
python models.py
The script will output the evaluation results for each model and save them to model_results.txt.

3. Predict on New Data
If you have a new dataset that you want to classify, ensure the data is in the same format as the training data. Place it in the project directory and name it preprocessed_flowdata11.csv. The models.py script will automatically detect the file, preprocess it, and use the trained Random Forest model to make predictions. The predictions will be saved in rf_predictions.csv.

Results
The results for each model, including accuracy and classification reports, are saved in the model_results.txt file.
