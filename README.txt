
IRIS FLOWER CLASSIFICATION PROJECT
==================================

Overview
--------
This project focuses on building a machine learning model to classify iris flowers into three species: Iris-setosa, Iris-versicolour, and Iris-virginica. Using four key features – Sepal Length, Sepal Width, Petal Length, and Petal Width – the model predicts the flower species accurately. The project incorporates several machine learning algorithms and provides a Tkinter-based GUI for user interaction, where users can input measurements and receive real-time predictions.

Technologies Used
-----------------
- Python: Core programming language
- Pandas & NumPy: Data manipulation and preprocessing
- Matplotlib & Seaborn: Data visualization
- Scikit-Learn: Machine learning models and evaluation
- Joblib: Model serialization (saving and loading)
- Tkinter: GUI development for user interaction

Features
--------
- Preprocessing: Handles missing values, duplicates, and outliers using the IQR method.
- Data Visualization: Scatter plots, box plots, and heatmaps provide insights into feature relationships.
- Machine Learning Models: K-Nearest Neighbors (KNN), Naive Bayes, Decision Tree, and Support Vector Machine (SVM).
- Hyperparameter Tuning: Optimization of KNN model for improved accuracy.
- GUI Integration: Tkinter-based interface for easy, real-time species prediction.
- Model Persistence: Save and load models using Joblib.

Installation and Setup
----------------------
1. Clone the repository:
   ```
   git clone https://github.com/your-username/iris-flower-classification.git
   cd iris-flower-classification
   ```

2. Install required libraries:
   Make sure you have Python installed. Then install the dependencies:
   ```
   pip install -r requirements.txt
   ```
   *(Include a requirements.txt file with dependencies such as pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, etc.)*

3. Run the application:
   Execute the code to launch the Tkinter GUI:
   ```
   python iris_classification_gui.py
   ```

Usage
-----
- GUI Prediction: Input the flower measurements (Sepal/Petal Length and Width) and click the "Predict Species" button to get the predicted species.
- Model Training and Evaluation: The code allows users to train, evaluate, and compare multiple machine learning models for accuracy.
- Visualization: Use the built-in data visualization functions to explore feature relationships.

Directory Structure
-------------------
```
iris-flower-classification/
│
├── IrisDataset.csv           # Iris flower dataset
├── iris_classification_gui.py # Tkinter-based GUI script
├── model_training.py          # Script for training and evaluating models
├── KNNIris.pkl                # Serialized KNN model
├── SVMIris.pkl                # Serialized SVM model
├── NaiveBayesIris.pkl         # Serialized Naive Bayes model
├── requirements.txt           # List of required dependencies
└── README.txt                 # Project documentation
```

Machine Learning Models Used
----------------------------
- K-Nearest Neighbors (KNN): Hyperparameter tuning with n_neighbors from 3 to 29
- Naive Bayes: Gaussian Naive Bayes classifier
- Decision Tree: Trained with both scaled and non-scaled data
- Support Vector Machine (SVM): Linear model for classification

Results and Accuracy
--------------------
The performance of each model is measured using accuracy scores on test data:
- KNN Model (k=9): ~100% accuracy
- Naive Bayes Model: ~94% accuracy
- SVM Model: ~97% accuracy

Contributing
------------
Contributions are welcome! If you find any issues or want to enhance the project, feel free to fork the repository and submit a pull request.

License
-------
This project is licensed under the MIT License – see the LICENSE file for details.
