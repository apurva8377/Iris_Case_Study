##########################################################
#
#  Case Study Name : Iris_Case_Study
#  Description     : Prediction & Evaluation – Predicts test data and measures performance using:
#                       Accuracy Score
#                       Confusion Matrix
#                       Classification Report
#  Auther          : Apurva Vilas Shinde
#  date            : 26/05/2026
#
##########################################################

import pandas as pd

import matplotlib.pylab as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*40

##############################################
# Step 1 : Load the Dadatset
##############################################
print(Border)
print("Step 1 : Load the dataset")
print(Border)

DatasetPath = "iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded succesfully...")
print("Initial entries from dataset : ")
print(df.head())

###########################################################
# Step 2 : Data Analysis (EDA - Exploratory Data Analysis)
###########################################################
print(Border)
print("Step 2 : Data Analysis")
print(Border)

print("Shape of dataset : ",df.shape)

print("Column Names : ",list(df.columns))

print("Missing values (Per Column) :")
print(df.isnull().sum())

print("Class Distribution (Species count) : ")
print(df["variety"].value_counts())

print("Statistical Report of dataset :")
print(df.describe())

###########################################################
# Step 3 : Decide Independent & Dependent variable
###########################################################
print(Border)
print("Step 3 : Decide Independent & Dependent variable")
print(Border)

# X : Independent variable
# Y : Dependent variable

feature_cols = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width"
]

X = df[feature_cols]
Y = df["variety"]

print("X shape : ",X.shape)
print("Y shape : ",Y.shape)

###########################################################
# Step 4 : Visualisation of dataset
###########################################################
print(Border)
print("Step 4 : Visualisation of dataset")
print(Border)

# Scatter plot
plt.figure(figsize=(7,5))

for va in df["variety"].unique():
    temp = df[df["variety"] == va]
    plt.scatter(temp["petal.length"], temp["petal.width"])

plt.title("Iris : Petal length vs petal width")

plt.xlabel("petal.length")
plt.ylabel("petal.width")

plt.legend()
plt.grid (True)
plt.show()

###########################################################
# Step 5 : Split the dataset for training and testing
###########################################################
print(Border)
print("Step 5 : Split the dataset for training and testing")
print(Border)

# total dataset : 150,5
# X : 150,4
# Y : 150,1
# Test size = 20%
# Train size = 80%

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Data splitting activity done : ")

print("X - Independent : ",X.shape)  # (150,4)
print("Y - Dependent : ",Y.shape)    # (150,)

print("X_train : ",X_train.shape)    # (120,4)
print("X_test : ",X_test.shape)      # (30,4)

print("Y_train : ",Y_train.shape)    # (120,)
print("Y_test : ",Y_test.shape)      # (30,)

###########################################################
# Step 6 : Build the model
###########################################################
print(Border)
print("Step 6 : Build the model")
print(Border)

print("We are going to use DecisionTreeClassifier")

Model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

print("Model succesfully created : ",Model)

###########################################################
# Step 7 : Train the model
###########################################################
print(Border)
print("Step 7 : Train the model")
print(Border)

Model.fit(X_train,Y_train)

print("Model training completed")

###########################################################
# Step 8 : Evaluate the model
###########################################################
print(Border)
print("Step 8 : Evaluate the model")
print(Border)

Y_pred = Model.predict(X_test)

print("Model evaluation (testing) complete")

print(Y_pred.shape)

print("Expected answers : ")
print(Y_test)

print("Predicted answers : ")
print(Y_pred)

