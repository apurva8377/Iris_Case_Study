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

