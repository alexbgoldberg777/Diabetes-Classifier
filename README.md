# Diabetes-Classifier

This project includes a trained decision tree model used to predict a user's likelihood of developing diabetes.

The Data folder contains data of 253,680 survey respondents about certain health indicators that pertain to diabetes, as well as whether or not they have diabetes/prediabetes. The data was found at: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download

The data was converted to an sqlite3 database with the csv_to_sql file. Within the Training_and_UI folder is the Model_Training file where Catboost is used to train a gradient-boosted decision tree model based on the dataset. This file also contains grid-based hypeparameter tuning that was performed to find the best parameters to train the model with, which was saved in the Models folder.

Also included in the Training_and_UI folder in the InputUI file is a user interface made using PySimpleGUI where a user can input their own health indicator data and see what their probability of developing diabetes/prediabetes is.

To run the project, download the code from the repository and simply run the InputUI file. A UI window will pop up prompting the user to input their data. After doing so, click the "Calculate Diabetes Chance" button to see your results.
