import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sqlite3
from scipy.stats import randint


SQL_PATH = 'Data/database.db'
MODEL_PATH = 'Models/'

# Create connection to database for reading in data
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except:
        print("Connection creation failed")
    return conn

# Read in data from database for training
conn = create_connection(SQL_PATH)
database_query = '''SELECT Diabetes_binary,
                        HighBP,
                        HighChol,
                        CholCheck,
                        BMI,
                        Smoker,
                        Stroke,
                        HeartDiseaseorAttack,
                        PhysActivity,
                        Fruits,
                        Veggies,
                        HvyAlcoholConsump,
                        AnyHealthcare,
                        NoDocbcCost,
                        GenHlth,
                        MentHlth,
                        PhysHlth,
                        DiffWalk,
                        Sex,
                        Age,
                        Education,
                        Income
                    FROM diabetes'''

training_data = pandas.read_sql(database_query, conn)
conn.close()
training_data.head()


# Split Training and Testing Data
columns, values = training_data.drop(['Diabetes_binary'], axis=1), training_data['Diabetes_binary']

training_columns, testing_columns, training_values,testing_values = train_test_split(columns, values, test_size=.2, random_state=13)
training_columns.head()

# Create the classifer and set with hard-coded parameters for initial testing
default_parameters = {'iterations': 5000,
    'loss_function': 'Logloss',
    'depth': 4,
    'early_stopping_rounds':20,
    'custom_loss': ['AUC', 'Accuracy']}

classifier = CatBoostClassifier(**default_parameters)

# Fit the classifier with the default parameters and train the model
classifier.fit(
    training_columns,
    training_values,
    eval_set=(testing_columns,testing_values),
    verbose=False,
    plot=True,
)
predictions = classifier.predict(testing_columns)

# Create graphs to showcase the features by importance within the default classifier
figure, axis = plt.subplots()
feature_importance_data = pandas.DataFrame({'feature':classifier.feature_names_, 'importance':classifier.feature_importances_})
feature_importance_data.sort_values('importance', ascending=False, inplace=True)
seaborn.barplot(x='importance', y='feature', data=feature_importance_data)

# Set the grid of hyperparameter possibilities. This is the second attempt, now with an overfitting detector
HPgrid = {'depth': [2,4,6,8],
    'iterations': [3000,5000,7500,10000],
    'early_stopping_rounds': [1,3,5,10],
    'l2_leaf_reg': [5,10,15,25],
    'od_type': ['IncToDec']
}

# Fit the model and test all possible combinations of hyperparameters found by random tuning to find those that produce the highest accuracy
# Uses grid-based hyperparameter tuning
grid_cross_val = GridSearchCV(estimator=classifier, param_grid=HPgrid, scoring='accuracy', cv=5)
grid_cross_val.fit(training_columns,training_values, verbose=False)

# Fit and train the model with the best parameters as found by the above hyperparameter tuning
best_parameters = grid_cross_val.best_params_
tuned_model = CatBoostClassifier(**best_parameters)
tuned_model.fit(
    training_columns, 
    training_values, 
    verbose = False, 
    eval_set=(testing_columns,testing_values)
    )

HP_predictions = tuned_model.predict(testing_columns)

# Save and export the tuned model to be used to make new predictions with the Input UI
tuned_model.save_model(MODEL_PATH + 'Tuned-Model-with-Overfitting-Detection')

# Create graphs to showcase the features by importance with the best parameters as found by the hyperparameter tuning
figure, axis = plt.subplots()
feature_importance_data = pandas.DataFrame({'feature':tuned_model.feature_names_, 'importance':tuned_model.feature_importances_})
feature_importance_data.sort_values('importance', ascending=False, inplace=True)
seaborn.barplot(x='importance', y='feature', data=feature_importance_data)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
# Showcase the best accuracy score and the best parameters as found by the hyperparameter tuning
print("Best Score: " + str(grid_cross_val.best_score_))
print("Best Params: " + str(grid_cross_val.best_params_))

# Showcase information about the predictions made in training of the most accurate model
print(classification_report(testing_values, predictions))
print(confusion_matrix(testing_values, predictions))