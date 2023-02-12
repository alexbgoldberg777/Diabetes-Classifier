import pandas as pd
from catboost import CatBoostClassifier
import PySimpleGUI as sg

# Trained model
model = CatBoostClassifier().load_model('Models/Tuned-Model-With-Overfitting-Detection')

# Create the layout so a user can input their values for all features used in the model
column_layout = [
    [sg.Text("Enter your values for each question, then click the button at the bottom to see your chances of diabetes or prediabetes.", font='bold')],
    [sg.Text("Are you biologically male or female?")],
    [sg.Combo(['Male', 'Female'], size=(10,2), key='Sex')],
    [sg.Text("Do you have high blood pressure?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='HighBP')],
    [sg.Text("Do you have high cholesterol?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='HighChol')],
    [sg.Text("Have you had a cholesterol check in the past 5 years?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='CholCheck')],
    [sg.Text("What is your BMI (body mass index)?")],
    [sg.InputText(size=(10,2), key='BMI', default_text=0.0)],
    [sg.Text("Have you smoked at least 100 cigarettes in your entire life?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='Smoker')],
    [sg.Text("Have you ever had a stroke?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='Stroke')],
    [sg.Text("Do you have CHD (coronary heart disease) or have had an MI (myocardial infraction)?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='HeartDiseaseorAttack')],
    [sg.Text("Have you done regular physical activity in the past 30 days?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='PhysActivity')],
    [sg.Text("Do you eat at least 1 serving of fruit a day?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='Fruits')],
    [sg.Text("Do you eat at least 1 serving of vegetables a day?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='Veggies')],
    [sg.Text("Do you have 14 or more alcoholic drinks a week (if you're biologically male), or 7 or more alcoholic drinks per week (if you're biologically female)?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='HvyAlcoholConsump')],
    [sg.Text("Do you have any healthcare coverage?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='AnyHealthcare')],
    [sg.Text("In the past 12 months, have you needed to go to the doctor but could not because of the cost?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key= 'NoDocbcCost')],
    [sg.Text("In general, rate your health from 1 to 5, with 1 being excellent and 5 being poor.")],
    [sg.Combo(['1', '2', '3', '4', '5'], size=(10,2), key='GenHlth', default_value=1)],
    [sg.Text("How many days of the past 30 days would you say your mental health has been poor?")],
    [sg.InputText(size=(10,2), key='MentHlth', default_text=0.0)],
    [sg.Text("How many days of the past 30 days have you dealt with a physical illness or injury?")],
    [sg.InputText(size=(10,2), key='PhysHlth', default_text=0.0)],
    [sg.Text("Do you have significant diffculty walking or climbing stairs?")],
    [sg.Combo(['Yes', 'No'], size=(10,2), key='DiffWalk')],
    [sg.Text("What age range do you fall under?")],
    [sg.Combo(['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'], size=(10,2), key='Age')],
    [sg.Text("What is your highlest level of education completed?")],
    [sg.Combo(['Never attended school or only kindergarten', 'Grades 1-8 (elementary/middle)', 'Grades 9-11 (some high school)', 'Grade 12 or GED (high school graduate)', 'Some college or technical school', 'College graduate'], size=(10,2), key='Education')],
    [sg.Text("What is your annual income level?")],
    [sg.Combo(['Less than $10,000', '$10,000 - $14,999', '$15,000 - $19,999', '$20,000 - $24,999', '$25,000 - $34,999', '$35,000 - $49,999', '$50,000 - $74,999', '$75,000 or more'], size=(10,2), key='Income')],
    [sg.Button('Calculate Diabetes Chance')],
    [sg.Text("Your chance of prediabetes or diabetes is: ")],
    [sg.Text("", key='Probability')],
    [sg.CloseButton("Close")]
]
UI_layout = [[sg.Column(column_layout, scrollable=True, vertical_scroll_only=True)]]

# Create the UI window with the above layout
window = sg.Window("Window", UI_layout, grab_anywhere=True)

while True:
    # values saves all inputs from the user for their health indicator answers
    # event tracks when the user selects the button to calculate their diabetes probability
    event, values = window.read()

    # Takes the values input from the user and converts them into a form the model can use
    HighBP = 1.0 if values['HighBP'] == 'Yes' else 0.0
    HighChol = 1.0 if values['HighChol'] == 'Yes' else 0.0
    CholCheck = 1.0 if values['CholCheck'] == 'Yes' else 0.0
    BodyMassIndex = float(values['BMI'])
    Smoker = 1.0 if values['Smoker'] == 'Yes' else 0.0
    Stroke = 1.0 if values['Stroke'] == 'Yes' else 0.0
    HeartDiseaseorAttack = 1.0 if values['HeartDiseaseorAttack'] == 'Yes' else 0.0
    PhysActivity = 1.0 if values['PhysActivity'] == 'Yes' else 0.0
    Fruits = 1.0 if values['Fruits'] == 'Yes' else 0.0
    Veggies = 1.0 if values['Veggies'] == 'Yes' else 0.0
    HvyAlcoholConsump = 1.0 if values['HvyAlcoholConsump'] == 'Yes' else 0.0
    AnyHealthcare = 1.0 if values['AnyHealthcare'] == 'Yes' else 0.0
    NoDocbcCost = 1.0 if values['NoDocbcCost'] == 'Yes' else 0.0
    GenHlth = float(values['GenHlth'])
    MentHlth = float(values['MentHlth'])
    PhysHlth = float(values['PhysHlth'])
    DiffWalk = 1.0 if values['DiffWalk'] == 'Yes' else 0.0
    Sex = 1.0 if values['Sex'] == 'Male' else 0.0

    if values['Age'] == '18-24':
        Age = 1.0
    elif values['Age'] == '25-29':
        Age = 2.0
    elif values['Age'] == '30-34':
        Age = 3.0
    elif values['Age'] == '35-39':
        Age = 4.0
    elif values['Age'] == '40-44':
        Age = 5.0
    elif values['Age'] == '45-49':
        Age = 6.0
    elif values['Age'] == '50-54':
        Age = 7.0
    elif values['Age'] == '55-59':
        Age = 8.0
    elif values['Age'] == '60-64':
        Age = 9.0
    elif values['Age'] == '65-69':
        Age = 10.0
    elif values['Age'] == '70-74':
        Age = 11.0
    elif values['Age'] == '75-79':
        Age = 12.0
    else:
        Age = 13.0

    if values['Education'] == 'Never attended school or only kindergarten':
        Education = 1.0
    elif values['Education'] == 'Grades 1-8 (elementary/middle)':
        Education = 2.0
    elif values['Education'] == 'Grades 9-11 (some high school)':
        Education = 3.0
    elif values['Education'] == 'Grade 12 or GED (high school graduate)':
        Education = 4.0
    elif values['Education'] == 'Never attended school or only kindergarten':
        Education = 5.0
    else:
        Education = 6.0

    if values['Income'] == 'Less than $10,000':
        Income = 1.0
    elif values['Income'] == '$10,000 - $14,999':
        Income = 2.0
    elif values['Income'] == '$15,000 - $19,999':
        Income = 3.0
    elif values['Income'] == '$20,000 - $24,999':
        Income = 4.0
    elif values['Income'] == '$25,000 - $34,999':
        Income = 5.0
    elif values['Income'] == '$35,000 - $49,999':
        Income = 6.0
    elif values['Income'] == '$50,000 - $74,999':
        Income = 7.0
    else:
        Income = 8.0

    # Maps the input data provided by the user into a dictionary to be used in the model
    user_data = {'HighBP': HighBP,
        'HighChol': HighChol,
        'CholCheck': CholCheck,
        'BMI': BodyMassIndex,
        'Smoker': Smoker,
        'Stroke': Stroke,
        'HeartDiseaseorAttack': HeartDiseaseorAttack,
        'PhysActivity': PhysActivity,
        'Fruits': Fruits,
        'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump,
        'AnyHealthcare': AnyHealthcare,
        'NoDocbcCost': NoDocbcCost,
        'GenHlth': GenHlth,
        'MentHlth': MentHlth,
        'PhysHlth': PhysHlth,
        'DiffWalk': DiffWalk,
        'Sex': Sex,
        'Age': Age,
        'Education': Education,
        'Income': Income}

    if event == 'Calculate Diabetes Chance':
        # Converts the data into a pandas dataframe for the next step
        dataframe = pd.DataFrame(data=user_data, index=[0])

        # Using the trained model, calculates the probability that the user has prediabetes or diabetes based on their input data
        window['Probability'].update(value = str(round(model.predict_proba(dataframe[model.feature_names_])[0][1] * 100, 2)) + " %")

