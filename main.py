import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from classification import classify
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_data():
    the_data = fetch_ucirepo(id=45)
    features_data = the_data.data.features
    target_data = pd.Series(the_data.data.targets.values.ravel())
    return features_data, target_data


def user_input():
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (1 for male, 0 for female): "))
    chestpain = int(input("Enter chest pain type (0-3): "))
    restingblood = int(input("Enter resting blood pressure: "))
    chol = int(input("Enter serum cholesterol(in the range of 100 to 300): "))
    fastingblood = int(input("Enter fasting blood sugar (1 if > 120 mg/dl, 0 otherwise): "))
    restingECG = int(input("Enter resting electrocardiographic results (0-2): "))
    maximumheartrate = int(input("Enter maximum heart rate achieved: "))
    angina = int(input("Enter exercise induced angina (1 for yes, 0 for no): "))
    STdepression = float(input("Enter ST depression induced by exercise relative to rest(0 to 6.2): "))
    slope = int(input("Enter the slope of the peak exercise ST segment (0-2): "))
    fluoroscopy = int(input("Enter number of major vessels colored by fluoroscopy (0-3): "))
    thallium = int(input("Enter thallium stress test result (0-3): "))

    return [age, gender, chestpain, restingblood, chol, fastingblood, restingECG, maximumheartrate, angina, STdepression, slope, fluoroscopy, thallium]


def predict_heart_disease(model, data):
    column_names = ['age', 'sex', 'chestpain', 'restingblood', 'chol', 'fastingblood', 'restingECG', 'maximumheartrate', 'angina', 'STdepression', 'slope',
                    'fluoroscopy', 'thallium']
    data_df = pd.DataFrame([data], columns=column_names)
    probability = model.predict_proba(data_df)[:, 1][0]

    # Debugging
    print("User input data:")
    print(data_df)

    print("Raw model output:")
    print(model.predict_proba(data_df))

    return probability


def main():
    features_data, target_data = load_data()

    # print
    print("Let's see our columns:")
    print(features_data.columns)

    # Q1: Percentage of patients with heart disease
    disease_counts = target_data.value_counts(normalize=True) * 100
    plt.figure()
    sns.barplot(x=disease_counts.index, y=disease_counts.values)
    plt.ylabel('Percentage')
    plt.xlabel('Heart Disease')
    plt.title('How Many People Have Heart Disease?')
    plt.show()

    # Q2: Age distribution of patients with heart disease
    people_with_disease = features_data[target_data == 1]
    plt.figure()
    sns.histplot(people_with_disease['age'], kde=True, bins=30)
    plt.title('How Old Are They?')
    plt.show()

    # Q3: Gender comparison in heart disease occurrence
    gender_distribution = people_with_disease['sex'].value_counts(normalize=True) * 100
    plt.figure()
    sns.barplot(x=gender_distribution.index, y=gender_distribution.values)
    plt.ylabel('Percentage')
    plt.xlabel('Gender (1 = Male, 0 = Female)')
    plt.title('Does Gender Play a Role?')
    plt.show()

    # Q4: Relationship between age and cholesterol level
    plt.figure()
    sns.scatterplot(x='age', y='chol', data=people_with_disease)
    plt.xlabel('How old are they?')
    plt.ylabel('What’s their cholesterol level?')
    plt.title('Does Age Affect Cholesterol Level?')
    plt.show()

    # Q5: High blood pressure in heart disease patients
    high_bp_people = people_with_disease[people_with_disease['trestbps'] > 130]
    high_bp_percent = (len(high_bp_people) / len(people_with_disease)) * 100
    plt.figure(figsize=(5, 5))
    sns.barplot(x=['High Blood Pressure', 'Normal Blood Pressure'],
                y=[high_bp_percent, 100 - high_bp_percent])
    plt.ylabel('How many % of them?')
    plt.title('High Blood Pressure and Heart Disease')
    plt.show()

    model = classify(features_data, target_data)

    user_data = user_input()
    probability = predict_heart_disease(model, user_data)
    print(f"The predicted probability of having heart disease is: {probability * 100:.2f}%")
    print(f"IGNORE WARNINGS")


if __name__ == "__main__":
    main()
