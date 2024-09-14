import numpy as np
import pandas as pd
import difflib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sqlite3
from datetime import datetime
import warnings

warnings.simplefilter("ignore")

# Path to SQLite database
db_path = "symptom_predictions.db"

# Initialize the database
def initialize_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT NOT NULL,
            Symptoms TEXT NOT NULL,
            Diagnosis TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

initialize_db()

def save_prediction(symptoms, diagnosis):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    symptoms_str = ', '.join(symptoms)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO predictions (timestamp, symptoms, diagnosis)
        VALUES (?, ?, ?)
    """, (timestamp, symptoms_str, diagnosis))
    
    conn.commit()
    conn.close()

def fetch_all_predictions():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM predictions")
    
    rows = cursor.fetchall()
    
    df = pd.DataFrame(rows, columns=['ID', 'Timestamp', 'Symptoms', 'Diagnosis'])
    
    conn.close()
    
    return df

def data_ripper(symptom_list):
    df = pd.read_csv("dataset.csv")
    df["Symptoms"] = df.apply(lambda row: [symptom for symptom in row[1:] if pd.notna(symptom)], axis=1)

    column_values = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
                        'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8',
                        'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12',
                        'Symptom_13', 'Symptom_14', 'Symptom_15', 'Symptom_16',
                        'Symptom_17']].values.ravel()

    symps = pd.unique(column_values)
    symps = symps.tolist()
    symps = [i for i in symps if pd.notna(i)]

    symptoms = pd.DataFrame(columns=symps, index=df.index)
    symptoms["Symptoms"] = df["Symptoms"]
    for i in symps:
        symptoms[i] = symptoms.apply(lambda x: 1 if i in x.Symptoms else 0, axis=1)

    symptoms["Disease"] = df["Disease"]
    symptoms = symptoms.drop("Symptoms", axis=1)

    train, test = train_test_split(symptoms, test_size=0.2)
    X_train = train.drop("Disease", axis=1)
    y_train = train["Disease"].copy()

    rnd_forest = RandomForestClassifier(
                                        n_estimators=200,
                                        #criterion='gini',
                                        max_depth=30,
                                        #min_samples_split=10, #10 after 2
                                        #min_samples_leaf=4, #4 after 1
                                        max_features='sqrt', # was 'sqrt'
                                        bootstrap=True,
                                        #random_state=42
                                        )
    rnd_forest.fit(X_train, y_train)

    input_vector = pd.DataFrame(0, index=[0], columns=symps)

    for symptom in symptom_list:
        if symptom in input_vector.columns:
            input_vector.at[0, symptom] = 1

    feature_importances = rnd_forest.feature_importances_
    important_features = pd.Series(feature_importances, index=X_train.columns).sort_values(ascending=False)
    #print(important_features)

    scores = cross_val_score(rnd_forest, X_train, y_train, cv=5)
    #print(f"Cross-validated accuracy: {scores.mean() * 100:.2f}%")

    prediction = rnd_forest.predict(input_vector)[0]
    #save_prediction(symptom_list, prediction)
    
    return prediction

def check_disease_symptoms_2(input_symptoms, input_disease):

    df = pd.read_csv("dataset.csv") # until line 2461

    df_new = pd.read_csv("dataset.csv")
    symptom_columns = [col for col in df_new.columns if "Symptom" in col]
    df_new["Symptoms"] = df_new[symptom_columns].apply(lambda row: [symptom for symptom in row if pd.notna(symptom)], axis=1)


    collected_symptoms = []
    
    # Loop through rows where the disease matches input_disease
    for _, row in df[df['Disease'] == input_disease].iterrows():
        # Extract symptoms from the row and append non-NaN, non-duplicate symptoms to the list
        for symptom in row[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
                            'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8',
                            'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12',
                            'Symptom_13', 'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17']]:
            if pd.notna(symptom) and symptom not in collected_symptoms:
                collected_symptoms.append(symptom)
    print(collected_symptoms)

    # Check if the input symptoms match the disease symptoms
    if set(input_symptoms).issubset(set(collected_symptoms)):
        print(f"The symptoms match the disease {input_disease}.")
        return input_disease
        #return f"The symptoms match the disease {input_disease}."
    else:
        # If the symptoms don't match, find the most similar disease based on symptoms
        df_new['symptom_match'] = df_new['Symptoms'].apply(lambda symps: len(set(symps).intersection(set(input_symptoms))))
        
        # Find the disease with the highest number of matching symptoms
        best_match = df_new.loc[df_new['symptom_match'].idxmax()]
        best_match_disease = best_match['Disease']

        print(f"The symptoms do not match the disease {input_disease}. The closest match is {best_match_disease}.")
        return best_match_disease
        #return f"The symptoms do not match the disease {input_disease}. The closest match is {best_match_disease}."



def predict_med(dis):
    import pandas as pd
    import warnings

    warnings.simplefilter("ignore")
    #df = pd.read_csv("Warehouse_Data/dataset.csv")
    df = pd.read_csv("dataset.csv")
    disease = df["Disease"].unique()
    disease_meds = {"Disease": disease}
    disease_meds = pd.DataFrame(disease_meds)
    disease_meds
    medicines = ["clotrimazole", "Cetirizine", "esomeprazole", "ursodiol", "Benadryl", "omeprazole",
                 "You Have To Consult Doctor", "Fortamet", "ondansetron", "A MED 10", "A MED 11", "A MED 12", "A MED 13",
                 "A MED 14", "A MED 15", "A MED 16", "A MED 17", "A MED 18", "A MED 19", "A MED 20", "A MED 21",
                 "A MED 22", "A MED 23", "A MED 24", "A MED 25", "A MED 26", "A MED 27", "A MED 28", "A MED 29",
                 "Aspirin", "A MED 31", "A MED 32", "A MED 34", "A MED 35", "A MED 36", "A MED 37", "A MED 38",
                 "Benzoyl peroxide", "A MED 40", "A MED 41", "A MED 42"]
    disease_meds["Medicine"] = medicines
    return disease_meds[disease_meds["Disease"] == dis]["Medicine"].item()

# Symptom list examples for the model
symptom_input = ["skin_rash", "nodal_skin_eruptions", "itching"]
#symptom_input = ["constipation", "pain_during_bowel_movements", "pain_in_anal_region"]
#symptom_input = ["skin_rash", "nodal_skin_eruptions", "itching"]
#symptom_input = ["chills", "vomiting", "fatigue"]
#symptom_input = ["patches_in_throat", "high_fever", "extra_marital_contacts"]
#symptom_input = ["red_spots_over_body", "high_fever", "headache"]
#symptom_input = ["constipation", "abdominal_pain", "diarrhoea", "toxic_look_(typhos)"]
#symptom_input = ["vomiting", "yellowish_skin", "dark_urine", "nausea"]
#symptom_input = ["breathlessness", "sweating", "fainting"]
#symptom_input = ["bruising", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"]
#symptom_input = ["sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger", "muscle_weakness"]

predicted_disease = data_ripper(symptom_input)
print(f"The predicted disease is: {predicted_disease}")

validated_disease = check_disease_symptoms_2(symptom_input, predicted_disease)
predicted_medicine = predict_med(validated_disease)

save_prediction(symptom_input, validated_disease)

print(f"Medicine recommendation: {predicted_medicine}")
