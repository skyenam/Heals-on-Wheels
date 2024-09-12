#run "python db.py" to query the database. 


import sqlite3
import pandas as pd

#path to sqLite database
db_path = "symptom_predictions.db"

def fetch_all_predictions():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM predictions")
    
    rows = cursor.fetchall()
    
    df = pd.DataFrame(rows, columns=['ID', 'Timestamp', 'Symptoms', 'Diagnosis'])
    
    conn.close()
    
    return df

predictions_df = fetch_all_predictions()

print(predictions_df)
