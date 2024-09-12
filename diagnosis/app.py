# This script sets up a web service that listens for GET requests at the /symptom_predict/<word> endpoint. 
# When a request is made to this endpoint with a symptom in the URL, the application queries OpenAI's GPT-3.5 model for advice or information related to that symptom. 
# The response is then returned as a JSON object, which could be used by other applications or interfaces.




#modified flask application to interact with SQLite database. 

#imports
from flask import Flask, request
from openai import OpenAI
import os
from dotenv import load_dotenv
import sqlite3 
from datetime import datetime


load_dotenv()

# Load OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

client = OpenAI(api_key=openai_api_key)
app = Flask(__name__)

db_path = "symptom_predictions.db"

#initialiing sqlite database 
def init_database():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp TEXT,
                           symptom TEXT,
                           diagnosis TEXT)''')
        conn.commit() 
init_database() 


@app.route("/symptom_predict/<word>")
def matching_symptoms(word):
    # Use GPT-3 turn-based chat to find matching symptoms

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical professional, here to advise me on relieving the symptom I have. Be concise."},
            {"role": "user", "content": word},
        ],
        max_tokens=100,
    )

    gpt3_response = completion.choices[0].message.content
    print(completion.choices[0].message.content)

    #saving this result to sqlite database 
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor() 
        cursor.execute("INSERT INTO predictions (timestamp, symptom, diagnosis) VALUES (?, ?, ?)",
                       (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), word, gpt3_response))
        conn.commit()

    return {"symptom": gpt3_response}

if __name__ == "__main__":
    app.run(debug=True)
