# Heals on Wheels

## Overview

**Heals on Wheels** is an autonomous medical assistant robot designed to enhance patient care. It independently navigates to patients using sensor fusion of LiDAR and ultrasonic sensors to avoid both static and dynamic obstacles. The robot collects biometric data from patients and diagnoses potential diseases using a **Random Forest Classifier** trained on a symptom-disease dataset. Additionally, it provides medication recommendations powered by **OpenAI’s GPT-3.5-turbo**. The system stores diagnoses and recommendations in a centralized **SQLite** database for easy access by medical staff, optimizing patient diagnostics and reducing the burden on healthcare professionals.

## Features

- **Autonomous Navigation**: Utilizes sensor fusion of LiDAR and ultrasonic sensors to navigate in dynamic environments like hospitals.
- **Symptom-Based Disease Diagnosis**: Uses a **Random Forest Classifier** to predict diseases based on patient-reported symptoms.
- **Medication Recommendation**: Integrates **OpenAI GPT-3.5-turbo** to provide human-readable medication advice.
- **Centralized Database**: Stores patient data, diagnoses, and medication recommendations using an **SQLite3** database.
- **Cross-Validation**: Ensures robust model performance with cross-validation techniques.

## Hardware

- **Microcontroller**: Arduino Uno R4
- **Sensors**:
  - **LiDAR**: TF-Luna for primary object detection.
  - **Ultrasonic Sensors**: HC-SR04 for supplementary obstacle detection.
- **Mecanum Wheels**: Provide enhanced mobility in tight spaces, crucial for hospital environments.

## Software

- **Disease Prediction Model**: Random Forest Classifier implemented with **scikit-learn**, trained on a symptom-disease dataset.
- **Medication Recommendations**: Powered by **OpenAI GPT-3.5-turbo**, offering suggestions based on predicted diagnoses.
- **Database**: **SQLite3** is used for storing patient data, diagnosis results, and medication advice.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heals-on-wheels.git

## Usage 
Launch the robot by uploading the Arduino code found in the arduino_code folder.
Run the Python script for diagnosis and medication recommendations
python heals_on_wheels.py


## Contributors

- **Dayoung (Skye) Nam** - University of Toronto Computer Engineering

- **James Kim** - University of Toronto Electrical Engineering 
  
- **Nicholas Carbones** - University of Toronto Computer Engineering

- **Richard Tan** - University of Toronto Computer Engineering

- **Jung Ho Ham** - University of Toronto Electrical Engineering

- **Ryan Zhu** - University of Toronto Computer Engineering

- **Christine Lee** - University of Toronto Biomedical Systems Engineering

- **Joon Park** - University of Toronto Electrical Engineering
