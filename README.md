# Obesity Risk Assessment

A simple, interactive app that estimates a person’s obesity risk from basic lifestyle and anthropometric inputs. Built with Python and Streamlit, with a light preprocessing layer and a trained ML model. The repository includes a Streamlit app, model and preprocessing artifacts, and a minimal dependency set for easy deployment.

Hosted Link to Try This Out : https://obesityriskassessment.streamlit.app/

## Features

- ML-powered risk prediction using a pre-trained model. 
- Input preprocessing kept under Preprocessing/ for reproducibility. 
- Streamlit UI for quick, form-based prediction. 
- Deploy-friendly: includes requirements.txt, a runtime.txt, and a .devcontainer/ for Codespaces/Containers

## Quick Start

    -Install Dependencies
    pip install -r requirements.txt

    -Locate the Random Forest Model in the Model Folder.
    -Run the file and it will create the Model required for this system.

    -Run the system
    streamlit run app.py
