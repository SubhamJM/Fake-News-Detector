# Fake News Detector (Deep learning using LSTM)

## Motivation for this Project

I wanted to implement my deep learning skills like NLP using TextVectorization and using RNN models like LSTM
I also wanted to learn how to handle model deployement using FastAPI to serve the model and Streamlit for a basic frontend for the user to interact and get predictions

## Tech stacks used

1. Model training:
   a. Tensorflow
   b. Pandas
   c. Numpy
   
2. Model Serving (Backend)
   FastAPI

3. User Interface (frontend)
   Streamlit


## How to use

Important: tensorflow is not available on the latest python 3.14 so this project uses python 3.13 for compatibility so use python 3.13 virtual environment before running anything

1. clone this github repository on your local machine
2. type this in the terminal in root project directory:
  on windows:
   '''pip install -r requirements.txt'''
   on mac/linux
   '''pip3 install -r requirements.txt'''
4. Run the main.py file using python3 in the api folder
5. In terminal in the root project directory type:
   '''streamlit run app/app.py'''

   This will open up streamlit interface in your browser. Congratulations the setup is complete!
   
