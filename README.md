# 📰 Fake News Detector using Deep Learning (LSTM)

A deep learning-based Fake News Detection application built using **TensorFlow**, **FastAPI**, and **Streamlit**. The model leverages **Natural Language Processing (NLP)** techniques with **TextVectorization** and an **LSTM (Long Short-Term Memory)** network to classify news articles as **Real** or **Fake**.

Note: I wasn't able to get any server hosting service for free which can handle my model loading with all those parameters thats why there is only a local hosting option here!

---

# 📌 Motivation

This project was built to strengthen my understanding of **Deep Learning** and **Natural Language Processing (NLP)** by implementing a complete end-to-end machine learning application.

The primary goals of this project were to:

* Learn text preprocessing using TensorFlow's **TextVectorization** layer.
* Build and train an **LSTM-based** neural network for text classification.
* Deploy a trained deep learning model using **FastAPI**.
* Create an interactive web interface using **Streamlit**.
* Understand how machine learning models are served in production.

---

# 🛠️ Tech Stack

## Model Training

* TensorFlow
* Pandas
* NumPy

## Backend (Model Serving)

* FastAPI

## Frontend

* Streamlit

---

# 📂 Project Workflow

```
News Article
      │
      ▼
TextVectorization
      │
      ▼
LSTM Model
      │
      ▼
FastAPI Backend
      │
      ▼
Streamlit Frontend
```

---

# 🚀 Getting Started

## Prerequisites

> **Note:** TensorFlow currently does not support **Python 3.14**. This project uses **Python 3.13**, so make sure to create and activate a Python 3.13 virtual environment before installing the dependencies.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SubhamJM/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2. Install the dependencies

**Windows**

```bash
pip install -r requirements.txt
```

**Linux/macOS**

```bash
pip3 install -r requirements.txt
```

---

## Running the Application

### Start the FastAPI server

From the project root directory, run:

```bash
python api/main.py
```

or on Linux/macOS:

```bash
python3 api/main.py
```

---

### Start the Streamlit frontend

Open another terminal in the project root and run:

```bash
streamlit run app/app.py
```

The Streamlit application will open automatically in your browser.

---

# 📖 Features

* Fake news classification using an LSTM neural network
* Text preprocessing with TensorFlow TextVectorization
* REST API powered by FastAPI
* Simple and interactive Streamlit interface

---

# 📷 Preview


* Example of a Fake news:
  
<img width="1286" height="1038" alt="image" src="https://github.com/user-attachments/assets/151eff02-c624-4b72-a70e-b3ece6d9321a" />




* Example of a Real news:

<img width="1286" height="1038" alt="image" src="https://github.com/user-attachments/assets/d354c1b0-7599-4d97-8e9d-1a4034945614" />


