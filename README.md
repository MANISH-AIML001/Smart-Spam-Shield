# Smart Spam Shield

Smart Spam Shield is a web-based spam detection system built with Flask and machine learning. It allows users to sign up or log in, paste a message, and get a spam-risk verdict such as safe, suspicious, or spam. The app combines a trained ML model to detect common spam patterns such as urgency words, links, capital-letter usage, and suspicious message structure.

## Project Overview

This project is designed to:

- detect spam messages in SMS, email, or chat-style text,
- classify messages with a confidence score,
- show explainable feature analysis behind the final decision,
- provide a simple login-based web interface for end users.

The application is built around a Flask backend and a trained Logistic Regression model saved in the `models` directory.

---

## Features

- User authentication with signup/login/logout using SQLite
- Real-time message scanning through a web UI
- Spam verdict with confidence levels
- Trigger word detection such as free, win, urgent, click, bonus, etc.
- Feature breakdown showing why a message was classified as suspicious or spam
- Training pipeline for updating the model with new datasets

---

## Tech Stack

- Python
- Flask
- SQLite
- scikit-learn
- pandas
- NumPy
- NLTK
- HTML/CSS/JavaScript

---

## Project Structure

```text
Spam Detector/
├── app.py                  # Main Flask application
├── training.py             # Model training script
├── utils.py                # Dataset cleaning and feature logic
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment config
├── report.md               # Project report summary
├── testing.py              # Testing / validation script
├── users.db                # SQLite database created at runtime
├── dataset/
│   ├── testing/
│   └── training/
├── models/
│   └── v2/
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   ├── index.html
│   ├── login.html
│   └── signup.html
└── README.md
```

---

## How It Works

### 1. User Authentication

When a user opens the app, they first sign up or log in. The app stores user details in a local SQLite database named `users.db`.

- Passwords are hashed before storage.
- Login sessions are managed with Flask session cookies.
- Protected routes are blocked unless a user is logged in.

### 2. Message Analysis

On the dashboard, the user enters a message and clicks the scan button. The backend receives the text through the `/api/v1/predict` route.

The message is processed with `extract_features()` from `utils.py`, which calculates:

- message length,
- number of digits,
- uppercase count,
- special characters,
- URLs found,
- phone number count,
- exclamation marks,
- uppercase ratio,
- urgency-word score.

### 3. Model Prediction

The trained model is loaded from `models/v2/lr_model.pkl` using `pickle`.

The model predicts whether the message is:

- safe,
- suspicious,
- spam.

The prediction is displayed together with confidence percentages and a list of detected trigger words.

### 4. Why the Verdict Was Given

The app also shows the feature breakdown for the scanned message. For example, if the text contains many urgency cues like free, winner, claim, or urgent, the app may mark it as suspicious or spam.

---

## Running the Project Locally

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd "Spam Detector"
```

### Step 2: Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create a .env file
``` bash
nano .env
```

.env

``` bash
SECRET_KEY=<your_secret_key>
```

### Step 5: Start the application

```bash
python app.py
```

### Step 6: Open the app in a browser

Visit:

```text
http://127.0.0.1:5000
```

You should see the login page. Create an account and sign in to use the spam detector.

---

## Training the Model

If you want to retrain the classifier using the provided dataset, run:

```bash
python training.py
```

This script:

- loads the dataset,
- cleans and standardizes the text,
- trains a TF-IDF + Logistic Regression pipeline,
- evaluates the model,
- saves the trained model in `models/v2/lr_model.pkl`.

---

## Important Notes

- The project creates `users.db` automatically when the app starts.
- The model file is expected to exist in `models/v2/lr_model.pkl`.
- If the model is missing, retrain it with `python training.py` before running the app.
- The app uses a lightweight local SQLite setup, so it is ideal for local development and demonstration.

---

## Recommended Workflow for New Developers

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies from `requirements.txt`
4. Run `python training.py` if the model is missing
5. Start the app with `python app.py`
6. Open the app in the browser and create a user account
7. Test with sample spam messages like:
   - "Congratulations! You have won a free prize. Claim now!"
   - "Urgent: click this link to receive your bonus"

---

## License

This project is for educational and demonstration use.

---

## Summary

Smart Spam Shield is a practical end-to-end spam detection project that combines machine learning, web development, and user authentication. It is a strong example of how a trained ML model can be wrapped in a user-friendly application and deployed as a local web service.
