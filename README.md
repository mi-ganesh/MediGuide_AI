# MediGuide AI — AI-Powered Healthcare Assistance Platform

## Overview

MediGuide AI — AI-Powered Healthcare Assistance Platform is a machine learning-based web application that predicts possible diseases based on user-reported symptoms. It combines a Python backend with a React frontend to provide an interactive symptom assessment experience.

The application uses a trained machine learning model to generate possible disease predictions and an adaptive questioning approach to gather additional symptom information.

**Note:** This project is intended for educational and research purposes. Predictions are preliminary and should not be considered a medical diagnosis.

## Features

- **Symptom Selection:** Select symptoms through an interactive user interface.
- **Disease Prediction:** Generate possible disease predictions using a trained machine learning model.
- **Adaptive Questioning:** Ask follow-up questions to gather more relevant symptom information.
- **Prediction Results:** Display possible conditions and model prediction information.
- **Disease Information:** Provide additional information about predicted conditions.
- **REST API:** Connect the React frontend to the Python backend.

## Technology Stack

### Frontend
- React.js
- Vite
- JavaScript
- CSS

### Backend
- Python
- Flask or the framework configured in `backend/app.py`
- REST API

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Trained classification model

### Data Storage
- CSV dataset for disease and symptom information
- JSON file for additional disease information

## Project Structure

```text
disease-predictor/
├── backend/
│   ├── app.py
│   ├── engine.py
│   ├── train_model.py
│   ├── requirements.txt
│   ├── disease_info.json
│   ├── disease_info.template.json
│   ├── data/
│   │   └── Diseases_and_Symptoms_dataset.csv
│   └── model/
│       ├── model.joblib
│       └── metrics.json
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

Install the following before running the project:

- Python 3.10 or a version compatible with the backend dependencies
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd disease-predictor
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Set Up the Backend

Open a terminal in the project root.

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Ensure the dataset and trained model are available at the paths expected by the backend. If the model has not been trained, follow the instructions in `backend/train_model.py` to generate it.

Start the backend using the command appropriate for the framework configured in `backend/app.py`. For example, if the application uses Flask:

```bash
python backend/app.py
```

### 3. Set Up the Frontend

Open another terminal:

```bash
cd frontend
npm install
```

If environment variables are required, copy `.env.example` to `.env` and configure the appropriate backend API URL. Never commit your actual `.env` file or secret credentials.

Start the development server:

```bash
npm run dev
```

Open the local URL displayed in the terminal by Vite.

## How It Works

1. The user selects or reports symptoms.
2. The frontend sends the symptom information to the backend.
3. The backend processes the input and uses the trained model to predict possible conditions.
4. The application may ask follow-up questions to collect additional symptom information.
5. The results and relevant disease information are returned to the frontend for display.

## Machine Learning Model

The project uses a trained classification model to map symptom inputs to possible disease categories. The saved model is loaded by the backend to generate predictions.

The `backend/model/metrics.json` file can contain model evaluation results, such as accuracy, precision, recall, and F1-score.

Evaluation metrics describe performance on the evaluated dataset; they do not establish clinical accuracy or reliability.

## Dataset

The dataset contains disease labels and corresponding symptom features. The backend uses these features to prepare symptom inputs for prediction.

If the dataset is not included in the repository, obtain it from an authorized source and place it at the path expected by the backend.

Review the dataset license and redistribution conditions before publishing it.

## Limitations

- Predictions depend on the quality and coverage of the training dataset.
- Some diseases have overlapping symptoms.
- Model confidence scores are not necessarily calibrated probabilities of having a disease.
- The application is not a substitute for a qualified healthcare professional.
- Real-world clinical validation is required before any medical use.

## Future Improvements

- Expand and validate the training dataset.
- Improve model evaluation and probability calibration.
- Add multilingual symptom input.
- Improve accessibility and mobile responsiveness.
- Integrate location-based healthcare recommendations.
- Add automated testing and deployment.

## Contributing

Contributions, suggestions, and bug reports are welcome. Please create an issue or submit a pull request describing the proposed change.

## License

No license has been specified yet. Add an appropriate license file before permitting others to reuse or redistribute this project.

## Disclaimer

This project is developed for educational and research purposes only. It does not provide professional medical advice, diagnosis, or treatment recommendations. Consult a qualified healthcare professional for medical concerns. Seek urgent medical care for emergency symptoms.
