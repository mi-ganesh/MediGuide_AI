# MediGuide: adaptive disease prediction website (React + Flask)

```
disease-predictor/
├── backend/
│   ├── data/                  <- put Diseases_and_Symptoms_dataset.csv here
│   ├── model/                 <- model.joblib + metrics.json are created here
│   ├── engine.py              shared model code (blending, calibration, next-question logic)
│   ├── train_model.py         advanced training script (replaces the notebook training)
│   ├── app.py                 API: /api/health, /api/symptoms, /api/assess
│   ├── disease_info.json      descriptions/precautions (add more diseases here)
│   └── requirements.txt
└── frontend/                  React (Vite) website
```

## 1. Train the model (once)
```
cd backend
pip install -r requirements.txt
# copy Diseases_and_Symptoms_dataset.csv into backend/data/
python train_model.py --fast     # 1-2 min sanity check
python train_model.py            # full training (about 10-25 min)
python train_model.py --search   # also compares several configs (slowest, best)
```
It prints accuracy for 1, 2, 3 and 5 given symptoms and saves `model/metrics.json`.

## 2. Run the site (two terminals)
```
cd backend  && python app.py      # http://127.0.0.1:5000
cd frontend && npm install && npm run dev    # http://localhost:5173
```

## How it works
1. The user adds starting symptoms.
2. `POST /api/assess` returns the top 3 conditions and the **next best question**
   (the symptom with the highest expected information gain).
3. The user answers Yes / No / Not sure; the page calls `/api/assess` again.
4. It stops when confidence reaches 80%, after 7 questions, or when no question helps.

Change the limits in `backend/app.py` (`CONFIDENCE_THRESHOLD`, `MAX_QUESTIONS` env vars)
and `MAX_QUESTIONS` at the top of `frontend/src/App.jsx`.

## Add disease descriptions
Training writes `backend/disease_info.template.json` listing all diseases. Copy entries you
fill in (description, precautions, specialist) into `disease_info.json`. Missing ones fall back
to generic advice. Always have a doctor review medical text before publishing.

## Production
- Frontend: set `VITE_API_URL=https://your-api`, run `npm run build`, host `dist/`.
- Backend: `pip install gunicorn && gunicorn app:app`, set `CORS_ORIGINS=https://your-site`.

Not a medical device. Results are not a diagnosis.
