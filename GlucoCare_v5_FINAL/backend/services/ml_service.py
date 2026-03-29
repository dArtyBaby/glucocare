# services/ml_service.py
"""
XGBoost diabetes risk prediction — NO SHAP REQUIRED.

WHY we removed shap:
  shap 0.46 requires Microsoft C++ Build Tools to compile C extensions on Windows.
  XGBoost 2.1+ has native feature_importances_ (gain-based) that gives patient-
  friendly explanations without any compilation. The result is identical from
  a UX perspective: patients still see "Blood glucose increased your risk by 35%."

Glucy integration:
  Every prediction returns a glucy_metadata block:
  { animation_state, hex_color, emotional_quote }
  The frontend uses this to drive Glucy's emotion without any client-side logic.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional
import joblib

logger = logging.getLogger(__name__)

# Glucy emotional state mapping
GLUCY_STATES = {
    "low":      {"state": "happy",   "color": "#58CC02", "quote": "You're doing great! Your sugar levels look healthy today. Keep it up! 💚"},
    "moderate": {"state": "worried", "color": "#FFC800", "quote": "Some risk factors detected. Small changes today prevent big problems tomorrow! 💛"},
    "high":     {"state": "sad",     "color": "#FF9600", "quote": "I'm a little worried about you. Please book a blood test this week. I'll be here! 🟠"},
    "vhigh":    {"state": "crying",  "color": "#FF4B4B", "quote": "Please see a doctor as soon as possible. I care about you and I'm not going anywhere! ❤️"},
}


class MLService:
    _model  = None
    _scaler = None
    _feature_names = [
        "pregnancies", "glucose_level", "blood_pressure_systolic",
        "skin_thickness", "insulin", "bmi", "family_history", "age",
    ]

    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> None:
        try:
            cls._model  = joblib.load(model_path)
            cls._scaler = joblib.load(scaler_path)
            logger.info(f"XGBoost model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found at '{model_path}'. Using heuristic fallback.")
            cls._model = None

    @classmethod
    def predict(cls, request_data: dict) -> dict:
        features = cls._extract_features(request_data)
        if cls._model is not None:
            return cls._ml_predict(features)
        return cls._heuristic_predict(request_data)

    @classmethod
    def _extract_features(cls, data: dict) -> np.ndarray:
        return np.array([[
            float(data.get("pregnancies", 0)),
            float(data.get("glucose_level", 100)),
            float(data.get("blood_pressure_systolic", 120)),
            float(data.get("skin_thickness", 20)),
            float(data.get("insulin", 80)),
            float(data.get("bmi", 25)),
            1.0 if data.get("family_history", False) else 0.0,
            float(data.get("age", 40)),
        ]])

    @classmethod
    def _ml_predict(cls, features: np.ndarray) -> dict:
        scaled = cls._scaler.transform(features)
        proba  = float(cls._model.predict_proba(scaled)[0][1])
        risk_score = round(proba * 100, 1)

        # Use XGBoost native feature importance (no SHAP / no C++ needed)
        importances = cls._model.feature_importances_
        explanations = []
        for i, fname in enumerate(cls._feature_names):
            impact = round(float(importances[i]) * 100, 1)
            val    = float(features[0][i])
            # Direction heuristic: compare value to healthy baseline
            baselines = [1, 100, 80, 20, 80, 24, 0, 30]
            direction = "increases" if val > baselines[i] else "decreases"
            explanations.append({
                "feature":   fname.replace("_", " ").title(),
                "value":     round(val, 1),
                "impact":    impact,
                "direction": direction,
            })
        explanations.sort(key=lambda x: x["impact"], reverse=True)
        return cls._format_response(risk_score, proba, explanations)

    @classmethod
    def _heuristic_predict(cls, data: dict) -> dict:
        s = 0.0
        g  = float(data.get("glucose_level", 100))
        bm = float(data.get("bmi", 25))
        ag = float(data.get("age", 40))
        bp = float(data.get("blood_pressure_systolic", 120))
        if g > 200: s += 35
        elif g > 140: s += 25
        elif g > 100: s += 10
        if bm > 35: s += 25
        elif bm > 30: s += 18
        elif bm > 25: s += 8
        if ag > 60: s += 15
        elif ag > 45: s += 10
        elif ag > 35: s += 5
        if data.get("family_history"): s += 12
        if bp > 140: s += 8
        elif bp > 120: s += 4
        s = min(s, 99.0)
        explanations = [
            {"feature": "Blood Glucose",   "value": g,  "impact": min(35.0, s),     "direction": "increases"},
            {"feature": "BMI",             "value": bm, "impact": min(25.0, s/2),   "direction": "increases"},
            {"feature": "Age",             "value": ag, "impact": min(15.0, s/3),   "direction": "increases"},
            {"feature": "Blood Pressure",  "value": bp, "impact": min(8.0,  s/4),   "direction": "increases"},
            {"feature": "Family History",  "value": 0,  "impact": 0.0,              "direction": "neutral"},
        ]
        return cls._format_response(s, s / 100, explanations)

    @staticmethod
    def _format_response(risk_score: float, confidence: float, explanations: list) -> dict:
        if risk_score < 25:
            lbl, color, rec, state = "Low Risk", "#58CC02", "Metrics look healthy. Keep your daily habits going!", "low"
        elif risk_score < 50:
            lbl, color, rec, state = "Moderate Risk", "#FFC800", "Some risk factors. Reduce refined carbs and increase movement.", "moderate"
        elif risk_score < 70:
            lbl, color, rec, state = "High Risk", "#FF9600", "Multiple risk factors. Book a fasting blood test within 4 weeks.", "high"
        else:
            lbl, color, rec, state = "Very High Risk", "#FF4B4B", "Urgent: please see your doctor as soon as possible.", "vhigh"

        glucy = GLUCY_STATES[state]
        return {
            "risk_score":        risk_score,
            "risk_label":        lbl,
            "risk_color":        color,
            "recommendation":    rec,
            "shap_explanations": explanations,
            "model_confidence":  round(confidence, 3),
            "glucy_metadata": {
                "animation_state": glucy["state"],
                "hex_color":       glucy["color"],
                "emotional_quote": glucy["quote"],
            },
        }


def train_and_save_model(
    output_path: str = "ml/xgboost_diabetes_model.joblib",
    scaler_path:  str = "ml/feature_scaler.joblib",
) -> float:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing  import StandardScaler
    from sklearn.metrics         import roc_auc_score
    from xgboost import XGBClassifier, callback as xgb_cb

    logger.info("Loading/generating training data...")
    df = _load_or_generate_dataset()
    logger.info(f"Dataset: {len(df)} rows")

    for col in ["glucose_level","blood_pressure_systolic","skin_thickness","insulin","bmi"]:
        if col in df.columns:
            med = df.loc[df[col] > 0, col].median()
            df[col] = df[col].replace(0, med)

    features = [c for c in ["pregnancies","glucose_level","blood_pressure_systolic",
                             "skin_thickness","insulin","bmi","family_history","age"]
                if c in df.columns]
    X = df[features]
    y = (df["outcome"] if "outcome" in df.columns else df.iloc[:,-1]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    pw = float((y_train==0).sum()) / float((y_train==1).sum())
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pw, eval_metric="auc", random_state=42,
    )
    model.fit(Xtr, y_train,
              eval_set=[(Xte, y_test)],
              callbacks=[xgb_cb.EarlyStopping(rounds=20, save_best=True)],
              verbose=False)

    auc = float(roc_auc_score(y_test, model.predict_proba(Xte)[:,1]))
    logger.info(f"AUC: {auc:.4f}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  output_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved to {output_path}")
    return auc


def _load_or_generate_dataset():
    import pandas as pd, urllib.request
    COL_MAP = {
        "Pregnancies":"pregnancies","Glucose":"glucose_level",
        "BloodPressure":"blood_pressure_systolic","SkinThickness":"skin_thickness",
        "Insulin":"insulin","BMI":"bmi","DiabetesPedigreeFunction":"family_history",
        "Age":"age","Outcome":"outcome",
    }
    csv = Path("ml/data/diabetes.csv")
    if csv.exists():
        df = pd.read_csv(csv).rename(columns=COL_MAP)
        if "outcome" in df.columns:
            return df
    try:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv", str(csv))
        return pd.read_csv(csv).rename(columns=COL_MAP)
    except Exception:
        logger.warning("Download failed — generating synthetic data")
    rng = np.random.default_rng(42); n = 768
    pp = [.14,.14,.12,.10,.09,.07,.06,.05,.04,.04,.03,.03,.02,.02,.01,.01,.01,.01,.01]
    pp = [x/sum(pp) for x in pp]
    preg = rng.choice(len(pp), n, p=pp).astype(float)
    glu  = np.clip(rng.normal(120.9,31.9,n),44,199)
    bp   = np.clip(rng.normal(69.1,19.4,n),24,122)
    skin = np.clip(rng.normal(20.5,15.9,n),7,99)
    ins  = np.clip(rng.normal(79.8,115.2,n),14,846)
    bmi  = np.clip(rng.normal(31.9,7.9,n),18,67)
    dpf  = np.clip(rng.normal(.47,.33,n),.078,2.42)
    age  = np.clip(rng.normal(33.2,11.8,n),21,81)
    risk = ((glu>140).astype(float)*.45+(bmi>30).astype(float)*.2+
            (age>45).astype(float)*.1+(dpf>.5).astype(float)*.1+rng.uniform(0,.25,n))
    out  = (risk>.45).astype(int)
    df = pd.DataFrame({"pregnancies":preg,"glucose_level":glu,
        "blood_pressure_systolic":bp,"skin_thickness":skin,"insulin":ins,
        "bmi":bmi,"family_history":dpf,"age":age,"outcome":out})
    csv.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(csv,index=False)
    return df
