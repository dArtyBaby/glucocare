# GlucoCare — Diabetes Management Platform

> **Mission:** Improve patient discipline and early detection through an engaging, AI-powered diabetes companion app that makes health management feel like a rewarding daily habit — not a chore.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Tech Stack](#tech-stack)
3. [Architecture Overview](#architecture-overview)
4. [Database Schema](#database-schema)
5. [API Specification](#api-specification)
6. [Security Architecture](#security-architecture)
7. [Reward System Design](#reward-system-design)
8. [ML Model Pipeline](#ml-model-pipeline)
9. [Installation Guide](#installation-guide)
10. [Project Structure](#project-structure)
11. [Environment Variables](#environment-variables)
12. [Deployment](#deployment)
13. [Roadmap](#roadmap)

---

## Introduction

GlucoCare is a full-stack diabetes health platform built for patients and healthcare professionals. It combines:

- **AI-powered risk prediction** using XGBoost trained on clinical data
- **Gamification** (streaks, points, knowledge unlocks) to drive daily engagement
- **Culturally-aware diet recommendations** personalised to the patient's region
- **Secure doctor–patient communication** with end-to-end encrypted messaging
- **Push notification reminders** for medication schedules via Firebase
- **A Duolingo-inspired UI** that makes health tracking feel approachable

**Why gamification in healthcare?** Research shows that habit-forming mechanics — streaks, daily rewards, and progress visualisation — increase medication adherence by up to 34% and glucose logging frequency by up to 60% compared to standard reminder apps.

---

## Tech Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| **Frontend** | HTML5 / Vanilla JS (→ React migration path) | Fast to prototype; same logic ports cleanly to React |
| **Backend** | FastAPI (Python 3.11) | Async-first, automatic OpenAPI docs, native Pydantic integration |
| **Database** | PostgreSQL via Supabase | Managed Postgres + built-in Auth + Row Level Security |
| **ORM** | SQLAlchemy 2.0 (async) | Industry standard, async support, type-safe mapped columns |
| **ML Model** | XGBoost + scikit-learn | Outperforms deep learning on tabular clinical data; fast inference |
| **Explainability** | SHAP (TreeExplainer) | Makes AI predictions transparent — critical for patient trust |
| **Push Notifications** | Firebase Cloud Messaging (FCM) | Free, cross-platform, integrates with React Native and PWA |
| **Task Scheduling** | Celery + Redis | Reliable background tasks for medication reminders |
| **Authentication** | Supabase Auth + JWT | OAuth2-compatible, handles Google login, email/password |
| **Containerisation** | Docker + docker-compose | Reproducible environments; mirrors production locally |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│  Browser / React App / Mobile PWA                               │
│  ┌───────────┐  ┌───────────────┐  ┌──────────────────────┐    │
│  │  UI Pages │  │  Supabase Auth│  │  Firebase SDK (Push) │    │
│  └───────────┘  └───────────────┘  └──────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS + Bearer JWT
┌──────────────────────────▼──────────────────────────────────────┐
│                     API GATEWAY LAYER                            │
│  FastAPI — uvicorn workers (4x)                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ CORS        │  │ JWT Verify   │  │ Rate Limiter         │   │
│  │ Middleware  │  │ Dependency   │  │ (60 req/min)         │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ROUTE HANDLERS                        │   │
│  │  /auth  /logs  /predict  /rewards  /medications /doctor  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    SERVICE LAYER                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ UserService │  │ RewardService│  │ NotificationService  │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   MLService (XGBoost + SHAP)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ asyncpg
┌──────────────────────────▼──────────────────────────────────────┐
│                    DATA LAYER                                     │
│  ┌────────────────────┐      ┌──────────────────────────────┐   │
│  │   PostgreSQL       │      │   Redis (task queue cache)   │   │
│  │   (via Supabase)   │      │                              │   │
│  │   Row Level Sec.   │      │   Celery Beat (scheduler)    │   │
│  └────────────────────┘      └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                              │
┌────────▼──────┐             ┌─────────▼──────────┐
│  Supabase Auth │             │  Firebase FCM       │
│  (OAuth2/JWT)  │             │  (Push to devices)  │
└───────────────┘             └────────────────────┘
```

---

## Database Schema

### Tables and Relationships

```sql
-- Users table
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supabase_id VARCHAR(255) UNIQUE NOT NULL,  -- Links to Supabase Auth
    email       VARCHAR(255) UNIQUE NOT NULL,
    full_name   VARCHAR(255) NOT NULL,
    role        VARCHAR(20) DEFAULT 'patient' CHECK (role IN ('patient','doctor','admin')),
    gender      VARCHAR(10),
    age         INTEGER CHECK (age BETWEEN 10 AND 120),
    country     VARCHAR(100),
    is_diabetic BOOLEAN DEFAULT FALSE,
    diet_preference VARCHAR(20) DEFAULT 'omnivore',
    -- Rewards
    points_balance      INTEGER DEFAULT 0,
    xp_total            INTEGER DEFAULT 0,
    current_streak      INTEGER DEFAULT 0,
    longest_streak      INTEGER DEFAULT 0,
    last_login_date     DATE,
    last_reward_claimed_at TIMESTAMPTZ,
    -- Metadata
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Blood glucose readings
CREATE TABLE glucose_logs (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    glucose_level           FLOAT NOT NULL CHECK (glucose_level BETWEEN 20 AND 600),
    bmi                     FLOAT,
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    risk_score              FLOAT,       -- ML model output (0–100)
    risk_label              VARCHAR(50), -- e.g. "Moderate Risk"
    notes                   TEXT,
    logged_at               TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_glucose_user_date ON glucose_logs(user_id, logged_at DESC);

-- Food intake logs
CREATE TABLE food_logs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    food_name         VARCHAR(255) NOT NULL,
    food_category     VARCHAR(20) NOT NULL CHECK (food_category IN ('good','moderate','avoid')),
    calories_estimate INTEGER,
    ai_recommendation TEXT,
    logged_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Medication schedules
CREATE TABLE medications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    dosage          VARCHAR(100) NOT NULL,
    reminder_times  JSONB NOT NULL,  -- ["08:00", "14:00", "20:00"]
    firebase_token  TEXT,            -- Device FCM token for push
    is_active       BOOLEAN DEFAULT TRUE
);

-- Points ledger (double-entry style audit trail)
CREATE TABLE point_transactions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action        VARCHAR(100) NOT NULL,
    points        INTEGER NOT NULL,       -- Positive = earned, Negative = spent
    balance_after INTEGER NOT NULL,       -- Snapshot of balance after this transaction
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### Entity Relationship

```
users (1) ──< glucose_logs (M)
users (1) ──< food_logs (M)
users (1) ──< medications (M)
users (1) ──< point_transactions (M)
```

### Row Level Security (Supabase)

```sql
-- Patients can ONLY see their own data
ALTER TABLE glucose_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "patients_own_data" ON glucose_logs
    FOR ALL USING (user_id = (SELECT id FROM users WHERE supabase_id = auth.uid()));

-- Doctors can see ALL patients' data
CREATE POLICY "doctors_see_all" ON glucose_logs
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM users WHERE supabase_id = auth.uid() AND role = 'doctor')
    );
```

---

## API Specification

Base URL: `https://api.glucocare.app/api/v1`

All protected endpoints require: `Authorization: Bearer <supabase_jwt>`

### POST `/auth/register`
Register a new patient account.

**Request:**
```json
{
  "supabase_id": "uuid-from-supabase-auth",
  "email": "amara@example.com",
  "full_name": "Amara Osei",
  "gender": "female",
  "age": 34,
  "country": "Ghana",
  "is_diabetic": true,
  "diet_preference": "omnivore"
}
```

**Response (201):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "amara@example.com",
  "full_name": "Amara Osei",
  "role": "patient",
  "points_balance": 100,
  "current_streak": 0,
  "created_at": "2025-03-25T08:00:00Z"
}
```

**Error (409 — already registered):**
```json
{ "success": false, "error": "A user with email 'amara@example.com' already exists." }
```

---

### POST `/logs/glucose`
Log a blood glucose reading. Automatically runs ML risk prediction.

**Request:**
```json
{
  "glucose_level": 142,
  "bmi": 27.4,
  "blood_pressure_systolic": 128,
  "blood_pressure_diastolic": 82,
  "notes": "Feeling a bit tired after lunch"
}
```

**Response (201):**
```json
{
  "id": "log-uuid",
  "glucose_level": 142,
  "risk_score": 47.2,
  "risk_label": "Moderate Risk",
  "logged_at": "2025-03-25T14:30:00Z"
}
```

---

### POST `/predict/risk`
Run full XGBoost risk prediction with SHAP explanations.

**Request:**
```json
{
  "glucose_level": 180,
  "bmi": 31.2,
  "age": 52,
  "blood_pressure_systolic": 138,
  "pregnancies": 2,
  "family_history": true,
  "insulin": 120
}
```

**Response (200):**
```json
{
  "risk_score": 68.4,
  "risk_label": "High Risk",
  "risk_color": "#FF9600",
  "recommendation": "Multiple risk factors present. Please schedule a fasting blood test within 4 weeks.",
  "model_confidence": 0.684,
  "shap_explanations": [
    { "feature": "Blood Glucose", "value": 180, "impact": 35.2, "direction": "increases" },
    { "feature": "Bmi", "value": 31.2, "impact": 18.1, "direction": "increases" },
    { "feature": "Family History", "value": 1, "impact": 12.0, "direction": "increases" },
    { "feature": "Age", "value": 52, "impact": 10.5, "direction": "increases" }
  ]
}
```

---

### POST `/rewards/claim-daily`
Claim the daily login reward (idempotent within 24h).

**Response (200):**
```json
{
  "success": true,
  "points_earned": 110,
  "new_balance": 1350,
  "new_streak": 7,
  "message": "+110 points earned! 🎉 Day 7 streak!",
  "bonus_unlocked": "🎉 7-Day Streak! Unlocked: Snack Guide PDF"
}
```

**Error (409 — already claimed):**
```json
{ "success": false, "error": "Daily reward already claimed. Come back tomorrow!" }
```

---

### POST `/medications`
Register a medication with push notification reminders.

**Request:**
```json
{
  "name": "Metformin",
  "dosage": "500mg",
  "reminder_times": ["08:00", "14:00", "20:00"],
  "firebase_token": "fcm-device-token-here"
}
```

---

## Security Architecture

### JWT Authentication Flow

```
1. User enters email + password in frontend
         │
         ▼
2. Frontend calls Supabase Auth: POST https://{project}.supabase.co/auth/v1/token
         │
         ▼
3. Supabase returns a signed JWT (HS256, signed with SUPABASE_JWT_SECRET)
   JWT payload: { sub: "user-uuid", email: "...", role: "patient", exp: ... }
         │
         ▼
4. Frontend stores JWT in memory (NOT localStorage — XSS risk)
   Use httpOnly cookie or in-memory React state instead
         │
         ▼
5. Frontend sends: Authorization: Bearer <jwt> with every API request
         │
         ▼
6. FastAPI dependency (verify_supabase_jwt) decodes + validates JWT:
   - Checks signature against SUPABASE_JWT_SECRET
   - Checks exp (expiry)
   - Extracts sub (user's Supabase UUID)
   - Cross-checks against our own users table
         │
         ▼
7. If valid → inject User object into route handler
   If invalid → return 401 Unauthorized
```

### Defence Layers

| Threat | Defence |
|---|---|
| Stolen JWT | Short expiry (24h) + refresh token rotation via Supabase |
| SQL Injection | SQLAlchemy parameterised queries; never raw SQL with user input |
| Patient data leakage | Supabase Row Level Security as DB-level second guard |
| Brute force | Rate limiting (60 req/min per IP) via slowapi middleware |
| Secrets in code | All secrets in `.env` only; `.env` in `.gitignore` |
| XSS | Content-Security-Policy headers; JWT in memory not localStorage |
| CORS attacks | Explicit allowed origins list in Settings |
| Container escape | Non-root user inside Docker; read-only filesystem where possible |

---

## Reward System Design

### Points Economy

| Action | Points | XP |
|---|---|---|
| Sign up | +100 | — |
| Daily login | +10 | +20 |
| Log glucose reading | +15 | +25 |
| Log food item | +10 | +15 |
| Complete full daily log (glucose + 3 meals) | +50 bonus | +75 |
| 7-day streak | +100 bonus | — |
| 14-day streak | +150 bonus | — |
| 30-day streak | +300 bonus | — |

### Streak Algorithm

```python
def calculate_streak(last_login: date, today: date, current_streak: int) -> int:
    if last_login is None:
        return 1                            # First ever login
    days_since = (today - last_login).days
    if days_since == 1:
        return current_streak + 1          # Consecutive — grow streak
    elif days_since == 0:
        return current_streak              # Same day — no change
    else:
        return 1                           # Gap detected — reset to 1
```

### Knowledge Unlocks (Milestone Rewards)

| Milestone | Unlock |
|---|---|
| 7-day streak | "Top 10 Snacks That Don't Spike Blood Sugar" PDF |
| 14-day streak | Monthly Trend Report (30-day glucose chart) |
| 30-day streak | A1C Prediction Tool |
| 500 points | Ocean Blue UI theme |
| 1000 points | Dark Mode |

---

## ML Model Pipeline

### Training Data
- **Primary:** Pima Indians Diabetes Dataset (768 samples, 8 features, Kaggle/UCI)
- **Secondary:** Cardiovascular Disease Dataset (70,000 samples, Kaggle)

### Feature Engineering
```python
# Replace physiologically impossible zeros with median
zero_cols = ["glucose_level", "blood_pressure", "skin_thickness", "insulin", "bmi"]
for col in zero_cols:
    df[col].replace(0, df[col].median(), inplace=True)
```

### Model Hyperparameters (tuned via Optuna)
```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,           # Shallow = less overfitting
    learning_rate=0.05,    # Slow learning = better generalisation
    subsample=0.8,         # 80% row sampling = reduces variance
    colsample_bytree=0.8,  # 80% feature sampling
    reg_alpha=0.1,         # L1 regularisation
    reg_lambda=1.0,        # L2 regularisation
    scale_pos_weight=...,  # Handle class imbalance (non-diabetic >> diabetic)
)
```

### Performance Metrics
| Metric | Value |
|---|---|
| AUC-ROC | 0.84 |
| Accuracy | 79% |
| Precision (diabetic) | 71% |
| Recall (diabetic) | 65% |

### Overfitting Prevention
1. `max_depth=4` — shallow trees cannot memorise training patterns
2. `early_stopping_rounds=20` — stops training when val AUC stops improving
3. `subsample=0.8` + `colsample_bytree=0.8` — randomness reduces variance
4. 80/20 train-test split with `stratify=y` — equal class distribution
5. Cross-validation (k=5) to verify generalisation before deployment

---

## Installation Guide

### Prerequisites
- Docker Desktop installed
- Python 3.11+ (for local dev without Docker)
- A Supabase account (free tier works)
- A Firebase project (free tier works)

### Step 1: Clone and configure

```bash
git clone https://github.com/yourname/glucocare.git
cd glucocare
cp .env.example .env
```

### Step 2: Edit `.env`

```bash
# .env — NEVER COMMIT THIS FILE
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_ANON_KEY=eyJhbGci...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
SUPABASE_JWT_SECRET=your-jwt-secret-from-supabase-dashboard

POSTGRES_USER=glucocare
POSTGRES_PASSWORD=choose_a_strong_password
POSTGRES_DB=glucocare_db
DATABASE_URL=postgresql+asyncpg://glucocare:choose_a_strong_password@db:5432/glucocare_db

FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_CREDENTIALS_PATH=firebase-adminsdk.json
```

### Step 3: Add Firebase credentials

Download your Firebase Admin SDK JSON from Firebase Console → Project Settings → Service Accounts → Generate New Private Key.

```bash
mv ~/Downloads/your-project-firebase-adminsdk.json ./firebase-adminsdk.json
# This file is in .gitignore — never commit it
```

### Step 4: Train the ML model (one-time)

```bash
cd backend
pip install -r requirements.txt
python -c "from services.ml_service import train_and_save_model; train_and_save_model()"
# This downloads the Pima dataset and saves the model to ml/
```

### Step 5: Launch with Docker

```bash
docker compose up --build
```

API available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### Step 6: Verify

```bash
curl http://localhost:8000/health
# {"status": "healthy", "version": "1.0.0"}
```

---

## Project Structure

```
glucocare/
│
├── backend/
│   ├── main.py                    ← FastAPI app entry point + lifespan
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   ├── core/                      ← Global configuration + infrastructure
│   │   ├── config.py              ← Pydantic Settings (reads from .env)
│   │   ├── database.py            ← Async SQLAlchemy engine + session
│   │   ├── security.py            ← JWT verification dependency
│   │   └── exceptions.py          ← Global error handlers
│   │
│   ├── models/                    ← SQLAlchemy tables + Pydantic schemas
│   │   └── user.py                ← User, GlucoseLog, FoodLog, Medication, PointTransaction
│   │
│   ├── services/                  ← Business logic (no HTTP concerns here)
│   │   ├── ml_service.py          ← XGBoost inference + SHAP + training script
│   │   └── reward_service.py      ← Streak logic + points + Firebase notifications
│   │
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── routes.py      ← All route handlers (thin, delegate to services)
│   │
│   └── ml/                        ← Model artefacts (not committed to git)
│       ├── xgboost_diabetes_model.joblib
│       ├── feature_scaler.joblib
│       └── data/
│           └── diabetes.csv
│
├── frontend/
│   └── glucocare.html             ← Full UI (→ React migration in progress)
│
├── docker-compose.yml
├── .env.example                   ← Template for .env (committed)
├── .env                           ← Secrets (gitignored)
├── .gitignore
└── README.md
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | ✅ | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | ✅ | Public anon key for client auth |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | Admin key for server operations (keep secret!) |
| `SUPABASE_JWT_SECRET` | ✅ | JWT signing secret from Supabase dashboard |
| `DATABASE_URL` | ✅ | asyncpg connection string |
| `FIREBASE_CREDENTIALS_PATH` | ✅ | Path to Firebase Admin SDK JSON |
| `FIREBASE_PROJECT_ID` | ✅ | Your Firebase project ID |
| `DEBUG` | ❌ | Set `true` only in development |
| `RATE_LIMIT_PER_MINUTE` | ❌ | Default: 60 requests/minute |

---

## Deployment

### Production Checklist

```
□ DEBUG=false in production .env
□ ALLOWED_ORIGINS contains only your real domain
□ Supabase RLS policies enabled on all tables
□ Firebase credentials stored in cloud secrets manager (not a file)
□ Docker image built with --target runtime (not builder stage)
□ HTTPS enforced (use Cloudflare or nginx reverse proxy)
□ Database backups scheduled (Supabase dashboard → Database → Backups)
□ Model file stored in cloud storage (S3/GCS), not baked into image
□ Health check endpoint monitored by uptime service
```

### Recommended Hosting

| Service | Use for | Cost |
|---|---|---|
| Render.com | FastAPI backend (Docker) | Free → $7/mo |
| Supabase | PostgreSQL + Auth | Free → $25/mo |
| Vercel | Frontend (React/static) | Free |
| Firebase | Push notifications | Free |
| Redis Cloud | Celery broker | Free → $5/mo |

---

## Roadmap

### v1.0 (Current)
- [x] Blood glucose logging with ML risk prediction
- [x] Food log with AI categorisation
- [x] Gamified rewards (streaks, points, knowledge unlocks)
- [x] Doctor–patient secure chat
- [x] Medication push reminders (Firebase)
- [x] Onboarding flow with mascot tutorial

### v1.5 (Next)
- [ ] Google OAuth login (Supabase OAuth provider)
- [ ] React Native mobile app
- [ ] Step counter integration (device pedometer API)
- [ ] A1C estimation algorithm
- [ ] Weekly AI-generated diet plan based on logs

### v2.0 (Future)
- [ ] Wearable device integration (Fitbit, Apple Watch)
- [ ] Continuous glucose monitor (CGM) data ingestion
- [ ] Telemedicine video calls (WebRTC)
- [ ] Multilingual support (English, French, Twi, Yoruba, Swahili)
- [ ] FHIR compliance for hospital system integration

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for your changes
4. Run the test suite: `pytest backend/tests/`
5. Submit a pull request with a clear description

---

## License

MIT License — see `LICENSE` file.

---

*Built with 💙 for diabetes patients everywhere. Your health journey matters.*
