#!/usr/bin/env python3
"""
GlucoCare Setup Script v4 — Windows / Mac / Linux
Run from inside the backend/ folder: python setup.py
"""
import sys, os, subprocess, shutil
from pathlib import Path

G  = lambda s: f"\033[92m{s}\033[0m"
Y  = lambda s: f"\033[93m{s}\033[0m"
R  = lambda s: f"\033[91m{s}\033[0m"
C  = lambda s: f"\033[96m{s}\033[0m"
B  = lambda s: f"\033[1m{s}\033[0m"
PY = sys.executable

def run(cmd, capture=True):
    return subprocess.run(cmd, shell=True, capture_output=capture, text=True)

def pip(*pkgs, quiet=True):
    q = "-q" if quiet else ""
    args = " ".join(pkgs)
    return run(f'"{PY}" -m pip install {args} --prefer-binary {q}')

print(C(B("\n╔══════════════════════════════════════════╗")))
print(C(B("║     GlucoCare Setup Script v4.0         ║")))
print(C(B("╚══════════════════════════════════════════╝\n")))
print(f"  Python: {PY}")
print(f"  Version: {sys.version.split()[0]}\n")

# ── 1. Python version check ────────────────────────────────────────
print(Y("[ 1/6 ] Checking Python version..."))
v = sys.version_info
if v < (3, 9):
    print(R("        ✗ Python 3.9+ required — https://python.org"))
    sys.exit(1)
print(G(f"        ✓ Python {v.major}.{v.minor} — compatible"))

# ── 2. Install packages ────────────────────────────────────────────
print(Y("\n[ 2/6 ] Installing dependencies..."))
print("        Upgrading pip...")
run(f'"{PY}" -m pip install --upgrade pip -q')

print("        Installing from requirements.txt...")
r = pip("-r requirements.txt")
if r.returncode == 0:
    print(G("        ✓ All packages installed"))
else:
    print(Y("        Installing in groups (fallback)..."))
    groups = [
        ["fastapi==0.115.6", "uvicorn[standard]==0.32.1"],
        ["sqlalchemy[asyncio]==2.0.36", "asyncpg==0.30.0", "alembic==1.14.0"],
        ["pydantic[email]==2.10.3", "pydantic-settings==2.6.1"],
        ["httpx==0.27.2", "supabase==2.10.0"],
        ["PyJWT==2.10.1", "cryptography==43.0.3"],
        ["numpy==2.1.3", "pandas==2.2.3", "joblib==1.4.2"],
        ["scikit-learn==1.5.2"],
        ["xgboost==2.1.3"],
        ["firebase-admin==6.6.0", "celery[redis]==5.4.0", "redis==5.2.1", "slowapi==0.1.9"],
        ["pytest==8.3.4", "pytest-asyncio==0.24.0", "pytest-cov==6.0.0"],
    ]
    for g in groups:
        label = g[0].split("==")[0]
        r2 = pip(*g)
        status = G(f"        ✓ {label} OK") if r2.returncode == 0 else Y(f"        ⚠ {label} partial")
        print(status)

# Verify
print("\n        Verifying imports...")
checks = [("fastapi","FastAPI"),("sqlalchemy","SQLAlchemy"),("pydantic","Pydantic"),
          ("xgboost","XGBoost"),("sklearn","scikit-learn"),("numpy","NumPy")]
missing = []
for mod, name in checks:
    r3 = run(f'"{PY}" -c "import {mod}; print({mod}.__version__)"')
    if r3.returncode == 0:
        print(G(f"          ✓ {name} {r3.stdout.strip()}"))
    else:
        print(R(f"          ✗ {name} not available"))
        missing.append(name)
if missing:
    print(Y(f"\n        Missing: {', '.join(missing)}"))
    print(Y(f"        Run manually: pip install {' '.join(m.lower().replace('-','_') for m in missing)}"))

# ── 3. .env ────────────────────────────────────────────────────────
print(Y("\n[ 3/6 ] Setting up .env file..."))
if not Path(".env").exists():
    src = ".env.example"
    if Path(src).exists():
        shutil.copy(src, ".env")
        print(G("        ✓ .env created from .env.example"))
    else:
        Path(".env").write_text(
            "APP_NAME=GlucoCare API\nDEBUG=true\n"
            "SUPABASE_URL=https://YOUR_PROJECT.supabase.co\n"
            "SUPABASE_ANON_KEY=REPLACE_ME\nSUPABASE_SERVICE_ROLE_KEY=REPLACE_ME\n"
            "SUPABASE_JWT_SECRET=REPLACE_ME\nPOSTGRES_USER=glucocare\n"
            "POSTGRES_PASSWORD=glucocare_dev_123\nPOSTGRES_DB=glucocare_db\n"
            "DATABASE_URL=postgresql+asyncpg://glucocare:glucocare_dev_123@db:5432/glucocare_db\n"
            "FIREBASE_CREDENTIALS_PATH=/app/firebase-adminsdk.json\n"
            "FIREBASE_PROJECT_ID=your-project\n"
            "MODEL_PATH=ml/xgboost_diabetes_model.joblib\n"
            "SCALER_PATH=ml/feature_scaler.joblib\nREDIS_URL=redis://redis:6379/0\n"
            "RATE_LIMIT_PER_MINUTE=60\nALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000\n"
        )
        print(G("        ✓ .env created with defaults"))
    print(Y("        ⚠  Edit .env: notepad .env"))
else:
    print(G("        ✓ .env exists"))

# ── 4. ML dirs ────────────────────────────────────────────────────
print(Y("\n[ 4/6 ] Preparing ML directories..."))
Path("ml/data").mkdir(parents=True, exist_ok=True)
# Ensure __init__.py files exist for imports to work
for d in ["core","models","services","api","api/v1","api/v1/endpoints","worker"]:
    Path(d).mkdir(parents=True, exist_ok=True)
    init = Path(d) / "__init__.py"
    if not init.exists():
        init.write_text("")
print(G("        ✓ Directories and __init__.py files ready"))

# ── 5. Train model ────────────────────────────────────────────────
print(Y("\n[ 5/6 ] Training XGBoost model..."))
model_file = Path("ml/xgboost_diabetes_model.joblib")
if model_file.exists():
    print(G("        ✓ Model already trained — skipping"))
else:
    print("        Training (30–90 sec, works offline)...")
    # Write a temp training script to avoid f-string-in-subprocess issues
    train_script = Path("_train_temp.py")
    train_script.write_text(
        "import sys, os\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from services.ml_service import train_and_save_model\n"
        "auc = train_and_save_model()\n"
        "print(f'AUC_RESULT:{auc:.4f}')\n"
    )
    r4 = run(f'"{PY}" _train_temp.py')
    train_script.unlink(missing_ok=True)
    out = (r4.stdout + r4.stderr).strip()
    if "AUC_RESULT:" in out:
        auc_val = [l for l in out.splitlines() if "AUC_RESULT:" in l][0].split("AUC_RESULT:")[1].strip()
        print(G(f"        ✓ Model trained! AUC = {auc_val}"))
        print(G(f"        ✓ Saved: ml/xgboost_diabetes_model.joblib"))
    else:
        print(Y("        ⚠  Training output:"))
        for line in out.splitlines()[-6:]:
            if line.strip():
                print(f"           {line}")
        print(Y("        App will use heuristic risk scoring (still fully functional)"))

# ── 6. Start instructions ──────────────────────────────────────────
print(Y("\n[ 6/6 ] Checking Docker..."))
dr = run("docker info")
if dr.returncode == 0:
    print(G("        ✓ Docker running"))
else:
    print(Y("        ⚠  Docker not running — use Option A"))

print(G(B("\n╔══════════════════════════════════════════╗")))
print(G(B("║         Setup Complete! 🎉              ║")))
print(G(B("╚══════════════════════════════════════════╝")))
print(f"""
  ━━━ START THE API (no Docker needed) ━━━
  {C('uvicorn main:app --reload --port 8000')}

  ━━━ THEN OPEN THESE IN YOUR BROWSER ━━━
  {C('http://127.0.0.1:8000/docs')}   ← Swagger API (use 127.0.0.1 not localhost)
  {C('http://127.0.0.1:8000/health')} ← Quick health check

  WHY 127.0.0.1 instead of localhost?
  Windows sometimes resolves localhost to IPv6 (::1) which fails.
  127.0.0.1 always works.

  ━━━ FILL IN CREDENTIALS ━━━
  {C('notepad .env')}   (Windows)
""")
