from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # project root
DATA = BASE / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

for p in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)