# Air System Problem - Machine Learning

Predictive maintenance model for truck air systems. The goal is to identify vehicles with air system defects before failure, reducing corrective maintenance costs.

## Problem

Trucks with air system defects generate high corrective maintenance costs ($500/failure). The model helps flag defective vehicles early so preventive action can be taken at a much lower cost ($25/vehicle).

## Dataset

| File | Description |
|------|-------------|
| `air_system_previous_years.csv` | Historical data used for training |
| `air_system_present_year.csv` | Current year data used for evaluation |

> Data files are not versioned (listed in `.gitignore`).

## Models Evaluated

- **Random Forest** — best precision for the `pos` class
- **SVC** — solid baseline
- **Gradient Boosting** — best recall for the `pos` class; also tuned with GridSearchCV

## Cost Analysis

| Event | Cost |
|-------|------|
| False Negative (missed defect) | $500 |
| False Positive (unnecessary maintenance) | $10 |
| True Positive (correct preventive action) | $25 |

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Project Structure

```
Air-system-problem/
├── main.py                          # Full pipeline: cleaning, training, evaluation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignores data files, cache, venvs
└── README.md
```
