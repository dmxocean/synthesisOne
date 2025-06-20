# Execution Guide

## Overview

This guide provides step-by-step instructions for executing the Translation Tasks Optimizer pipeline from raw data to running the application.

## Prerequisites

Ensure you have completed the installation steps in [INSTALL.md](INSTALL.md):
- Conda environment `Synthesis` is created and activated
- All dependencies are installed

## Execution Workflow

### Step 1: Data Preparation

#### Split Raw Data
If you have the raw Excel file (`data/raw/data.xlsx`), split it into individual CSV files:

```bash
python src/data/data_splitter.py
```

This will create four CSV files in `data/interim/`:
- `clients.csv` - Client profiles and requirements
- `schedules.csv` - Translator availability schedules
- `data.csv` - Historical translation task data
- `translatorsCostPairs.csv` - Translator rates by language pair

#### Verify Data Files
Check that all files are created:
```bash
ls data/interim/
```

### Step 2: Data Preprocessing

#### Base Preprocessing
Process the raw data for basic features:

```bash
python src/preprocessing/base.py
```

This creates processed files in `data/processed/base/`

#### Ranking Preprocessing (for ML Model)
If using the machine learning ranking model:

```bash
python src/preprocessing/ranking.py
```

This creates ranking-specific features in `data/processed/ranking/`

### Step 3: Model Training (Optional)

If you need to retrain the models with new data:

#### Train LightGBM Ranking Model
```bash
python src/models/ranking.py
```

This will:
- Load preprocessed data
- Train the LightGBM model
- Save the model to `models/ranking/lgbm_ranker_model.pkl`
- Save metadata to `models/ranking/metadata.json`

#### SAT Model
The SAT (Constraint Satisfaction) model doesn't require training as it uses rule-based constraints.

### Step 4: Run the Application

#### Launch Streamlit App
```bash
streamlit run app/app.py
```

Or alternatively:
```bash
python app/app.py
```

The application will:
- Open in your default web browser (usually at `http://localhost:8501`)
- Load the processed data and trained models
- Provide an interface for task assignment

#### Using the Application
- Select a project from the dropdown
- View available tasks for the project
- Choose assignment method (SAT or ML-based)
- Review and confirm assignments
- Export results if needed