# 🏀 NBA Match Outcome Prediction using Machine Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project for predicting NBA game outcomes. This repository details a full data pipeline, from exploratory data analysis and advanced feature engineering to hyperparameter tuning, model comparison, and deployment via a REST API. The project analyzes 8,281 games across 7 seasons (2017/18–2023/24) and implements a robust methodology to prevent data leakage in time-series forecasting.

---

## 📖 Project Overview

This project tackles the challenge of predicting NBA match winners by leveraging a rich dataset of team and player statistics. The core of the project is a sophisticated feature engineering pipeline that generates rolling statistical averages over various time windows, ensuring that models are trained only on historical data to prevent lookahead bias.

Two distinct models, a **Logistic Regression** classifier and a **Multi-Layer Perceptron (MLP) Neural Network**, are trained and rigorously evaluated. A key finding is the superior predictive power of advanced basketball metrics (e.g., True Shooting Percentage, Offensive/Defensive Rating) over traditional counting stats (e.g., total points, rebounds). The final, best-performing model is automatically selected and served via a FastAPI application.

**Key Features**:
-   **End-to-End MLOps Pipeline**: From data exploration (`1_EDA.ipynb`) and feature engineering (`2_Feature_Engineering.ipynb`, `3_Feature_Engineering2.ipynb`) to automated training (`train_*.py`) and API deployment (`predict_api.py`).
-   **Robust Time-Series Validation**: Employs `TimeSeriesSplit` cross-validation to ensure model performance is realistically evaluated on out-of-time data.
-   **Data Leakage Prevention**: Meticulously crafted rolling averages (20, 30, 40-game windows and season-long) that strictly use past data for feature creation.
-   **Comprehensive Hyperparameter Tuning**: Utilizes `GridSearchCV` for Logistic Regression and `RandomizedSearchCV` for the Neural Network to find optimal model configurations across different data variants and feature sets.
-   **Model Comparison**: A data-driven approach automatically selects the best model (Logistic Regression) based on cross-validation performance to be used in the final API.
-   **REST API for Predictions**: A static FastAPI (`predict_api.py`) serves predictions for any game within the 2017-2024 dataset, demonstrating a practical application of the trained models.

---

## 📁 Project Structure

The repository is organized to follow a logical machine learning workflow:

```plaintext
.
├── data/
│   ├── start/                  # Raw SQLite database
│   └── transformed/            # Processed data, including merged moving averages
├── graphs/                     # Output charts, including correlation and confusion matrices
├── models/                     # Saved model artifacts (pipelines, configs, thresholds)
├── notebooks/
│   ├── 1_EDA.ipynb             # Exploratory Data Analysis
│   ├── 2_Feature_Engineering.ipynb # Initial feature creation (streaks, rolling stats)
│   └── 3_Feature_Enginnering2.ipynb # Merging and finalizing feature sets
├── src/                        # Helper Python modules
│   ├── data_processing.py      # Functions for data cleaning and feature engineering
│   └── model_utils.py          # Functions for model training, evaluation, and artifact saving
├── train_logistic_regression.py # Script for training and evaluating the LR model
├── train_neural_network.py    # Script for training and evaluating the NN model
├── predict_api.py               # FastAPI application for serving predictions
└── requirements.txt             # Project dependencies

```
---

## ⚙️ Methodology

### 1. Data Sourcing and EDA
The dataset, containing 8,281 games over 7 seasons (2017/18–2023/24), was built using scraped data from basketball-reference.com.

-   **Primary Scraper**: [luke-lite's scraping notebook](https://github.com/luke-lite/NBA-Web-Scraper/blob/main/NBA-Web-Scraper-Notebook.ipynb) for game and player box scores.
-   **Supplementary Scraper**: An improved version of [kyleskom's scraper](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/tree/master/src/Process-Data) for team rankings and form indicators.

Initial analysis (`1_EDA.ipynb`) revealed the distribution of key statistics and identified outliers and data errors, such as incorrect values in `AST%`, which were subsequently cleaned.

### 2. Feature Engineering
This was the most critical phase, focused on creating meaningful predictors without leaking future information. The process is detailed in `2_Feature_Engineering.ipynb` and `3_Feature_Enginnering2.ipynb`.

-   **Temporal Aggregation**: To model team form, rolling averages were calculated for all statistical columns over 20, 30, and 40-game windows, as well as an expanding window for the entire season.
-   **Contextual Features**: Win/loss streaks (`streak`) and wins in the last 10 games (`last10`) were engineered to capture short-term team momentum.
-   **Handling Initial Data**: For a team's first game of a season, statistics from the previous season's average were used for imputation, providing a realistic starting point. For the very first season in the dataset (2017-18), league-wide averages were used.

### 3. Model Training and Tuning
The training process (`train_logistic_regression.py`, `train_neural_network.py`) was designed to systematically find the best model, feature set, and hyperparameters.

-   **Data Split**: Data was split chronologically: games before **October 1, 2022** were used for training, and games after were used for testing.
-   **Pipelines**: `scikit-learn` Pipelines were used to chain preprocessing (scaling, PCA) with the model.
-   **Hyperparameter Search**: `GridSearchCV` (LR) and `RandomizedSearchCV` (NN) were combined with `TimeSeriesSplit` to robustly evaluate thousands of potential configurations. The search space included:
    -   **Data Variants**: 20, 30, 40-game, and season-long rolling averages.
    -   **Feature Sets**: All features, advanced stats only (`no_totals`), and basic stats only (`no_prc`).
    -   **Model Parameters**: Regularization strength (LR), network architecture (NN), optimizers, learning rates, and dropout rates (NN).
-   **Best Configuration**: The search concluded that the **`team_last_20`** data variant with **advanced stats only (`no_totals`)** yielded the highest cross-validation accuracy for both models.

### 4. Final Model Configurations

#### Logistic Regression (Best Performing Model)
-   **Accuracy (CV)**: **63.24%**
-   **Data**: 20-game rolling averages
-   **Features**: Advanced stats only
-   **PCA Components**: 15
-   **Solver**: `liblinear`
-   **Regularization**: L1 penalty with `C=0.25`

#### Neural Network (MLP)
-   **Accuracy (CV)**: 63.07%
-   **Data**: 20-game rolling averages
-   **Features**: Advanced stats only
-   **PCA Components**: 5
-   **Architecture**: 3 hidden layers (60, 20, 10 neurons with `tanh` activation)
-   **Optimizer**: `AdamW` (lr=0.001)
-   **Regularization**: Dropout (0.3, 0.3, 0.1)

---

## 📊 Results

Based on the superior balanced accuracy during cross-validation (63.24% vs. 63.07%), the **Logistic Regression model was selected as the final model for the API**. Below is the performance of both models on the unseen test set (seasons 2022/23 and 2023/24).

### Performance on Test Set

| Metric | Logistic Regression | Neural Network |
| :--- | :--- | :--- |
| **Accuracy** | 62.27% | 60.39% |
| Precision (Home Win) | 70.6% | **71.5%** |
| Recall (Home Win) | **56.4%** | 49.1% |
| Precision (Away Win) | **55.5%** | 53.4% |
| Recall (Away Win) | 69.8% | **74.9%** |


### Key Findings
-   **Accuracy Boost**: The final model provides a ~7 percentage point improvement over the baseline of simply predicting a home team win (which occurs ~55-56% of the time).
-   **Advanced Stats are Key**: Both models performed best when trained exclusively on advanced metrics (like `ORtg`, `eFG%`, `TS%`), indicating that raw counting stats introduce more noise than signal.
-   **Model Selection**: While the NN showed slightly higher precision for home wins, its overall accuracy and recall were lower on the test set. The LR model, chosen for its higher cross-validation score, proved to be more balanced and robust.
-   **PCA Impact**: Using PCA slightly reduced training time without a significant negative impact on performance, proving useful for dimensionality reduction.

---

## 🛠️ Installation and Usage

You can run this project in two ways: using Docker for quick deployment of the API, or setting up a local environment to explore the notebooks and training scripts.

### Option 1: Run with Docker (Recommended for API Deployment)
This is the simplest way to get the prediction API running, as the Docker image contains all necessary data, models, and dependencies.

**Prerequisites**:
-   [Docker](https://docs.docker.com/get-docker/) installed and running on your machine.

**Steps**:
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/marekk13/nba_predictor.git
    cd nba-predictor
    ```

2.  **Build the Docker image**:
    From the root directory of the project, run the following command to build the image.
    ```bash
    docker build -t nba-predictor .
    ```

3.  **Run the Docker container**:
    This command starts the container and maps port 8000 on your local machine to port 8000 inside the container.
    ```bash
    docker run -p 8000:8000 nba-predictor
    ```

4.  **Access the API**:
    The API is now running. You can access the interactive documentation at `http://127.0.0.1:8000/docs` to test the endpoints.

### Option 2: Local Development Setup (For Exploration and Retraining)
Follow these steps if you want to run the Jupyter notebooks or execute the training scripts from scratch.

**Prerequisites**:
-   Python 3.9 or higher.

**Steps**:
1.  **Clone the repository and set up the environment**:
    ```bash
    # Clone the repository
    git clone https://github.com/marekk13/nba_predictor.git
    cd nba-predictor

    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **(Optional) Reproduce the entire pipeline**:
    To generate the datasets and retrain the models from scratch, run the notebooks and scripts in the following order:
    ```bash
    # 1. Run the Jupyter notebooks for EDA and feature engineering.
    #    - `notebooks/1_EDA.ipynb`
    #    - `notebooks/2_Feature_Engineering.ipynb`
    #    - `notebooks/3_Feature_Enginnering2.ipynb`

    # 2. Run the training scripts to perform hyperparameter tuning and save the best models.
    python train_logistic_regression.py
    python train_neural_network.py
    ```

3.  **Run the Prediction API locally**:
    After ensuring the model artifacts exist in the `models/` directory, start the FastAPI server with uvicorn.
    ```bash
    uvicorn predict_api:app --reload
    ```

### Example API Request
Once the API is running (either via Docker or locally), you can send a `POST` request to the `/predict` endpoint. Here is an example using `curl`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "home_team_abbr": "GSW",
  "away_team_abbr": "HOU",
  "game_date": "2023-12-25"
}'
```

---

## 🚧 Limitations & Future Directions

### Current Limitations
-   **Player Availability**: The models do not account for injuries or player absences, which are significant factors in real-world outcomes.
-   **Static Dataset**: The API serves predictions from a fixed historical dataset. It does not have a live data pipeline for upcoming games.
-   **Data Gaps**: 74 game records were missing from the raw data due to scraper inconsistencies.

### Future Improvements
-   **Integrate Injury Reports**: Incorporate player injury and availability data as a crucial feature.
-   **Live Data Pipeline**: Develop a daily-run pipeline to scrape data for upcoming games and make real-time predictions.
