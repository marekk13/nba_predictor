# 🏀 NBA Match Outcome Prediction using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

Predict NBA game results using machine learning models. Analyzes 7 seasons (2017/18–2023/24) with advanced feature engineering and model comparison.

---

## 📖 Project Overview  
**Key Challenges**:
- Temporal data aggregation for 8,281 games - preventing data leakage
- Imbalanced precision (Home: 72-79% vs Away: 44-52%)

**Core Components**:
- Data pipeline with rolling averages and PCA
- Logistic Regression vs Neural Network comparison
- Advanced stats (TS%, ORtg) as key predictors

---

## 🛠️ Installation  
### Requirements
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset
### Structure
- **8,281 games** (5 seasons training / 2 seasons testing)
- **78 features** including:
  - Basic stats: PTS, REB, AST
  - Advanced metrics: TS%, PER, ORtg
  - Contextual: Win streaks, rest days

### Sources
- Primary: basketball-reference.com - used [luke-lite's scraping notebook](https://github.com/luke-lite/NBA-Web-Scraper/blob/main/NBA-Web-Scraper-Notebook.ipynb)
- Supplementary: Team rankings, form indicators; [kyleskom's scraper](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/tree/master/src/Process-Data) was improved and run to get the data


---

## ⚙️ Methodology
### Feature Engineering
1. **Temporal Features**:
   - 20/30/40-game rolling averages
   - Season-long cumulative stats
2. **Contextual Enhancements**:
   - Win/loss streaks
   - Days-rest differentials

### Preprocessing
```plaintext
- StandardScaler (Logistic Regression)
- MinMaxScaler (Neural Networks)
- PCA (15 components)
```

### Model Configurations
#### Logistic Regression
```plaintext
Best Parameters:
- Solver: saga
- Regularization: C=0.4
- Accuracy: 63.18%
```

#### Neural Network (MLP)
```plaintext
Architecture:
- Layers: [40, 15, 1]
- Dropout: 20% (Layer 1), 50% (Layer 2)
- Optimizer: Adam (lr=0.003)
- Epochs: 40
- Accuracy: 63.72%
```


---

## 📊 Results
### Performance Comparison
| Metric            | Logistic Regression | Neural Network |
|-------------------|---------------------|----------------|
| **Accuracy**      | 63.18%              | **63.72%**     |
| Precision (Home)  | 72%                 | 79%            |
| Recall (Home)     | 70%                 | 68%            |
| Precision (Away)  | 52%                 | 44%            |

### Key Findings
- ~7 percentage point accuracy boost over baseline (home win probability)
- Best results were achieved using advanced stats and deleting basic stats
- PCA reduces training time, but has no significant influence on the results

---

## 🧠 Insights
### Model Tradeoffs
```plaintext
Logistic Regression:
+ Interpretable coefficients
- Limited non-linear patterns

Neural Network:
+ Captures complex interactions
- Requires GPU for optimal training
```

---

## 🚧 Limitations
- No use of player availability data
- Limited to box score statistics
- 74 missing records in raw data due to inconsistencies in scrapers

---

## 🔮 Future Directions
```plaintext
1. Integrate injury reports
2. Experiment with Bayesian methods
3. Address class imbalance via focal loss
```
---