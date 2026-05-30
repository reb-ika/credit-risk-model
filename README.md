cat > README.md << 'EOF'
# Credit Risk Probability Model for Alternative Data
**Kifiya AI Training Program — Week 4**
*Bati Bank — Buy-Now-Pay-Later Credit Scoring*

---

## Credit Scoring Business Understanding

### 1. How does Basel II influence the need for interpretable and well-documented models?

The Basel II Capital Accord requires banks to hold capital reserves proportional to their credit risk exposure. This means every risk model must produce results that can be explained to regulators, auditors, and internal risk committees. A black-box model that produces accurate predictions but cannot explain *why* a customer is high-risk fails Basel II requirements because:

- **Pillar 1** requires validated risk measurement methods with documented assumptions
- **Pillar 2** requires supervisory review — regulators must understand model logic
- **Pillar 3** requires public disclosure of risk measurement approaches

Practically, this means Logistic Regression with Weight of Evidence (WoE) transformations is often preferred in regulated contexts over Gradient Boosting, even if the latter has better AUC — because every coefficient has a clear business interpretation and the model can be audited line by line.

### 2. Why is a proxy variable necessary, and what business risks does it introduce?

The Xente dataset contains no direct "default" label — we do not know which customers failed to repay loans because this is transaction data, not loan data. A proxy variable is necessary to create a training signal for the credit risk model.

We use RFM (Recency, Frequency, Monetary) behavioral patterns as a proxy:
- **Recency**: How recently did the customer transact? (disengaged customers = higher risk)
- **Frequency**: How often do they transact? (low frequency = lower engagement = higher risk)
- **Monetary**: How much do they spend? (low monetary = limited financial activity = higher risk)

**Business risks of proxy-based prediction:**
- The proxy may not capture actual default behavior — a customer could be disengaged but still creditworthy
- Systematic bias may be introduced if disengagement correlates with demographics (age, location) rather than creditworthiness
- Regulatory challenge — we must document clearly that this is a proxy, not a ground-truth label
- Model drift — the relationship between RFM patterns and actual default may change over time

### 3. Trade-offs between interpretable and high-performance models

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost) |
|-----------|--------------------------|----------------------------|
| Interpretability | High — coefficients directly interpretable | Low — requires SHAP for explanation |
| Performance | Moderate AUC | Higher AUC |
| Regulatory compliance | Strong — Basel II preferred | Requires additional documentation |
| Maintenance | Simple — stable scorecards | Complex — requires monitoring |
| Speed | Very fast | Slower |
| Feature engineering | Requires WoE binning | Handles raw features |

**Recommendation**: Use Logistic Regression as the primary model for regulatory compliance, with Gradient Boosting as a challenger model. If the performance gap justifies the complexity, document the GBM thoroughly with SHAP values for explainability.

---

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml    # CI/CD pipeline
├── data/raw/                   # Raw data (not committed)
├── data/processed/             # Processed data (not committed)
├── notebooks/eda.ipynb         # Exploratory analysis
├── src/
│   ├── data_processing.py      # Feature engineering pipeline
│   ├── train.py                # Model training with MLflow
│   ├── predict.py              # Inference
│   └── api/
│       ├── main.py             # FastAPI application
│       └── pydantic_models.py  # Request/response schemas
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
---

## Setup Instructions

```bash
git clone https://github.com/reb-ika/credit-risk-model.git
cd credit-risk-model
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

---

## Author
**Rebika Woldeyesus**
Kifiya AI Training Program — Week 4
GitHub: [@reb-ika](https://github.com/reb-ika)
