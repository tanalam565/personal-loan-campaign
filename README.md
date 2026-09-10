# Personal Loan Campaign

This project uses customer banking data to identify liability customers who are most likely to accept a personal loan offer. It combines exploratory data analysis with decision-tree classification to support more targeted bank marketing campaigns.

## Business Objective

AllLife Bank wants to expand its personal-loan customer base while retaining customers as depositors. The goal is to predict whether a customer will accept a personal loan and identify the customer attributes that are most useful for targeting.

## Dataset

The main dataset is [`Loan_Modelling.csv`](Loan_Modelling.csv), containing 5,000 customers and 14 columns:

- `ID`: Customer identifier
- `Age`: Customer age in years
- `Experience`: Years of professional experience
- `Income`: Annual income in thousands of dollars
- `ZIPCode`: Home ZIP code
- `Family`: Family size
- `CCAvg`: Average monthly credit-card spending in thousands of dollars
- `Education`: Education level (`1``, `2``, or `3`)
- `Mortgage`: Mortgage value in thousands of dollars
- `Personal_Loan`: Target variable; whether the customer accepted the loan (`0` or `1`)
- `Securities_Account`: Whether the customer has a securities account
- `CD_Account`: Whether the customer has a certificate of deposit account
- `Online`: Whether the customer uses online banking
- `CreditCard`: Whether the customer has a credit card from another bank

## Analysis Workflow

The notebook [`personal_loan.ipynb`](personal_loan.ipynb) covers:

1. Dataset inspection, descriptive statistics, and missing-value checks.
2. Data cleaning, including correction of negative experience values.
3. Conversion of education codes into readable categories.
4. ZIP-code feature engineering by retaining the first two digits.
5. Univariate and bivariate exploratory analysis with charts.
6. Removal of the unique customer ID and the highly correlated `Experience` feature.
7. One-hot encoding of `ZIPCode` and `Education`.
8. A 70/30 train-test split.
9. Decision-tree classification evaluated with accuracy, precision, recall, F1 score, and confusion matrices.
10. Hyperparameter tuning with recall as the grid-search scoring metric.
11. Decision-tree complexity analysis using cost-complexity pruning.

Recall is emphasized because failing to identify a customer who would accept a loan represents a missed marketing opportunity.

## Key Findings

- Income, education, family size, average credit-card spending, and CD-account ownership are useful indicators of personal-loan interest.
- Customers with income above approximately `$116k` and family sizes greater than two are highlighted as stronger campaign targets in the tuned tree's decision rules.
- Customers with higher education levels are more likely to accept a personal loan than undergraduates.
- Customers with a CD account show a substantially higher loan-need rate in the exploratory analysis.
- Online banking, external credit-card ownership, securities-account ownership, and ZIP-code group contribute relatively little to the tuned tree's decisions.
- The tuned, pre-pruned decision tree is simpler and more interpretable than the unrestricted tree while retaining a reported test recall of approximately `0.92`.

These findings are exploratory and should be validated with current campaign data before being used for operational targeting.

## Getting Started

### Requirements

Python 3.9 or later is recommended. Install the notebook dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the notebook

1. Clone or download this repository.
2. Open `personal_loan.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
3. Update the data-loading cell to use the repository dataset:

   ```python
   Loan = pd.read_csv("Loan_Modelling.csv")
   ```

4. Run the notebook cells from top to bottom.

The notebook currently contains Google Colab-specific mounting and file-path code. The local path above replaces those cells when running from this repository.

## Repository Contents

```text
.
├── Loan_Modelling.csv       # Customer data used by the notebook
├── personal_loan.ipynb      # EDA, modeling, evaluation, and recommendations
├── more_files/              # Additional datasets included in the workspace
└── README.md
```

## Limitations

- The analysis uses a historical campaign outcome and should not be treated as a production lending decision system.
- The notebook prioritizes recall, so campaign teams should also review precision, cost of outreach, and customer eligibility.
- Decision-tree feature importance describes the model's behavior and does not establish causation.
- The notebook does not include a saved model artifact or a production scoring pipeline.