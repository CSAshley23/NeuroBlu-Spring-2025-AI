import neuroblu as nb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

if __name__ == "__main__":
    # 1. Load data
    df = nb.get_df('health_equity_new')

    # 2. Filter out incomplete sociodemographics
    df = df[
        df['race_concept_id'] != 0
    ].dropna(subset=['zip_code_source_value'])

    # 3. Normalize visit frequency and emergency ratio
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['avg_visits_per_year', 'emergency_ratio']])
    df['avg_visits_per_year_norm'] = scaled[:, 0]
    df['emergency_ratio_norm'] = scaled[:, 1]

    # 4. Invert scores to reflect risk
    X1 = 1 - df['avg_visits_per_year_norm']           # Lower visits = higher risk
    X2 = df['emergency_ratio_norm']                   # Higher ER = higher risk
    X3 = 1 - df['condition_resolved']                 # Unresolved = higher risk

    features = np.vstack([X1, X2, X3]).T

    # 5. Define your proxy score (to learn weights from)
    df['inequity_score'] = (
        0.4 * X1 +
        0.4 * X2 +
        0.2 * X3
    )

    target = df['inequity_score']

    # 6. Define loss and optimize weights
    def loss(weights):
        predicted = features @ weights
        return ((predicted - target) ** 2).mean()

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * 3
    initial = np.array([0.33, 0.33, 0.34])

    result = minimize(loss, initial, bounds=bounds, constraints=constraints)
    learned_weights = result.x
    print("\n=== Learned Weights ===")
    print(f"Visits weight:     {learned_weights[0]:.4f}")
    print(f"ER Ratio weight:   {learned_weights[1]:.4f}")
    print(f"Unresolved weight: {learned_weights[2]:.4f}")

    # 7. Apply weights to get data-driven score
    df['data_driven_score'] = features @ learned_weights

    # 8. Train/test split to validate regression using the learned score
    X = df[['avg_visits_per_year', 'emergency_ratio', 'condition_resolved', 'gender_concept_id',
           'race_concept_id','ethnicity_concept_id']]
    y = df['data_driven_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Regression Evaluation on Data-Driven Score ===")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

    print("\nCoefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print("Intercept:", model.intercept_)

    # 9. OLS summary (optional)
    X_sm = sm.add_constant(X)
    model_ols = sm.OLS(y, X_sm).fit()
    print("\n=== Statsmodels OLS Summary ===")
    print(model_ols.summary())

    # 10. View predictions
    df['predicted_data_driven_score'] = model.predict(X)
    print("\n=== Sample Predictions ===")
    print(df[['person_id', 'data_driven_score', 'predicted_data_driven_score']].head())

