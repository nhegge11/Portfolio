import numpy as np
import pandas as pd
import pyodbc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data_from_db(server, database, user, password, table_name):
    

    connection_string = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password}"
    )
    cnxn = pyodbc.connect(connection_string)
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, cnxn)
    cnxn.close()
    return data

def train_delinquency_model(data, features, target):
    """
    Trains a RandomForest model to predict delinquency.
    :param data: Pandas DataFrame with the necessary columns.
    :param features: List of feature column names.
    :param target: The name of the target (delinquency) column.
    :return: (model, X_test, y_test)
    """
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Evaluation on Test Set")
    print("============================")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test

def generate_delinquency_alerts(model, new_data, threshold=0.5):
    """
    Generates alerts based on the probability of delinquency.
    :param model: Trained RandomForest model.
    :param new_data: Pandas DataFrame of new observations.
    :param threshold: Probability threshold for triggering an alert.
    :return: DataFrame with delinquency_probability and alert columns.
    """
    probabilities = model.predict_proba(new_data)[:, 1]
    alerts = (probabilities >= threshold).astype(int)

    results = new_data.copy()
    results['delinquency_probability'] = probabilities
    results['alert'] = alerts
    return results

def main():
    """
    Main function to load data, train the model, and simulate alerts.
    Update parameters as needed for your environment.
    """
    server = 'REDACTED_SERVER_NAME'
    database = 'REDACTED_DB_NAME'
    user = 'REDACTED_USERNAME'
    password = 'REDACTED_PASSWORD'
    table_name = 'REDACTED_TABLE_NAME'

    # Load data from the database
    data = load_data_from_db(server, database, user, password, table_name)

    # Columns to use as features for delinquency prediction
    feature_columns = [
        'REDACTED_CREDIT_SCORE_COLUMN',
        'REDACTED_INCOME_COLUMN',
        'REDACTED_ACCOUNT_AGE_COLUMN',
        'REDACTED_UTILIZATION_COLUMN',
        'REDACTED_PAYMENT_HISTORY_COLUMN'
    ]

    # Column indicating whether the account is delinquent (0 or 1)
    target_column = 'REDACTED_DELINQUENCY_COLUMN'

    # Train the model
    model, X_test, y_test = train_delinquency_model(data, feature_columns, target_column)

    new_data = X_test.head(10)

    # Generate alerts
    alert_results = generate_delinquency_alerts(model, new_data, threshold=0.6)
    print("\nDelinquency Alerts")
    print("==================")
    print(alert_results[['delinquency_probability', 'alert']])

if __name__ == "__main__":
    main()
