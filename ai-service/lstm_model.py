import pandas as pd
import numpy as np
import psycopg2
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt


# 1. Data Loading & Feature Engineering

def fetch_daily_data():
    conn = psycopg2.connect(
        host="localhost",
        database="budget_data",
        user="postgres",
        password="postgres"
    )
    query = "SELECT date, category, amount FROM real_budget ORDER BY date"
    df = pd.read_sql_query(query, conn, parse_dates=['date'])
    conn.close()

    df['date'] = df['date'].dt.date
    df_agg = df.groupby(['date', 'category']).agg({'amount': 'sum'}).reset_index()
    df_pivot = df_agg.pivot(index='date', columns='category', values='amount').fillna(0)

    # Reindex to preserve all days
    full_index = pd.date_range(start=pd.to_datetime(df_pivot.index.min()),
                               end=pd.to_datetime(df_pivot.index.max()), freq='D')
    df_pivot = df_pivot.reindex(full_index, fill_value=0)
    df_pivot.index.name = 'date'
    df_pivot = df_pivot.reset_index()

    subcats = list(df_pivot.columns)
    subcats.remove('date')

    # Clip extreme values
    for col in subcats:
        q_low = df_pivot[col].quantile(0.01)
        q_high = df_pivot[col].quantile(0.99)
        df_pivot[col] = df_pivot[col].clip(q_low, q_high)

    # Rolling features: 7-day moving average and standard deviation
    feat_dict = {}
    for col in subcats:
        feat_dict[f"{col}_ma7"] = df_pivot[col].rolling(window=7, min_periods=1).mean()
        feat_dict[f"{col}_std7"] = df_pivot[col].rolling(window=7, min_periods=1).std().fillna(0)
    feats_df = pd.DataFrame(feat_dict)

    # Seasonal features: sin/cos of day-of-week and month
    df_dates = df_pivot[['date']].copy()
    df_dates['dayofweek'] = pd.to_datetime(df_dates['date']).dt.dayofweek
    df_dates['sin_dow'] = np.sin(2 * np.pi * df_dates['dayofweek'] / 7)
    df_dates['cos_dow'] = np.cos(2 * np.pi * df_dates['dayofweek'] / 7)
    df_dates['month'] = pd.to_datetime(df_dates['date']).dt.month
    df_dates['sin_month'] = np.sin(2 * np.pi * df_dates['month'] / 12)
    df_dates['cos_month'] = np.cos(2 * np.pi * df_dates['month'] / 12)

    combined_df = pd.concat([df_pivot, feats_df, df_dates[['sin_dow', 'cos_dow', 'sin_month', 'cos_month']]], axis=1)
    return combined_df, subcats

def visualize_data(df, subcats):
    """
    Visualizes monthly total expenses and category breakdown for 2024.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['Total'] = df[subcats].sum(axis=1)

    df_2024 = df[df['year'] == 2024].copy()
    monthly_total = df_2024.groupby('month')['Total'].sum()
    monthly_total = monthly_total[monthly_total > 0]

    plt.figure(figsize=(12, 6))
    monthly_total.plot(kind='bar', color='lightgreen')
    plt.title('Total Expenses by Month (2024)')
    plt.xlabel('Month')
    plt.ylabel('Total Amount Spent')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    monthly_category = df_2024.groupby('month')[subcats].sum()
    monthly_category = monthly_category.loc[monthly_total.index]
    monthly_category.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Paired')
    plt.title('Monthly Expenses by Category (2024)')
    plt.xlabel('Month')
    plt.ylabel('Total Amount Spent')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# 2. Create Sliding-Window Samples (Log Transform Targets)

def create_samples(df, subcats, input_window=120, target_window=30):
    """
    Creates sliding-window samples.
    X: Sequence of input_window days (all features).
    y: For each sample, the sum over the next target_window days (per category),
       then log1p-transformed to reduce skew.
    """
    X_all = df.drop(columns=['date']).values
    n_sub = len(subcats)
    y_all = X_all[:, :n_sub]  # Raw category columns as targets
    num_days = X_all.shape[0]

    X_samples, y_samples, target_dates = [], [], []
    dates = pd.to_datetime(df['date']).values

    for i in range(num_days - input_window - target_window + 1):
        x_seq = X_all[i: i + input_window, :]
        y_val = np.sum(y_all[i + input_window: i + input_window + target_window, :], axis=0)
        y_val = np.log1p(y_val)  # Log transform
        target_date = dates[i + input_window + target_window - 1]
        X_samples.append(x_seq)
        y_samples.append(y_val)
        target_dates.append(target_date)

    return np.array(X_samples), np.array(y_samples), target_dates


# 3. Train & Save Model

def train_and_save_model():
    df, subcats = fetch_daily_data()
    X, y, _ = create_samples(df, subcats)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale inputs using RobustScaler
    scaler_X = RobustScaler()
    X_train_flat = X_train.reshape(-1, X.shape[2])
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    # Scale targets (log-transformed)
    scaler_y = RobustScaler()
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Build model with your specified architecture (output vector length = number of categories)
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Conv1D(filters=32, kernel_size=3, activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(len(subcats), activation='relu')  # ReLU forces non-negative outputs
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae', 'mse'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    checkpoint = ModelCheckpoint('lstm_budget_multivariate.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=100,
        batch_size=256,
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Evaluate on the test set 
    loss, mae_scaled, mse_scaled = model.evaluate(X_test_scaled, y_test_scaled)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MSE: {mse_scaled:.4f}")
    print(f"Test MAE: {mae_scaled:.4f}")

    # Visualizations on test set
    # 1. Training vs. Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    # 2. Scatter plot: Aggregated predictions vs. actual totals (aggregated over categories)
    y_test_total = y_test_scaled.sum(axis=1)
    y_pred_total = model.predict(X_test_scaled).sum(axis=1)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_total, y_pred_total, alpha=0.6, color='blue')
    plt.plot([y_test_total.min(), y_test_total.max()],
             [y_test_total.min(), y_test_total.max()], 'r--')
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Aggregated Prediction vs Actual")
    plt.tight_layout()
    plt.show()

    # 3. Bar chart of evaluation metrics
    metrics = [mae_scaled, mse_scaled]
    labels = ["MAE", "MSE"]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, metrics, color=['blue', 'orange'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.show()

    # Save model, scalers, and subcategories list
    model.save('lstm_budget_multivariate.keras')
    with open("scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
    with open("subcats.pkl", "wb") as f:
        pickle.dump(subcats, f)
    with open("feature_cols.pkl", "wb") as f:
        pickle.dump(df.drop(columns=['date']).columns.tolist(), f)

    print("Model trained and saved successfully.")
    return model, subcats


# 4. Prediction Function (Category-wise & Total in Original Scale)

def predict_next_month_expenses(category=None):
    """
    Uses the most recent window to predict the next 30-day sum for each category.

    """
    df, _ = fetch_daily_data()
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[['date'] + feature_cols]

    # Use the most recent 120 days as input
    recent = df[-120:].copy()
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    with open("subcats.pkl", "rb") as f:
        subcats = pickle.load(f)

    X_input = recent.drop(columns=['date']).values.reshape(1, 120, -1)
    X_input_scaled = scaler_X.transform(X_input.reshape(-1, X_input.shape[2])).reshape(X_input.shape)

    model = load_model("lstm_budget_multivariate.keras")
    y_pred_scaled = model.predict(X_input_scaled)
    # Inverse scale and reverse the log transform to obtain original amounts
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
    y_pred_orig = np.expm1(y_pred_log)[0]

    pred_dict = {cat: float(pred) for cat, pred in zip(subcats, y_pred_orig)}
    pred_dict['total'] = sum(pred_dict.values())

    if category:
        category = category.strip()
        if category.lower() == 'total':
            return {"predicted_amount": pred_dict['total']}
        elif category in pred_dict:
            return {category: pred_dict[category]}
        else:
            return {"error": f"Invalid category. Valid options are: {list(pred_dict.keys())}"}
    return {"predicted_amount": pred_dict['total'], "categories": pred_dict}


# 5. API Wrapper

def predict_next_value(category=None):
    return predict_next_month_expenses(category)


# Entry Point

if __name__ == "__main__":
    model, subcats = train_and_save_model()
    df, subcats = fetch_daily_data()
    visualize_data(df, subcats)

    # Example: Print total expense forecast on original scale
    result_total = predict_next_value("total")
    print("\nNext 30-Day Total Forecast (Original Scale):")
    print(result_total)

    # Example: Print category-wise forecast (original scale)
    result_all = predict_next_value()
    print("\nNext 30-Day Category-wise Forecast:")
    print(result_all)
