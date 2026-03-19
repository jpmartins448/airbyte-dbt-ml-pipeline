import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ----------------------------
# 1. Connect to Postgres
# ----------------------------

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="airbyte",
    user="postgres",
    password="postgres"
)

# ----------------------------
# 2. Load feature table
# ----------------------------

df = pd.read_sql("SELECT * FROM commit_features", conn)

print("Raw data preview:")
print(df.head())

# ----------------------------
# 3. Data cleaning
# ----------------------------

# Remove rows without previous commit
df = df.dropna(subset=["time_since_last_commit"])

# Convert timedelta to seconds
df["time_since_last_commit"] = df["time_since_last_commit"].dt.total_seconds()

# Convert datetime
df["committed_at"] = pd.to_datetime(df["committed_at"])

# ----------------------------
# 4. Feature engineering
# ----------------------------

df["commit_hour"] = df["committed_at"].dt.hour
df["commit_day_of_week"] = df["committed_at"].dt.dayofweek

# Convert author names to numbers
df["author_id"] = df["author_name"].astype("category").cat.codes

print("\nProcessed data preview:")
print(df.head())

# ----------------------------
# 5. Define ML features
# ----------------------------

X = df[[
    "commit_hour",
    "commit_day_of_week",
    "author_id"
]]

y = df["time_since_last_commit"]

# ----------------------------
# 6. Train / Test split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 7. Train model
# ----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training complete!")

# ----------------------------
# 8. Evaluate model
# ----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error (seconds):", mae)
print("Mean Absolute Error (hours):", mae / 3600)

# ----------------------------
# 9. Example prediction
# ----------------------------

sample = X_test.iloc[0:1]

prediction = model.predict(sample)

print("\nExample prediction:")
print("Features:", sample.to_dict())
print("Predicted seconds until next commit:", prediction[0])
print("Predicted hours:", prediction[0] / 3600)