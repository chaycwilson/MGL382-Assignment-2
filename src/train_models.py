import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Load your data
df = pd.read_csv('../data/apple_stock.csv')
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Original features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Target']

# StandardScaler for original features
scaler_basic = StandardScaler()
X_scaled = scaler_basic.fit_transform(X)

# Splitting the dataset for the basic model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Model definition for the basic model
model = Sequential([
    Input(shape=(len(features),)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Save the basic model and its scaler
joblib.dump(model, '../artifacts/model_1.pkl')
joblib.dump(scaler_basic, '../artifacts/scaler_basic.pkl')

# Additional features
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(window=14).mean() / 
                            -df['Close'].diff().clip(upper=0).rolling(window=14).mean()))
df['Momentum'] = df['Close'].diff(periods=10)
additional_features = ['MA10', 'MA50', 'RSI', 'Momentum']
new_features = features + additional_features
X_new = df[new_features].fillna(0)

# StandardScaler for engineered features
scaler_engineered = StandardScaler()
X_scaled_new = scaler_engineered.fit_transform(X_new)

# Splitting the dataset for the engineered model
X_train_new, X_test_new, y_train, y_test = train_test_split(X_scaled_new, y, test_size=0.2, random_state=0)

# Model definition for the engineered model
model_new = Sequential([
    Input(shape=(len(new_features),)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_new.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_new.fit(X_train_new, y_train, validation_data=(X_test_new, y_test), epochs=50, batch_size=32)
loss_new, accuracy_new = model_new.evaluate(X_test_new, y_test)
print(f'Improved Test Accuracy with Feature Engineering: {accuracy_new*100:.2f}%')

# Save the engineered model and its scaler
joblib.dump(model_new, '../artifacts/model_2.pkl')
joblib.dump(scaler_engineered, '../artifacts/scaler_engineered.pkl')
