import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


# Load dataset
file_path = "LDO_data.csv"  # Update with your file path
df = pd.read_csv('/home/seb/Desktop/ANN1/LDO_data.csv')

# Split data into inputs (performance parameters) and outputs (device parameters)
X = df.iloc[:, 8:].values  # Performance parameters (now input)
y = df.iloc[:, :8].values  # Device parameters (now output)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Define Reverse ANN model
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # Hidden layer 1
    Dense(16, activation='relu'),                    # Hidden layer 2
    Dense(8, activation='linear')                    # Output layer (8 device parameters)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest MAE: {test_mae}")

#Select 3 Performance Sets for Prediction
num_samples = 3
random_indices = random.sample(range(len(X_test)), num_samples)  # Select random indices
random_inputs = X_test[random_indices]  # Get corresponding performance parameters

# Predict Device Parameters
predicted_device_params = scaler_y.inverse_transform(model.predict(random_inputs))

# Print Results
print("\ Performance Parameters (Input):")
print(scaler_X.inverse_transform(random_inputs))  # Convert back to original scale
print("\nPredicted Device Parameters (Output):")
print(predicted_device_params)

# Compute R² Score for Reverse Model
y_pred = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred)  # Convert back to original scale
y_test_original = scaler_y.inverse_transform(y_test)  # Convert test values back

r2 = r2_score(y_test_original, y_pred)
print(f"\nReverse ANN R² Score: {r2:.4f}")


#Overfitting check

# 1. Create figure with loss and MAE progression
plt.figure(figsize=(14, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Loss Progression')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Progression')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Quantitative overfitting analysis
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
loss_ratio = final_val_loss / final_train_loss

# 3. Final metrics comparison
print("\nOverfitting Check:")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Test Loss: {test_loss:.4f}\n")

if history.history['val_loss'][-1] > history.history['loss'][-1] * 1.15:
    print("Potential overfitting (validation loss >15% higher than training loss)")
else:
    print("No significant overfitting detected")