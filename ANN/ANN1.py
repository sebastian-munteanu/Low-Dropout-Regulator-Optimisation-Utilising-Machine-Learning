import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
file_path = "LDO_data.csv"  # Update with your file path
df = pd.read_csv('/home/seb/Desktop/ANN1/LDO_data.csv')

# Split data into inputs (X) and outputs (y)
X = df.iloc[:, :8].values  # First 8 columns as input
y = df.iloc[:, 8:].values  # Last 4 columns as output

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Define ANN model
model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),  # Hidden layer 1
    Dense(16, activation='relu'),                    # Hidden layer 2
    Dense(4, activation='linear')                    # Output layer (4 outputs, linear activation for regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test MAE: {test_mae}")

# Predict example input (10 outputs)
example_input = X_test[:10]
predicted_outputs = scaler_y.inverse_transform(model.predict(example_input))
print("Predicted Performance Parameters:\n", predicted_outputs)

# Get predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions back to original scale
y_pred = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Compute R² score
r2 = r2_score(y_test_original, y_pred)
print(f"R² Score: {r2:.4f}")

#Scatter plot

# Define performance parameter names
labels = ["PM (deg)", "UGBW (kHz)", "PSRR (dB)","Iq (μA)"]

# Generate group numbers (assuming sequential test samples)
group_numbers = np.arange(len(y_test_original))

# Create subplots (2x2 layout)
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# Plot each performance parameter
for i, ax in enumerate(axs.flat):
    ax.scatter(group_numbers, y_test_original[:, i], color='blue', label="Real", s=10)  # Real values
    ax.scatter(group_numbers, y_pred[:, i], color='orange', label="Predict", s=10)  # Predicted values
    ax.set_xlabel("Group number")
    ax.set_ylabel(labels[i])
    ax.legend()
    ax.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()


#Overfitting check

plt.figure(figsize=(12, 5))

# 1. Loss progression plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend()

# 2. MAE progression plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# 3. Final metrics comparison
print("\nOverfitting Check:")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Test Loss: {test_loss:.4f}\n")

if history.history['val_loss'][-1] > history.history['loss'][-1] * 1.15:
    print("Potential overfitting (validation loss >15% higher than training loss)")
else:
    print("No significant overfitting detected")