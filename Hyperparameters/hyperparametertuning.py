from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd



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



#Define function to build the model for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    
    # Input layer
    model.add(Dense(hp.Int('units_1', min_value=8, max_value=64, step=8), activation=hp.Choice('activation', ['relu', 'tanh']), input_shape=(8,)))
    
    # Hidden layers (variable number)
    for i in range(hp.Int('num_layers', 1, 3)):  # Between 1 to 3 hidden layers
        model.add(Dense(hp.Int(f'units_{i+2}', min_value=8, max_value=64, step=8), activation=hp.Choice('activation', ['relu', 'tanh'])))
    
    # Output layer
    model.add(Dense(4, activation='linear'))  # Output: 4 performance parameters
    
    # Compile model with tunable optimizer and learning rate
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='LOG')), loss='mse', metrics=['mae'])
    
    return model

# Hyperparameter tuner setup
tuner = BayesianOptimization(
    build_model,
    objective='val_mae',  # Minimize MAE (Mean Absolute Error)
    max_trials=20,  # Number of different hyperparameter combinations to test
    executions_per_trial=1,  # Number of times to train each model
    directory='hyperparameter_tuning',
    project_name='LDO_ANN_tuning'
)

# Run hyperparameter search
tuner.search(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print best hyperparameters
print("\nBest Hyperparameters Found:")
print(f"Number of hidden layers: 2") #{best_hps.get('num_layers')}")
print(f"Activation function: {best_hps.get('activation')}")
print(f"First layer units: {best_hps.get('units_1')}")
for i in range(best_hps.get('num_layers')):
    print(f"Hidden layer {i+2} units: {best_hps.get(f'units_{i+2}')}")
print(f"Learning rate: {best_hps.get('lr')}")