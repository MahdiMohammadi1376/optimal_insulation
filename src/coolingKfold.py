import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.regularizers import Regularizer
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="1/1", category=UserWarning)

# Custom regularizer to penalize weights except for r_wall
class CustomRegularizer(Regularizer):
    def __init__(self, strength=0.2):
        self.strength = strength

    def __call__(self, x):
        # Penalize all features except for the first one (r_wall)
        penalty = tf.reduce_sum(tf.square(x[:, 2:]))
        return self.strength * penalty

    def get_config(self):
        return {'strength': self.strength}


# Load data
file_path = 'D:/work/insulation_article/insulationNSGA/files/scatter.xlsx'
df = pd.read_excel(file_path, sheet_name="regression")

# Transform the r_wall to include both original and log-transformed
df['log_r_wall'] = np.log(df['r_wall'])

# Define independent variables and dependent variable
x = df[['r_wall', 'log_r_wall', 'SHGC', 'A4', 'u_glass', 'A5']]
y = df['Cooling']

# Train-test splitting (hold-out test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8, random_state=15)

# ----------------- K-Fold cross-validation on training set -----------------
# Minimal changes: perform 5-fold CV on x_train/y_train, fit scaler inside each fold,
# train same model architecture in each fold, report MAE/RMSE/R2 per fold,
# then retrain final model on full training set and proceed as before.

kf = KFold(n_splits=5, shuffle=True, random_state=15)

cv_mae = []
cv_rmse = []
cv_r2 = []

print("Starting 5-fold CV on training set...")

for fold, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train), 1):
    # Prepare fold data (as numpy arrays)
    X_tr = x_train.iloc[train_idx].values
    y_tr = y_train.iloc[train_idx].values
    X_val = x_train.iloc[val_idx].values
    y_val = y_train.iloc[val_idx].values

    # Fit scaler on fold training only (no leakage)
    scaler_fold = StandardScaler().fit(X_tr)
    X_tr_s = scaler_fold.transform(X_tr)
    X_val_s = scaler_fold.transform(X_val)

    # Build the ANN model (same architecture as original)
    fold_model = Sequential()
    fold_model.add(Dense(16, activation='sigmoid', input_shape=[X_tr_s.shape[1]],
                         kernel_regularizer=CustomRegularizer(strength=0.2)))
    fold_model.add(BatchNormalization())
    fold_model.add(Dense(16, activation='sigmoid',
                         kernel_regularizer=CustomRegularizer(strength=0.2)))
    fold_model.add(BatchNormalization())
    fold_model.add(Dense(1, activation='linear'))

    fold_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss='mean_squared_error')

    # Train
    history_fold = fold_model.fit(X_tr_s, y_tr, epochs=200, batch_size=10,
                                  verbose=0, validation_data=(X_val_s, y_val))

    # Evaluate on validation fold
    y_val_pred = fold_model.predict(X_val_s, verbose=0).flatten()
    mae_f = mean_absolute_error(y_val, y_val_pred)
    rmse_f = np.sqrt(mean_squared_error(y_val, y_val_pred))
    #rmse_f = mean_squared_error(y_val, y_val_pred)
    r2_f = r2_score(y_val, y_val_pred)

    cv_mae.append(mae_f)
    cv_rmse.append(rmse_f)
    cv_r2.append(r2_f)

    print(f"Fold {fold}: MAE={mae_f:.4f}, RMSE={rmse_f:.4f}, R2={r2_f:.4f}")

    # Clean session to free memory before next fold
    tf.keras.backend.clear_session()

# Report CV summary
print("\nCross-validation results (5 folds):")
print(f"MAE:  mean={np.mean(cv_mae):.4f}, std={np.std(cv_mae):.4f}")
print(f"RMSE: mean={np.mean(cv_rmse):.4f}, std={np.std(cv_rmse):.4f}")
print(f"R2:   mean={np.mean(cv_r2):.4f}, std={np.std(cv_r2):.4f}")

# ------------ Train final model on full training set (same architecture) ----------
# Fit scaler on full training set
scaler = StandardScaler().fit(x_train.values)
x_train_scaled = scaler.transform(x_train.values)
x_test_scaled = scaler.transform(x_test.values)

# Build the final model (same architecture)
model = Sequential()
model.add(Dense(16, activation='sigmoid', input_shape=[x_train_scaled.shape[1]],
                kernel_regularizer=CustomRegularizer(strength=0.2)))
model.add(BatchNormalization())
model.add(Dense(16, activation='sigmoid',
                kernel_regularizer=CustomRegularizer(strength=0.2)))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error')

# Train final model on full training set
history = model.fit(x_train_scaled, y_train, epochs=200, batch_size=10,
                    verbose=0, validation_data=(x_test_scaled, y_test))

# Save the final model 
#model.save('coolingKfold2.h5')
#joblib.dump(scaler, 'scaler_thermal.pkl')

# Evaluation (same as before)
y_train_hat = model.predict(x_train_scaled, verbose=0).flatten()
y_test_hat = model.predict(x_test_scaled, verbose=0).flatten()

r2_train = r2_score(y_train, y_train_hat)
r2_test = r2_score(y_test, y_test_hat)
print('\nFinal model performance on train/test:')
print('r2_train:', r2_train)
print('r2_test:', r2_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_hat))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_hat))
print(f"RMSE_train: {rmse_train:.4f}")
print(f"RMSE_test: {rmse_test:.4f}")
mae_train = mean_absolute_error(y_train,y_train_hat)
mae_test = mean_absolute_error(y_train,y_train_hat)
print(f"MAE_train: {mae_train:.4f}")
print(f"MAE_test: {mae_test:.4f}")

# ------------------ rest of your code unchanged (prediction loop, plotting) ------------------
"------------------------------------------------------------------------"
r_wall = 1.5
u_glass = 2.665
shgc = 0.703
λ_insulation = 0.042
A1 = 730
A2 = 910
A6 = 0.29

A4 = 910*A6
A5 = 910 - A4
r_insulation = 0.01/λ_insulation


i = 0
while i < 11:
    r_wall_i = r_wall + i * r_insulation
    r_wall_i2 = r_wall + (i + 1) * r_insulation
    x = np.array([[r_wall_i, np.log(r_wall_i), shgc, A4, u_glass, A5]])
    x2 = np.array([[r_wall_i2, np.log(r_wall_i2), shgc, A4, u_glass, A5]])
    x_scaled = scaler.transform(x)
    x2_scaled = scaler.transform(x2)

    # Predict using the model
    cooling = model.predict([x_scaled], verbose=0).flatten()
    cooling2 = model.predict([x2_scaled], verbose=0).flatten()
    cooling = float(cooling.item())
    cooling2 = float(cooling2.item())
    print(f'cooling({i}): {cooling}')
    c = cooling - cooling2
    print(f'Cminus: {c}')
    print('--------------------------')
    i += 1
"---------------------------------------------------------------------------"


# Plot r_square train vs. test
fig, axs = plt.subplots(1, 2)

axs[0].scatter(y_train, y_train_hat, 15, 'grey', 'o')
axs[0].plot(y_train, y_train, color='black', linestyle='-', linewidth=1)
axs[1].scatter(y_test, y_test_hat, 15, 'grey', 'o')
axs[1].plot(y_test, y_test, color='black', linestyle='-', linewidth=1)

axs[0].text(0.05, 0.95, f'R-squared= {r2_train:.3f}',
            transform=axs[0].transAxes, fontsize=15, verticalalignment='top')
axs[1].text(0.05, 0.95, f'R-squared= {r2_test:.3f}',
            transform=axs[1].transAxes, fontsize=15, verticalalignment='top')
axs[0].set_title('Cooling Load (train data)')
axs[0].set_xlabel('Actual Value (MWh)')
axs[0].set_ylabel('Predicted Value (MWh)')
axs[1].set_title('Cooling Load (test data)')
axs[1].set_xlabel('Actual Value (MWh)')
axs[1].set_ylabel('Predicted Value (MWh)')
plt.show()

# Plot loss
plt.plot(history.history['loss'][50:], 'green')
plt.plot(history.history['val_loss'][50:], 'blue')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
