import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import joblib
from keras.layers import BatchNormalization
import tensorflow as tf

# load data
file_path = 'd:/work/insulation_article/insulationNSGA/files/convert.xlsx'
df = pd.read_excel(file_path, sheet_name="Sheet1")

# train_test splitting
x = df[['A', 'P', 'wwr', 'index']]
y = df['real']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8, random_state=16)

# K-Fold cross-validation on training set
kf = KFold(n_splits = 5, shuffle=True, random_state=16)
cv_mae = []
cv_rmse = []
cv_r2 = []

print("Starting 5-fold CV on training set...")

for fold, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train), 1):
    X_tr = x_train.iloc[train_idx].values
    y_tr = y_train.iloc[train_idx].values
    X_val = x_train.iloc[val_idx].values
    y_val = y_train.iloc[val_idx].values

    # Fit scalar on fold training only (no leakage)
    scalar_fold = StandardScaler().fit(X_tr)
    X_tr_s = scalar_fold.transform(X_tr)
    X_val_s = scalar_fold.transform(X_val)

    # build the ANN model
    fold_model = Sequential()
    fold_model.add(Dense(8, activation='sigmoid'))
    fold_model.add(BatchNormalization())
    fold_model.add(Dense(8, activation='sigmoid'))
    fold_model.add(BatchNormalization())
    fold_model.add(Dense(8, activation='sigmoid'))
    fold_model.add(BatchNormalization())
    fold_model.add(Dense(1, activation='linear'))

    fold_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # train
    history_fold = fold_model.fit(X_tr_s, y_tr, epochs=500, batch_size=8, verbose=0, validation_data=(X_val_s, y_val))

    # evaluate on validation fold
    y_val_pred = fold_model.predict(X_val_s, verbose=0).flatten()
    mae_f = mean_absolute_error(y_val, y_val_pred)
    rmse_f = root_mean_squared_error(y_val,y_val_pred)
    r2_f = r2_score(y_val, y_val_pred)

    cv_mae.append(mae_f)
    cv_rmse.append(rmse_f)
    cv_r2.append(r2_f)
    print(f"Fold {fold}: MAE={mae_f:.4f}, RMSE={rmse_f:.4f}, R2={r2_f:.4f}")
    tf.keras.backend.clear_session()

# Report CV summary
print("\nCross-validation results (5 folds):")
print(f"MAE:  mean={np.mean(cv_mae):.4f}, std={np.std(cv_mae):.4f}")
print(f"RMSE: mean={np.mean(cv_rmse):.4f}, std={np.std(cv_rmse):.4f}")
print(f"R2:   mean={np.mean(cv_r2):.4f}, std={np.std(cv_r2):.4f}")

# train final model
# standardize data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# build the ANN model
model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=[4]))
model.add(BatchNormalization())
model.add(Dense(8, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(8, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

# compile the model
optimizar = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizar, loss='mean_squared_error')

# storing the best weight
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=150, restore_best_weights=True)

# train the model
history = model.fit(x_train_scaled, y_train, epochs=500, batch_size=8,
                    verbose=0, validation_data=(x_test_scaled, y_test), callbacks=[early_stop])

# predictions
y_train_hat = model.predict(x_train_scaled).flatten()
y_test_hat = model.predict(x_test_scaled).flatten()

# evaluation
r2_train = r2_score(y_train, y_train_hat)
r2_test = r2_score(y_test, y_test_hat)
print('r2_train:', r2_train)
print('r2_test:', r2_test)
bias_train = np.mean(y_train - y_train_hat)
bias_test = np.mean(y_test - y_test_hat)
variance_train = np.var(y_train_hat)
variance_test = np.var(y_test_hat)
mae_train = mean_absolute_error(y_train, y_train_hat)
mae_test = mean_absolute_error(y_test, y_test_hat)
rmse_train = root_mean_squared_error(y_train, y_train_hat)
rmse_test = root_mean_squared_error(y_test, y_test_hat)
mse_train = mean_squared_error(y_train, y_train_hat)
mse_test = mean_squared_error(y_test, y_test_hat)
MSE_TEST = model.evaluate(x_test_scaled, y_test)
print(f"bias_train: {bias_train:.4f}")
print(f"bias_test: {bias_test:.4f}")
print(f"variance_train: {variance_train:.4f}")
print(f"variance_test: {variance_test:.4f}")
print(f"MAE_train: {mae_train:.4f}")
print(f"MAE_test: {mae_test:.4f}")
print(f"RMSE_train: {rmse_train:.4f}")
print(f"RMSE_test: {rmse_test:.4f}")
print(f"MSE_train: {mse_train:.4f}")
print(f"MSE_test: {mse_test:.4f}")
print(f'MSE_TEST: {MSE_TEST}')

# plot r_square train vs. test
fig, axs = plt.subplots(1, 2)

axs[0].scatter(y_train, y_train_hat, 10, 'grey', 'o')
axs[0].plot(y_train, y_train, color='black', linestyle='-', linewidth=1)
axs[1].scatter(y_test, y_test_hat, 10, 'grey', 'o')
axs[1].plot(y_test, y_test, color='black', linestyle='-', linewidth=1)

axs[0].text(0.05, 0.95, f'R-squared= {r2_train:.3f}',
            transform=axs[0].transAxes, fontsize=15, verticalalignment='top')
axs[1].text(0.05, 0.95, f'R-squared= {r2_test:.3f}',
            transform=axs[1].transAxes, fontsize=15, verticalalignment='top')

axs[0].set_title('Coefficient (train data)')
axs[0].set_xlabel('Actual Value (ratio)')
axs[0].set_ylabel('Predicted Value (ratio)')
axs[1].set_title('Coefficient Load (test data)')
axs[1].set_xlabel('Actual Value (ratio)')
axs[1].set_ylabel('Predicted Value (ratio)')
plt.show()

# other evalute
x1 = np.array([[2.5, 1.571429, 0.3, 1]])
x1_scaled = scaler.transform(x1)
result1 = model.predict(x1_scaled).flatten()
x2 = np.array([[2.5, 1.571429, 0.275, 1]])
x2_scaled = scaler.transform(x2)
result2 = model.predict(x2_scaled).flatten()
x3 = np.array([[2.5, 1.571429, 0.25, 1]])
x3_scaled = scaler.transform(x3)
result3 = model.predict(x3_scaled).flatten()
print('cooling: ', result1, result2, result3)

x4 = np.array([[2.5, 1.571429, 0.3, 2]])
x4_scaled = scaler.transform(x4)
result4 = model.predict(x4_scaled).flatten()
x5 = np.array([[2.5, 1.571429, 0.275, 2]])
x5_scaled = scaler.transform(x5)
result5 = model.predict(x5_scaled).flatten()
x6 = np.array([[2.5, 1.571429, 0.25, 2]])
x6_scaled = scaler.transform(x6)
result6 = model.predict(x6_scaled).flatten()
print('heating: ', result4, result5, result6)

x7 = np.array([[3.701, 2.067, 0.29, 1]])
x7_scaled = scaler.transform(x7)
result7 = model.predict(x7_scaled).flatten()
x8 = np.array([[3.701, 2.067, 0.29, 2]])
x8_scaled = scaler.transform(x8)
result8 = model.predict(x8_scaled).flatten()
print('ideal :', result7, result8)

# plot loss
plt.plot(history.history['loss'][50:], 'blue')
plt.plot(history.history['val_loss'][50:], 'red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

model.save('coefKfold10.h5')
#joblib.dump(scaler, 'scaler_convertor.pkl')
