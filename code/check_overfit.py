"""
Quick overfitting check across the three best models.
Reports R²_train vs R²_test gap for ANN P4-C, LSTM P5, and Optuna P9.

Input:  data/dataset{1-4}.csv — same 70/30 temporal split as all phases
Output: console table — R²_train, R²_test, gap, and pass/warn/fail label per dataset
        (no files saved — diagnostic-only script)

P9 best-params are hardcoded here from the saved JSON in results/phase9_optuna/
so this script can run standalone without re-running Optuna.
"""
import numpy as np
import tensorflow as tf
import keras
import optuna
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE, TEMP_LABEL

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS = [0, 42, 123]
WIN   = 20

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]

# P9 best params (from results)
P9_PARAMS = {
    1: {'batch_size':128, 'n_layers':1, 'lr':0.0021554, 'dropout':0.3,  'units_0':128},
    2: {'batch_size':128, 'n_layers':3, 'lr':0.0030097, 'dropout':0.0,  'units_0':64,  'units_1':128, 'units_2':16},
    3: {'batch_size':64,  'n_layers':4, 'lr':0.0012336, 'dropout':0.0,  'units_0':64,  'units_1':32,  'units_2':16, 'units_3':32},
    4: {'batch_size':32,  'n_layers':1, 'lr':0.0032653, 'dropout':0.0,  'units_0':128},
}

def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_p4c():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m

def build_lstm():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(WIN, 2)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m

def build_p9(params):
    layers = [tf.keras.Input(shape=(2,))]
    for i in range(params['n_layers']):
        layers.append(tf.keras.layers.Dense(params[f'units_{i}'], activation="relu"))
        if params['dropout'] > 0:
            layers.append(tf.keras.layers.Dropout(params['dropout']))
    layers.append(tf.keras.layers.Dense(1))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(params['lr']), loss="mse")
    return m

def gap_label(gap):
    if gap < 0.05:  return "✅ OK"
    if gap < 0.15:  return "⚠️ nhẹ"
    return "❌ overfit"

print("=" * 70)
print("OVERFITTING CHECK — R²_train vs R²_test (mean over 3 seeds)")
print("=" * 70)

for model_name, builder, use_seq, extra in [
    ("ANN P4-C", build_p4c,  False, None),
    ("LSTM P5",  build_lstm, True,  None),
    ("Optuna P9",build_p9,   False, None),
]:
    print(f"\n[{model_name}]")
    print(f"  {'Dataset':<14} {'R²_train':>9} {'R²_test':>9} {'gap':>7} {'Status'}")
    print(f"  {'-'*50}")

    for ds_id in DATASETS:
        df   = load_dataset(ds_id)
        X    = df[INPUT_FEATURES].values
        y    = df[OUTPUT_FEATURE].values
        split = int(len(X) * 0.7)
        X_tr_raw, X_te_raw = X[:split], X[split:]
        y_tr, y_te          = y[:split], y[split:]

        scaler  = MinMaxScaler().fit(X_tr_raw)
        X_tr_sc = scaler.transform(X_tr_raw)
        X_te_sc = scaler.transform(X_te_raw)

        train_r2s, test_r2s = [], []
        for seed in SEEDS:
            tf.random.set_seed(seed); np.random.seed(seed)

            if model_name == "Optuna P9":
                m = build_p9(P9_PARAMS[ds_id])
                batch = P9_PARAMS[ds_id]['batch_size']
            else:
                m = builder()
                batch = 64

            if use_seq:
                X_tr_s, y_tr_s = make_sequences(X_tr_sc, y_tr, WIN)
                X_te_s, y_te_s = make_sequences(X_te_sc, y_te, WIN)
                m.fit(X_tr_s, y_tr_s, epochs=300, batch_size=batch,
                      validation_split=0.1, callbacks=CALLBACKS, verbose=0)
                train_r2s.append(r2_score(y_tr_s, m.predict(X_tr_s, verbose=0)))
                test_r2s.append( r2_score(y_te_s, m.predict(X_te_s, verbose=0)))
            else:
                m.fit(X_tr_sc, y_tr, epochs=300, batch_size=batch,
                      validation_split=0.1, callbacks=CALLBACKS, verbose=0)
                train_r2s.append(r2_score(y_tr, m.predict(X_tr_sc, verbose=0)))
                test_r2s.append( r2_score(y_te, m.predict(X_te_sc, verbose=0)))

        tr = np.mean(train_r2s)
        te = np.mean(test_r2s)
        gap = tr - te
        print(f"  ds{ds_id} ({TEMP_LABEL[ds_id]})   {tr:>9.4f} {te:>9.4f} {gap:>+7.4f}  {gap_label(gap)}")
