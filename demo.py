"""
Fast Trigger ML Simulation 

This script runs the entire pipeline:
1. Generate synthetic data
2. Train baseline and ML models
3. Evaluate performance
4. Generate visualizations

"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("="*70)
print("  FAST TRIGGER ML SIMULATION - DEMO")
print("  Machine Learning for High-Energy Physics Triggers")
print("="*70)
print()

# ============================================================================
# 1. SETUP AND CONFIGURATION
# ============================================================================

print("[1/6] Loading configuration...")

config = {
    'random_seed': 42,
    'n_signal_events': 10000,  # Reduced for quick demo
    'n_background_events': 10000,
    'signal': {
        'pt_mean': 45.0, 'pt_std': 20.0, 'pt_min': 20.0, 'pt_max': 150.0,
        'eta_mean': 0.0, 'eta_std': 1.2, 'eta_max': 2.4,
        'phi_mean': 0.0, 'phi_std': 1.8
    },
    'background': {
        'pt_mean': 12.0, 'pt_std': 8.0, 'pt_min': 3.0, 'pt_max': 40.0,
        'eta_mean': 0.0, 'eta_std': 1.5, 'eta_max': 2.4,
        'phi_mean': 0.0, 'phi_std': 1.8
    },
    'baseline': {
        'pt_threshold': 20.0, 'eta_max': 2.4, 'chi2_max': 10.0,
        'min_hits': 8, 'isolation_max': 0.3
    },
    'bdt': {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8
    },
    'neural_net': {
        'architecture': [16, 32, 16, 1], 'activation': 'relu',
        'dropout': 0.3, 'batch_norm': True, 'learning_rate': 0.001,
        'batch_size': 256, 'epochs': 30, 'validation_split': 0.2,
        'early_stopping_patience': 10
    }
}

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("✓ Configuration loaded")
print(f"  - Signal events: {config['n_signal_events']:,}")
print(f"  - Background events: {config['n_background_events']:,}")
print()

# ============================================================================
# 2. DATA GENERATION
# ============================================================================

print("[2/6] Generating synthetic collision events...")

class EventGenerator:
    """Generate synthetic muon events."""
    
    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState(config['random_seed'])
    
    def generate_signal(self, n):
        cfg = self.config['signal']
        pt = self.rng.exponential(cfg['pt_mean']/2, n).clip(cfg['pt_min'], cfg['pt_max'])
        eta = self.rng.uniform(-cfg['eta_max'], cfg['eta_max'], n)
        phi = self.rng.uniform(-np.pi, np.pi, n)
        n_hits = self.rng.poisson(15, n).clip(8, 25)
        chi2 = self.rng.gamma(2, 1.5, n).clip(0.1, 20)
        isolation = self.rng.gamma(1, 0.15, n).clip(0, 1)
        n_vertices = self.rng.poisson(35, n).clip(1, 80)
        
        return pd.DataFrame({
            'pt': pt, 'eta': eta, 'phi': phi, 'n_hits': n_hits,
            'chi2': chi2, 'isolation': isolation, 'n_vertices': n_vertices,
            'label': 1
        })
    
    def generate_background(self, n):
        cfg = self.config['background']
        pt = self.rng.exponential(cfg['pt_mean']/2, n).clip(cfg['pt_min'], cfg['pt_max'])
        eta = self.rng.uniform(-cfg['eta_max'], cfg['eta_max'], n)
        eta = eta + self.rng.normal(0, 0.5, n)
        eta = eta.clip(-cfg['eta_max'], cfg['eta_max'])
        phi = self.rng.uniform(-np.pi, np.pi, n)
        n_hits = self.rng.poisson(10, n).clip(4, 20)
        chi2 = self.rng.gamma(4, 2.5, n).clip(0.1, 50)
        isolation = self.rng.gamma(2, 0.25, n).clip(0, 1)
        n_vertices = self.rng.poisson(35, n).clip(1, 80)
        
        return pd.DataFrame({
            'pt': pt, 'eta': eta, 'phi': phi, 'n_hits': n_hits,
            'chi2': chi2, 'isolation': isolation, 'n_vertices': n_vertices,
            'label': 0
        })

generator = EventGenerator(config)
signal = generator.generate_signal(config['n_signal_events'])
background = generator.generate_background(config['n_background_events'])

data = pd.concat([signal, background], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split
train_size = int(0.7 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

X_train = train_data.drop('label', axis=1)
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1)
y_test = test_data['label'].values

print("✓ Data generated")
print(f"  - Training set: {len(train_data):,} events")
print(f"  - Test set: {len(test_data):,} events")
print(f"  - Signal fraction: {y_test.mean():.1%}")
print()

# ============================================================================
# 3. TRAIN BASELINE MODEL (Cut-based)
# ============================================================================

print("[3/6] Training baseline cut-based trigger...")

class CutBasedTrigger:
    """Traditional trigger using simple cuts."""
    
    def __init__(self, config):
        self.pt_threshold = config['pt_threshold']
        self.eta_max = config['eta_max']
        self.chi2_max = config['chi2_max']
        self.min_hits = config['min_hits']
        self.isolation_max = config['isolation_max']
    
    def predict(self, X):
        mask = (
            (X['pt'] >= self.pt_threshold) &
            (np.abs(X['eta']) <= self.eta_max) &
            (X['chi2'] <= self.chi2_max) &
            (X['n_hits'] >= self.min_hits) &
            (X['isolation'] <= self.isolation_max)
        )
        return mask.astype(int).values
    
    def predict_proba(self, X):
        pred = self.predict(X)
        proba = np.zeros((len(pred), 2))
        proba[:, 1] = pred
        proba[:, 0] = 1 - pred
        return proba

baseline_model = CutBasedTrigger(config['baseline'])
baseline_pred = baseline_model.predict(X_test)

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_auc = roc_auc_score(y_test, baseline_pred)

print("✓ Baseline model trained")
print(f"  - Accuracy: {baseline_acc:.1%}")
print(f"  - AUC: {baseline_auc:.3f}")
print()

# ============================================================================
# 4. TRAIN ML MODELS
# ============================================================================

print("[4/6] Training machine learning models...")

# BDT Model
print("  Training BDT...")
from xgboost import XGBClassifier

bdt_model = XGBClassifier(
    n_estimators=config['bdt']['n_estimators'],
    max_depth=config['bdt']['max_depth'],
    learning_rate=config['bdt']['learning_rate'],
    subsample=config['bdt']['subsample'],
    colsample_bytree=config['bdt']['colsample_bytree'],
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

bdt_model.fit(X_train, y_train, verbose=False)
bdt_pred = bdt_model.predict(X_test)
bdt_pred_proba = bdt_model.predict_proba(X_test)

bdt_acc = accuracy_score(y_test, bdt_pred)
bdt_auc = roc_auc_score(y_test, bdt_pred_proba[:, 1])

print(f"  ✓ BDT: Accuracy={bdt_acc:.1%}, AUC={bdt_auc:.3f}")

# Neural Network Model
print("  Training Neural Network...")
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Silence TF warnings
tf.get_logger().setLevel('ERROR')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

history = nn_model.fit(
    X_train_scaled, y_train,
    batch_size=256,
    epochs=30,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
    ],
    verbose=0
)

nn_pred_proba = nn_model.predict(X_test_scaled, verbose=0)
nn_pred = (nn_pred_proba >= 0.5).astype(int).flatten()

nn_acc = accuracy_score(y_test, nn_pred)
nn_auc = roc_auc_score(y_test, nn_pred_proba)

print(f"  ✓ Neural Network: Accuracy={nn_acc:.1%}, AUC={nn_auc:.3f}")
print()

# ============================================================================
# 5. PERFORMANCE EVALUATION
# ============================================================================

print("[5/6] Evaluating model performance...")

from sklearn.metrics import confusion_matrix, roc_curve

def calculate_metrics(y_true, y_pred):
    """Calculate signal efficiency and background rejection."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    signal_efficiency = tp / (tp + fn)  # True positive rate
    background_rejection = tn / (tn + fp)  # True negative rate
    
    return {
        'signal_efficiency': signal_efficiency,
        'background_rejection': background_rejection,
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }

baseline_metrics = calculate_metrics(y_test, baseline_pred)
bdt_metrics = calculate_metrics(y_test, bdt_pred)
nn_metrics = calculate_metrics(y_test, nn_pred)

print("✓ Performance metrics calculated")
print()
print("  Model Comparison:")
print("  " + "-"*66)
print(f"  {'Model':<20} {'Signal Eff':<15} {'Bkg Rejection':<15} {'Accuracy':<15}")
print("  " + "-"*66)
print(f"  {'Cut-based':<20} {baseline_metrics['signal_efficiency']:>13.1%} "
      f"{baseline_metrics['background_rejection']:>14.1%} {baseline_metrics['accuracy']:>14.1%}")
print(f"  {'BDT':<20} {bdt_metrics['signal_efficiency']:>13.1%} "
      f"{bdt_metrics['background_rejection']:>14.1%} {bdt_metrics['accuracy']:>14.1%}")
print(f"  {'Neural Network':<20} {nn_metrics['signal_efficiency']:>13.1%} "
      f"{nn_metrics['background_rejection']:>14.1%} {nn_metrics['accuracy']:>14.1%}")
print("  " + "-"*66)
print()

# ============================================================================
# 6. GENERATE VISUALIZATIONS
# ============================================================================

print("[6/6] Generating visualizations...")

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. ROC Curves
ax1 = fig.add_subplot(gs[0, :2])
fpr_base, tpr_base, _ = roc_curve(y_test, baseline_model.predict_proba(X_test)[:, 1])
fpr_bdt, tpr_bdt, _ = roc_curve(y_test, bdt_pred_proba[:, 1])
fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_pred_proba)

ax1.plot(fpr_base, tpr_base, 'b-', label=f'Cut-based (AUC={baseline_auc:.3f})', linewidth=2)
ax1.plot(fpr_bdt, tpr_bdt, 'g-', label=f'BDT (AUC={bdt_auc:.3f})', linewidth=2)
ax1.plot(fpr_nn, tpr_nn, 'r-', label=f'Neural Network (AUC={nn_auc:.3f})', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate (Background Acceptance)', fontsize=11)
ax1.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=11)
ax1.set_title('ROC Curves - Trigger Performance Comparison', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Signal vs Background pT
ax2 = fig.add_subplot(gs[0, 2])
signal_mask = y_test == 1
ax2.hist(X_test[signal_mask]['pt'], bins=30, alpha=0.6, label='Signal', color='blue', density=True)
ax2.hist(X_test[~signal_mask]['pt'], bins=30, alpha=0.6, label='Background', color='red', density=True)
ax2.axvline(config['baseline']['pt_threshold'], color='black', linestyle='--', 
            label=f'Cut at {config["baseline"]["pt_threshold"]} GeV', linewidth=2)
ax2.set_xlabel('Transverse Momentum pT (GeV)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('pT Distribution', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Feature Importance (BDT)
ax3 = fig.add_subplot(gs[1, :2])
importance = bdt_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importance)[::-1]
ax3.barh(range(len(importance)), importance[indices], color='steelblue')
ax3.set_yticks(range(len(importance)))
ax3.set_yticklabels([feature_names[i] for i in indices])
ax3.set_xlabel('Feature Importance', fontsize=11)
ax3.set_title('BDT Feature Importance', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 4. Confusion Matrix - Neural Network
ax4 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, nn_pred)
im = ax4.imshow(cm, cmap='Blues', aspect='auto')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Background', 'Signal'])
ax4.set_yticklabels(['Background', 'Signal'])
ax4.set_xlabel('Predicted', fontsize=10)
ax4.set_ylabel('True', fontsize=10)
ax4.set_title('Neural Network\nConfusion Matrix', fontsize=11, fontweight='bold')

# Annotate confusion matrix
for i in range(2):
    for j in range(2):
        text = ax4.text(j, i, f'{cm[i, j]}',
                       ha="center", va="center", color="black", fontsize=12)

# 5. Performance Comparison Bar Chart
ax5 = fig.add_subplot(gs[2, :])
metrics_names = ['Signal Efficiency', 'Background Rejection', 'Accuracy']
x = np.arange(len(metrics_names))
width = 0.25

baseline_vals = [baseline_metrics['signal_efficiency'], 
                baseline_metrics['background_rejection'], 
                baseline_metrics['accuracy']]
bdt_vals = [bdt_metrics['signal_efficiency'], 
           bdt_metrics['background_rejection'], 
           bdt_metrics['accuracy']]
nn_vals = [nn_metrics['signal_efficiency'], 
          nn_metrics['background_rejection'], 
          nn_metrics['accuracy']]

ax5.bar(x - width, baseline_vals, width, label='Cut-based', color='steelblue')
ax5.bar(x, bdt_vals, width, label='BDT', color='green')
ax5.bar(x + width, nn_vals, width, label='Neural Network', color='red')

ax5.set_ylabel('Score', fontsize=11)
ax5.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names, fontsize=10)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0.5, 1.0])

# Add value labels on bars
for i, (b, bd, n) in enumerate(zip(baseline_vals, bdt_vals, nn_vals)):
    ax5.text(i - width, b + 0.01, f'{b:.2%}', ha='center', va='bottom', fontsize=8)
    ax5.text(i, bd + 0.01, f'{bd:.2%}', ha='center', va='bottom', fontsize=8)
    ax5.text(i + width, n + 0.01, f'{n:.2%}', ha='center', va='bottom', fontsize=8)

plt.savefig('figures/trigger_ml_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to figures/trigger_ml_results.png")

plt.show()

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================

print()
print("="*70)
print("  DEMO COMPLETED SUCCESSFULLY")
print("="*70)
print()
print("KEY RESULTS:")
print(f"  • Neural Network achieved {nn_metrics['signal_efficiency']:.1%} signal efficiency")
print(f"  • {((nn_metrics['signal_efficiency'] - baseline_metrics['signal_efficiency']) / baseline_metrics['signal_efficiency'] * 100):.1f}% improvement over cut-based baseline")
print(f"  • Background rejection improved from {baseline_metrics['background_rejection']:.1%} to {nn_metrics['background_rejection']:.1%}")
print()
print("FILES GENERATED:")
print("  • figures/trigger_ml_results.png - Comprehensive visualization")
print()
print("NEXT STEPS:")
print("  1. Run full training with 50k events: python scripts/train_models.py")
print("  2. Explore Jupyter notebooks for detailed analysis")
print("  3. Implement quantized model for FPGA deployment")
print("  4. Test with real CERN Open Data")
print()
print("For questions or issues, check README.md or open a GitHub issue.")
print("="*70)