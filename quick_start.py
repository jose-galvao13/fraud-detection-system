#!/usr/bin/env python3
"""
QUICK START: Unsupervised Fraud Detection
==========================================
Simplified execution script with built-in error handling.

Usage:
    python quick_start.py creditcard.csv
    
Or (if file is in same directory):
    python quick_start.py
"""

import sys
import os

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              UNSUPERVISED FRAUD DETECTION - QUICK START                   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# Check Python version
if sys.version_info < (3, 7):
    print("❌ ERROR: Python 3.7+ required")
    sys.exit(1)

print("✓ Python version OK")

# Check required libraries
print("\nChecking required libraries...")
missing_libs = []

try:
    import pandas as pd
    print("  ✓ pandas")
except ImportError:
    missing_libs.append('pandas')
    print("  ✗ pandas")

try:
    import numpy as np
    print("  ✓ numpy")
except ImportError:
    missing_libs.append('numpy')
    print("  ✗ numpy")

try:
    import sklearn
    print("  ✓ scikit-learn")
except ImportError:
    missing_libs.append('scikit-learn')
    print("  ✗ scikit-learn")

try:
    import matplotlib.pyplot as plt
    print("  ✓ matplotlib")
except ImportError:
    missing_libs.append('matplotlib')
    print("  ✗ matplotlib")

try:
    import seaborn as sns
    print("  ✓ seaborn")
except ImportError:
    missing_libs.append('seaborn')
    print("  ✗ seaborn")

if missing_libs:
    print(f"\n❌ Missing libraries: {', '.join(missing_libs)}")
    print(f"\nInstall them with:")
    print(f"  pip install {' '.join(missing_libs)}")
    sys.exit(1)

print("\n✓ All libraries available")

# Determine CSV path
csv_path = None

# 1. Command line argument
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
# 2. Common locations
else:
    candidates = [
        './creditcard.csv',
        '../creditcard.csv',
        'creditcard.csv'
    ]
    
    for path in candidates:
        if os.path.exists(path):
            csv_path = path
            break

if not csv_path or not os.path.exists(csv_path):
    print("\n❌ ERROR: creditcard.csv not found!")
    print("\nPlease place creditcard.csv in one of these locations:")
    print("  • ./creditcard.csv")
    print("\nOr provide the path as argument:")
    print("  python quick_start.py /path/to/creditcard.csv")
    sys.exit(1)

print(f"\n✓ Found CSV file: {csv_path}")

# Create output directory
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)
print(f"✓ Output directory: {output_dir}")

# Import and run main script
print("\n" + "=" * 80)
print("STARTING FRAUD DETECTION ANALYSIS")
print("=" * 80)

# We'll create a simplified version right here
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import confusion_matrix


# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n📂 Loading {csv_path}...")
try:
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} transactions")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    sys.exit(1)

    df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    print(f"⚡ Subsampling to 50,000 rows for speed...")

# Validate
required_cols = {'Time', 'Amount', 'Class'}
pca_cols = {f'V{i}' for i in range(1, 29)}

missing = required_cols - set(df.columns)
if missing:
    print(f"❌ Missing columns: {missing}")
    sys.exit(1)

print(f"✓ Data validation passed")
print(f"  - Normal: {(df['Class']==0).sum():,}")
print(f"  - Fraud: {(df['Class']==1).sum():,}")

# ============================================================================
# PREPROCESS
# ============================================================================

print(f"\n🔧 Preprocessing...")
feature_cols = [col for col in df.columns if col not in ['Time', 'Class']]
X = df[feature_cols].values
y = df['Class'].values

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Features normalized")

# ============================================================================
# TRAIN MODELS
# ============================================================================

print(f"\n🤖 Training models...")

CONTAM = 0.01

# Isolation Forest
print("  🌲 Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,
    max_samples='auto', 
    contamination=CONTAM, 
    random_state=42, 
    n_jobs=-1
)
pred_if = (iso.fit_predict(X_scaled) == -1).astype(int)
print(f"    ✓ Detected: {pred_if.sum():,}")

# PCA
print("  🔍 PCA Outlier Detector")
pca_detector = PCA(n_components=5) # Reduzimos para captar o "padrão"
X_pca = pca_detector.fit_transform(X_scaled)
X_reconstructed = pca_detector.inverse_transform(X_pca)

# Calcula o erro de reconstrução (distância entre o original e o reconstruído)
mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
# O limiar é o percentil da nossa contaminação (ex: top 1% maiores erros)
threshold_pca = np.percentile(mse, 100 * (1 - CONTAM))
pred_lof = (mse > threshold_pca).astype(int) # Mantemos o nome pred_lof para não quebrar o resto do seu código
print(f"    ✓ Detected: {pred_lof.sum():,}")

# Elliptic
print("  ⭕ Elliptic Envelope...")
ee = EllipticEnvelope(contamination=0.002, random_state=42)
pred_ee = (ee.fit_predict(X_scaled) == -1).astype(int)
print(f"    ✓ Detected: {pred_ee.sum():,}")

# ============================================================================
# ENSEMBLE
# ============================================================================

print(f"\n🎯 Creating ensemble...")
ensemble = (pred_if + pred_lof + pred_ee >= 1).astype(int)
print(f"✓ Ensemble (voting): {ensemble.sum():,} frauds")

# ============================================================================
# METRICS
# ============================================================================

print(f"\n📊 Calculating metrics...")

def calc_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).ravel()
    tn, fp, fn, tp = cm
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'recall': recall, 'precision': precision, 'f1': f1}

m_if = calc_metrics(y, pred_if)
m_lof = calc_metrics(y, pred_lof)
m_ee = calc_metrics(y, pred_ee)
m_ens = calc_metrics(y, ensemble)

print(f"\nENSEMBLE RESULTS:")
print(f"  ✓ True Positives: {m_ens['tp']:,}")
print(f"  ✗ False Positives: {m_ens['fp']:,}")
print(f"  ✗ False Negatives: {m_ens['fn']:,}")
print(f"  Recall: {m_ens['recall']:.1%}")
print(f"  Precision: {m_ens['precision']:.1%}")
print(f"  F1-Score: {m_ens['f1']:.1%}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n💾 Saving results...")

results = pd.DataFrame({
    'TransactionID': range(len(df)),
    'Time': df['Time'],
    'Amount': df['Amount'],
    'True_Class': y,
    'Isolation_Forest': pred_if,
    'LOF': pred_lof,
    'Elliptic_Envelope': pred_ee,
    'Ensemble': ensemble,
})

results.to_csv(f'{output_dir}/results.csv', index=False)
print(f"✓ Saved: {output_dir}/results.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print(f"\n📊 Creating visualizations...")

# 1. PCA 3D
print("  Creating 3D visualization...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(16, 10))
for idx, (title, pred) in enumerate([
    ('Ground Truth', y),
    ('Isolation Forest', pred_if),
    ('LOF', pred_lof),
    ('Elliptic Envelope', pred_ee),
    ('Ensemble', ensemble),
], 1):
    ax = fig.add_subplot(2, 3, idx, projection='3d')
    normal = pred == 0
    anomaly = pred == 1
    ax.scatter(X_pca[normal, 0], X_pca[normal, 1], X_pca[normal, 2],
        c='blue', alpha=0.05, s=1) 
    ax.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1], X_pca[anomaly, 2],
        c='red', alpha=0.5, s=20, marker='x') 
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

ax = fig.add_subplot(2, 3, 6)
ax.bar(['PC1', 'PC2', 'PC3'], pca.explained_variance_ratio_ * 100)
ax.set_ylabel('Variance %')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_3d_visualization.png', dpi=100)
plt.close()
print("  ✓ Saved: 01_3d_visualization.png")

# 2. Performance
print("  Creating performance metrics...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

models = ['Isolation\nForest', 'LOF', 'Elliptic\nEnvelope', 'Ensemble']
recalls = [m_if['recall'], m_lof['recall'], m_ee['recall'], m_ens['recall']]
precisions = [m_if['precision'], m_lof['precision'], m_ee['precision'], m_ens['precision']]
f1s = [m_if['f1'], m_lof['f1'], m_ee['f1'], m_ens['f1']]

axes[0, 0].bar(models, recalls, color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
axes[0, 0].set_ylabel('Recall')
axes[0, 0].set_title('Sensitivity (Fraud Detection Rate)')
axes[0, 0].set_ylim([0, 1])

axes[0, 1].bar(models, precisions, color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Quality of Alerts')
axes[0, 1].set_ylim([0, 1])

x = np.arange(len(models))
width = 0.35
axes[1, 0].bar(x - width/2, recalls, width, label='Recall', alpha=0.7)
axes[1, 0].bar(x + width/2, precisions, width, label='Precision', alpha=0.7)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Recall vs Precision')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(models)
axes[1, 0].legend()
axes[1, 0].set_ylim([0, 1])

axes[1, 1].bar(models, f1s, color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_title('Balanced Score')
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/02_performance.png', dpi=100)
plt.close()
print("  ✓ Saved: 02_performance.png")

# 3. Amount analysis
print("  Creating amount analysis...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6, label='Normal', density=True)
axes[0].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.6, label='Fraud', density=True)
axes[0].set_xlabel('Amount (€)')
axes[0].set_ylabel('Density')
axes[0].set_title('Amount Distribution')
axes[0].legend()

tp_amt = df[(ensemble==1) & (y==1)]['Amount']
fp_amt = df[(ensemble==1) & (y==0)]['Amount']
fn_amt = df[(ensemble==0) & (y==1)]['Amount']

axes[1].hist(tp_amt, bins=30, alpha=0.7, label=f'TP: {len(tp_amt)}', color='green')
axes[1].hist(fp_amt, bins=30, alpha=0.7, label=f'FP: {len(fp_amt)}', color='orange')
axes[1].hist(fn_amt, bins=30, alpha=0.7, label=f'FN: {len(fn_amt)}', color='red')
axes[1].set_xlabel('Amount (€)')
axes[1].set_ylabel('Count')
axes[1].set_title('Ensemble Detection by Amount')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/03_amount_analysis.png', dpi=100)
plt.close()
print("  ✓ Saved: 03_amount_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\n📁 Results saved to: {output_dir}")
print(f"\nGenerated files:")
print(f"  • results.csv              (Detailed predictions)")
print(f"  • 01_3d_visualization.png  (3D PCA visualization)")
print(f"  • 02_performance.png       (Performance metrics)")
print(f"  • 03_amount_analysis.png   (Amount analysis)")

print(f"\n📊 Key Findings:")
print(f"  • Detected {ensemble.sum():,} anomalies")
print(f"  • Recall: {m_ens['recall']:.1%} (caught {m_ens['tp']:,}/{m_ens['tp']+m_ens['fn']:,} frauds)")
print(f"  • Precision: {m_ens['precision']:.1%} ({m_ens['tp']:,} correct out of {m_ens['tp']+m_ens['fp']:,} alerts)")
print(f"  • F1-Score: {m_ens['f1']:.1%}")

print(f"\n✓ Next steps:")
print(f"  1. Review results.csv for individual cases")
print(f"  2. Check visualizations for patterns")
print(f"  3. Validate with domain experts")
print(f"  4. Adjust thresholds if needed")
print(f"  5. Deploy with human oversight")

print(f"\n" + "=" * 80)

# ============================================================================
# FINAL EXECUTION & REPORT GENERATION
# ============================================================================

def get_metrics_dict(y_true, pred):
    cm = confusion_matrix(y_true, pred).ravel()
    tn, fp, fn, tp = cm
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'recall': recall, 'precision': precision, 'f1': f1}

def create_html_report(df, y_true, m_if, m_pca, m_ee, m_ens, output_dir):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>🔐 Fraud Detection Report - Dark Mode</title>
        <style>
            :root {{
                --bg-body: #0f172a;
                --bg-container: #1e293b;
                --bg-card: #334155;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --accent: #818cf8;
                --neon-green: #4ade80;
                --neon-red: #fb7185;
                --gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            }}
            
            body {{ 
                font-family: 'Segoe UI', system-ui, sans-serif; 
                background-color: var(--bg-body); 
                color: var(--text-main); 
                margin: 0; 
                padding: 20px; 
            }}
            
            .container {{ 
                max-width: 1100px; 
                margin: auto; 
                background: var(--bg-container); 
                padding: 40px; 
                border-radius: 20px; 
                box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
                border: 1px solid rgba(255,255,255,0.05);
            }}
            
            header {{ 
                text-align: center; 
                background: var(--gradient); 
                padding: 40px; 
                border-radius: 16px; 
                margin-bottom: 40px;
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);
            }}
            
            h1 {{ margin: 0; font-size: 2.5em; letter-spacing: -1px; }}
            h2 {{ color: var(--accent); border-bottom: 1px solid var(--bg-card); padding-bottom: 10px; margin-top: 50px; }}
            
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
                gap: 20px; 
                margin-bottom: 40px; 
            }}
            
            .card {{ 
                background: var(--bg-card); 
                padding: 25px; 
                border-radius: 15px; 
                border-bottom: 4px solid var(--accent); 
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .card:hover {{ transform: translateY(-5px); }}
            .card h3 {{ margin: 0; font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }}
            .card .value {{ font-size: 32px; font-weight: bold; color: #fff; margin-top: 10px; }}
            
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin-top: 20px; 
                background: rgba(0,0,0,0.2);
                border-radius: 10px;
                overflow: hidden;
            }}
            
            th, td {{ padding: 18px; text-align: left; border-bottom: 1px solid var(--bg-card); }}
            th {{ background: rgba(79, 70, 229, 0.3); color: var(--accent); font-size: 14px; }}
            tr:hover {{ background: rgba(255,255,255,0.02); }}
            
            .highlight {{ background: rgba(129, 140, 248, 0.15) !important; font-weight: bold; }}
            .highlight td {{ color: var(--neon-green); }}

            .viz {{ text-align: center; margin-top: 50px; }}
            .viz h3 {{ color: var(--text-muted); font-weight: 400; margin-bottom: 20px; }}
            
            /* Efeito para as imagens (que têm fundo branco) não chocarem com o fundo escuro */
            .viz img {{ 
                max-width: 100%; 
                border-radius: 15px; 
                box-shadow: 0 20px 25px -5px rgba(0,0,0,0.3); 
                margin-bottom: 60px; 
                border: 8px solid #fff; /* Moldura estilo Polaroid */
            }}
            
            footer {{ text-align: center; margin-top: 40px; color: var(--text-muted); font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔐 Fraud Detection System</h1>
                <p>Unsupervised Security Analysis • {len(df):,} Transactions processed</p>
            </header>

            <h2>📊 Executive Summary (Ensemble)</h2>
            <div class="grid">
                <div class="card"><h3>Total Correct (TP)</h3><div class="value" style="color:var(--neon-green)">{m_ens['tp']:,}</div></div>
                <div class="card"><h3>Recall Rate</h3><div class="value">{m_ens['recall']:.1%}</div></div>
                <div class="card"><h3>False Alarms (FP)</h3><div class="value" style="color:var(--neon-red)">{m_ens['fp']:,}</div></div>
                <div class="card"><h3>Precision</h3><div class="value">{m_ens['precision']:.1%}</div></div>
            </div>

            <h2>🤖 Algorithm Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>DETECTION MODEL</th>
                        <th>TRUE POSITIVES</th>
                        <th>FALSE ALARMS</th>
                        <th>RECALL</th>
                        <th>PRECISION</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Isolation Forest</td><td>{m_if['tp']}</td><td>{m_if['fp']}</td><td>{m_if['recall']:.1%}</td><td>{m_if['precision']:.1%}</td></tr>
                    <tr><td>PCA Reconstruction</td><td>{m_pca['tp']}</td><td>{m_pca['fp']}</td><td>{m_pca['recall']:.1%}</td><td>{m_pca['precision']:.1%}</td></tr>
                    <tr><td>Elliptic Envelope</td><td>{m_ee['tp']}</td><td>{m_ee['fp']}</td><td>{m_ee['recall']:.1%}</td><td>{m_ee['precision']:.1%}</td></tr>
                    <tr class="highlight"><td>VOTING ENSEMBLE (FINAL)</td><td>{m_ens['tp']}</td><td>{m_ens['fp']}</td><td>{m_ens['recall']:.1%}</td><td>{m_ens['precision']:.1%}</td></tr>
                </tbody>
            </table>

            <div class="viz">
                <h2>📈 Deep Visual Analysis</h2>
                
                <h3>3D Anomalies in PCA Space</h3>
                <img src="01_3d_visualization.png">
                
                <h3>Performance Comparison Metrics</h3>
                <img src="02_performance.png">
                
                <h3>Transaction Amount Distribution</h3>
                <img src="03_amount_analysis.png">
            </div>
            
            <footer>
                Generated by Unsupervised Fraud Detection System • 2026
            </footer>
        </div>
    </body>
    </html>
    """
    with open(f'{output_dir}/report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

# 1. Gerar os dicionários de métricas com os resultados que calculamos antes
metrics_if = get_metrics_dict(y, pred_if)
metrics_pca = get_metrics_dict(y, pred_lof) # pred_lof é o seu PCA
metrics_ee = get_metrics_dict(y, pred_ee)
metrics_ens = get_metrics_dict(y, ensemble)

# 2. Chamar a função para criar o arquivo HTML
create_html_report(df, y, metrics_if, metrics_pca, metrics_ee, metrics_ens, output_dir)

print(f"✅ FINAL STEP: HTML Report generated at {output_dir}report.html")
print(f"\n" + "=" * 80)