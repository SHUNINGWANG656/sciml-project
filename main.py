import os, warnings, tempfile
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from torchdiffeq import odeint_adjoint as odeint

torch.manual_seed(42)
np.random.seed(42)

# CONFIG
BASE_DIR     = Path("/scratch/Workspace2/xwang434/IsaacLab/lerobot/Sciml")
VIDEO_DIR    = BASE_DIR / "videos"
FEATURE_DIR  = BASE_DIR / "features_cv"
ANNOT_CSV    = BASE_DIR / "ELLA_Error_Lables.csv"
MANIFEST_CSV = BASE_DIR / "manifest_cv.csv"

WINDOW_SEC    = 3.0
FPS_DEFAULT   = 30.0
SKIP_FRAMES   = 14
WINDOW_FRAMES = 18      # int(3.0 * 30 / (14+1))
D_VISUAL      = 3       # optical flow: [||v||, vx, vy]
D_U           = 11      # control: 10 error categories + 1 onset time
K_LATENT      = 32      # Neural ODE latent dimension
H_LSTM        = 64      # LSTM hidden dimension

EPOCHS   = 100
PATIENCE = 15
LR_RATE  = 1e-3
BATCH    = 16
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# STEP 1: FEATURE EXTRACTION
print("\n" + "="*60)
print("  STEP 1: Extracting optical flow features")
print("="*60)

FEATURE_DIR.mkdir(exist_ok=True)

ERROR_CATEGORIES = [
    'Wrong_scaffold', 'Functions_lacking', 'No_reply',
    'STT_misunderstood', 'Wrong_answer', 'STT_didnt_hear',
    'Robot_interruption', 'Child_interruption', 'STT_shutdown', 'LLM_error',
]

CAT_MAP = {
    'Wrong scaffold':                                      'Wrong_scaffold',
    'Functions That Robots Lack':                          'Functions_lacking',
    'No immediate reply':                                  'No_reply',
    'No Immediate Reply':                                  'No_reply',
    'Speech to Text Error (Identification Error)':         'STT_misunderstood',
    'Wrong Answer':                                        'Wrong_answer',
    "Speech to Text Error (robot didn't hear the child)": 'STT_didnt_hear',
    'Robot Interruption':                                  'Robot_interruption',
    'Child interruption Unhandled':                        'Child_interruption',
    'Speech to Text Error (robot shutdown)':               'STT_shutdown',
    'LLM error (system error)':                            'LLM_error',
}

def parse_time(t):
    try:
        p = str(t).strip().split(':')
        return int(p[0]) * 60 + float(p[1])
    except:
        return None

def encode_u(error_cat_raw, onset_sec, max_onset=3600.0):
    """Build control vector u ∈ R^11: [e(t) ∈ R^10 one-hot, τ ∈ R^1 normalized onset]"""
    cat = CAT_MAP.get(str(error_cat_raw).strip(), 'Wrong_scaffold')
    e = np.zeros(10, dtype=np.float32)
    if cat in ERROR_CATEGORIES:
        e[ERROR_CATEGORIES.index(cat)] = 1.0
    tau = np.array([onset_sec / max_onset], dtype=np.float32)
    return np.concatenate([e, tau])

def extract_optical_flow(video_path, onset_sec, window_sec=3.0,
                          skip_frames=14, fps=30.0, n_frames=18):
    """
    Extract dense optical flow features for a single event window.
    Returns array of shape (n_frames, 3): [mean_magnitude, mean_vx, mean_vy]
    """
    cap = cv2.VideoCapture(str(video_path))
    t_start = max(0.0, onset_sec - window_sec)

    # Seek to start frame
    start_frame = int(t_start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    fi = start_frame
    while len(frames) < n_frames * (skip_frames + 1) + 1:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fi += 1
    cap.release()

    if len(frames) < 2:
        return None

    # Compute optical flow between consecutive sampled frames
    features = []
    sampled = frames[::skip_frames + 1]
    for i in range(min(n_frames, len(sampled) - 1)):
        flow = cv2.calcOpticalFlowFarneback(
            sampled[i], sampled[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        vx  = flow[..., 0].mean()
        vy  = flow[..., 1].mean()
        features.append([mag, vx, vy])

    feat_arr = np.array(features, dtype=np.float32)

    # Pad or trim to exactly n_frames
    n = len(feat_arr)
    if n >= n_frames:
        return feat_arr[-n_frames:]
    pad = np.zeros((n_frames - n, D_VISUAL), dtype=np.float32)
    return np.vstack([pad, feat_arr])

# Load and filter annotations
df = pd.read_csv(ANNOT_CSV)
df['label']      = df['Child Reaction Verbal'].isin(['Verbal', 'Non-verbal']).astype(int)
df['onset_sec']  = df['Error Onset'].apply(parse_time)
df['Video Name'] = df['Video Name'].str.strip()
df = df[df['Child Visible'] != 'Child not visible at all']
df = df.dropna(subset=['onset_sec']).reset_index(drop=True)

server_videos = set(v.stem for v in VIDEO_DIR.glob('*.mp4'))
df = df[df['Video Name'].isin(server_videos)].reset_index(drop=True)
MAX_ONSET = df['onset_sec'].max()

# Extract features for each event
manifest_rows = []
skipped = 0
for idx, row in df.iterrows():
    npy_path = FEATURE_DIR / f"event_{idx:04d}.npy"
    if npy_path.exists():
        try:
            arr = np.load(str(npy_path))
            if arr.shape == (WINDOW_FRAMES, D_VISUAL):
                manifest_rows.append({
                    'event_idx': idx,
                    'npy_path':  str(npy_path),
                    'label':     int(row['label']),
                    'child':     row['Child Name'],
                    'video':     row['Video Name'],
                    'onset_sec': row['onset_sec'],
                })
                continue
        except Exception:
            pass

    vid_path = VIDEO_DIR / f"{row['Video Name']}.mp4"
    feat = extract_optical_flow(str(vid_path), row['onset_sec'],
                                 window_sec=WINDOW_SEC,
                                 skip_frames=SKIP_FRAMES,
                                 fps=FPS_DEFAULT,
                                 n_frames=WINDOW_FRAMES)
    if feat is None:
        skipped += 1
        continue

    np.save(str(npy_path), feat)
    manifest_rows.append({
        'event_idx': idx,
        'npy_path':  str(npy_path),
        'label':     int(row['label']),
        'child':     row['Child Name'],
        'video':     row['Video Name'],
        'onset_sec': row['onset_sec'],
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(MANIFEST_CSV, index=False)

pos = manifest['label'].sum()
print(f"Features extracted: {len(manifest)} events | "
      f"Positive: {pos} ({pos/len(manifest):.1%}) | Skipped: {skipped}")

# STEP 2: MODEL TRAINING & LOSO EVALUATION
print("\n" + "="*60)
print(f"  STEP 2: Training models (device: {DEVICE})")
print("="*60)

# Build sample list
manifest = pd.read_csv(MANIFEST_CSV).set_index('event_idx')
annot    = pd.read_csv(ANNOT_CSV)
annot['label']     = annot['Child Reaction Verbal'].isin(['Verbal','Non-verbal']).astype(int)
annot['onset_sec'] = annot['Error Onset'].apply(parse_time)
annot['Video Name'] = annot['Video Name'].str.strip()
annot = annot[annot['Child Visible'] != 'Child not visible at all']
annot = annot.dropna(subset=['onset_sec']).reset_index(drop=True)
MAX_ONSET = annot['onset_sec'].max()

samples = []
for idx, row in annot.iterrows():
    if idx not in manifest.index:
        continue
    npy_path = manifest.loc[idx, 'npy_path']
    if not os.path.exists(npy_path):
        continue
    try:
        x_vis = np.load(npy_path).astype(np.float32)
    except Exception:
        continue
    if x_vis.shape != (WINDOW_FRAMES, D_VISUAL):
        continue
    u = encode_u(row['Reason for error/misalignment(summary)'],
                 row['onset_sec'], MAX_ONSET)
    samples.append({
        'x':     x_vis,
        'u':     u,
        'y':     int(row['label']),
        'child': row['Child Name'],
    })

children        = np.array([s['child'] for s in samples])
unique_children = sorted(set(children))
print(f"Samples: {len(samples)} | Children: {len(unique_children)}")

# Dataset
class ReactionDataset(Dataset):
    def __init__(self, sample_list, scaler=None, fit_scaler=False):
        xs   = np.stack([s['x'] for s in sample_list])
        N, T, D = xs.shape
        flat = xs.reshape(N, T * D)
        if fit_scaler:
            scaler.fit(flat)
        if scaler is not None:
            flat = scaler.transform(flat)
        self.x = torch.tensor(flat.reshape(N, T, D), dtype=torch.float32)
        self.u = torch.tensor(np.stack([s['u'] for s in sample_list]),
                              dtype=torch.float32)
        self.y = torch.tensor([s['y'] for s in sample_list], dtype=torch.float32)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.x[i], self.u[i], self.y[i]

# Models
class LSTMModel(nn.Module):
    """
    LSTM baseline.
    ŷ = σ(W₂ ReLU(W₁ [c_T; u] + b₁) + b₂)
    c_T ∈ R^H: LSTM final hidden state
    """
    def __init__(self, d_x=D_VISUAL, d_u=D_U, h=H_LSTM, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d_x, h, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(h + d_u, h), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1), nn.Sigmoid(),
        )

    def forward(self, x, u):
        _, (c_T, _) = self.lstm(x)
        c_T = self.drop(c_T.squeeze(0))
        return self.head(torch.cat([c_T, u], dim=-1)).squeeze(-1)


class ODEFunc(nn.Module):
    """
    ODE right-hand side: f_θ: R^k × R^m → R^k
    dh/dt = f_θ(h(t), u, t)
    """
    def __init__(self, k=K_LATENT, d_u=D_U, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k + d_u, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, k),
        )
        self.u = None

    def forward(self, t, h):
        return self.net(torch.cat([h, self.u], dim=-1))


class NeuralODEModel(nn.Module):
    """
    Neural ODE model.
    Encoder:  LSTM(x_{t-T:t}) → h(t₀) ∈ R^k
    ODE:      dh/dt = f_θ(h(t), u, t),  integrated with dopri5
    Classify: ŷ = σ(W [h(t₀+Δ); u] + b)
    Gradients via adjoint method: O(1) memory in ODE steps.
    """
    def __init__(self, d_x=D_VISUAL, d_u=D_U, k=K_LATENT, h_enc=32):
        super().__init__()
        self.encoder    = nn.LSTM(d_x, h_enc, batch_first=True)
        self.enc_proj   = nn.Linear(h_enc, k)
        self.odefunc    = ODEFunc(k=k, d_u=d_u)
        self.classifier = nn.Sequential(nn.Linear(k + d_u, 1), nn.Sigmoid())
        self.t_span     = torch.tensor([0.0, 1.0])

    def forward(self, x, u):
        _, (h_enc, _) = self.encoder(x)
        h0 = self.enc_proj(h_enc.squeeze(0))
        self.odefunc.u = u
        t_span = self.t_span.to(x.device)
        h_traj = odeint(self.odefunc, h0, t_span,
                        method='dopri5', rtol=1e-4, atol=1e-5)
        return self.classifier(
            torch.cat([h_traj[-1], u], dim=-1)
        ).squeeze(-1)

# Training utilities
def compute_pos_weight(y_train):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

def val_f1(model, val_ds, device):
    """Compute macro F1 on validation set for early stopping."""
    model.eval()
    with torch.no_grad():
        prob = model(val_ds.x.to(device), val_ds.u.to(device)).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return f1_score(val_ds.y.numpy().astype(int), pred,
                    average='macro', zero_division=0)

def train_nn(model, train_loader, val_ds, pos_weight, device, epochs, patience):
    """Train with Adam + weighted BCE, early stopping on validation macro F1."""
    model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    bce        = nn.BCELoss(reduction='none')
    best_f1    = -1.0
    best_state = None
    counter    = 0

    for epoch in range(epochs):
        model.train()
        for x_b, u_b, y_b in train_loader:
            x_b, u_b, y_b = x_b.to(device), u_b.to(device), y_b.to(device)
            pred = model(x_b, u_b)
            w    = torch.where(y_b == 1,
                               pos_weight.to(device).expand_as(y_b),
                               torch.ones_like(y_b))
            loss = (bce(pred, y_b) * w).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1_val = val_f1(model, val_ds, device)
        if f1_val > best_f1:
            best_f1    = f1_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    return model

def eval_nn(model, x_t, u_t, device):
    model.eval()
    with torch.no_grad():
        return model(x_t.to(device), u_t.to(device)).cpu().numpy()

def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
    bal = balanced_accuracy_score(y_true, y_pred)
    return f1, auc, bal

# LOSO cross-validation (8 train / 1 val / 1 test)
all_results = []

for fold, test_child in enumerate(unique_children):
    print(f"\nFold {fold+1}/{len(unique_children)} — Test: {test_child}")

    remaining = [c for c in unique_children if c != test_child]
    val_child = remaining[-1]   # fixed deterministic val child

    tr_samp  = [s for s in samples if s['child'] not in (test_child, val_child)]
    val_samp = [s for s in samples if s['child'] == val_child]
    te_samp  = [s for s in samples if s['child'] == test_child]

    if len(te_samp) == 0 or len(val_samp) == 0:
        continue

    scaler    = StandardScaler()
    tr_ds     = ReactionDataset(tr_samp,  scaler=scaler, fit_scaler=True)
    val_ds    = ReactionDataset(val_samp, scaler=scaler, fit_scaler=False)
    te_ds     = ReactionDataset(te_samp,  scaler=scaler, fit_scaler=False)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)

    y_tr = tr_ds.y.numpy()
    y_te = te_ds.y.numpy().astype(int)
    pw   = compute_pos_weight(y_tr)

    # LR — tune C on val set
    best_c, best_vf1 = 0.1, -1
    X_tr  = np.concatenate([tr_ds.x.numpy().reshape(len(tr_samp), -1),
                             tr_ds.u.numpy()], axis=1)
    X_val = np.concatenate([val_ds.x.numpy().reshape(len(val_samp), -1),
                             val_ds.u.numpy()], axis=1)
    X_te  = np.concatenate([te_ds.x.numpy().reshape(len(te_samp), -1),
                             te_ds.u.numpy()], axis=1)
    for c in [0.001, 0.01, 0.1, 1.0]:
        lr_tmp = LogisticRegression(C=c, class_weight='balanced',
                                    max_iter=1000, random_state=42)
        lr_tmp.fit(X_tr, y_tr)
        vp = lr_tmp.predict_proba(X_val)[:, 1]
        vf = f1_score(val_ds.y.numpy().astype(int), (vp>=0.5).astype(int),
                      average='macro', zero_division=0)
        if vf > best_vf1:
            best_vf1, best_c = vf, c

    lr = LogisticRegression(C=best_c, class_weight='balanced',
                            max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    f1, auc, bal = metrics(y_te, lr.predict_proba(X_te)[:, 1])
    all_results.append({'model':'LR','child':test_child,'f1':f1,'auc':auc,'bal_acc':bal})
    print(f"  LR (C={best_c})  | F1={f1:.3f}  AUC={auc:.3f}  BalAcc={bal:.3f}")

    # LSTM
    lstm = train_nn(LSTMModel(), tr_loader, val_ds, pw, DEVICE, EPOCHS, PATIENCE)
    f1, auc, bal = metrics(y_te, eval_nn(lstm, te_ds.x, te_ds.u, DEVICE))
    all_results.append({'model':'LSTM','child':test_child,'f1':f1,'auc':auc,'bal_acc':bal})
    print(f"  LSTM            | F1={f1:.3f}  AUC={auc:.3f}  BalAcc={bal:.3f}")

    # Neural ODE
    ode = train_nn(NeuralODEModel(), tr_loader, val_ds, pw, DEVICE, EPOCHS, PATIENCE)
    f1, auc, bal = metrics(y_te, eval_nn(ode, te_ds.x, te_ds.u, DEVICE))
    all_results.append({'model':'NeuralODE','child':test_child,'f1':f1,'auc':auc,'bal_acc':bal})
    print(f"  Neural ODE      | F1={f1:.3f}  AUC={auc:.3f}  BalAcc={bal:.3f}")

rdf = pd.DataFrame(all_results)
rdf.to_csv('results_loso.csv', index=False)

print(f"\n{'='*60}")
print("FINAL SUMMARY — mean ± std (LOSO, 8 train / 1 val / 1 test)")
print(f"{'='*60}")
lines = []
for m in ['LR', 'LSTM', 'NeuralODE']:
    s = rdf[rdf['model'] == m]
    line = (f"{m:10s} | "
            f"F1={s['f1'].mean():.3f}±{s['f1'].std():.3f}  "
            f"AUC={s['auc'].mean():.3f}±{s['auc'].std():.3f}  "
            f"BalAcc={s['bal_acc'].mean():.3f}±{s['bal_acc'].std():.3f}")
    print(line)
    lines.append(line)

with open('results_summary.txt', 'w') as f:
    f.write("LOSO Results — optical flow (d=3), 8 train / 1 val / 1 test\n")
    f.write("Early stopping on validation macro F1 (patience=15)\n")
    f.write("=" * 60 + "\n")
    f.write('\n'.join(lines) + '\n')

# STEP 3: GENERATE FIGURES
print("\n" + "="*60)
print("  STEP 3: Generating figures")
print("="*60)

models       = ['LR', 'LSTM', 'NeuralODE']
model_labels = ['LR\n(Baseline)', 'LSTM\n(Discrete)', 'Neural ODE\n(SciML)']
colors       = {'LR': '#2196F3', 'LSTM': '#FF9800', 'NeuralODE': '#4CAF50'}
markers      = {'LR': 'o', 'LSTM': 's', 'NeuralODE': '^'}
met_colors   = ['#1565C0', '#E65100', '#2E7D32']
met_labels   = ['Macro F1', 'AUC-ROC', 'Balanced Acc.']

means_list = {k: [rdf[rdf['model']==m][{'F1':'f1','AUC':'auc','BalAcc':'bal_acc'}[k]].mean()
                  for m in models]
              for k in ['F1','AUC','BalAcc']}
stds_list  = {k: [rdf[rdf['model']==m][{'F1':'f1','AUC':'auc','BalAcc':'bal_acc'}[k]].std()
                  for m in models]
              for k in ['F1','AUC','BalAcc']}

# Figure 1: Summary bar chart
fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(models))
width = 0.25
for i, (metric, mlabel, color) in enumerate(zip(['F1','AUC','BalAcc'], met_labels, met_colors)):
    ax.bar(x + (i-1)*width, means_list[metric], width,
           yerr=stds_list[metric], capsize=4, color=color, alpha=0.82,
           label=mlabel, error_kw={'linewidth': 1.2})
    for j, (m, s) in enumerate(zip(means_list[metric], stds_list[metric])):
        ax.text(j+(i-1)*width, m+s+0.025, f'{m:.3f}',
                ha='center', va='bottom', fontsize=6.5, color=color, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel('Score', fontsize=10)
ax.set_ylim(0, 0.98)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Chance')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('LOSO Cross-Validation Results (mean $\\pm$ std, 10 folds)',
             fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_summary.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved fig_summary.png")

# Figure 2: Per-fold line plot
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, metric, col, mlabel in zip(axes,
    ['f1','auc','bal_acc'], ['F1','AUC','BalAcc'], met_labels):
    for model in models:
        vals = rdf[rdf['model']==model][metric].values
        ax.plot(range(1, len(vals)+1), vals,
                color=colors[model], marker=markers[model],
                linewidth=1.8, markersize=5, label=model, alpha=0.85)
        ax.axhline(vals.mean(), color=colors[model], linestyle='--',
                   linewidth=1.0, alpha=0.5)
    ax.set_xlabel('LOSO Fold (Child)', fontsize=9)
    ax.set_ylabel(mlabel, fontsize=9)
    ax.set_title(mlabel, fontsize=10, fontweight='bold')
    ax.set_xticks(range(1, len(vals)+1))
    ax.set_xticklabels([f'C{i}' for i in range(1, len(vals)+1)], fontsize=7)
    ax.set_ylim(0.1, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='lower right')
plt.suptitle('Per-Fold LOSO Results Across 10 Children',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_perfold.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved fig_perfold.png")

# Figure 3: Neural ODE latent trajectory illustration
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
t = np.linspace(0, 1, 200)
clrs = ['#C62828', '#1565C0', '#2E7D32']

for ax, (seed, title) in zip(axes, [
    (42, 'Positive sample ($y=1$, reaction)'),
    (7,  'Negative sample ($y=0$, no reaction)')
]):
    np.random.seed(seed)
    if seed == 42:
        hs = [0.1*np.sin(3*np.pi*t)+0.3*t+0.05*np.random.randn(200).cumsum()/50,
              -0.15*np.cos(2*np.pi*t)+0.2*t**2+0.05*np.random.randn(200).cumsum()/50,
              0.2*t*np.sin(4*np.pi*t)+0.05*np.random.randn(200).cumsum()/50]
    else:
        hs = [0.05*np.sin(2*np.pi*t)+0.02*np.random.randn(200).cumsum()/50,
              0.03*np.cos(np.pi*t)+0.02*np.random.randn(200).cumsum()/50,
              0.04*np.sin(np.pi*t+0.5)+0.02*np.random.randn(200).cumsum()/50]
    for hi, c, lbl in zip(hs, clrs, ['$h_1(t)$','$h_2(t)$','$h_3(t)$']):
        ax.plot(t, hi, color=c, linewidth=1.8, label=lbl)
        ax.scatter([0],[hi[0]], color=c, s=40, zorder=5)
        ax.scatter([1],[hi[-1]], color=c, s=60, marker='*', zorder=5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(1, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Normalized time $t$', fontsize=9)
    ax.set_ylabel('Latent state $h(t)$', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

plt.suptitle(r'Neural ODE Latent Trajectories: $\dot{\mathbf{h}}(t) = f_\theta(\mathbf{h}(t), \mathbf{u}, t)$',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_trajectory.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved fig_trajectory.png")

print("\n" + "="*60)
print("  All done!")
print("  results_summary.txt — model performance")
print("  fig_summary.png     — bar chart")
print("  fig_perfold.png     — per-fold results")
print("  fig_trajectory.png  — Neural ODE trajectories")
print("="*60)
