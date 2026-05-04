import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from torchdiffeq import odeint_adjoint as odeint

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# CONFIG 
BASE_DIR      = "/scratch/Workspace2/xwang434/IsaacLab/lerobot/Sciml"
ANNOT_CSV     = f"{BASE_DIR}/ELLA_Error_Lables.csv"
MANIFEST_CSV  = f"{BASE_DIR}/manifest_cv.csv"
FEATURE_DIR   = f"{BASE_DIR}/features_cv"

# x(t) dimension: optical flow (magnitude, vx, vy)
D_VISUAL      = 3
# u(t): 10 error categories one-hot + 1 normalized onset time
D_U           = 11
# Neural ODE latent dimension
K_LATENT      = 32
# LSTM hidden dimension
H_LSTM        = 64
# Window frames (WINDOW_SEC=3.0, FPS=30, SKIP=14 → 18 frames)
WINDOW_FRAMES = 18

EPOCHS   = 100
PATIENCE = 15       # patience on validation F1
LR_RATE  = 1e-3
BATCH    = 16
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# ERROR CATEGORY ENCODING 
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

def encode_u(error_cat_raw, onset_sec, max_onset=3600.0):
    cat = CAT_MAP.get(str(error_cat_raw).strip(), 'Wrong_scaffold')
    e = np.zeros(10, dtype=np.float32)
    if cat in ERROR_CATEGORIES:
        e[ERROR_CATEGORIES.index(cat)] = 1.0
    tau = np.array([onset_sec / max_onset], dtype=np.float32)
    return np.concatenate([e, tau])

# DATA LOADING 
def parse_time(t):
    try:
        p = str(t).strip().split(':')
        return int(p[0]) * 60 + float(p[1])
    except:
        return None

annot = pd.read_csv(ANNOT_CSV)
annot['label']     = annot['Child Reaction Verbal'].isin(['Verbal','Non-verbal']).astype(int)
annot['onset_sec'] = annot['Error Onset'].apply(parse_time)
annot['Video Name'] = annot['Video Name'].str.strip()
annot = annot[annot['Child Visible'] != 'Child not visible at all']
annot = annot.dropna(subset=['onset_sec']).reset_index(drop=True)

MAX_ONSET = annot['onset_sec'].max()

manifest = pd.read_csv(MANIFEST_CSV).set_index('event_idx')
print(f"Manifest: {len(manifest)} events")

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

print(f"Samples: {len(samples)} | "
      f"Positive: {sum(s['y'] for s in samples)} "
      f"({sum(s['y'] for s in samples)/len(samples):.1%})")

children        = np.array([s['child'] for s in samples])
unique_children = sorted(set(children))
print(f"Children ({len(unique_children)}): {unique_children}")

# DATASET 
class ReactionDataset(Dataset):
    def __init__(self, sample_list, scaler=None, fit_scaler=False):
        xs = np.stack([s['x'] for s in sample_list])
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

# MODELS
class LSTMModel(nn.Module):
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

# TRAINING UTILITIES 
def compute_pos_weight(y_train):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

def val_f1(model, val_ds, device):
    """Compute macro F1 on validation set."""
    model.eval()
    with torch.no_grad():
        prob = model(val_ds.x.to(device), val_ds.u.to(device)).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    y    = val_ds.y.numpy().astype(int)
    return f1_score(y, pred, average='macro', zero_division=0)

def train_nn(model, train_loader, val_ds, pos_weight, device, epochs, patience):
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

        # Early stopping on validation macro F1 
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
        prob = model(x_t.to(device), u_t.to(device)).cpu().numpy()
    return prob

def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
    bal = balanced_accuracy_score(y_true, y_pred)
    return f1, auc, bal

# LOSO CROSS-VALIDATION (8 train / 1 val / 1 test)
all_results = []

for fold, test_child in enumerate(unique_children):
    print(f"\n{'='*55}")
    print(f"  Fold {fold+1}/{len(unique_children)} — Test: {test_child}")

    # Remaining children after removing test child
    remaining = [c for c in unique_children if c != test_child]
    # Use the last remaining child as validation (fixed, deterministic)
    val_child = remaining[-1]

    tr_samp  = [s for s in samples if s['child'] not in (test_child, val_child)]
    val_samp = [s for s in samples if s['child'] == val_child]
    te_samp  = [s for s in samples if s['child'] == test_child]

    print(f"  Train: {len(tr_samp)} | Val: {len(val_samp)} ({val_child}) | Test: {len(te_samp)}")

    if len(te_samp) == 0 or len(val_samp) == 0:
        print("  Skipping fold (empty split).")
        continue

    scaler   = StandardScaler()
    tr_ds    = ReactionDataset(tr_samp,  scaler=scaler, fit_scaler=True)
    val_ds   = ReactionDataset(val_samp, scaler=scaler, fit_scaler=False)
    te_ds    = ReactionDataset(te_samp,  scaler=scaler, fit_scaler=False)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)

    y_tr = tr_ds.y.numpy()
    y_te = te_ds.y.numpy().astype(int)
    pw   = compute_pos_weight(y_tr)

    n_pos = int(y_te.sum()); n_neg = int((y_te == 0).sum())
    print(f"  Test pos={n_pos}, neg={n_neg}")
    print(f"{'='*55}")

    # LR 
    # LR: pick C on val set (small grid)
    best_c, best_val_f1 = 0.1, -1
    for c in [0.001, 0.01, 0.1, 1.0]:
        lr_tmp = LogisticRegression(C=c, class_weight='balanced',
                                    max_iter=1000, random_state=42)
        X_tr = np.concatenate([tr_ds.x.numpy().reshape(len(tr_samp), -1),
                                tr_ds.u.numpy()], axis=1)
        X_val = np.concatenate([val_ds.x.numpy().reshape(len(val_samp), -1),
                                 val_ds.u.numpy()], axis=1)
        lr_tmp.fit(X_tr, y_tr)
        val_prob = lr_tmp.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        vf1 = f1_score(val_ds.y.numpy().astype(int), val_pred,
                       average='macro', zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1, best_c = vf1, c

    X_tr  = np.concatenate([tr_ds.x.numpy().reshape(len(tr_samp), -1),
                             tr_ds.u.numpy()], axis=1)
    X_te  = np.concatenate([te_ds.x.numpy().reshape(len(te_samp), -1),
                             te_ds.u.numpy()], axis=1)
    lr    = LogisticRegression(C=best_c, class_weight='balanced',
                               max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_te)[:, 1]
    f1, auc, bal = metrics(y_te, lr_prob)
    all_results.append({'model':'LR', 'child':test_child,
                        'f1':f1, 'auc':auc, 'bal_acc':bal})
    print(f"  LR (C={best_c})  | F1={f1:.3f}  AUC={auc:.3f}  BalAcc={bal:.3f}")

    # LSTM 
    lstm      = LSTMModel()
    lstm      = train_nn(lstm, tr_loader, val_ds, pw, DEVICE, EPOCHS, PATIENCE)
    lstm_prob = eval_nn(lstm, te_ds.x, te_ds.u, DEVICE)
    f1, auc, bal = metrics(y_te, lstm_prob)
    all_results.append({'model':'LSTM', 'child':test_child,
                        'f1':f1, 'auc':auc, 'bal_acc':bal})
    print(f"  LSTM            | F1={f1:.3f}  AUC={auc:.3f}  BalAcc={bal:.3f}")

    # Neural ODE
    ode_model = NeuralODEModel()
    ode_model = train_nn(ode_model, tr_loader, val_ds, pw, DEVICE, EPOCHS, PATIENCE)
    ode_prob  = eval_nn(ode_model, te_ds.x, te_ds.u, DEVICE)
    f1, auc, bal = metrics(y_te, ode_prob)
    all_results.append({'model':'NeuralODE', 'child':test_child,
                        'f1':f1, 'auc':auc, 'bal_acc':bal})
    print(f"  Neural ODE      | F1={f1:.3f}  AUC={auc:.3f}  BalAcc={bal:.3f}")

# SUMMARY
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
    f.write("LOSO Results — optical flow features (d=3), 8 train / 1 val / 1 test\n")
    f.write("Early stopping on validation macro F1 (patience=15)\n")
    f.write("=" * 60 + "\n")
    f.write('\n'.join(lines) + '\n')

print("\nSaved: results_loso.csv, results_summary.txt")