import cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE      = Path('/scratch/Workspace2/xwang434/IsaacLab/lerobot/Sciml')
VIDEO_DIR = BASE / 'videos'
FEAT_DIR  = BASE / 'features_cv'
FEAT_DIR.mkdir(exist_ok=True)

WINDOW_SEC    = 3.0
FPS_DEFAULT   = 30.0
SAMPLE_EVERY  = 5
WINDOW_FRAMES = int(WINDOW_SEC * FPS_DEFAULT / SAMPLE_EVERY)  # 18
D_VISUAL      = 3

def parse_time(t):
    try:
        p = str(t).strip().split(':')
        return int(p[0])*60+float(p[1])
    except:
        return None

df = pd.read_csv(BASE/'ELLA_Error_Lables.csv')
df['label']      = df['Child Reaction Verbal'].isin(['Verbal','Non-verbal']).astype(int)
df['onset_sec']  = df['Error Onset'].apply(parse_time)
df['Video Name'] = df['Video Name'].str.strip()
df = df[df['Child Visible'] != 'Child not visible at all']
df = df.dropna(subset=['onset_sec']).reset_index(drop=True)
server_videos = set(v.stem for v in VIDEO_DIR.glob('*.mp4'))
df = df[df['Video Name'].isin(server_videos)].reset_index(drop=True)
pos = df['label'].sum()
print(f'Events: {len(df)} | Positive: {pos}')

unique_videos = df['Video Name'].unique()
video_cache = {}

for vid_name in tqdm(unique_videos, desc='Videos'):
    vid_path   = VIDEO_DIR / f'{vid_name}.mp4'
    cache_path = FEAT_DIR / f'_cv_{vid_name}.npy'
    if cache_path.exists():
        video_cache[vid_name] = np.load(cache_path, allow_pickle=True).item()
        continue
    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = FPS_DEFAULT
    prev_gray = None; frame_feats = {}; frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % SAMPLE_EVERY == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                frame_feats[frame_count/fps] = np.array([mag.mean(), mag.std(), mag.max()], dtype=np.float32)
            prev_gray = gray
        frame_count += 1
    cap.release()
    np.save(cache_path, frame_feats)
    video_cache[vid_name] = frame_feats

manifest_rows = []
for idx, row in df.iterrows():
    vid_name  = row['Video Name']
    onset_sec = row['onset_sec']
    if vid_name not in video_cache: continue
    frame_feats = video_cache[vid_name]
    t_start = max(0.0, onset_sec - WINDOW_SEC)
    window_frames = [v for t,v in sorted(frame_feats.items()) if t_start <= t < onset_sec]
    if len(window_frames) == 0: continue
    feat_arr = np.array(window_frames, dtype=np.float32)
    n = feat_arr.shape[0]
    if n >= WINDOW_FRAMES:
        feat_arr = feat_arr[-WINDOW_FRAMES:]
    else:
        pad = np.zeros((WINDOW_FRAMES-n, D_VISUAL), dtype=np.float32)
        feat_arr = np.vstack([pad, feat_arr])
    npy_path = FEAT_DIR / f'event_{idx:04d}.npy'
    np.save(npy_path, feat_arr)
    manifest_rows.append({'event_idx':idx,'child':row['Child Name'],'video':vid_name,
        'onset_sec':onset_sec,'error_cat':str(row['Reason for error/misalignment(summary)']).strip(),
        'error_dur':float(row['Error Offset'].split(':')[0])*60+float(row['Error Offset'].split(':')[1])-onset_sec if pd.notna(row['Error Offset']) else 0,
        'adult_present':1 if pd.notna(row.get('Others Present ','')) else 0,
        'story_num':row['Story Number'],'label':int(row['label']),
        'npy_path':str(npy_path)})

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(BASE/'manifest_cv.csv', index=False)
print(f'Done: {len(manifest)} events, shape ({WINDOW_FRAMES},{D_VISUAL})')