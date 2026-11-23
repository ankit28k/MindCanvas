import os, re, json, math, base64, hashlib, urllib.parse, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# <-- [ADDED] Mount local test_images/ directory so paired test stimulus images load directly
os.makedirs("test_images", exist_ok=True)
app.mount("/test_images", StaticFiles(directory="test_images"), name="test_images")

os.makedirs("gen_cache", exist_ok=True)
app.mount("/gen_cache", StaticFiles(directory="gen_cache"), name="gen_cache")

N_CHANNELS, N_TIMEPOINTS, CLIP_DIM = 17, 100, 768

class BlockEmbedding(nn.Module):
    def __init__(self, n_channels, block_size, d_model):
        super().__init__()
        self.block_size = block_size
        self.proj = nn.Sequential(nn.Linear(n_channels * block_size, d_model), nn.LayerNorm(d_model), nn.GELU())
    def forward(self, x):
        B, T, C = x.shape
        n_blocks = T // self.block_size
        x = x[:, :n_blocks * self.block_size, :].reshape(B, n_blocks, C * self.block_size)
        return self.proj(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, seq):
        w = F.softmax(self.attn(seq).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * seq).sum(dim=1), w

class EEGtoClipLSTM(nn.Module):
    def __init__(self, n_channels, block_size, hidden_dim=256, num_layers=3, clip_dim=768, dropout=0.3):
        super().__init__()
        self.embed = BlockEmbedding(n_channels, block_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attn = TemporalAttention(hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, clip_dim))
    def forward(self, x):
        x, _ = self.lstm(self.embed(x))
        ctx, w = self.attn(x)
        return F.normalize(self.proj(ctx), dim=-1), w

class EEGtoClipBiLSTM(nn.Module):
    def __init__(self, n_channels, block_size, hidden_dim=256, num_layers=3, clip_dim=768, dropout=0.3):
        super().__init__()
        self.embed = BlockEmbedding(n_channels, block_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = TemporalAttention(hidden_dim * 2)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, clip_dim))
    def forward(self, x):
        x, _ = self.lstm(self.embed(x))
        ctx, w = self.attn(x)
        return F.normalize(self.proj(ctx), dim=-1), w

class EEGtoClipGRU(nn.Module):
    def __init__(self, n_channels, block_size, hidden_dim=256, num_layers=3, clip_dim=768, dropout=0.3):
        super().__init__()
        self.embed = BlockEmbedding(n_channels, block_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attn = TemporalAttention(hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, clip_dim))
    def forward(self, x):
        x, _ = self.gru(self.embed(x))
        ctx, w = self.attn(x)
        return F.normalize(self.proj(ctx), dim=-1), w

class EEGtoClipBiGRU(nn.Module):
    def __init__(self, n_channels, block_size, hidden_dim=256, num_layers=3, clip_dim=768, dropout=0.3):
        super().__init__()
        self.embed = BlockEmbedding(n_channels, block_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = TemporalAttention(hidden_dim * 2)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, clip_dim))
    def forward(self, x):
        x, _ = self.gru(self.embed(x))
        ctx, w = self.attn(x)
        return F.normalize(self.proj(ctx), dim=-1), w

class EEGtoClipTransformer(nn.Module):
    def __init__(self, n_channels, block_size, d_model=256, n_heads=8, n_layers=4, clip_dim=768, dropout=0.3):
        super().__init__()
        n_blocks = N_TIMEPOINTS // block_size
        self.embed = BlockEmbedding(n_channels, block_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_blocks, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn = TemporalAttention(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, clip_dim))
    def forward(self, x):
        x = self.encoder(self.embed(x) + self.pos)
        ctx, w = self.attn(x)
        return F.normalize(self.proj(ctx), dim=-1), w

with open("inference_config.json") as f:
    INF_CONFIG = json.load(f)
with open("eeg_demo_slices.json") as f:
    DEMO_DATA = json.load(f)

GT_CLIP = np.load("vectors/test_clip_gt_embeddings.npy")
ALL_CONCEPTS = np.load("vectors/test_concept_names.npy")

ARCH_MAP = {
    "Transformer": (EEGtoClipTransformer, {"d_model": 256, "n_heads": 8, "n_layers": 4}),
    "GRU":         (EEGtoClipGRU,         {"hidden_dim": 256, "num_layers": 3}),
    "BiGRU":       (EEGtoClipBiGRU,       {"hidden_dim": 256, "num_layers": 3}),
    "BiLSTM":      (EEGtoClipBiLSTM,      {"hidden_dim": 256, "num_layers": 3}),
    "LSTM":        (EEGtoClipLSTM,        {"hidden_dim": 256, "num_layers": 3}),
}

MODELS = {}
for r_key, info in INF_CONFIG.items():
    m_name = info["model_name"]
    cls, kw = ARCH_MAP[m_name]
    model = cls(N_CHANNELS, info["block_size"], **kw, clip_dim=CLIP_DIM)
    model.load_state_dict(torch.load(f"ranked_checkpoints/{info['checkpoint_file']}", map_location="cpu", weights_only=True))
    model.eval()
    MODELS[m_name] = model
    info["params"] = int(sum(p.numel() for p in model.parameters()))  # <-- [ADDED] Exact parameter count from .pt weights
    info["n_tokens"] = int(N_TIMEPOINTS // info["block_size"])        # <-- [ADDED] Exact token count (5 or 4)

class DecodeReq(BaseModel):
    subject: str = "Grand_Avg"
    concept_idx: Union[str, int] = "0"
    model_name: str = "Transformer"
    noise_sigma: float = 0.0
    mask_pts: int = 0
    generate_image: bool = True  # [ADDED] Lets slider drags update EEG/metrics without spamming image API

# [ADDED] Strip numeric dataset prefix like "00171_spoon" -> "spoon"
def clean_label(raw: str) -> str:
    return re.sub(r"^\d+[\s_]+", "", str(raw)).replace("_", " ").strip()

# [ADDED] Generate via Cloudflare Workers AI (FLUX.1-schnell) and cache to gen_cache/
def synthesize_image(prompt_text: str) -> str:
    h = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:12]
    out_path = os.path.join("gen_cache", f"{h}.jpg")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return f"/gen_cache/{h}.jpg"

    cf_acc = os.environ.get("CF_ACCOUNT_ID", "").strip()
    cf_tok = os.environ.get("CF_API_TOKEN", "").strip()
    if cf_acc and cf_tok:
        url = f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        payload = json.dumps({"prompt": prompt_text, "steps": 4}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {cf_tok}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                img_b64 = data.get("result", {}).get("image")
                if img_b64:
                    with open(out_path, "wb") as f:
                        f.write(base64.b64decode(img_b64))
                    return f"/gen_cache/{h}.jpg"
        except Exception as e:
            print(f"[Cloudflare AI error] {e}")

    # Fallback if Cloudflare env vars are not set
    return f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt_text)}?width=384&height=384&nologo=true"

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/api/init")
def get_init():
    return {"config": INF_CONFIG, "demo": DEMO_DATA}

@app.post("/api/decode")
def run_decode(req: DecodeReq):
    idx_str = str(req.concept_idx)
    idx = int(req.concept_idx)
    gt_concept = str(DEMO_DATA["concept_names"][idx_str])

    model = MODELS[req.model_name]
    subs = ["sub-01", "sub-02"] if req.subject == "Grand_Avg" else [req.subject]

    preds, attns, raw_waves = [], [], []
    for s in subs:
        x = np.array(DEMO_DATA["eeg_slices"][s][idx_str], dtype=np.float32)
        if req.noise_sigma > 0:
            x = x + np.random.normal(0, req.noise_sigma, x.shape).astype(np.float32)
        if req.mask_pts > 0:
            x[35:35 + req.mask_pts, :] = 0.0
        raw_waves.append(x)
        with torch.no_grad():
            p, a = model(torch.tensor(x).unsqueeze(0))
            preds.append(p.numpy()[0])
            attns.append(a.numpy()[0])

    pred_vec = np.mean(preds, axis=0)
    pred_vec = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)
    attn_vec = np.mean(attns, axis=0)
    wave_vis = np.mean(raw_waves, axis=0)

    sims = np.dot(GT_CLIP, pred_vec)
    top5_idx = np.argsort(sims)[::-1][:5]
    top5 = [
        {
            "concept": clean_label(ALL_CONCEPTS[i]),  # [CHANGED] Clean human-readable label (no "00171")
            "sim": round(float(sims[i]), 4),
            "is_gt": int(i) == idx,
        }
        for i in top5_idx
    ]

    local_img_path = os.path.join("test_images", f"{gt_concept}.jpg")
    gt_url = f"/test_images/{gt_concept}.jpg" if os.path.exists(local_img_path) else ""

    # [CHANGED] Build clean prompt without numeric prefixes & generate on backend only when requested
    c1, c2 = top5[0]["concept"], top5[1]["concept"]
    prompt_text = f"realistic studio photograph of a {c1}, subtle visual resemblance to {c2}, natural lighting, sharp focus"
    gen_url = synthesize_image(prompt_text) if req.generate_image else None

    return {
        "gt_concept": gt_concept,
        "gt_similarity": round(float(sims[idx]), 4),
        "attention": np.round(attn_vec, 4).tolist(),
        "waveform_8ch": np.round(wave_vis[:, :8], 3).tolist(),
        "top5": top5,
        "generated_image_url": gen_url,
        "gt_image_url": gt_url,
    }
# class DecodeReq(BaseModel):
#     subject: str = "Grand_Avg"
#     concept_idx: Union[str, int] = "0"  # <-- [CHANGED] Accepts both int and str safely
#     model_name: str = "Transformer"
#     noise_sigma: float = 0.0
#     mask_pts: int = 0

# @app.get("/")
# def home():
#     return FileResponse("index.html")

# @app.get("/api/init")
# def get_init():
#     return {"config": INF_CONFIG, "demo": DEMO_DATA}

# @app.post("/api/decode")
# def run_decode(req: DecodeReq):
#     # <-- [ADDED] Normalize concept_idx as both string key and integer index
#     idx_str = str(req.concept_idx)
#     idx = int(req.concept_idx)
#     gt_concept = str(DEMO_DATA["concept_names"][idx_str])

#     model = MODELS[req.model_name]
#     subs = ["sub-01", "sub-02"] if req.subject == "Grand_Avg" else [req.subject]
    
#     preds, attns, raw_waves = [], [], []
#     for s in subs:
#         x = np.array(DEMO_DATA["eeg_slices"][s][idx_str], dtype=np.float32) # (100, 17)
#         if req.noise_sigma > 0:
#             x = x + np.random.normal(0, req.noise_sigma, x.shape).astype(np.float32)
#         if req.mask_pts > 0:
#             x[35:35 + req.mask_pts, :] = 0.0
#         raw_waves.append(x)
#         with torch.no_grad():
#             p, a = model(torch.tensor(x).unsqueeze(0))
#             preds.append(p.numpy()[0])
#             attns.append(a.numpy()[0])

#     # Late-Fusion in 768-D CLIP Space
#     pred_vec = np.mean(preds, axis=0)
#     pred_vec = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)
#     attn_vec = np.mean(attns, axis=0)
#     wave_vis = np.mean(raw_waves, axis=0)

#     sims = np.dot(GT_CLIP, pred_vec)
#     top5_idx = np.argsort(sims)[::-1][:5]
#     top5 = [
#         {
#             "concept": str(ALL_CONCEPTS[i]).replace("_", " "),
#             "sim": round(float(sims[i]), 4),
#             "is_gt": int(i) == idx
#         }
#         for i in top5_idx
#     ]

    # <-- [CHANGED] Serve local paired test image if present in test_images/, else clean fallback URL
    local_img_path = os.path.join("test_images", f"{gt_concept}.jpg")
    clean_gt = gt_concept.replace("_", " ")
    if os.path.exists(local_img_path):
        gt_url = f"/test_images/{gt_concept}.jpg"
    else:
        gt_url = f"https://image.pollinations.ai/prompt/studio%20photograph%20of%20{urllib.parse.quote(clean_gt)}%20isolated%20object?width=400&height=400&seed={idx+100}&nologo=true"

    # <-- [CHANGED] Fixed prompt_encoded variable name so gen_url builds without NameError
    c1, c2 = top5[0]["concept"], top5[1]["concept"]
    prompt_encoded = urllib.parse.quote(f"photorealistic {c1}, subtle elements of {c2}, neural dream reconstruction, cinematic lighting, high detail")
    seed = idx * 10 + len(req.model_name) + int(req.noise_sigma * 100) + (req.mask_pts * 7)
    gen_url = f"https://image.pollinations.ai/prompt/{prompt_encoded}?width=400&height=400&seed={seed}&nologo=true"

    return {
        "gt_concept": gt_concept,
        "gt_similarity": round(float(sims[idx]), 4),
        "attention": np.round(attn_vec, 4).tolist(),
        "waveform_8ch": np.round(wave_vis[:, :8], 3).tolist(),
        "top5": top5,
        "generated_image_url": gen_url,
        "gt_image_url": gt_url,
    }