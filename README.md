# MindCanvas — Zero-Shot EEG-to-CLIP Visual Reconstruction Workbench

**Live Demo:** [https://3.25.239.118.nip.io/](https://3.25.239.118.nip.io/)

MindCanvas is an end-to-end neural decoding and visualization workbench that projects **17-channel occipital/parietal EEG signals (100 Hz, 1.0 s window)** from the **THINGS-EEG2** dataset directly into a **768-D L2-normalized CLIP ViT-L/14** latent space. It performs zero-shot semantic retrieval across **200 unseen test concepts**, visualizes temporal attention over Event-Related Potentials (ERPs), and synthesizes visual reconstructions conditioned on the Top-2 decoded CLIP concepts.

---

## Key Features

- **5 Trained Sequence Encoders (Live PyTorch Inference):** Switch in real time between Temporal Transformer, GRU, BiGRU, BiLSTM, and LSTM checkpoints loaded on CPU.
- **Multi-Subject Late-Fusion:** Compare individual scalp streams (`sub-01`, `sub-02`) against Grand-Average Late-Fusion, which averages and L2-normalizes predicted 768-D CLIP embeddings across subjects.
- **Live Signal Perturbation & ERP Masking:**
  - **Gaussian Voltage Noise ($\sigma \in [0.00, 0.20]$):** Injects $\mathcal{N}(0, \sigma^2)$ noise across all 17 channels to evaluate embedding stability and SNR degradation.
  - **Temporal ERP Masking (`0 ms` to `250 ms`):** Zeroes out post-stimulus timepoints starting at `+150 ms` (targeting the N170/P300 visual recognition window) and reports real-time cosine similarity drift (`ΔCosSim`) against the clean baseline.
- **Interactive Oscilloscope & Attention Map:** Renders an 8-channel EEG waveform canvas (`-200 ms` baseline to `+800 ms` post-stimulus) alongside normalized block-level temporal attention weights ($\alpha$).
- **Paired Ground-Truth vs. FLUX.1-schnell Synthesis:** Displays the exact hold-out THINGS-EEG2 test stimulus beside a visual reconstruction generated via Cloudflare Workers AI (`@cf/black-forest-labs/flux-1-schnell`) with content-addressed MD5 disk caching (`gen_cache/`).

---

## Benchmark Results (200-Way Zero-Shot Retrieval)

Evaluated on the 200 hold-out test concepts in THINGS-EEG2 (`sub-01` and `sub-02`). All models use `BlockEmbedding` + sequence encoder + `TemporalAttention` + MLP projection head to 768-D L2-normalized CLIP space.

| Rank  | Architecture                                 | Block Size (Tokens) | Parameters | Grand-Avg Top-10 | Grand-Avg Top-1 | `sub-01` Top-10 | `sub-02` Top-10 |
| :---: | :------------------------------------------- | :-----------------: | :--------: | :--------------: | :-------------: | :-------------: | :-------------: |
| **1** | **Temporal Transformer** (4 layers, 8 heads) |   `20` (5 tokens)   | 3,544,321  |    **53.00%**    |   **14.50%**    |   **46.00%**    |     41.00%      |
| **2** | **Unidirectional GRU** (3 layers)            |   `25` (4 tokens)   | 1,590,017  |      52.00%      |     14.00%      |     45.50%      |     40.00%      |
| **3** | **Bidirectional GRU (BiGRU)** (3 layers)     |   `25` (4 tokens)   | 3,659,009  |      51.00%      |     12.50%      |     45.50%      |   **43.00%**    |
| **4** | **Bidirectional LSTM (BiLSTM)** (3 layers)   |   `25` (4 tokens)   | 4,710,657  |      47.00%      |     10.50%      |     43.50%      |     37.00%      |
| **5** | **Unidirectional LSTM** (3 layers)           |   `25` (4 tokens)   | 1,984,769  |      45.00%      |     11.50%      |     41.00%      |     39.00%      |

---

## System Architecture

```text
17-Ch EEG Input (100 × 17, 100 Hz)
  │
  ├──► Optional Live Perturbations: Gaussian Noise (σ) + Post-Stimulus Zero-Mask (+150ms..)
  ▼
BlockEmbedding (Linear + LayerNorm + GELU) ──► 4 or 5 Temporal Tokens
  ▼
Sequence Encoder (Transformer / GRU / BiGRU / BiLSTM / LSTM)
  ▼
TemporalAttention (α weights over sequence blocks)
  ▼
Projection Head + L2 Normalization ──► 768-D CLIP ViT-L/14 Embedding
  │
  ├──► Grand-Average Late-Fusion (mean of sub-01 & sub-02, re-normalized)
  ▼
Cosine Similarity vs. 200 Test CLIP GT Embeddings ──► Top-5 Zero-Shot Retrieval
  ▼
Top-2 Decoded Concepts ──► Cloudflare Workers AI (FLUX.1-schnell) + MD5 Disk Cache
```

---

## Repository Structure

```text
neuraldream/
├── MindCanvas.ipynb                 # Experimentation and model development notebook
├── NeuralDream_kaggle.ipynb         # Full training pipeline for all 5 architectures
├── NeuralDream_Summary.pdf          # Project summary report
├── extract_slices.py                # Extracts demo EEG slices for web inference
└── deploy_app/
    ├── server.py                    # FastAPI inference server, retrieval, and FLUX synthesis
    ├── index.html                   # Single-file interactive workbench UI (Canvas + Fetch API)
    ├── inference_config.json        # Ranked checkpoint metadata, block sizes, and test metrics
    ├── eeg_demo_slices.json         # 20 test EEG slices (sub-01 & sub-02, 100 × 17)
    ├── prepare_pairs.py             # Pairs demo EEG slices with hold-out THINGS-EEG2 test images
    ├── ranked_checkpoints/          # Trained PyTorch .pt weights for all 5 architectures
    ├── vectors/                     # 200-concept CLIP GT embeddings & concept labels (.npy)
    ├── test_images/                 # 20 paired hold-out stimulus images (.jpg)
    └── gen_cache/                   # Auto-created disk cache for generated FLUX reconstructions
```

---

## API Endpoints

| Method | Endpoint              | Description                                                                                                                                                                                         |
| :----- | :-------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/`                   | Serves the MindCanvas frontend workbench (`index.html`).                                                                                                                                            |
| `GET`  | `/api/init`           | Returns `inference_config.json` (with live parameter and token counts) and demo slice metadata.                                                                                                     |
| `POST` | `/api/decode`         | Runs PyTorch EEG-to-CLIP inference with optional `noise_sigma`, `mask_pts`, and `generate_image` flags; returns Top-5 retrieval, `gt_similarity`, attention weights, 8-ch waveform, and image URLs. |
| `GET`  | `/test_images/{file}` | Serves paired hold-out THINGS-EEG2 ground-truth images.                                                                                                                                             |
| `GET`  | `/gen_cache/{file}`   | Serves cached FLUX.1-schnell reconstructions.                                                                                                                                                       |

---

## Running Locally

### 1. Install Dependencies

```bash
cd deploy_app
pip install fastapi uvicorn torch numpy
```

### 2. Prepare Paired Test Images (Optional if `test_images/` is already populated)

```bash
python3 prepare_pairs.py
```

### 3. Configure Cloudflare Workers AI (for FLUX.1-schnell Synthesis)

```bash
export CF_ACCOUNT_ID="your_cloudflare_account_id"
export CF_API_TOKEN="your_cloudflare_workers_ai_token"
```

### 4. Start the FastAPI Server

```bash
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser. _(Do not open `index.html` via `file://`, as browser security blocks relative `/api/_` requests.)\*

---

## Production Deployment

Deployed on **AWS EC2 (Amazon Linux 2023)** running `uvicorn` on `127.0.0.1:8000` behind a **Caddy** reverse proxy with automatic TLS/HTTPS (`3.25.239.118.nip.io`).

---

## Tech Stack

- **ML & Signal Processing:** PyTorch, NumPy, CLIP ViT-L/14 embeddings, THINGS-EEG2
- **Backend & Synthesis:** FastAPI, Uvicorn, Pydantic, Cloudflare Workers AI (`FLUX.1-schnell`)
- **Frontend:** Vanilla HTML5, CSS3, JavaScript, HTML5 Canvas (real-time EEG oscilloscope)
- **Infrastructure:** AWS EC2, Caddy (reverse proxy & automated HTTPS)
