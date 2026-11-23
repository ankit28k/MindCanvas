import os, json, shutil, ssl, urllib.request, urllib.parse

os.makedirs("test_images", exist_ok=True)
ssl_ctx = ssl._create_unverified_context()  # <-- [ADDED] Prevents macOS Python SSL errors

with open("eeg_demo_slices.json", "r") as f:
    demo = json.load(f)

# <-- [CHANGED] Read exact {idx: concept_name} pairs directly from eeg_demo_slices.json
concept_map = demo["concept_names"]
LOCAL_DATA_DIR = os.path.abspath(os.path.join("..", "data"))  # <-- [CHANGED] Auto-scan neuraldream/data

def find_local_test_image(cname):
    if not os.path.exists(LOCAL_DATA_DIR):
        return None
    targets = {cname.lower(), cname.lower().replace("_", " "), cname.lower().replace(" ", "_")}
    for root, _, files in os.walk(LOCAL_DATA_DIR):
        folder_name = os.path.basename(root).lower()
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            stem = os.path.splitext(fn)[0].lower()
            if any(t in folder_name or t in stem for t in targets):
                return os.path.join(root, fn)
    return None

def download_fallback(cname, save_path):
    q = urllib.parse.quote(cname.replace("_", " "))
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=filetype:bitmap%20{q}&gsrlimit=1&prop=imageinfo&iiprop=url&iiurlwidth=400&format=json"
    req = urllib.request.Request(api_url, headers={"User-Agent": "MindCanvasResearch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=12, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
            for p in data.get("query", {}).get("pages", {}).values():
                img_url = p.get("imageinfo", [{}])[0].get("thumburl") or p.get("imageinfo", [{}])[0].get("url")
                if img_url:
                    img_req = urllib.request.Request(img_url, headers={"User-Agent": "MindCanvasResearch/1.0"})
                    with urllib.request.urlopen(img_req, timeout=12, context=ssl_ctx) as r:
                        with open(save_path, "wb") as out:
                            out.write(r.read())
                    return True
    except Exception as e:
        print(f"  Fallback failed for {cname}: {e}")
    return False

print(f"Building pairs for {len(concept_map)} EEG slices (scanning {LOCAL_DATA_DIR})...")
for idx_str, cname in sorted(concept_map.items(), key=lambda kv: int(kv[0])):
    cname = str(cname)
    out_file = os.path.join("test_images", f"{cname}.jpg")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"[CACHED] Slice #{int(idx_str)+1:03d} ({cname})")
        continue

    local_hit = find_local_test_image(cname)
    if local_hit:
        shutil.copy(local_hit, out_file)
        print(f"[LOCAL TEST IMG] Slice #{int(idx_str)+1:03d} ({cname}) <- {local_hit}")
    else:
        ok = download_fallback(cname, out_file)
        print(f"[{'DOWNLOADED' if ok else 'MISSING'}] Slice #{int(idx_str)+1:03d} ({cname})")

print("Total images in test_images/:", len(os.listdir("test_images")))
