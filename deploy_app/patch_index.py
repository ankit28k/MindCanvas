with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# CHANGE 1: Fix raw LaTeX string under Gaussian Noise slider
html = html.replace(
    r"($\mathcal{N}(0,\sigma^2)$)",
    "(N(0, σ<sup>2</sup>))"
)

# CHANGE 2: Fix dropdown population to read Object.keys(data.demo.concept_names) sorted numerically
old_init = """        const indices = data.demo.demo_indices;
        const names = data.demo.concept_names;
        sel.innerHTML = indices.map(idx => {
          const label = names[idx] || `concept_${idx}`;
          return `<option value="${idx}">Slice #${String(idx + 1).padStart(3, "0")} — ${label} (100t x 17ch)</option>`;
        }).join("");"""

new_init = """        const names = data.demo.concept_names || {};
        const indices = Array.isArray(data.demo.demo_indices)
          ? data.demo.demo_indices
          : Object.keys(names).sort((a, b) => Number(a) - Number(b));
        sel.innerHTML = indices.map(idx => {
          const label = names[String(idx)] || `concept_${idx}`;
          return `<option value="${idx}">Slice #${String(Number(idx) + 1).padStart(3, "0")} — ${label} (100t x 17ch)</option>`;
        }).join("");"""

assert old_init in html, "Could not find old_init block in index.html"
html = html.replace(old_init, new_init)

# CHANGE 3: Prevent broken-image icons before load & surface exact error text if fetch fails
html = html.replace(
    '<img id="gtImg" alt="Paired Test Set Stimulus" />',
    '<img id="gtImg" alt="" style="opacity:0;transition:opacity 0.2s" onload="this.style.opacity=1" />'
)
html = html.replace(
    '<img id="genImg" alt="EEG Decoded Reconstruction" />',
    '<img id="genImg" alt="" style="opacity:0;transition:opacity 0.2s" onload="this.style.opacity=1" />'
)
html = html.replace(
    'statusTag.textContent = "BACKEND OFFLINE (OPEN VIA HTTP:// OR HTTPS://, NOT FILE://)";',
    'statusTag.textContent = "ERROR: " + err.message;'
)
html = html.replace(
    'document.getElementById("statusTag").textContent = "RUN VIA UVICORN SERVER (NOT FILE://)";',
    'document.getElementById("statusTag").textContent = "INIT ERROR: " + e.message;'
)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("index.html patched cleanly!")
