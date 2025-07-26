# --- add this line ---
$envDir = "$PSScriptRoot\.venv"   # or any path you prefer
# ---------------------

# 0. Elevate execution policy for *this* process only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

# 1. Prefer a specific interpreter
$py = (Get-Command python3 -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $py) { Write-Error "Python not found"; exit 1 }

# 2. Create venv with that interpreter
if (!(Test-Path $envDir)) {
    Write-Host "Creating virtual environment..."
    & $py -m venv $envDir
}

# 3. Activate with error check
try {
    & "$envDir\Scripts\Activate.ps1"
    if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
        throw "venv activation failed or pip missing"
    }
} catch {
    Write-Error $_
    exit 1
}

# 4. Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 5. Embed / ingest
python data/scripts/embed_and_upsert_qdrant.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 6. Start services in *this* window (or use Start-Job for background)
# Start-Job -Name backend  -ScriptBlock { uvicorn rag.retrieval:app --host 0.0.0.0 --port 8000 }
# Start-Job -Name frontend -ScriptBlock { streamlit run app.py --server.port 8501 }

# Write-Host "Backend and frontend started. Run 'Get-Job' to see status, 'Stop-Job -Name backend|frontend' to stop."