$ErrorActionPreference = "Stop"
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
Write-Host "Setup complete. Run .\run.ps1"
