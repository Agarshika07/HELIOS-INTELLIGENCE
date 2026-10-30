$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "Virtual environment not found. Run .\setup.ps1 first." }
& $py -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Dashboard was not started." }
Start-Process "http://127.0.0.1:8000"
& $py -m helios.server
