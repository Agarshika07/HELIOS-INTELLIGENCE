# HELIOS INTELLIGENCE V1.2

Adaptive machine-intelligence prototype for the VAULT rover. **This release runs on simulated data only.** No ESP32/MPU9250 hardware has been connected or validated.

Pipeline: simulated sensors -> validation -> window features -> Random Forest motion-state classifier + Isolation Forest anomaly detector -> rule-based risk score -> advisory decision -> explanation -> dashboard.

## Requirements
- Windows 10/11, macOS, or Linux
- Python 3.10+
- NumPy, scikit-learn, and joblib (installed by setup script)
- **Pandas is not required.** V1.2 uses NumPy and Python's standard library for sensor data handling and CSV input.

## Run on Windows
Open PowerShell in this project folder, then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\run.ps1
```

The dashboard opens at `http://127.0.0.1:8000`. Keep the PowerShell window open while using it.

## Train, evaluate, and test
```powershell
.\.venv\Scripts\python.exe -m scripts.train_model
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dashboard features: select a simulated terrain scenario, start/pause/reset, inspect sensor values and movement state, and export a JSON session.

## CSV input
A recording CSV must contain columns `t,ax,ay,az,gx,gy,gz` with numeric values. Run:

```powershell
.\.venv\Scripts\python.exe -m helios.core --csv path\to\recording.csv
```

## Important limitations
- All dashboard scenarios and generated sensor values are simulations, not measurements from the physical rover.
- Synthetic macro-F1 is not real-world accuracy and must not be presented as validated performance.
- The risk value is a rule-based score, not a failure probability. An anomaly is not proof of hardware failure.
- No ESP32/MPU9250 hardware adapter is included; see `docs/HARDWARE_INTEGRATION.md`.
- Decisions are advisory only. This software does not command rover motors.
- The frontend is plain HTML/JavaScript served by Python's standard-library HTTP server.

## GitHub
Do not commit `.venv/`, `__pycache__/`, `.pyc`, or machine-specific files. These are covered by `.gitignore`.
