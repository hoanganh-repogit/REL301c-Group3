# Snake DQN web demo

The demo serves the frozen `checkpoint_best.pt` policy through FastAPI and animates its episode trace in the browser. It uses CPU inference, so it can deploy to a standard container host.

## Local run

On Windows with Python 3.11 installed, clone the repository and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\web_demo\run_local.ps1
```

`-ExecutionPolicy Bypass` applies only to this PowerShell process; it does not
change the execution policy configured on the machine.

The script creates an isolated `.venv-web-demo`, installs only the web-demo
dependencies, and starts the server. Python 3.11 must be installed first. To
install it with WinGet:

```powershell
winget install --id Python.Python.3.11 -e
```

Open a new PowerShell window after installation. If Python 3.11 is installed in
Conda but not via the Windows Python launcher, set `PYTHON` to its executable
first. For example: `$env:PYTHON = 'C:\\miniconda3\\envs\\myenvs\\python.exe'`.
Or, from the repository root, run:

```powershell
python -m pip install -r web_demo/requirements.txt
python -m uvicorn web_demo.app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`. Set `MODEL_CHECKPOINT` before startup to use another compact model checkpoint.

## Docker

Build from the repository root, then open `http://127.0.0.1:8000`:

```powershell
docker build -f web_demo/Dockerfile -t snake-dqn-demo .
docker run --rm -p 8000:8000 snake-dqn-demo
```

For a container host, use the Dockerfile, expose port `8000`, and set a health check at `/api/health`. The root `.dockerignore` packages only `checkpoint_best.pt` (about 1.2 MB), not `checkpoint_latest.pt`, which contains the replay buffer.

`checkpoint_best.pt` is tracked in Git, so a fresh clone has the model required
by the demo. The same checkpoint is used by the report and by
`scripts/play_final_dqn.py`. The Docker image excludes the other results.
