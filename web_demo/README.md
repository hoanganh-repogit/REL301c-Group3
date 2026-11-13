# Snake DQN web demo

The demo serves the frozen `checkpoint_best.pt` policy through FastAPI and animates its episode trace in the browser at three steps every 100 ms. Each level uses the deterministic seed with the highest score in the cross-level evaluation set; the default Level 4 uses seed `1275`. The seed is selected automatically and is not exposed as a demo control. It uses CPU inference, so it can deploy to a standard container host.

`checkpoint_best.pt` is tracked in Git, so a fresh clone has the model required
by the demo. The same checkpoint is used by the report and by
`scripts/play_final_dqn.py`.

## Run on Windows (step by step)

### 1. Install Python 3.11 (once)

Check whether it is already installed:

```powershell
py -0p
```

If no line starts with `-V:3.11`, install it, then **close and reopen
PowerShell**:

```powershell
winget install --id Python.Python.3.11 -e
```

If Python 3.11 lives only in Conda, point the script at it instead, for example
`$env:PYTHON = 'C:\Users\<you>\miniconda3\envs\<env>\python.exe'`.

### 2. Make sure port 8000 is free

The demo listens on port 8000. If an old demo server or a Docker container
already uses that port, **the browser opens the old program and you will not
see the current demo**. Windows does not report an error in this case.

```powershell
docker ps                                                # is a snake-dqn-demo container running?
docker stop <CONTAINER ID>                               # if yes, stop it
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
```

The last command must print nothing. If it still lists a process, close the
PowerShell window of the old server (or press `Ctrl+C` in it). The script in the
next step also checks this and stops with a message naming the program that
holds the port. To avoid the conflict entirely, use another port, e.g.
`-Port 8001`.

### 3. Start the server

Open PowerShell **in the repository root** (the folder that contains
`web_demo`), then run:

```powershell
cd C:\path\to\REL-SnakeAI
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\web_demo\run_local.ps1
```

Use another port with `... run_local.ps1 -Port 8001`.

- `-ExecutionPolicy Bypass` applies only to this PowerShell process; it does not
  change the execution policy configured on the machine.
- The first run creates `.venv-web-demo` and downloads PyTorch, which can take
  several minutes. Later runs start in a few seconds.
- The server is ready when the window shows:

  ```text
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```

- **Keep this window open.** Closing it or pressing `Ctrl+C` stops the demo.

### 4. Open the demo in a browser

The script does not open a browser. Open this address yourself (use the port
you chose):

```text
http://127.0.0.1:8000
```

The board is **empty until you start an episode**. Choose a level and click
**Run episode**; the snake starts moving once the server has sent the episode.
Use **Pause** to pause or resume.

Right after startup the server computes the episodes of all six levels in the
background (Level 4 first). On a laptop this takes about 10–20 seconds for
Level 4 and under a minute for all levels. If you click **Run episode** before a
level is ready, the message under the buttons counts the seconds
(`Running the policy on the server… 7 s`) until it is; just wait. After that,
every click plays immediately until you stop the server.

You are on the current version when the page has **no Seed field**, only the
Level menu and the two buttons. If you still see a Seed field, the browser
reached an old server or an old Docker image: repeat step 2, then press
`Ctrl+F5` on the page.

Expected results (the same on every run, because the seeds are fixed):

| Level | Final score | Outcome | Playback length |
| --- | ---: | --- | --- |
| 1 | 25 | target reached | ~8 s |
| 2 | 35 | target reached | ~14 s |
| 3 | 50 | target reached | ~25 s |
| 4 (default) | 185 | hits a wall | ~4 min |
| 5 | 100 | target reached | ~1.5 min |
| 6 | 150 | target reached | ~3 min |

## Run without the script

From the repository root, in any Python 3.11 environment:

```powershell
python -m pip install -r web_demo/requirements.txt
python -m uvicorn web_demo.app:app --host 127.0.0.1 --port 8000
```

Then follow step 4. Set `MODEL_CHECKPOINT` before startup to use another compact
model checkpoint.

## Docker

A Docker image contains the code from the moment it was built. **After every
`git pull`, rebuild it**, otherwise the container keeps serving the old demo.
From the repository root:

```powershell
docker build -f web_demo/Dockerfile -t snake-dqn-demo .
docker run --rm -p 8000:8000 snake-dqn-demo
```

Then open `http://127.0.0.1:8000`. Stop the container with `Ctrl+C` (or
`docker stop <CONTAINER ID>`) before running the demo with the script, because
both use port 8000.

For a container host, use the Dockerfile, expose port `8000`, and set a health
check at `/api/health`. The root `.dockerignore` packages only
`checkpoint_best.pt` (about 1.2 MB), not `checkpoint_latest.pt`, which contains
the replay buffer, and excludes the other results.

## Troubleshooting

| What you see | Cause | Fix |
| --- | --- | --- |
| Script stops with `Port 8000 is already in use by: ...` | Another program uses the port | Stop it (step 2) or add `-Port 8001` |
| Page shows a **Seed** field / old layout | Browser reached an old Docker container or server | Step 2, then `Ctrl+F5`; rebuild the Docker image after pulling |
| Browser says "This site can't be reached" | Server not running or still installing | Wait for `Application startup complete`; keep the window open |
| Page opens but the board is empty | No episode started yet | Click **Run episode** |
| `Running the policy on the server…` stays for a while | The level is still being computed after startup | Wait: the seconds counter shows it is working (about 10–20 s for Level 4 on a laptop) |
| `Running the policy on the server…` without a seconds counter | Browser uses an old cached `app.js` | Press `Ctrl+F5` |
| `Python 3.11 was not found` | Python 3.11 missing from the `py` launcher | Step 1 |
| `Error: ...` under the buttons | Server error | Read the traceback in the server window; check `http://127.0.0.1:8000/api/health` |
