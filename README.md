# DSAI_305_XAI

Explainable AI project for the DSAI 305 course.

---

## Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package and project manager

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/DEVOLOPER-1/DSAI_305_XAI.git
cd DSAI_305_XAI
```

### 2. Create the virtual environment
```bash
uv venv
```
This will create a `.venv` directory using your system's Python 3.12 interpreter.

### 3. Activate the environment

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
uv sync
```
This reads `pyproject.toml` and installs all required packages into your virtual environment.

### 5. Lock dependencies *(first-time or after adding packages)*
```bash
uv lock
```
Regenerates `uv.lock` to pin exact versions. Commit this file — it ensures every collaborator runs the same environment.

---

## Daily Workflow

```bash
# Pull latest changes
git pull

# Re-sync your environment after any dependency changes
uv sync

# Run your script
python your_script.py
```
```bash
ruff check .           # see all lint violations
ruff check --fix .     # auto-fix everything fixable
ruff format .          # format all files
```

---

## Adding a New Dependency

```bash
uv add <package-name>
uv lock
```

Then commit both `pyproject.toml` and `uv.lock`.

---

## Useful `uv` Commands

| Command | Description |
|---|---|
| `uv venv` | Create a virtual environment |
| `uv sync` | Install / update all dependencies from lock file |
| `uv lock` | Regenerate the lock file |
| `uv add <pkg>` | Add a new dependency |
| `uv remove <pkg>` | Remove a dependency |
| `uv run <script>` | Run a script inside the environment without activating |
| `uv help` | Full documentation |

---

## Project Structure

```
DSAI_305_XAI/
├── main.py                          # Entry point
├── scrapping/
│   └── cbio_porta_downloader.py     # cBioPortal Playwright scraper & downloader
├── downloads_valid/                 # Validated .tar.gz datasets (gitignored)
│   ├── acc_tcga.tar.gz
│   ├── acc_tcga_gdc.tar.gz
│   └── ...
├── downloads_tmp/                   # In-progress browser downloads (gitignored)
├── downloads_bad/                   # Failed / invalid downloads for inspection (gitignored)
├── logs/                            # Timestamped run logs (gitignored)
│   └── downloader_YYYYMMDD_HHMMSS.log
├── pyproject.toml                   # Project metadata and dependencies
├── uv.lock                          # Pinned dependency versions (committed)
└── README.md
```

### What gets committed

| Path | Committed |
|---|---|
| `main.py`, `scrapping/` | ✅ Yes |
| `pyproject.toml`, `uv.lock` | ✅ Yes |
| `README.md` | ✅ Yes |
| `.venv/` | ❌ No |
| `downloads_*/`, `logs/` | ❌ No — generated at runtime |

---

## Notes

- Never commit `.venv/` — it is listed in `.gitignore`
- Never commit `downloads_*/` or `logs/` — they are generated at runtime and can be large
- Always commit `uv.lock` after adding or updating dependencies
- If your environment gets into a broken state, delete `.venv/` and re-run `uv venv && uv sync`
