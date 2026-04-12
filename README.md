# XAI · WhiteBox — TMB Prediction

This is the single source of truth for our tracker codebase. Read this document once, top to bottom, the first time you set up. After that it doubles as a reference for commands and conventions.

---

## Table of Contents

1. [Competition at a Glance](#1-competition-at-a-glance)
2. [Prerequisites](#2-prerequisites)
3. [Clone & First-Time Setup](#3-clone--first-time-setup)
4. [Project Structure](#4-project-structure)
5. [Team Standards — Read Before You Code](#5-team-standards--read-before-you-code)
6. [Create Your Member Folder](#6-create-your-member-folder)
7. [Get the Dataset](#7-get-the-dataset)
8. [Daily Experiment Workflow](#8-daily-experiment-workflow)
9. [Submitting to Kaggle](#9-submitting-to-kaggle)
10. [Updating the Team Leaderboard](#10-updating-the-team-leaderboard)
11. [Make Command Reference](#11-make-command-reference)
12. [Scoring Formula](#12-scoring-formula)
13. [Gotchas & FAQ](#13-gotchas--faq)

---
## 2. Prerequisites

You need the following installed on your machine before anything else.

### 2.1 Python 3.13

```bash
python3 --version   # must be >= 3.13
```

If not, install from [python.org](https://www.python.org/downloads/) or via your system package manager. On Ubuntu:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt install python3.13 python3.13-venv
```

### 2.2 uv — our dependency manager

`uv` replaces `pip` + `venv` with a single, dramatically faster tool. If you have never used it before, here is the mental model:

| Old way | uv equivalent |
|---------|--------------|
| `python -m venv .venv` | `uv venv` |
| `pip install -e ".[dev]"` | `uv pip install -e ".[dev]"` |
| `pip install package` | `uv pip install package` |
| `pip freeze > requirements.txt` | handled by `uv.lock` automatically |

Install it once, globally:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

> uv is 10–100× faster than pip because it resolves and installs packages in parallel using a Rust-based resolver. The `uv.lock` file in this repo pins every transitive dependency so every teammate gets the exact same environment.

### 2.3 Git

```bash
git --version   # any recent version is fine
```

### 2.4 make

```bash
make --version
```

On Windows, install via [Git for Windows](https://gitforwindows.org/) (includes `make`) or use WSL2.

---

## 3. Clone & First-Time Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/DEVOLOPER-1/DSAI_305_XAI
cd DSAI_305_XAI
```

### Step 2 — Create the virtual environment and install core dependencies

```bash
make setup
```

This runs:
```
uv venv          # creates .venv/ in the project root
uv sync          # installs all dependencies pinned in uv.lock
uv lock          # re-locks if pyproject.toml changed
```

Your terminal should show something like:
```
✓ Core dependencies installed
  Run 'make setup-dl' to also install torch/timm
```

> **Activate the environment** after setup:
> ```bash
> source .venv/bin/activate      # Linux / macOS
> .venv\Scripts\activate         # Windows
> ```
> You must activate it every time you open a new terminal in this project. Your IDE (VS Code, PyCharm) can do this automatically — point it at `.venv/`.

### Step 3 — (Optional) Install the deep-learning stack

Only needed if your approach uses PyTorch models:

```bash
make setup-dl
```

This installs `torch`, `torchvision`, `timm` (backbone zoo), `einops`, and the FLOPs profiling tools. Skip this if you are working on a classical-CV approach — it is a large download (~2 GB).

### Step 5 - Weights and Biases (wandb) login
We use W&B for centralized experiment tracking. Every run you execute appears on our team dashboard, allowing us to compare architectures and see who has the best AUC in real-time.

### 5.0.1 Individual Authentication (ONLY ONCE per member)
W&B API keys are personal. **Never** share your key or commit it to Git.
1. Create an account at [wandb.ai](https://wandb.ai/site) using your university email.
2. Generate API key from the "Settings" page.
3. In your terminal (with your `.venv` activated), run:
   ```bash
   wandb login
   ```
4. Paste your API key when prompted.

### Step 5 — Install developer tools

```bash
uv pip install -e ".[dev]"
```

This installs `pre-commit`, `jupyter` and ipykernel.

---

## 4. Project Structure

```
aic4-tracker/
│
├── pyproject.toml              ← Project metadata, all dependencies, ruff config
├── uv.lock                     ← Pinned dependency graph (committed — do not edit manually)
├── Makefile                    ← All team commands (setup, lint, submit, leaderboard…)
├── STANDARDS.md                ← Team conventions — READ THIS (see §5)
├── README.md                   ← You are here
├── main.py                     ← Top-level entry point (delegates to src/)
│
├── configs/                    ← All experiment configs live here — never hardcode hyperparameters
│   ├── _base.yaml              ← Shared defaults: dataset paths, eval protocol, logging
│   └── experiments/
│       └── siamfc_mobile.yaml  ← Example: inherits _base, overrides model + training params
│
├── src/                        ← Shared source code — discuss changes with the team before pushing
│   ├── config.py               ← Loads and validates YAML configs
│   ├── data_loader.py          ← Parses MP4 sequences + annotation.txt bounding boxes
│   ├── train.py                ← Universal training loop (W&B integrated)
│   ├── inference.py            ← Online-only tracker evaluation (no future frame access)
│   ├── utils.py                ← IoU, bbox helpers, FLOPs/latency measurement
│   ├── models/
│       ├── __init__.py         ← Model factory / registry (add new models here)
│       └── baselines.py        ← Reference implementations
│
├── data/                       ← Raw dataset (gitignored — download separately, see §7)
│   ├── train/
│   └── public_lb/
│
├── logs/
│   ├── leaderboard.csv         ← Team leaderboard (committed — update after every submission)
│   └── runs/                   ← Per-run logs (gitignored)
│
├── members/                    ← One folder per team member (generated by `make new-member`)
│   └── {your_name}/
│       ├── README.md           ← Your personal experiment log
│       ├── best_experiment.ipynb   ← Your best result, clean, no debug cells
│       └── experiments/
│           └── YYYY-MM-DD_description.ipynb
│
└── cookiecutter-member/        ← Template used by `make new-member` — do not edit manually
    ├── cookiecutter.json
    └── {{cookiecutter.member_name}}/
        ├── README.md
        ├── best_experiment.ipynb
        └── experiments/
```

**The key split to understand:**

- `src/` is **team code** — changes here affect everyone. Coordinate before touching it.
- `members/{your_name}/` is **your sandbox** — experiment freely, commit often, no coordination needed.
- `configs/experiments/` is **your config** — add a new YAML for each distinct experiment.

---

## 5. Team Standards — Read Before You Code

> 📋 **[`STANDARDS.md`](STANDARDS.md) is the team contract.** Read it now, before writing a single line of code or creating a single notebook.

It covers:

- **Git branch naming** (`{name}/feature`, `{name}/fix` — never commit to `main` directly)
- **Commit message format** (`exp:`, `feat:`, `fix:`, `refactor:`, `docs:`)
- **Notebook naming convention** (`YYYY-MM-DD_kebab-case-description.ipynb`)
- **The `best_experiment.ipynb` rule** (always a clean copy of your best run)
- **Leaderboard update rules** (max 2 rows per member, always sorted)
- **`src/` change policy** (discuss in chat before pushing)

This document exists so that 5 people working in parallel don't create merge conflicts, orphaned files, or experiments nobody can reproduce.

---

## 6. Create Your Member Folder

Once you have read `STANDARDS.md`, generate your personal workspace:

```bash
make new-member
```

You will be prompted for three things:

```
member_name  [your_name]:   ahmed          ← used as the folder name, lowercase, no spaces
full_name    [Your Full Name]: Ahmed Mohsen
focus_area   [e.g. transformer-based, classical-CV, lightweight-CNN]: classical-CV
```

This creates `members/ahmed/` with the correct structure already in place. Then:

1. Open `members/ahmed/README.md` and fill in your name and focus area at the top.
2. Commit your empty folder immediately so teammates can see you exist:

```bash
git checkout -b ahmed/setup
git add members/ahmed/
git commit -m "feat: add ahmed member folder"
git push -u origin ahmed/setup
```

---

## 7. Get the Dataset

The dataset is **not in the repository** (W&E files are too large). 

After downloading, your `data/` folder should look like:

---

## 8. Daily Experiment Workflow

The loop you will repeat every day:

### 8.1 Pull latest changes first

```bash
git checkout main && git pull
git checkout -b {your_name}/my-experiment-idea
```

### 8.2 Model Addition
1. Add a new file under `src/models/`, e.g. `src/models/my_tracker.py`.
   Export one callable (class or function) that accepts a DotDict config
   and returns a tracker object.
2. the tracker object must inherit and implement the `TrackerProtocol` methods (init and update) defined in `src/models/__init__.py` (see existing trackers for reference).
3. Import it in the `src/models/__init__.py`
4. Add one line to `_REGISTRY` with Key = the string you put in your YAML under `model.type`  & Value = a callable (cfg) -> TrackerProtocol:
    > "my_tracker": lambda cfg: MyTracker(cfg)
    > or if specific imports required head to:     "siamfc": _load_siamfc example,


### 8.2 Write a config

Every experiment must be driven by a YAML config — no hardcoded hyperparameters in `.py` files.

Copy the base and override what you need:

```bash
cp configs/_base.yaml configs/experiments/ahmed_siamfc_v1.yaml
```

Edit `ahmed_siamfc_v1.yaml` to override model, learning rate, batch size, etc. The config loader in `src/config.py` merges your experiment config on top of `_base.yaml`.

Config naming: `{member}_{method}_{variant}.yaml` — lowercase, underscores.

### 8.3 Create a notebook for this experiment

```bash
# inside members/{your_name}/experiments/
cp template_or_previous.ipynb "2026-04-12_siamfc-mobilenet-v1.ipynb"
```

Naming rule from `STANDARDS.md`: `YYYY-MM-DD_kebab-case-description.ipynb` — always date-prefixed, always kebab-case.

### 8.4 Run training

```bash
python main.py --config configs/experiments/ahmed_siamfc_v1.yaml
# or via the entry point:
train-tracker --config configs/experiments/ahmed_siamfc_v1.yaml
```

W&B will log your run automatically. Your run ID will appear in the terminal.

### 8.5 Evaluate on validation split
```bash
python main.py --config configs/experiments/siamfc_mobile.yaml --mode eval
```
### 8.6 Generate Kaggle submission CSV from public-LB sequences
```bash
python main.py --config configs/experiments/siamfc_mobile.yaml --mode predict --output submission.csv
```
This generates a CSV in the format required by Kaggle.

### 8.7 Quick check — print the resolved config and registered models, then exit
```bash
python main.py --config configs/experiments/siamfc_mobile.yaml --mode info
```


### 8.6 Check your evaluation numbers
After evaluation, check your AUC, S_acc, and final score. Record these numbers in your notebook and in the team leaderboard (next section).

### 8.7 Commit your notebook and config

```bash
# Strip notebook outputs before committing (keeps git history small)
jupyter nbconvert --clear-output --inplace members/ahmed/experiments/2026-04-12_siamfc-mobilenet-v1.ipynb

git add members/ahmed/experiments/2026-04-12_siamfc-mobilenet-v1.ipynb
git add configs/experiments/ahmed_siamfc_v1.yaml
git commit -m "exp: siamfc mobilenet v1 — AUC 0.61, latency 22ms"
git push
```

---
## 10. Updating the Team Leaderboard

After every Kaggle submission, open `logs/leaderboard.csv` and add your row:

```
rank,member,date,experiment_name,model_notes,auc,norm_prec,s_acc,flops_g,params_m,latency_ms,size_gb,s_eff,final_score,kaggle_public_lb,notebook_path,config_path,notes
```

**Rules (from `STANDARDS.md`):**
- Keep only your **best 2 rows** — delete older, worse runs

View the current standings at any time:

```bash
make lb
```

Output:
```
Rank  Member       Experiment                     S_acc    S_eff    Final    Kaggle LB
---------------------------------------------------------------------
1     ahmed        siamfc-mobilenet-v1            0.5820   0.4100   0.4990   0.5100
2     omar         csrt-baseline                  0.5210   0.6200   0.3970   0.4100
…
```

Commit the updated leaderboard:

```bash
git add logs/leaderboard.csv
git commit -m "docs: update leaderboard — ahmed siamfc v1 0.499"
git push
```

---

## 11. Make Command Reference

| Command | What it does |
|---------|-------------|
| `make setup` | Create `.venv`, sync all core dependencies from `uv.lock` |
| `make setup-dl` | Install PyTorch, torchvision, timm, einops, FLOPs profilers |
| `make lint` | Run `ruff check src/` — shows style and bug warnings |
| `make format` | Run `ruff format src/` + `ruff check --fix src/` — auto-fixes everything it can |
| `make new-member` | Interactively generate your `members/{name}/` folder via cookiecutter |
| `make lb` | Print the team leaderboard sorted by final score |
| `make submit FILE=… MSG=…` | Submit a CSV to the Kaggle competition |

---

## 12. Scoring Formula

---

## 13. Gotchas & FAQ

**Q: `make setup` fails with "uv: command not found"**
→ You skipped §2.2. Install uv first, then open a fresh terminal.

**Q: `ruff` is not found after `make setup`**
→ `ruff` is in the `dev` optional group. Run `uv pip install -e ".[dev]"` explicitly, or activate your venv first (`source .venv/bin/activate`) before running make.

**Q: My notebook's output is huge and slowing down git**
→ Strip outputs before committing: `jupyter nbconvert --clear-output --inplace path/to/notebook.ipynb`

**Q: Should I use `uv pip install` or just `pip install`?**
→ Always `uv pip install` inside this project. Using `pip` directly installs into a different location and will not respect the `uv.lock` pin, causing version drift across the team.

**Q: I changed `pyproject.toml` — do I need to re-run setup?**
→ Run `uv sync` to update the environment. Run `uv lock` to regenerate `uv.lock` and commit both files.

**Q: The deep-learning stack install fails on my machine**
→ PyTorch sometimes needs a platform-specific install URL (especially for CUDA). Check [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command for your OS + CUDA version, install torch separately, then run `make setup-dl`.

**Q: Can I add a new package for my experiment?**
→ Add it to the appropriate group in `pyproject.toml` (core `dependencies` if everyone needs it, `[dl]` if it requires torch, otherwise a new named group). Then `uv lock` and commit both `pyproject.toml` and `uv.lock`.

**Q: Where do I put classical-CV approaches (CSRT, KCF, MOSSE)?**
→ They are first-class citizens. Put them in `src/models/` like any other approach and register them in the model factory (`src/models/__init__.py`). Drive them via a YAML config just like a neural network. `torch` and `timm` are optional installs specifically to support classical methods without requiring the full DL stack.

---

*Team WhiteBox · Data Science & Artificial Intelligence · Zewail City of Science and Technology*
