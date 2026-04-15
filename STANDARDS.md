# Team Standards — WhiteBox / XAI

> One document. Read it once. Follow it always.
> The goal: zero merge conflicts from naming chaos, zero "what does this do" confusion.

---

## 1. Git Workflow

### Branches
```
main              ← stable, always runnable
{name}/feature    ← your personal branch, e.g. ahmed/siamfc-backbone
{name}/fix        ← bug fixes
```

Never commit directly to `main`. Open a PR (even if you merge it yourself) so there's a record.

### Commit messages — Conventional Commits style
```
feat: add MobileNetV4 backbone to model registry
exp:  siamfc with cosine LR schedule — AUC 0.61
fix:  off-by-one in annotation parser frame index
refactor: split data_loader into loader + augment
docs: update leaderboard with ahmed's best run
```

Prefix `exp:` is for experiment commits — these may include notebook changes, config tweaks, WandB run IDs.

### What NOT to commit
- Raw data (`data/`, `*.mp4`, `*.jpg`)  ← gitignored
- Model weights (`*.pth`, `*.pt`) ← gitignored
- WandB cache (`wandb/`) ← gitignored
- Notebook outputs with large embedded images ← strip with `jupyter nbconvert --clear-output`

---

## 2. Member Folder Structure

Each team member has exactly one folder under `members/`. Create yours once with:

```bash
make new-member
# or directly:
cookiecutter cookiecutter-member/
```

This generates:
```
members/
└── {your_name}/
    ├── README.md               ← your personal experiment log (keep updated)
    ├── best_experiment.ipynb   ← your single best result (clean, no debug cells)
    └── experiments/
        ├── 2026-04-05_eda-uav123.ipynb
        ├── 2026-04-07_siamfc-baseline.ipynb
        └── 2026-04-09_mobilenet-backbone.ipynb
```

### Notebook naming convention
```
YYYY-MM-DD_kebab-case-description.ipynb
```
- Date prefix = sortable chronologically by default in every file browser
- Kebab-case = no spaces, no underscores, lowercase
- Description = what you tried, not what you hoped for

✓ `2026-04-09_siamfc-mobilenetv3-cosine-lr.ipynb`
✓ `2026-04-12_csrt-baseline-uav123.ipynb`
✗ `new notebook.ipynb`
✗ `test2_FINAL_v3.ipynb`

### Best experiment rule
- `best_experiment.ipynb` is always a **clean copy** of your best notebook from `experiments/`
- Strip all debug/print cells before copying
- Add a summary table at the top (see the template)
- Update `members/{your_name}/README.md` to match

---

## 3. Team Leaderboard (`logs/leaderboard.csv`)

Update this file after **every submission**. It is the team's source of truth.

```
rank, member, date, experiment_name, model_notes,
auc, s_acc,
notebook_path, config_path, notes
```

Rules:
- **Only keep your best 2 rows** per member (delete older worse runs)
- `final_score` = compute locally: `s_acc - 0.2 * s_eff`
- `kaggle_public_lb` = the score shown on the Kaggle leaderboard after submission
- Keep the file sorted by `final_score` descending (the `make lb` command does this for you)

---

## 4. Config Files

All experiments must be reproducible from a YAML config. No hardcoded hyperparameters in `.py` files.

```
configs/
├── _base.yaml              ← shared defaults (dataset paths, eval protocol)
└── experiments/
    ├── siamfc_mobile.yaml  ← inherits _base, overrides model + LR
    └── csrt_baseline.yaml
```

Naming: `{method}_{variant}.yaml` in lowercase kebab-case.

---

## 5. Core `src/` Changes

The `src/` directory is **shared code** — a bug there breaks everyone.

Rules:
- Discuss changes to `src/` in the team chat before pushing
- Any change to `src/models/__init__.py` (the registry) needs a PR
- New models go in `src/models/` with a self-contained file — don't modify `baselines.py`, add new files

---
