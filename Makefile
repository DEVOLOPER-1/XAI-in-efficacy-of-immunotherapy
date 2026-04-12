.PHONY: setup setup-dl lint format new-member lb submit

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	uv venv
	uv sync
	uv lock
# 	uv pip install -e ".[dev]"
	@echo "✓ Core dependencies installed"
	@echo "  Run 'make setup-dl' to also install torch/timm"

setup-dl:
	uv pip install -e ".[dl,profiling]"
	@echo "✓ Deep-learning stack installed"



# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	ruff check src/

format:
	ruff format src/
	ruff check --fix src/

# ── Member onboarding ──────────────────────────────────────────────────────────
new-member:
	@echo "Creating your member folder..."
	cookiecutter cookiecutter-member/ --output-dir members/
	@echo "✓ Done. Update members/{your_name}/README.md and start experimenting."

# ── Team leaderboard ───────────────────────────────────────────────────────────
# Shows leaderboard sorted by final_score descending
lb:
	@python3 -c "\
import csv, sys; \
rows = list(csv.DictReader(open('logs/leaderboard.csv'))); \
rows.sort(key=lambda r: float(r.get('final_score') or 0), reverse=True); \
print(f\"{'Rank':<5} {'Member':<12} {'Experiment':<30} {'S_acc':<8} {'S_eff':<8} {'Final':<8} {'Kaggle LB'}\"); \
print('-' * 85); \
[print(f\"{i+1:<5} {r['member']:<12} {r['experiment_name']:<30} {r['s_acc']:<8} {r['s_eff']:<8} {r['final_score']:<8} {r['kaggle_public_lb']}\") for i, r in enumerate(rows) if r['member'] != 'example'] \
"

# ── Kaggle submission ──────────────────────────────────────────────────────────
# Usage: make submit FILE=submissions/my_preds.csv MSG="siamfc mobilenet v2"
submit:
	@test -n "$(FILE)"  || (echo "Usage: make submit FILE=path/to/submission.csv MSG='description'"; exit 1)
	@test -n "$(MSG)"   || (echo "Usage: make submit FILE=path/to/submission.csv MSG='description'"; exit 1)
	kaggle competitions submit mtc-aic-4-phase-i -f $(FILE) -m "$(MSG)"
	@echo "✓ Submitted. Update logs/leaderboard.csv with your public LB score."
