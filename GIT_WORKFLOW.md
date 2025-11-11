# Git Workflow for LILY Project

This document defines our commit workflow to avoid large "catch-up" commits and maintain better project history.

---

## Commit Frequency Rule

**COMMIT AFTER EACH MAJOR TASK** - not at the end of the day/week.

### What Counts as a "Major Task"?

1. **Experiment completion** - Any ML training run that produces results (success or failure)
2. **Script creation** - New analysis/training/visualization script is ready to use
3. **Documentation** - Complete a documentation file or major section
4. **Dataset creation** - Generate a new dataset file
5. **Bug fix** - Fix a broken script or model
6. **Analysis completion** - Finish a visualization, report, or measurement analysis

**Rule of thumb:** If you can describe it in 1-2 sentences, it's a commit.

---

## Commit Workflow

### When Starting Work

```bash
# Check current status first
git status
```

If there are uncommitted changes from last session, commit them before starting new work.

### During Work

After completing each major task:

```bash
# 1. Check what changed
git status

# 2. Review changes (optional but recommended)
git diff [file_to_review]

# 3. Add files for this specific task only
git add [specific_files]

# OR add everything if it's all related
git add -A

# 4. Commit with clear, concise message
git commit -m "Brief description of what was done"
```

### End of Session

```bash
# Commit any remaining work
git add -A
git commit -m "Work in progress: [describe current state]"

# Push to remote (if applicable)
git push
```

---

## Commit Message Format

### Good Commit Messages

**Format:** `<action> <what>: <brief details>`

**Examples:**

```
Add VAE v2.6.8 fuzzy matching experiment: failed at ARI=0.087

Create RGB prediction models: R²=0.72 but downstream failure in VAE

Document semi-supervised classifier: 16.24% vs 42.32% baseline

Fix dataset creation bug in create_vae_v2_7_dataset.py

Train architecture comparison: 10D latent optimal over 8D

Analyze borehole coverage: 212/437 have complete measurements

Update CLAUDE.md: add v2.6.7 as production model
```

### Commit Message Template

For experiments:
```
<Experiment name>: <key metric result>

- <optional detail 1>
- <optional detail 2>
```

For code:
```
<Action> <file/feature>: <what it does>

- <optional implementation detail>
```

For documentation:
```
Document <topic>: <key finding/summary>
```

### Bad Commit Messages

❌ `update`
❌ `work in progress`
❌ `changes`
❌ `fixed stuff`
❌ `asdf`

---

## When to Group vs Split Commits

### Group into ONE commit:

- Experiment + its training script + its dataset + results CSV
- Script + its log file
- Documentation file + related CLAUDE.md update
- Multiple files that implement the same feature

### Split into SEPARATE commits:

- Multiple independent experiments (commit after each finishes)
- Unrelated documentation updates
- Bug fixes in different scripts
- Different analysis tasks

**Example timeline:**
```
10:00 AM - Train VAE v2.6.8 → commit "Add VAE v2.6.8 fuzzy matching..."
11:30 AM - Create RGB predictor → commit "Create RGB prediction models..."
02:00 PM - Write documentation → commit "Document semi-supervised classifier..."
04:00 PM - Fix bug in analysis → commit "Fix depth binning in coverage analysis"
```

---

## Special Cases

### Failed Experiments

**DO commit failed experiments!** They're valuable negative results.

```
Add VAE v2.6.10 RGB prediction experiment: failed at ARI=0.093

Expanded dataset to 396K samples using predicted RGB (R²=0.72),
but 28% unexplained variance corrupted cross-modal learning.
Conclusion: feature quality > dataset size.
```

### Work in Progress

If you must stop mid-task:

```bash
git add -A
git commit -m "WIP: [describe current state and next steps]"
```

Example:
```
WIP: RGB prediction - models trained, need to test on VAE

Models achieve R²=0.72 on validation set.
Next: create expanded dataset and train VAE v2.6.10
```

### Multiple Small Changes

If you have 3-5 small, unrelated changes, commit them separately:

```bash
git add script1.py
git commit -m "Fix typo in script1.py"

git add script2.py
git commit -m "Add logging to script2.py"

git add doc.md
git commit -m "Update doc.md with v2.6.7 results"
```

---

## Quick Reference Commands

```bash
# See what changed
git status              # High-level overview
git status --short      # Compact view
git diff                # All changes
git diff --stat         # Just file summaries
git diff file.py        # Specific file

# Commit workflow
git add file.py         # Stage specific file
git add folder/         # Stage entire folder
git add -A              # Stage everything
git commit -m "msg"     # Commit with message

# Undo mistakes (use carefully!)
git restore file.py            # Discard changes to file
git restore --staged file.py   # Unstage file
git reset --soft HEAD~1        # Undo last commit (keep changes)
git commit --amend             # Modify last commit message

# History
git log --oneline -10   # Last 10 commits
git log --stat          # With file summaries
git show                # Last commit details
```

---

## Workflow Example: Typical Day

```bash
# Morning: Start work
$ git status
# (Clean working tree)

# Task 1: Train new experiment
$ python train_vae_v2_8.py > vae_v2_8_training.log
# (Training completes)
$ git add train_vae_v2_8.py vae_v2_8_training.log ml_models/checkpoints/vae_v2_8.pth vae_v2_8_results.csv
$ git commit -m "Add VAE v2.8 experiment: tested layer normalization, ARI=0.184"

# Task 2: Write documentation
$ vim VAE_V2_8_SUMMARY.md
# (Finish writing)
$ git add VAE_V2_8_SUMMARY.md CLAUDE.md
$ git commit -m "Document VAE v2.8: layer normalization reduced ARI by 6%"

# Task 3: Fix bug in old script
$ vim analyze_coverage.py
# (Fix bug)
$ git add analyze_coverage.py
$ git commit -m "Fix off-by-one error in borehole depth calculation"

# Task 4: Run coverage analysis
$ python analyze_coverage.py > coverage.log
$ python visualize_coverage.py
# (Generates plots)
$ git add analyze_coverage.py coverage.log coverage_plots/
$ git commit -m "Analyze borehole coverage: 89% have GRA+MS, only 54% have RGB"

# End of day: Push to remote
$ git push
```

Result: **4 clear commits** instead of one giant "end of day" commit.

---

## Benefits of This Workflow

1. **Clear history** - Easy to see what was done when
2. **Easy rollback** - Can revert specific experiments/changes
3. **Better diffs** - Smaller commits = easier to review
4. **Documentation** - Commit messages serve as lab notebook
5. **Collaboration** - Others can follow progress

---

## Claude's Responsibilities

When working with Claude Code, Claude will:

1. **Remind you to commit** after each major task completion
2. **Suggest commit message** based on the work done
3. **Batch related changes** (e.g., experiment + log + results)
4. **Ask before committing** if unclear what should be included

**Example:**
```
Claude: "I've completed training VAE v2.8 (ARI=0.184). Should I commit this now with:
- train_vae_v2_8.py
- vae_v2_8_training.log
- ml_models/checkpoints/vae_v2_8.pth
- vae_v2_8_results.csv

Suggested message: 'Add VAE v2.8 experiment: tested layer normalization, ARI=0.184'

Proceed with commit?"
```

---

## Emergency: "I Have 50 Uncommitted Files"

If you find yourself with many uncommitted changes again:

1. **Group by topic:**
   ```bash
   git add *fuzzy* && git commit -m "Add fuzzy matching experiments"
   git add *rgb* && git commit -m "Add RGB prediction models"
   git add *semi* && git commit -m "Add semi-supervised classifier"
   git add *.md && git commit -m "Update documentation"
   ```

2. **Or just commit everything:**
   ```bash
   git add -A
   git commit -m "Bulk commit: [list major items]"
   ```

3. **Then resume normal workflow** - don't let it happen again!

---

## Summary

✅ **DO:**
- Commit after each major task (experiment, script, documentation)
- Write clear, descriptive commit messages
- Commit failed experiments (they're valuable!)
- Push regularly to remote

❌ **DON'T:**
- Wait until end of day/week to commit
- Use vague commit messages
- Skip commits for "small" changes
- Leave WIP uncommitted overnight

**Golden rule:** If you can describe what you did in 1-2 sentences, commit it now.
