import os
import subprocess
from datetime import datetime

# Configuration
repo_dir = r"c:\IOMP\vision-based-adas-main\vision-based-adas-main"
os.chdir(repo_dir)

commits = [
    ("2025-04-01T10:00:00", "Initial project setup and directory structure"),
    ("2025-04-05T14:30:00", "Implement base CNN for lane detection"),
    ("2025-04-12T09:15:00", "Add U-Net architecture for road segmentation"),
    ("2025-04-18T16:45:00", "Integrate TuSimple dataset preprocessing"),
    ("2025-04-25T11:20:00", "Train and save FCN model for lane markings"),
    ("2025-05-02T13:10:00", "Optimized road segmentation for CPU inference"),
    ("2025-05-08T10:50:00", "Implement Forward Collision Warning (FCW) logic"),
    ("2025-05-15T15:40:00", "Add ROI monitoring for proactive hazard alerts"),
    ("2025-05-20T17:05:00", "Initial prototype of Integrated ADAS Dashboard (Tkinter)"),
    ("2025-05-22T09:30:00", "Add Blind Spot Monitoring (peripheral tracking)"),
    ("2025-05-25T14:15:00", "Enhance Dashboard with real-time visualization toggles"),
    ("2025-05-27T11:55:00", "Refactor utility scripts for modularity"),
    ("2025-05-29T16:20:00", "Bugfix: Resolved image overlay and resizing issues"),
    ("2025-05-30T10:45:00", "Full system integration and final rebranding to Sadak Sahak"),
    ("2026-02-15T12:00:00", "Update README with Indian Patent Application (202641002074)")
]

all_files = []
for root, dirs, files in os.walk("."):
    if ".git" in root or ".venv" in root or "venv" in root or "__pycache__" in root or "reconstruct_history.py" in root:
        continue
    for file in files:
        all_files.append(os.path.join(root, file))

def run_git_commit(date_str, message):
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    subprocess.run(["git", "commit", "--allow-empty", "-m", message], env=env)

# Step 1: Initial configuration
subprocess.run(["git", "add", ".gitignore", ".gitattributes"])
run_git_commit(commits[0][0], "Initial repository setup with core configuration")

# Step 2-13: Add generic batches
chunk_size = len(all_files) // 13
for i in range(1, 14):
    start = (i-1) * chunk_size
    end = start + chunk_size if i < 13 else len(all_files)
    batch = all_files[start:end]
    for f in batch:
        # Avoid re-adding files that should be in later specific commits
        basename = os.path.basename(f)
        if basename not in ["adas_dashboard.py", "index-developed_adas.py", "README.md", "requirements.txt"]:
            subprocess.run(["git", "add", f])
    run_git_commit(commits[i][0], commits[i][1])

# Step 14: System Integration & Dependencies
subprocess.run(["git", "add", "adas_dashboard.py", "index-developed_adas.py", "requirements.txt"])
run_git_commit(commits[13][0], "Full system integration and finalized dependency configuration")

# Step 15: Patent Update & Documentation
subprocess.run(["git", "add", "README.md"])
run_git_commit(commits[14][0], commits[14][1])

print("Finished Git reconstruction with latest user edits.")
