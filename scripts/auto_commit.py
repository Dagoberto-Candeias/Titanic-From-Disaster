import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutoCommitHandler(FileSystemEventHandler):
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.last_commit_time = 0
        self.cooldown = 5  # seconds between commits

    def on_modified(self, event):
        if event.is_directory:
            return

        # Skip certain files/directories
        skip_patterns = [
            '.git', '__pycache__', '.pytest_cache', 'output', 'catboost_info',
            'htmlcov', 'venv', '.vscode', 'titanic_ml.log', 'pipeline_execution.log',
            'run_full_pipeline.log', 'full_run.log', 'pipeline_test_output.log',
            'pipeline_output.txt', 'flake8_reporting.txt'
        ]

        if any(pattern in event.src_path for pattern in skip_patterns):
            return

        current_time = time.time()
        if current_time - self.last_commit_time < self.cooldown:
            return

        self.commit_changes(event.src_path)

    def commit_changes(self, file_path):
        try:
            # Add all changes
            subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)

            # Check if there are changes to commit
            result = subprocess.run(['git', 'status', '--porcelain'], cwd=self.repo_path, capture_output=True, text=True)
            if not result.stdout.strip():
                print("No changes to commit")
                return

            # Commit with message
            relative_path = os.path.relpath(file_path, self.repo_path)
            commit_message = f"Auto-commit: changes in {relative_path}"
            subprocess.run(['git', 'commit', '-m', commit_message], cwd=self.repo_path, check=True, capture_output=True)

            # Push
            subprocess.run(['git', 'push'], cwd=self.repo_path, check=True, capture_output=True)

            print(f"Committed and pushed changes for {relative_path}")
            self.last_commit_time = time.time()

        except subprocess.CalledProcessError as e:
            print(f"Error during commit: {e}")

if __name__ == "__main__":
    repo_path = os.getcwd()
    event_handler = AutoCommitHandler(repo_path)
    observer = Observer()
    observer.schedule(event_handler, repo_path, recursive=True)
    observer.start()

    print("Auto-commit watcher started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
