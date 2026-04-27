import sys
import subprocess
from pathlib import Path
import shutil

EMBERCLONE = "https://github.com/FutureComputing4AI/EMBER2024.git"
CWD = Path.cwd()
EMBER_DIR = CWD / "EMBER2024"
DATA_DEST_ABS = CWD / "Data"


def ensure_gitpython():
    try:
        from git import Repo
        return Repo
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
        from git import Repo
        return Repo


def clone_repo():
    Repo = ensure_gitpython()

    if EMBER_DIR.exists():
        raise FileExistsError(
            f"Target repo directory (EMBER2024) already exists: {EMBER_DIR}"
        )

    DATA_DEST_ABS.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(EMBERCLONE, EMBER_DIR)


def install_thrember():
    if not EMBER_DIR.is_dir():
        raise FileNotFoundError(
            f"EMBER2024 repo directory not found: {EMBER_DIR}"
        )

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "."],
        cwd=EMBER_DIR
    )


def download_data():
    try:
        import thrember
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "thrember is not installed after pip install."
        ) from e

    thrember.download_dataset(str(DATA_DEST_ABS), file_type="PE")


def cleanup_repo():
    shutil.rmtree(EMBER_DIR, ignore_errors=True)


def main():
    try:
        clone_repo()
        install_thrember()
        download_data()
        print(f"Dataset downloaded successfully to: {DATA_DEST_ABS}")
    finally:
        cleanup_repo()


if __name__ == "__main__":
    main()