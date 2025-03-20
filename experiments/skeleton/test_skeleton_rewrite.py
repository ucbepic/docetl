import os
from pathlib import Path
from dotenv import load_dotenv
from docetl.runner import DSLRunner

def rewrite(yaml_file: Path, max_threads=None, resume=False, save_path=None):
    # Get the current working directory (where the user called the command)
    cwd = os.getcwd()

    # Load .env file from the current working directory
    env_file = os.path.join(cwd, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)

    runner = DSLRunner.from_yaml(str(yaml_file), max_threads=max_threads)
    runner.rewrite()

def main():
    yaml_file = Path("experiments/skeleton/map_test.yaml")
    rewrite(yaml_file)

if __name__ == "__main__":
    main()