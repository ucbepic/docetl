import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from docetl.runner import DSLRunner
from docetl.utils import decrypt, load_config

def main():
    parser = argparse.ArgumentParser(description="Build and optimize a DocETL pipeline from YAML config.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML config file.")
    parser.add_argument("--max_threads", type=int, default=None, help="Maximum number of threads to use for running operations.")
    parser.add_argument("--resume", action="store_true", help="Resume optimization from a previous build that may have failed.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the optimized pipeline configuration.")
    args = parser.parse_args()

    # Load .env from current working directory if it exists
    env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)

    # yaml_file = str(args.yaml_path)
    # base_name = yaml_file.rsplit(".", 1)[0]
    # suffix = yaml_file.split("/")[-1].split(".")[0]
    # config = load_config(yaml_file)

    # print(config)
    

    # Run an DocETL optimization.
    try:
        runner = DSLRunner.from_yaml(str(args.yaml_path), max_threads=args.max_threads)
        
        print("GOOD")
        runner.optimize(
            save=True,
            return_pipeline=False,
            resume=args.resume,
            save_path=args.save_path,
        )
        print("\n[BUILD SUCCESS] Pipeline built and optimized successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"[BUILD FAILED] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 