import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    
    result = subprocess.run([sys.executable, script_name], check=True)
    if result.returncode != 0:
        raise Exception(f"Error running {script_name}")

def main():
    # List of scripts to run in order with correct path
    scripts = [
        "preprocessing.py",
        "feature_extraction.py",
        "data_splitting.py",
        "classifier.py"
    ]
    
    # Check if all files exist
    for script in scripts:
        if not os.path.exists(script):
            raise FileNotFoundError(f"Required script {script} not found!")
    
    # Run scripts in sequence
    for script in scripts:
        run_script(script)
        
if __name__ == "__main__":
    try:
        main()
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        sys.exit(1) 