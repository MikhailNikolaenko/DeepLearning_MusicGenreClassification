# train_all.py
import subprocess

def run(cmd):
    print(f"\n===== RUNNING: {cmd} =====\n")
    
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

print("Training CNN...")
run("python train_musicrecnet_kaggle.py")

print("\nExtracting features...")
run("python extract_features_kaggle.py")

print("\nTraining SVM...")
run("python train_svm.py")

print("\n===== ALL MODELS TRAINED SUCCESSFULLY =====")
