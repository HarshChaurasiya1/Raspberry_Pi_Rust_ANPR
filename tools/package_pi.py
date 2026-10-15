import zipfile
import os

zip_name = r"D:\Raspberry_Pi_Rust\pi_rust_anpr.zip"
exclude_dirs = {'target', '.git', '__pycache__', 'test_results', 'benchmarks'}
exclude_files = {'my_car.mp4', 'pi_rust_anpr.zip', 'anpr_kaggle_ready.zip'}
include_roots = ['.cargo', 'configs', 'models', 'samples', 'src', 'tests', 'tools']
include_files = ['build.rs', 'Cargo.lock', 'Cargo.toml', 'COMPLETE_BEGINNER_GUIDE.md', 'install.sh', 'README.md']

print("Building package...")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in include_files:
        if os.path.exists(f):
            zipf.write(f, f)
            print(f"Added {f}")
    for root_dir in include_roots:
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file in exclude_files or file.endswith('.pyc') or file.endswith('.log'):
                    continue
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path)

size_mb = os.path.getsize(zip_name) / (1024 * 1024)
print(f"Successfully packaged: {zip_name} ({size_mb:.2f} MB)")
