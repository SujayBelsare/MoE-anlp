"""
Setup script for ANLP Assignment 3
Creates directory structure and verifies environment
"""

import os
import sys
import subprocess


def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'models',
        'pipelines',
        'utils',
        'notebooks',
        'results',
        'visualizations',
        'logs',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory in ['models', 'pipelines', 'utils']:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f'"""{directory} module"""\n')
    
    print("✓ Directory structure created")


def verify_python_version():
    """Verify Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"  Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Available GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠ CUDA not available, will use CPU (training will be slow)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False


def install_dependencies(check_only=False):
    """Install or check dependencies"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'peft',
        'rouge-score',
        'bert-score',
        'nltk',
        'matplotlib',
        'seaborn',
        'pandas',
        'numpy',
        'pyyaml',
        'tqdm',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    
    if missing:
        if check_only:
            print(f"\n⚠ Missing packages: {', '.join(missing)}")
            print("Run: pip install -r requirements.txt")
            return False
        else:
            print(f"\nInstalling missing packages: {', '.join(missing)}")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing)
            print("✓ Dependencies installed")
    else:
        print("\n✓ All dependencies installed")
    
    return True


def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("\nDownloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"⚠ Failed to download NLTK data: {e}")
        return False


def verify_config():
    """Verify config.yaml exists and is valid"""
    if not os.path.exists('config.yaml'):
        print("✗ config.yaml not found")
        return False
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_sections = ['data', 'model', 'training', 'baselines', 'evaluation', 'output']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing section in config.yaml: {section}")
                return False
        
        print("✓ config.yaml is valid")
        return True
    except Exception as e:
        print(f"✗ Error reading config.yaml: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review and adjust config.yaml for your hardware")
    print("   - Adjust batch sizes based on GPU memory")
    print("   - Set train_samples to small number for testing")
    print("   - Configure number of epochs and experts")
    print()
    print("2. Test with a small subset:")
    print("   python main.py --config config.yaml --task bart_inference")
    print()
    print("3. Follow sequence.txt for full task execution:")
    print("   cat sequence.txt")
    print()
    print("4. Start with baselines:")
    print("   python main.py --config config.yaml --task all_baselines")
    print()
    print("5. Train MoE models:")
    print("   python main.py --config config.yaml --task all_moe")
    print()
    print("6. Run evaluation:")
    print("   python main.py --config config.yaml --task all_eval")
    print()
    print("For help: python main.py --help")
    print("="*60)


def main():
    """Main setup function"""
    print("="*60)
    print("ANLP Assignment 3 - Setup")
    print("="*60)
    
    # Verify Python version
    if not verify_python_version():
        sys.exit(1)
    
    # Create directories
    create_directory_structure()
    
    # Verify config
    verify_config()
    
    # Check/install dependencies
    install_dependencies(check_only=True)
    
    # Check CUDA
    check_cuda()
    
    # Download NLTK data
    download_nltk_data()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()