import os
import sys
sys.path.append(os.getcwd())

import subprocess
import argparse
import scripts.config


parser = argparse.ArgumentParser(
  prog="Setup Project",
  description="This script is used to setup the virtual environment and its dependencies",
)
parser.add_argument("--install", dest="is_install", action="store_const", default=False, const=True, help="Installs the dependencies into the virtual environment. Your global environment will not be polluted.")
args = parser.parse_args()

if not os.path.exists(scripts.config.VIRTUALENV_NAME):
  subprocess.run([scripts.config.PYTHON_NAME, "-m", "venv", scripts.config.VIRTUALENV_NAME], check=True)

venvpaths = scripts.config.VirtualEnvPath.create(scripts.config.VIRTUALENV_NAME)
os.chmod(venvpaths.activate, 0o744)
subprocess.run(scripts.config.run_bash_script([venvpaths.activate]), check=True)

if args.is_install:
  if not os.path.exists(venvpaths.pip_compile) or not os.path.exists(venvpaths.pip_sync):
    print(f"Found no existing pip-tools installation. Installing them from PyPy...")
    subprocess.run([venvpaths.pip, "install", "pip-tools"], check=True)
  subprocess.run([venvpaths.pip_compile, scripts.config.REQUIREMENTS_PATH, "-o", scripts.config.REQUIREMENTS_LOCK_PATH, "--verbose"])
  subprocess.run([venvpaths.pip_sync, scripts.config.REQUIREMENTS_LOCK_PATH, "--verbose"])
else:
  print(f"WARNING: If this is your first time running this project, please run python scripts/setup.py --install.")


