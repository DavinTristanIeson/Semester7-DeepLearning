import os
import sys
sys.path.append(os.getcwd())

import subprocess
import argparse
from retina.log import Ansi
import scripts.config


parser = argparse.ArgumentParser(
  prog="Setup Project",
  description="This script is used to setup the virtual environment and its dependencies",
)
parser.add_argument("--install", dest="is_install", action="store_const", default=False, const=True, help="Installs the dependencies into the virtual environment. Your global environment will not be polluted.")
args = parser.parse_args()

if not os.path.exists(scripts.config.VIRTUALENV_NAME):
  subprocess.run(["python", "-m", "venv", scripts.config.VIRTUALENV_NAME], check=True)

subprocess.run([os.path.join(scripts.config.VIRTUALENV_SCRIPTS, "activate.bat")], check=True)

if args.is_install:
  PIP_COMPILE = os.path.join(scripts.config.VIRTUALENV_SCRIPTS, "pip-compile.exe")
  PIP_SYNC = os.path.join(scripts.config.VIRTUALENV_SCRIPTS, "pip-sync.exe")
  if not os.path.exists(PIP_COMPILE) or not os.path.exists(PIP_SYNC):
    print(f"{Ansi.Error}Found no existing pip-tools installation. Installing them from PyPy...{Ansi.End}")
    subprocess.run([scripts.config.VENVPIP, "install", "pip-tools"], check=True)
  subprocess.run([PIP_COMPILE, scripts.config.REQUIREMENTS_PATH, "-o", scripts.config.REQUIREMENTS_LOCK_PATH, "--verbose"])
  subprocess.run([PIP_SYNC, scripts.config.REQUIREMENTS_LOCK_PATH, "--verbose"])
else:
  print(f"{Ansi.Warning}WARNING: If this is your first time running this project, please run python scripts/setup.py --install.{Ansi.End}")


