import os
import sys
from dataclasses import dataclass
from typing import Sequence

# Virtual Environment
VIRTUALENV_NAME = "Venv"
PYTHON_NAME = "python3"
PIP_NAME = "pip3"

REQUIREMENTS_PATH = "requirements.in"
REQUIREMENTS_LOCK_PATH = "requirements.lock"

@dataclass
class VirtualEnvPath:
  name: str
  path: str
  python: str
  pip: str
  pip_compile: str
  pip_sync: str
  activate: str

  @staticmethod
  def create(name: str):
    VIRTUALENV_SCRIPTS = os.path.join(name, "Scripts")
    VIRTUALENV_BIN = os.path.join(name, "bin")

    path: str
    activate_script: str
    pip_compile_script: str
    pip_sync_script: str
    if sys.platform == "windows":
      path = VIRTUALENV_SCRIPTS
      activate_script = "activate.bat"
      pip_compile_script = "pip-compile.exe"
      pip_sync_script = "pip-sync.exe"
    else:
      path = VIRTUALENV_BIN
      activate_script = "activate"
      pip_compile_script = "pip-compile"
      pip_sync_script = "pip-sync"

    return VirtualEnvPath(
      name=name,
      path=path,
      python=os.path.join(path, PYTHON_NAME),
      pip=os.path.join(path, PIP_NAME),
      pip_compile=os.path.join(path, pip_compile_script),
      pip_sync=os.path.join(path, pip_sync_script),
      activate=os.path.join(path, activate_script)
    )


def run_bash_script(params: Sequence[str]):
  if sys.platform == 'windows':
    return params
  else:
    return ['bash', *params]