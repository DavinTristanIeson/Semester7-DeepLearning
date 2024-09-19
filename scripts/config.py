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

  @property
  def activate_venv(self)->Sequence[str]:
    try:
      os.chmod(self.activate, 0o744)
    except Exception as e:
      print(e)
    filename = os.path.basename(self.activate)
    if filename.endswith('.bat'):
      return [self.activate]
    else:
      return ['bash', self.activate]

  @staticmethod
  def detect_venv(name: str):
    '''This function should be run only after a virtual environment has been created'''
    VIRTUALENV_SCRIPTS = os.path.join(name, "Scripts")
    VIRTUALENV_BIN = os.path.join(name, "bin")
    if os.path.exists(VIRTUALENV_BIN):
      path = VIRTUALENV_BIN
      return VirtualEnvPath(
        name=name,
        path=path,
        python=os.path.join(path, PYTHON_NAME),
        pip=os.path.join(path, PIP_NAME),
        pip_compile=os.path.join(path, "pip-compile"),
        pip_sync=os.path.join(path, "pip-sync"),
        activate=os.path.join(path, "activate"),
      )
  
    if os.path.exists(VIRTUALENV_SCRIPTS):
      path = VIRTUALENV_SCRIPTS
      return VirtualEnvPath(
        name=name,
        path=path,
        python=os.path.join(path, PYTHON_NAME),
        pip=os.path.join(path, PIP_NAME),
        pip_compile=os.path.join(path, "pip-compile.exe"),
        pip_sync=os.path.join(path, "pip-sync.exe"),
        activate=os.path.join(path, "activate.bat"),
      )
    
    raise Exception("Unable to detect any supported virtual-env installation in the local computer.")