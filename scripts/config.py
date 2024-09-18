import os

# Virtual Environment
VIRTUALENV_NAME = "Venv"
VIRTUALENV_SCRIPTS = os.path.join(VIRTUALENV_NAME, "Scripts")
VENVPYTHON = os.path.join(VIRTUALENV_SCRIPTS, "python")
VENVPIP = os.path.join(VIRTUALENV_SCRIPTS, "pip")
REQUIREMENTS_PATH = "requirements.in"
REQUIREMENTS_LOCK_PATH = "requirements.lock"
