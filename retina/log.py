class Ansi:
  Error = '\033[91m'
  Warning = '\033[93m'
  Success = '\033[92m'
  End = '\033[0m'
  Grey = "\033[38;5;243m"
  Bold = "\033[1m"
  Underline = "\033[4m"

def show_progress(action: str, message: str, step: int, total: int):
  print(f"{Ansi.Bold}{action}{Ansi.End}{Ansi.Grey}: {message} ({step + 1}/{total}){Ansi.End}")
