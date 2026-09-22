#!/bin/bash
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Opening python.org - install it, then double-click this file again."
  open https://www.python.org/downloads/
  read -p "Press Enter to close"
  exit 1
fi

# skip Streamlit's first-run email question
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
  mkdir -p "$HOME/.streamlit"
  printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

if [ ! -f .installed ]; then
  echo "First run: installing components, this takes a minute or two..."
  python3 -m pip install -r requirements.txt || { echo "Install failed."; read -p "Press Enter"; exit 1; }
  touch .installed
fi

echo "Scanner is starting. Your browser will open shortly. Keep this window open; close it to stop."
python3 -m streamlit run app.py
