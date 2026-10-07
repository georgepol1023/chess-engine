@echo off
rem Launcher for chess GUIs: point the GUI's "add engine" dialog at this file.
cd /d "%~dp0"
".venv\Scripts\python.exe" uci.py --checkpoint checkpoints\run\model.pt
