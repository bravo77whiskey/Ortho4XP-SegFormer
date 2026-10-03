@echo off
echo Setting up Ortho4XP...
echo:

echo Creating the Python virtual environment (.venv)
python -m venv .venv
echo:

echo Activating the Python virtual environment
call .venv\Scripts\activate.bat
echo:

echo Installing Python dependencies
pip install -r requirements.txt
echo:

echo Installing the optional GDAL wheel (it only installs on the Python version in its file name)
for %%f in (Utils\win\gdal-*.whl) do pip install "%%f"
echo:

echo Ortho4XP setup complete!
echo:

echo Use start_windows.bat to start Ortho4XP
echo:

call deactivate
pause
