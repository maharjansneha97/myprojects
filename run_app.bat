@echo off
echo Starting Intelligent Environment Image Interpreter...
echo.
echo 1. Checking dependencies...
pip install -r requirements.txt
echo.
echo 2. Starting Web Application...
start http://127.0.0.1:5000
cd web_app
python app.py
pause
