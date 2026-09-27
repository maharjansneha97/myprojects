@echo off
echo Starting Intelligent Environment Image Interpreter (Django Edition)...
echo.
echo 1. Checking dependencies...
pip install -r requirements.txt
echo.
echo 2. Running Migrations...
cd django_app
python manage.py migrate
echo.
echo 3. Starting Django Application...
start http://127.0.0.1:8000
python manage.py runserver
pause
