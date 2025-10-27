@echo off
REM Smart Meter Anomaly Detection - Quick Launch Script
REM This script starts the Flask API and opens the dashboard

echo ============================================================
echo   Smart Meter Anomaly Detection System
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Found virtual environment, activating...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

REM Check if database has data
echo Checking database status...
python -c "from pymongo import MongoClient; db = MongoClient()['smart_meter_db']; count = db.meter_readings.count_documents({}); print(f'Database records: {count:,}'); exit(0 if count > 0 else 1)" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo WARNING: Database appears to be empty
    echo Would you like to load the data now? This takes 5-10 minutes.
    choice /C YN /M "Load database"
    if errorlevel 2 goto skip_load
    if errorlevel 1 goto load_data
)

goto start_api

:load_data
echo.
echo Loading database...
python updated_mongodb_setup.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to load database
    pause
    exit /b 1
)
echo.
echo Database loaded successfully!
goto start_api

:skip_load
echo Skipping database load...

:start_api
echo.
echo ============================================================
echo   Starting Flask API Server
echo ============================================================
echo.
echo API will be available at: http://localhost:5000
echo Dashboard will open automatically
echo.
echo Press Ctrl+C to stop the server
echo.

REM Open dashboard in default browser (in background)
start "" frontend\dashboard.html

REM Wait a moment for browser to start
timeout /t 2 /nobreak >nul

REM Start Flask API (this will block)
python scripts\flask_api_setup.py

REM If we get here, the server was stopped
echo.
echo Server stopped.
pause
