@echo off
echo Starting Forgery Detection System...

REM ---- Activate Virtual Environment ----
call venv\Scripts\activate

REM ---- Start Flask Backend in New Window ----
echo Starting Flask API...
start cmd /k "call venv\Scripts\activate && python main.py"

REM ---- Wait 2 Seconds to Ensure Backend Boots ----
timeout /t 2 >nul

REM ---- Start Streamlit UI in New Window ----
echo Starting Streamlit App...
start cmd /k "call venv\Scripts\activate && streamlit run app.py"

echo ---
echo System launched successfully.
echo Close this window if both apps are running.
pause
