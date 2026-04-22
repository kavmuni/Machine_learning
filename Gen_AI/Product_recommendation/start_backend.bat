@echo off
cd /d "C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\backend"
echo Starting Product Recommendation FastAPI Backend...
echo.
echo This window will show the server logs.
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn main:app --reload --port 8000
pause
