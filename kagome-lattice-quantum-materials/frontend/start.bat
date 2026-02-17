@echo off
REM Quick Start Script for Kagome Lattice App
REM 快速启动脚本

echo ======================================================================
echo Kagome Lattice Optimization - Quick Start
echo Kagome晶格优化 - 快速启动
echo ======================================================================
echo.

echo This script will start both backend and frontend
echo 此脚本将同时启动后端和前端
echo.

REM Check if in correct directory
if not exist "package.json" (
    echo ERROR: package.json not found!
    echo Please run this script from the frontend directory
    echo 请在frontend目录中运行此脚本
    pause
    exit /b 1
)

echo [1/3] Checking if backend is running...
curl -s http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Backend not detected on port 5000
    echo 警告: 未检测到5000端口的后端服务
    echo.
    echo Please start backend first in another terminal:
    echo 请先在另一个终端启动后端:
    echo   cd backend
    echo   python app_pytorch.py
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Backend is running
)

echo.
echo [2/3] Installing dependencies (if needed)...
if not exist "node_modules" (
    echo Installing npm packages...
    call npm install
) else (
    echo ✅ Dependencies already installed
)

echo.
echo [3/3] Starting frontend development server...
echo ======================================================================
echo.
echo Frontend will open at: http://localhost:3000
echo Backend running at:    http://localhost:5000
echo.
echo Press Ctrl+C to stop
echo.
echo ======================================================================

call npm run dev
