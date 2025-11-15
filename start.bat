@echo off
chcp 65001 >nul
cls

echo ========================================
echo A股多维度金融模型训练系统
echo A-Share Multi-Dimensional Financial Model
echo ========================================
echo.
echo 作者 / Author: eddy
echo 版本 / Version: 2.0
echo 更新 / Updated: 2025-11-14
echo.
echo ========================================
echo.
echo 正在启动 Gradio 界面...
echo Starting Gradio interface...
echo.
echo 访问地址 / Access URL:
echo http://127.0.0.1:7860
echo.
echo 按 Ctrl+C 停止服务器
echo Press Ctrl+C to stop the server
echo ========================================
echo.

call .\venv\Scripts\activate.bat
python gradio_app.py

pause

