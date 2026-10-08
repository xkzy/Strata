@echo off
rem Strata native runner: runs standalone Go binary if available, else builds it or falls back to START-HERE.bat
setlocal
cd /d "%~dp0"
if exist "bin\strata.exe" (
    bin\strata.exe %*
    exit /b %errorlevel%
)
if exist "bin\strata" (
    bin\strata %*
    exit /b %errorlevel%
)
where go >nul 2>nul
if not errorlevel 1 (
    echo Building native Strata binary...
    if not exist "bin" mkdir "bin"
    go build -o bin\strata.exe .\cmd\strata
    if exist "bin\strata.exe" (
        bin\strata.exe %*
        exit /b %errorlevel%
    )
)
call START-HERE.bat %*
