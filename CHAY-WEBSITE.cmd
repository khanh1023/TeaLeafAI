@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Please install Node.js LTS from https://nodejs.org then reopen this file.
  pause
  exit /b 1
)
if not exist "frontend\dist\index.html" (
  echo Preparing TeaLeafAI for the first time. Internet connection required...
  pushd frontend
  call npm ci
  if errorlevel 1 (popd & pause & exit /b 1)
  call npm run build
  if errorlevel 1 (popd & pause & exit /b 1)
  popd
)
node frontend/scripts/serve-dist.mjs --open
pause
