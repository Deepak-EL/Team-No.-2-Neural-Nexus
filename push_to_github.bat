@echo off
REM One-time upload of this folder to https://github.com/Deepak-EL/Team-No.-2-Neural-Nexus
REM Requires Git for Windows (https://git-scm.com/download/win). You will be asked to sign in to GitHub in a browser window.
cd /d "%~dp0"
git init
git config user.name "Deepak E L"
git config user.email "deepakel176@gmail.com"
git add .
git commit -m "Coral reef health classification: CNN vs ViT benchmark - code, results, models, paper"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/Deepak-EL/Team-No.-2-Neural-Nexus.git
git config http.postBuffer 524288000
git push -u origin main
pause
