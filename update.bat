@echo off

if exist ".git" (
    git fetch --prune
    git reset --hard origin/main
) else (
    git init
    git remote add origin https://github.com/nadare881/voras-webui-beta.git
    git fetch --prune
    git reset --hard origin/main
)

pause
