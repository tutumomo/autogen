@echo off
d:
cd \TOMO.Project\autogen

call ac
REM pip install -U pyautogen autogenstudio jupyter-ai ipython
start /b autogenstudio ui --port 8081

timeout /t 3
start http://127.0.0.1:8081/
start https://platform.flowgen.app/flows


