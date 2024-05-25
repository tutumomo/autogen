REM https://github.com/tutumomo/autogen
@echo off
start https://github.com/tutumomo/autogen
start https://github.com/tutumomo/autogen-ui
start https://github.com/tutumomo/autogen-agi/tree/master
start https://github.com/tutumomo/xforceai-ide
start https://github.com/tutumomo/flowgen
start https://github.com/tutumomo/autogenstudio-skills
pause

git pull 

call ac

pip install -U pyautogen autogenstudio jupyter-ai ipython yfinance panda pyautogui python-dotenv langchain llama_index google-search-results colored duckduckgo-search pypdf Pillow docx2txt

REM pip install -U git+https://github.com/microsoft/autogen.git@gemini

pause

call sub_update "autogenstudio-skills"

call sub_update "autogen-ui"

call sub_update "autogen-agi"

call sub_update "xforceai-ide"

call sub_update "flowgen"

pause



