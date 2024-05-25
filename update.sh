#!/bin/bash

# 打开浏览器并访问GitHub仓库
open "https://github.com/tutumomo/autogen"
open "https://github.com/tutumomo/autogen-ui"
open "https://github.com/tutumomo/autogen-agi"
open "https://github.com/tutumomo/xforceai-ide"
open "https://github.com/tutumomo/flowgen"

# 等待一段时间以确保浏览器加载完毕
sleep 5

# 执行git pull
git pull

# 调用ac脚本（假设ac是一个Shell脚本文件）
./ac.sh

# 安装Python包
pip3 install -U pyautogen autogenstudio jupyter-ai ipython yfinance panda pyautogui
pip3 install -U python-dotenv langchain llama_index
pip3 install -U google-search-results colored duckduckgo-search pypdf2 Pillow docx2txt

# 克隆或更新GitHub仓库
clone_or_pull() {
    if [ ! -d "$1" ]; then
        git clone "$2"
    else
        cd "$1"
        git pull
        cd ..
    fi
}

clone_or_pull "autogen-ui" "https://github.com/tutumomo/autogen-ui.git"
clone_or_pull "autogen-agi" "https://github.com/tutumomo/autogen-agi.git"
clone_or_pull "xforceai-ide" "https://github.com/tutumomo/xforceai-ide.git"
clone_or_pull "flowgen" "https://github.com/tutumomo/flowgen.git"

echo "完成所有操作"
