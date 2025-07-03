#!/bin/bash

# TradingAgents CLI 启动脚本
# 自动设置必要的环境变量并运行CLI

# 设置项目根目录到Python路径
export PYTHONPATH="/Users/lingxiao/PycharmProjects/TradingAgents:$PYTHONPATH"

# 读取配置文件中的LLM提供商
if [ -f "config.yaml" ]; then
    LLM_PROVIDER=$(grep "^llm_provider:" config.yaml | sed 's/llm_provider:[[:space:]]*["'\'']*\([^"'\'']*\)["'\'']*$/\1/' | tr -d '"'\''')
    echo "检测到LLM提供商: $LLM_PROVIDER"
else
    LLM_PROVIDER="openai"
    echo "未找到config.yaml，默认使用OpenAI"
fi

# 根据LLM提供商设置API密钥
if [ "$LLM_PROVIDER" = "ollama" ]; then
    # 对于ollama，设置虚拟API密钥（程序需要但不会实际使用）
    export OPENAI_API_KEY="ollama-placeholder-key"
    echo "使用Ollama本地服务，设置虚拟API密钥"
elif [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: OPENAI_API_KEY 环境变量未设置"
    echo "请设置您的OpenAI API密钥:"
    echo "export OPENAI_API_KEY=your_api_key_here"
    echo ""
    echo "或者在运行此脚本前设置:"
    echo "OPENAI_API_KEY=your_api_key ./run_cli.sh"
    echo ""
    echo "继续使用测试密钥运行..."
    export OPENAI_API_KEY="sk-test-key"
fi

# 运行CLI
echo "启动 TradingAgents CLI..."
~/miniconda3/envs/testTrading/bin/python cli/main.py "$@"