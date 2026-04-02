#!/bin/bash

echo "=================================="
echo "仅仿真模式安装脚本"
echo "=================================="

# 检查 Python
python3 --version

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装最小依赖
pip install -r requirements_sim_only.txt

echo ""
echo "=================================="
echo "✓ 安装完成！"
echo "=================================="
echo ""
echo "快速开始:"
echo "  source venv/bin/activate"
echo "  python simple_sim_demo.py"