"""中介化算力交易网络：多智能体强化学习动态定价（完整实验代码）。"""
import os

# Windows 上 PyTorch 与 NumPy/MKL 各自带一份 Intel OpenMP 运行库，同时加载时会触发
# "OMP: Error #15 ... libiomp5md.dll already initialized" 并直接杀死 Jupyter 内核。
# 允许重复加载可以避免内核崩溃（PyTorch 官方 FAQ 给出的通用处理）。
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
