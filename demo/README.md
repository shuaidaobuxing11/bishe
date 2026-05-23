# 可视化 Demo（Streamlit + JSBSim/Stub）

与 `envs/`、`offline_rl/`、`online_rl/` 训练代码解耦，可单独扩展。

## 安装

在项目根目录 `code/`：

```bash
pip install -r requirements-demo.txt
```

若使用真实 JSBSim（可选）：

```bash
pip install jsbsim
# Windows 若 pip 无可用 wheel，需按 JSBSim 官方文档自行编译/安装
set JSBSIM_ROOT=C:\path\to\jsbsim-data-root
```

## 运行

```bash
streamlit run demo/streamlit_app.py
```

浏览器打开提示的本地地址即可。

- 不勾选「优先使用真实 JSBSim」时，使用内置 **Stub** 纵向通道仿真，无需任何 C++ 依赖。
- 勾选且环境可用时，尝试加载 `c172p` 等模型；失败会自动退回 Stub。

## 后续扩展建议

- 多页 `demo/pages/`：实验曲线、超参表、视频/GIF
- JSBSim：固定机型 XML、标准初始条件、与 RL 状态对齐的观测映射（仍放在 `demo/` 内）
