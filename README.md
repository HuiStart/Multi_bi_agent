# 🎓 多智能体双向幻觉抑制教育平台 (Multi-Agent Bi-Directional Hallucination Suppression)

本项目实现了针对教育场景的「多智能体双向幻觉抑制架构」。该架构通过事前（生成中）与事后（迭代中）的双维度全链路防控机制，有效抑制事实性、逻辑性与教学适配性幻觉。

## 🚀 核心架构设计

### 1. 正向抑制链路 (Forward Suppression - 事前/生成中)
*   **权威知识锚定智能体**：从 RAG 检索到的权威知识库中提取金标准，划定知识边界。
*   **教育内容生成智能体**：严格基于金标准生成内容，确保适配学段认知水平。
*   **生成中并行校验智能体**：实时逐句校验生成内容，一旦偏离金标准立即拦截修正。

### 2. 反向抑制链路 (Backward Suppression - 生成后/迭代中)
*   **多维度幻觉检测智能体**：对生成内容进行事实、逻辑、适配性全方位扫描。
*   **溯源修正智能体**：基于检测报告定位幻觉根源，并依据金标准进行二次精准修正。
*   **反向迭代智能体**：将幻觉案例、修正结果及人工反馈，反向更新至知识库与 Prompt 体系。

## 📦 环境搭建

1.  **Python 版本**：推荐 Python 3.10+
2.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```
3.  **配置环境变量**：
    将 `.env.example` 重命名为 `.env` 并填写您的 API Key (OpenAI/DeepSeek/Tongyi/Wenxin)。

## 🖥️ 运行说明

### 1. 启动 Web 可视化界面 (Gradio)
```bash
python -m src.app
```
打开浏览器访问：`http://localhost:7860`

### 2. 命令行快速测试
```bash
python test_run.py
```

## 📁 目录结构
*   `src/`: 核心代码目录
    *   `agents.py`: 7 个智能体的角色定义与 Prompt
    *   `knowledge_base.py`: 基于 ChromaDB 的知识库管理
    *   `workflow.py`: 双向全链路流程调度
    *   `config.py`: 多模型配置与加载
    *   `app.py`: Gradio Web 交互界面
*   `data/`: 存放权威教材、课标文档 (PDF/Docx/Txt)
*   `db/`: 向量数据库存储目录
*   `requirements.txt`: 依赖项
*   `.env`: 配置文件

## 📝 实验示例
1.  在界面左侧上传 `data/physics_kb.txt`。
2.  在提问框输入：`请讲解高中物理牛顿第二定律。`
3.  点击「开始双向抑制生成」，观察生成过程中的各维度检测报告。
4.  若发现生成的例子超纲，可在「人工反馈」处输入：`该案例包含微积分，不适合高中。`
5.  点击「提交反馈并优化系统」，系统将自动更新规则库与 Prompt 策略。
