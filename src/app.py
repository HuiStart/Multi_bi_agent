import gradio as gr
import os
from .knowledge_base import KnowledgeBaseManager
from .workflow import BiDirectionalWorkflow

# Initialize system
kb_manager = KnowledgeBaseManager()
workflow = BiDirectionalWorkflow(kb_manager)

def upload_file(files):
    if not files:
        return "请先选择文件上传。"
    count = 0
    for file in files:
        count += kb_manager.add_document(file.name)
    return f"成功解析并上传 {len(files)} 个文件，共计 {count} 个知识分块。"

def process_query(query, academic_level):
    if not query:
        return "请输入提问内容。", "", "", "", ""
    
    result = workflow.process_query(query, academic_level)
    
    return (
        result["final_content"],
        result["gold_standard"],
        result["detection_report"],
        result["kb_source"],
        result["optimization_log"]
    )

def handle_feedback(query, final_content, gold_standard, feedback):
    if not feedback:
        return "请输入反馈内容。"
    log = workflow.handle_feedback(query, final_content, gold_standard, feedback)
    return log

# Gradio Interface
with gr.Blocks(title="多智能体双向幻觉抑制架构") as demo:
    gr.Markdown("# 🎓 多智能体双向幻觉抑制教育平台")
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📚 权威知识库管理")
            file_input = gr.File(label="上传教材/课标 (PDF/Docx/Txt)", file_count="multiple")
            upload_btn = gr.Button("🚀 上传并向量化")
            upload_status = gr.Textbox(label="上传状态", interactive=False)
            
            gr.Markdown("### 🔍 提问配置")
            academic_level = gr.Dropdown(
                ["小学", "初中", "高中", "大学"], 
                label="选择对应学段", 
                value="高中"
            )
            query_input = gr.Textbox(label="请输入您的教育类问题", lines=4, placeholder="例如：请讲解高中物理牛顿第二定律的应用场景")
            process_btn = gr.Button("✨ 开始双向抑制生成", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 📝 生成结果展示")
            final_output = gr.Markdown(label="最终合规内容")
            
            with gr.Accordion("📋 幻觉检测报告 & 权威来源", open=False):
                with gr.Row():
                    gold_standard_out = gr.Textbox(label="权威金标准", interactive=False, lines=5)
                    detection_report_out = gr.Textbox(label="幻觉检测报告", interactive=False, lines=5)
                kb_source_out = gr.Textbox(label="检索知识来源 (RAW)", interactive=False, lines=3)
            
            gr.Markdown("### 🔄 反向迭代与反馈")
            feedback_input = gr.Textbox(label="人工反馈/幻觉标注", placeholder="若发现幻觉，请在此输入正确知识，点击反馈触发反向迭代")
            feedback_btn = gr.Button("🔁 提交反馈并优化系统")
            optimization_log_out = gr.Textbox(label="系统优化日志", interactive=False, lines=5)

    # Event Handlers
    upload_btn.click(
        upload_file, 
        inputs=[file_input], 
        outputs=[upload_status]
    )
    
    process_btn.click(
        process_query,
        inputs=[query_input, academic_level],
        outputs=[final_output, gold_standard_out, detection_report_out, kb_source_out, optimization_log_out]
    )
    
    feedback_btn.click(
        handle_feedback,
        inputs=[query_input, final_output, gold_standard_out, feedback_input],
        outputs=[optimization_log_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
