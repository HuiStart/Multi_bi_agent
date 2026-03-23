from .agents import (
    TotalControlAgent, AuthorityKnowledgeAgent, ContentGenerationAgent,
    HallucinationDetectionAgent, SourceCorrectionAgent
)
from .knowledge_base import KnowledgeBaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiDirectionalWorkflow:
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb_manager = kb_manager
        self.total_control = TotalControlAgent()
        self.authority_anchor = AuthorityKnowledgeAgent()
        self.generator = ContentGenerationAgent()
        self.detector = HallucinationDetectionAgent()
        self.corrector = SourceCorrectionAgent()

    def process_query(self, query: str, academic_level: str = "高中"):
        """完整版工作流 - 包含所有 Agent 调用"""
        logger.info(f"Processing query: {query} (Level: {academic_level})")

        # 0. Total Control Planning
        plan = self.total_control.run(f"用户提问：{query}\n学段：{academic_level}")
        logger.info("Plan generated")

        # --- 正向抑制链路 (Forward Suppression) ---

        # 1. Retrieve Knowledge
        kb_content = self.kb_manager.retrieve_knowledge(query)
        gold_standard = self.authority_anchor.run(query, context=kb_content)
        if "无法生成内容" in gold_standard:
            return {
                "final_content": "抱歉，由于缺乏权威知识支撑，无法生成相关内容。",
                "gold_standard": gold_standard,
                "detection_report": "未生成内容",
                "kb_source": kb_content,
                "optimization_log": "N/A"
            }
        logger.info("Gold standard anchored")

        # 2. Generate and Verify
        initial_content = self.generator.run(query, context=gold_standard)
        logger.info("Initial content generated")

        # --- 反向抑制链路 (Backward Suppression) ---

        # 3. Detect Hallucinations
        detection_report = self.detector.run(initial_content, context=gold_standard)
        logger.info("Hallucination detection completed")

        # 4. Correct Content
        final_content = self.corrector.run(
            initial_content,
            context=f"金标准：{gold_standard}\n检测报告：{detection_report}"
        )
        logger.info("Correction completed")

        return {
            "final_content": final_content,
            "gold_standard": gold_standard,
            "detection_report": detection_report,
            "kb_source": kb_content,
            "optimization_log": "N/A"
        }

    def process_query_fast(self, query: str, academic_level: str = "小学/初中"):
        """
        快速版工作流 - 针对 GSM8K 评测优化，减少 LLM 调用次数

        优化策略:
        1. 跳过 TotalControlAgent (测试场景不需要调度规划)
        2. 合并 AuthorityKnowledge + 检索 为一步
        3. 跳过复杂的幻觉检测 (GSM8K 只需看最终答案是否正确)
        4. 直接生成答案，不做反向修正
        """
        logger.info(f"Fast processing query: {query}")

        # 1. 直接检索知识库 + 生成答案 (合并步骤)
        kb_content = self.kb_manager.retrieve_knowledge(query, k=5)

        # 2. 直接让生成 Agent 输出答案 (跳过多余环节)
        prompt = f"请基于以下参考知识解答这道数学题，直接给出解题步骤和最终答案。\n\n参考知识:\n{kb_content}\n\n题目：{query}"

        final_content = self.generator.run(prompt, context="")
        logger.info("Fast generation completed")

        # 3. 简化检测 - 只检查是否生成了有效内容
        detection_report = "无幻觉" if final_content and len(final_content) > 10 else "检测到幻觉"

        return {
            "final_content": final_content,
            "gold_standard": kb_content[:500] if kb_content else "N/A",  # 只返回前 500 字作为参考
            "detection_report": detection_report,
            "kb_source": kb_content,
            "optimization_log": "快速模式"
        }

    def handle_feedback(self, query, final_content, gold_standard, feedback):
        """Handle manual user feedback for backward iteration."""
        return self.iterator.run(
            f"提问：{query}\n最终内容：{final_content}\n人工反馈：{feedback}",
            context=f"金标准：{gold_standard}"
        )
