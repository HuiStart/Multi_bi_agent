from .agents import (
    TotalControlAgent, AuthorityKnowledgeAgent, ContentGenerationAgent,
    InProcessVerificationAgent, HallucinationDetectionAgent,
    SourceCorrectionAgent, BackwardIterationAgent
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
        self.verifier = InProcessVerificationAgent()
        self.detector = HallucinationDetectionAgent()
        self.corrector = SourceCorrectionAgent()
        self.iterator = BackwardIterationAgent()

    def process_query(self, query: str, academic_level: str = "高中"):
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

        # 2. Generate and Verify (Iterative loop for in-process verification)
        # Simplified: one-pass generation with parallel verification for this implementation
        # In a real streaming app, this would be chunk-by-chunk
        initial_content = self.generator.run(query, context=gold_standard)
        logger.info("Initial content generated")
        
        verification_result = self.verifier.run(initial_content, context=gold_standard)
        if "拦截" in verification_result:
            logger.warning("Forward verification failed, regenerating...")
            initial_content = self.generator.run(
                f"修正要求: {verification_result}\n原内容: {initial_content}", 
                context=gold_standard
            )
        logger.info("Forward suppression completed")

        # --- 反向抑制链路 (Backward Suppression) ---
        
        # 3. Detect Hallucinations
        detection_report = self.detector.run(initial_content, context=gold_standard)
        logger.info("Hallucination detection completed")

        # 4. Correct Content
        final_content = self.corrector.run(
            initial_content, 
            context=f"金标准: {gold_standard}\n检测报告: {detection_report}"
        )
        logger.info("Correction completed")

        # 5. Backward Iteration (System Optimization)
        optimization_log = self.iterator.run(
            f"提问: {query}\n最终内容: {final_content}", 
            context=f"检测报告: {detection_report}\n金标准: {gold_standard}"
        )
        logger.info("Backward iteration completed")

        return {
            "final_content": final_content,
            "gold_standard": gold_standard,
            "detection_report": detection_report,
            "kb_source": kb_content,
            "optimization_log": optimization_log
        }

    def handle_feedback(self, query, final_content, gold_standard, feedback):
        """Handle manual user feedback for backward iteration."""
        return self.iterator.run(
            f"提问: {query}\n最终内容: {final_content}\n人工反馈: {feedback}",
            context=f"金标准: {gold_standard}"
        )
