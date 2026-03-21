from src.knowledge_base import KnowledgeBaseManager
from src.workflow import BiDirectionalWorkflow
import os
from dotenv import load_dotenv

load_dotenv()

def test_system():
    # 1. Initialize KB
    kb_manager = KnowledgeBaseManager()
    
    # 2. Add sample document
    print("--- 1. Uploading KB Document ---")
    kb_file = "data/physics_kb.txt"
    if os.path.exists(kb_file):
        count = kb_manager.add_document(kb_file)
        print(f"Uploaded {kb_file}, splits: {count}")
    else:
        print(f"Warning: {kb_file} not found.")

    # 3. Process Query
    print("\n--- 2. Processing Query: Newton's Second Law ---")
    workflow = BiDirectionalWorkflow(kb_manager)
    query = "请详细讲解高中物理牛顿第二定律，并给出一个解题示例。"
    result = workflow.process_query(query, "高中")

    print("\n[Final Content]:")
    print(result["final_content"])
    
    print("\n[Gold Standard]:")
    print(result["gold_standard"])
    
    print("\n[Hallucination Detection Report]:")
    print(result["detection_report"])
    
    print("\n[Optimization Log]:")
    print(result["optimization_log"])

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in .env or environment
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("QIANFAN_AK"):
        print("Please set your API key in .env first!")
    else:
        test_system()
