import json
import os
from .knowledge_base import KnowledgeBaseManager

class GSM8KManager:
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb_manager = kb_manager

    def ingest_jsonl(self, file_path, limit=None):
        """Ingest GSM8K JSONL file into the knowledge base."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GSM8K file not found at: {file_path}")

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data = json.loads(line)
                question = data.get("question", "")
                answer = data.get("answer", "")
                
                # Combine question and answer as a 'Gold Standard' entry
                # Format: Question: [Q] \n\n Gold Standard Answer: [A]
                content = f"问题: {question}\n\n权威解答步骤 (金标准): {answer}"
                self.kb_manager.update_knowledge(content, source=f"GSM8K_{os.path.basename(file_path)}")
                count += 1
        
        print(f"Successfully ingested {count} GSM8K entries into Knowledge Base.")
        return count

if __name__ == "__main__":
    # Example usage for manual ingestion
    from .knowledge_base import KnowledgeBaseManager
    kb = KnowledgeBaseManager()
    gsm8k = GSM8KManager(kb)
    gsm8k.ingest_jsonl("data/gsm8k/train.jsonl", limit=100) # Ingest first 100 as seeds
