import json
import os
from .knowledge_base import KnowledgeBaseManager

class GSM8KManager:
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb_manager = kb_manager

    def ingest_jsonl(self, file_path, limit=None, batch_size=100, clear_before=False):
        """
        Ingest GSM8K JSONL file into the knowledge base using batch processing.

        Args:
            file_path: 输入文件路径
            limit: 限制处理的条数 (None 表示全部)
            batch_size: 批量处理的大小
            clear_before: 是否在注入前清空数据库
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GSM8K file not found at: {file_path}")

        # 可选：清空数据库
        if clear_before:
            print("正在清空数据库...")
            self.kb_manager.clear_database()

        print(f"当前数据库文档数量：{self.kb_manager.get_document_count()}")

        contents = []
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data = json.loads(line)
                question = data.get("question", "")
                answer = data.get("answer", "")

                # Combine question and answer as a 'Gold Standard' entry
                content = f"问题：{question}\n\n 权威解答步骤 (金标准): {answer}"
                contents.append((content, f"GSM8K_{os.path.basename(file_path)}"))
                count += 1

                # Process in batches
                if len(contents) >= batch_size:
                    print(f"Processing batch {count // batch_size}...")
                    self.kb_manager.update_knowledge_batch(contents)
                    contents = []

        # Process remaining items
        if contents:
            print(f"Processing final batch...")
            self.kb_manager.update_knowledge_batch(contents)

        print(f"Successfully ingested {count} GSM8K entries into Knowledge Base.")
        return count

if __name__ == "__main__":
    # Example usage for manual ingestion
    from .knowledge_base import KnowledgeBaseManager
    kb = KnowledgeBaseManager()
    gsm8k = GSM8KManager(kb)
    gsm8k.ingest_jsonl("data/gsm8k/train.jsonl", limit=100) # Ingest first 100 as seeds
