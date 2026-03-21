from src.knowledge_base import KnowledgeBaseManager
from src.gsm8k_manager import GSM8KManager
import os
import time

def run_full_training():
    """Ingest the ENTIRE GSM8K training set into the knowledge base."""
    train_file = "data/gsm8k/train.jsonl"
    if not os.path.exists(train_file):
        print(f"❌ 错误: 找不到训练集文件 {train_file}")
        return

    print("🚀 --- 开始 GSM8K 全量数据训练 (知识注入) ---")
    print(f"正在处理文件: {train_file}")
    
    start_time = time.time()
    
    # 1. Initialize
    kb = KnowledgeBaseManager()
    gsm8k = GSM8KManager(kb)
    
    # 2. Ingest without limit
    # Note: GSM8K train.jsonl has 7473 samples
    try:
        count = gsm8k.ingest_jsonl(train_file, limit=None)
        
        duration = time.time() - start_time
        print(f"\n✅ 全量训练完成！")
        print(f"入库总数: {count} 条解题金标准")
        print(f"总耗时: {duration:.2f} 秒 (约 {duration/60:.2f} 分钟)")
        print(f"平均速度: {count/duration:.2f} 条/秒")
        print(f"现在系统已具备完整的 GSM8K 知识背景。")
        
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")

if __name__ == "__main__":
    run_full_training()
