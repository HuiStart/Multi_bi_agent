from src.knowledge_base import KnowledgeBaseManager
from src.gsm8k_manager import GSM8KManager
import os

def run_training(limit=500):
    print(f"--- 开始 GSM8K 知识注入 (训练阶段) ---")
    print(f"目标入库条数: {limit}")
    
    # 1. 初始化
    kb = KnowledgeBaseManager()
    gsm8k = GSM8KManager(kb)
    
    # 2. 执行入库
    train_file = "data/gsm8k/train.jsonl"
    if os.path.exists(train_file):
        count = gsm8k.ingest_jsonl(train_file, limit=limit)
        print(f"✅ 训练完成！已将 {count} 条权威解题思路存入向量数据库。")
        print(f"现在的系统已具备处理类似数学问题的「金标准」依据。")
    else:
        print(f"❌ 错误: 找不到训练文件 {train_file}")

if __name__ == "__main__":
    # 建议初次训练注入 500 条
    run_training(limit=500)
