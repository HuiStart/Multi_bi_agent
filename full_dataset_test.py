import json
import os
import re
import time
from src.knowledge_base import KnowledgeBaseManager
from src.workflow import BiDirectionalWorkflow
from dotenv import load_dotenv

load_dotenv()

def extract_final_answer(text):
    """Extract final numerical answer from GSM8K format (after ####)."""
    # Pattern: #### [number]
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Try looking for just the last number in case of different formats
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return None

def run_full_evaluation(output_file="gsm8k_evaluation_report.json"):
    """Evaluate system accuracy on the ENTIRE GSM8K test set (1319 samples)."""
    print("🚀 --- 开始 GSM8K 全量数据评测 (Evaluation) ---")
    
    test_file = "data/gsm8k/test.jsonl"
    if not os.path.exists(test_file):
        print(f"❌ 错误: 找不到测试集文件 {test_file}")
        return

    # 1. Initialize
    kb_manager = KnowledgeBaseManager()
    workflow = BiDirectionalWorkflow(kb_manager)
    
    # 2. Results tracking
    all_results = []
    correct_count = 0
    total_processed = 0
    
    # Check for existing results to resume (checkpointing)
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as rf:
            try:
                all_results = json.load(rf)
                total_processed = len(all_results)
                correct_count = sum(1 for r in all_results if r["is_correct"])
                print(f"检测到断点，已从第 {total_processed} 条样本恢复...")
            except:
                print("加载断点文件失败，将从头开始评测。")
    
    # 3. Process test file
    start_time = time.time()
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Skip samples already processed
            if i < total_processed:
                continue
            
            data = json.loads(line)
            question = data["question"]
            ground_truth_answer = data["answer"]
            expected_val = extract_final_answer(ground_truth_answer)

            print(f"\n[{i+1}/1319] 正在处理问题: {question[:100]}...")
            
            try:
                # Process via Bi-Directional Suppression
                output = workflow.process_query(question, academic_level="小学/初中")
                final_text = output["final_content"]
                predicted_val = extract_final_answer(final_text)

                is_correct = (predicted_val == expected_val)
                if is_correct:
                    correct_count += 1
                
                hallucination_detected = "无幻觉" not in output["detection_report"]
                
                result_entry = {
                    "index": i + 1,
                    "question": question,
                    "expected": expected_val,
                    "predicted": predicted_val,
                    "is_correct": is_correct,
                    "hallucination_detected": hallucination_detected,
                    "detection_report": output["detection_report"],
                    "gold_standard": output["gold_standard"],
                    "final_content": final_text
                }
                all_results.append(result_entry)
                total_processed += 1

                # Checkpoint: Save every 5 samples
                if total_processed % 5 == 0:
                    with open(output_file, 'w', encoding='utf-8') as wf:
                        json.dump(all_results, wf, ensure_ascii=False, indent=2)
                    
                    # Performance tracking
                    elapsed = time.time() - start_time
                    avg_time = elapsed / total_processed
                    remaining = (1319 - total_processed) * avg_time
                    accuracy = (correct_count / total_processed) * 100
                    print(f"进度: {total_processed}/1319 | 准确率: {accuracy:.2f}% | 预计剩余时间: {remaining/60:.2f} 分钟")

            except Exception as e:
                print(f"❌ 处理样本 {i+1} 时出错: {e}")
                continue

    # 4. Final Finalization
    with open(output_file, 'w', encoding='utf-8') as wf:
        json.dump(all_results, wf, ensure_ascii=False, indent=2)
    
    duration = time.time() - start_time
    final_accuracy = (correct_count / total_processed) * 100 if total_processed > 0 else 0
    
    print("\n" + "="*50)
    print(f"🏁 全量评测完成！")
    print(f"最终准确率 (Accuracy): {final_accuracy:.2f}% ({correct_count}/{total_processed})")
    print(f"总耗时: {duration:.2f} 秒 (约 {duration/60:.2f} 分钟)")
    print(f"评测报告已保存至: {output_file}")
    print("="*50)

if __name__ == "__main__":
    run_full_evaluation()
