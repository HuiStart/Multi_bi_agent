import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def process_single_sample(workflow, question, ground_truth_answer, index, total):
    """处理单个测试样本"""
    try:
        # 使用快速模式处理
        output = workflow.process_query_fast(question, academic_level="小学/初中")
        final_text = output["final_content"]
        predicted_val = extract_final_answer(final_text)
        expected_val = extract_final_answer(ground_truth_answer)

        is_correct = (predicted_val == expected_val)
        hallucination_detected = "无幻觉" not in output["detection_report"]

        return {
            "index": index,
            "question": question,
            "expected": expected_val,
            "predicted": predicted_val,
            "is_correct": is_correct,
            "hallucination_detected": hallucination_detected,
            "final_content": final_text
        }
    except Exception as e:
        print(f"❌ 处理样本 {index} 时出错：{e}")
        return {
            "index": index,
            "question": question,
            "expected": extract_final_answer(ground_truth_answer),
            "predicted": None,
            "is_correct": False,
            "hallucination_detected": True,
            "error": str(e)
        }

def run_full_evaluation(output_file="gsm8k_evaluation_report.json", use_fast_mode=True, max_workers=1):
    """
    Evaluate system accuracy on the ENTIRE GSM8K test set (1319 samples).

    Args:
        output_file: 输出报告文件路径
        use_fast_mode: 是否使用快速模式 (跳过部分 Agent 调用)
        max_workers: 并行处理的 worker 数量 (受限于 LLM 并发能力)
    """
    print("🚀 --- 开始 GSM8K 全量数据评测 (Evaluation) ---")
    print(f"模式：{'快速模式' if use_fast_mode else '完整模式'} | 并行 workers: {max_workers}")

    test_file = "data/gsm8k/test.jsonl"
    if not os.path.exists(test_file):
        print(f"❌ 错误：找不到测试集文件 {test_file}")
        return

    # 1. Initialize
    kb_manager = KnowledgeBaseManager()
    workflow = BiDirectionalWorkflow(kb_manager)

    # 2. Results tracking
    all_results = []
    correct_count = 0

    # Check for existing results to resume
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as rf:
            try:
                all_results = json.load(rf)
                start_index = len(all_results)
                correct_count = sum(1 for r in all_results if r.get("is_correct", False))
                print(f"✅ 检测到断点，已从第 {start_index + 1} 条样本恢复...")
            except:
                print("加载断点文件失败，将从头开始评测。")
                all_results = []
                start_index = 0

    # Load all test samples first
    test_samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_samples.append({
                "question": data["question"],
                "answer": data["answer"]
            })

    print(f"📊 测试集总数：{len(test_samples)} 条，已处理：{start_index} 条，剩余：{len(test_samples) - start_index} 条")

    # 3. Process test samples
    start_time = time.time()

    if max_workers > 1:
        # 并行处理模式
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(start_index, len(test_samples)):
                sample = test_samples[i]
                future = executor.submit(
                    process_single_sample,
                    workflow,
                    sample["question"],
                    sample["answer"],
                    i + 1,
                    len(test_samples)
                )
                futures[future] = i

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                if result.get("is_correct"):
                    correct_count += 1

                # 定期保存和输出进度
                if len(all_results) % 10 == 0:
                    save_results(all_results, output_file)
                    print_progress(all_results, correct_count, start_time, len(test_samples))
    else:
        # 串行处理模式 (默认)
        for i in range(start_index, len(test_samples)):
            sample = test_samples[i]
            print(f"\r[{i + 1}/{len(test_samples)}] 正在处理：{sample['question'][:50]}...", end="", flush=True)

            result = process_single_sample(
                workflow,
                sample["question"],
                sample["answer"],
                i + 1,
                len(test_samples)
            )
            all_results.append(result)

            if result.get("is_correct"):
                correct_count += 1

            # 定期保存和输出进度
            if (i + 1) % 10 == 0:
                save_results(all_results, output_file)
                print_progress(all_results, correct_count, start_time, len(test_samples))

    # 4. Final save
    save_results(all_results, output_file)

    duration = time.time() - start_time
    total_processed = len(all_results)
    final_accuracy = (correct_count / total_processed) * 100 if total_processed > 0 else 0

    print("\n" + "="*50)
    print(f"🏁 全量评测完成！")
    print(f"最终准确率 (Accuracy): {final_accuracy:.2f}% ({correct_count}/{total_processed})")
    print(f"总耗时：{duration:.2f} 秒 (约 {duration/60:.2f} 分钟)")
    print(f"评测报告已保存至：{output_file}")
    print("="*50)

def save_results(results, output_file):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as wf:
        json.dump(results, wf, ensure_ascii=False, indent=2)

def print_progress(all_results, correct_count, start_time, total_samples):
    """打印进度信息"""
    elapsed = time.time() - start_time
    total_processed = len(all_results)
    avg_time = elapsed / total_processed if total_processed > 0 else 0
    remaining = (total_samples - total_processed) * avg_time
    accuracy = (correct_count / total_processed) * 100 if total_processed > 0 else 0

    print(f"\n📈 进度：{total_processed}/{total_samples} | "
          f"准确率：{accuracy:.2f}% | "
          f"平均速度：{avg_time:.2f}秒/题 | "
          f"预计剩余：{remaining/60:.1f}分钟")

if __name__ == "__main__":
    # 快速模式 + 单线程 (适合本地 Ollama)
    run_full_evaluation(use_fast_mode=True, max_workers=1)

    # 如果有强大的 API 服务，可以开启并行:
    # run_full_evaluation(use_fast_mode=True, max_workers=4)
