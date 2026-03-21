import json
import os
import re
from src.knowledge_base import KnowledgeBaseManager
from src.workflow import BiDirectionalWorkflow
from dotenv import load_dotenv

load_dotenv()

def extract_final_answer(text):
    """Extract final numerical answer from GSM8K format (after ####)."""
    # Look for '#### [number]' pattern
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1)
    
    # Try looking for just the last number in case of different formats
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]
    
    return None

def test_gsm8k_accuracy(num_samples=5):
    """Evaluate system accuracy on GSM8K samples."""
    print(f"--- Starting GSM8K Accuracy Test ({num_samples} samples) ---")
    
    # 1. Initialize System
    kb_manager = KnowledgeBaseManager()
    workflow = BiDirectionalWorkflow(kb_manager)
    
    # 2. Load Test Data
    test_file = "data/gsm8k/test.jsonl"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    results = []
    correct_count = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            data = json.loads(line)
            question = data["question"]
            ground_truth_answer = data["answer"]
            expected_val = extract_final_answer(ground_truth_answer)

            print(f"\n[Test Case {i+1}] Question: {question[:100]}...")
            
            # Process via Bi-Directional Suppression
            output = workflow.process_query(question, academic_level="小学/初中")
            final_text = output["final_content"]
            predicted_val = extract_final_answer(final_text)

            is_correct = (predicted_val == expected_val)
            if is_correct:
                correct_count += 1
            
            hallucination_detected = "无幻觉" not in output["detection_report"]

            results.append({
                "question": question,
                "expected": expected_val,
                "predicted": predicted_val,
                "is_correct": is_correct,
                "hallucination_detected": hallucination_detected,
                "detection_report": output["detection_report"]
            })

            print(f"Expected: {expected_val}, Predicted: {predicted_val}, Correct: {is_correct}")
            print(f"Hallucination Detected: {hallucination_detected}")

    # 3. Final Summary
    accuracy = (correct_count / num_samples) * 100
    print("\n" + "="*50)
    print(f"GSM8K ACCURACY: {accuracy:.2f}% ({correct_count}/{num_samples})")
    print("="*50)

if __name__ == "__main__":
    # First, let's ensure we have some context by running a small ingestion
    # This is the "Training" (Knowledge Ingestion) part
    from src.gsm8k_manager import GSM8KManager
    kb = KnowledgeBaseManager()
    gsm8k = GSM8KManager(kb)
    
    print("Pre-loading some seeds into Knowledge Base (Training Phase)...")
    gsm8k.ingest_jsonl("data/gsm8k/train.jsonl", limit=20) 
    
    # Then Run Accuracy Test
    test_gsm8k_accuracy(num_samples=5)
