import json
import requests

API_URL  = "http://127.0.0.1:8000/query"

def load_golden_set(path):
    with open(path) as f:
        return json.load(f)
    

def evaluate_case(case):
    response = requests.post(API_URL,json = {'question': case['question']})
    result = response.json()
    
    answer = result.get('answer',"")
    sources = result.get('sources',[])
    
    if case["should_be_answerable"]:
        
        keyword_found = all(kw.lower() in answer.lower() for kw in case ['expected_keywords'])
        correctly_gated = len(sources) >0
        passed = keyword_found and correctly_gated
        
    else:
        passed = len(sources) ==0
    
    return {
        "question": case["question"],
        "passed": passed,
        "answer": answer,
        "sources": sources
    }
    
def run_eval(golden_set_path = 'data/raw/golden_set.json'):
    cases = load_golden_set(golden_set_path)
    results = [evaluate_case(c) for c in cases]
        
    passed_count = sum(r['passed'] for r in results)
    total = len(results)
        
    print(f"\n{'='*50}")
    print(f"EVAL RESULTS: {passed_count}/{total} passed ({passed_count/total*100:.0f}%)")
    print(f"{'='*50}\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['question']}")
        if not r["passed"]:
            print(f"       Answer: {r['answer'][:150]}")

    return results


if __name__ == "__main__":
    
    run_eval()
        
    
    