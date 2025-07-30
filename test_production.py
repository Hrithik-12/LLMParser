import requests
import json
import time

def test_complete_system():
    base_url = "http://localhost:8000"
    token = "1946e5edb566278a8419b7529c46cd12f704f8d440a584ebd07201ec32fcbfd0"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing Complete Production System")
    print("=" * 60)
    
    # 1. Health Check
    print("1️⃣ Health Check...")
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 2. System Status
    print("\n2️⃣ System Status...")
    response = requests.get(f"{base_url}/api/v1/status", headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        status = response.json()
        print(f"   System Ready: {status['system_ready']}")
        print(f"   Documents Processed: {status['stats']['documents_processed']}")
    
    # 3. Main Test with Real Document
    print("\n3️⃣ Processing Real Document with Questions...")
    
    # Using a publicly accessible PDF with substantial content for testing
    test_data = {
        "documents": "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
        "questions": [
            "What is this document about?",
            "What are the main topics covered?",
            "Can you summarize the key points?",
            "What information is provided?",
            "What is the purpose of this document?"
        ]
    }
    
    print(f"   Document: {test_data['documents']}")
    print(f"   Questions: {len(test_data['questions'])}")
    
    start_time = time.time()
    response = requests.post(
        f"{base_url}/api/v1/hackrx/run",
        headers=headers,
        json=test_data,
        timeout=120  # 2 minute timeout for processing
    )
    processing_time = time.time() - start_time
    
    print(f"   Processing Time: {processing_time:.2f} seconds")
    print(f"   Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS! Here are the results:")
        print("=" * 60)
        
        for i, (question, answer) in enumerate(zip(test_data['questions'], result['answers']), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
            print("-" * 40)
        
        # Save results
        with open("production_test_results.json", "w") as f:
            json.dump({
                "request": test_data,
                "response": result,
                "processing_time": processing_time,
                "timestamp": time.time()
            }, f, indent=2)
        
        print(f"\n💾 Results saved to 'production_test_results.json'")
        print(f"🎉 System is working perfectly!")
        
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_complete_system()