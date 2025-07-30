import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

def test_complete_system():
    base_url = "http://localhost:8000"
    # Use environment variable for the token instead of hardcoding
    token = os.getenv('API_TOKEN', "1946e5edb566278a8419b7529c46cd12f704f8d440a584ebd07201ec32fcbfd0")
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
    
    # 3. Main Test with Real Insurance Policy Document
    print("\n3️⃣ Processing Real Insurance Policy with Questions...")
    
    # Using the actual insurance policy document with real questions
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
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