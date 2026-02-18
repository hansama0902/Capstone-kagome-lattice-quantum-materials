"""
Diagnostic Script for Backend Issues
后端问题诊断脚本
"""

import requests
import json

BASE_URL = "http://localhost:5000"

print("="*70)
print("Backend Diagnostic Tool")
print("="*70)

# Test 1: Health Check
print("\n[1/4] Testing health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"✅ Status Code: {response.status_code}")
    print(f"✅ Response: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Test 2: Simple DOS
print("\n[2/4] Testing simple DOS computation...")
try:
    response = requests.post(
        f"{BASE_URL}/api/compute_dos",
        json={'t_a': -0.3, 't_b': -0.2},
        timeout=10
    )
    print(f"✅ Status Code: {response.status_code}")
    data = response.json()
    print(f"✅ DOS points: {len(data['dos'])}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Generate Target
print("\n[3/4] Testing target generation...")
try:
    response = requests.post(
        f"{BASE_URL}/api/generate_target_dos",
        json={'t_a': -0.3, 't_b': -0.2},
        timeout=10
    )
    print(f"✅ Status Code: {response.status_code}")
    data = response.json()
    dos_target = data['dos']
    bins_target = data['bins']
    print(f"✅ Target generated: {len(dos_target)} points")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Test 4: Start Optimization (the problematic one)
print("\n[4/4] Testing optimization initialization...")
print("This is where the error occurred...")

try:
    payload = {
        'dos_target': dos_target[:100],  # 使用较小的数据集
        'bins_target': bins_target[:101],
        'n_initial': 2,  # 减少初始点
        'n_iterations': 1,
        'metric': 'mse'
    }
    
    print(f"Sending payload with:")
    print(f"  - dos_target: {len(payload['dos_target'])} points")
    print(f"  - bins_target: {len(payload['bins_target'])} points")
    print(f"  - n_initial: {payload['n_initial']}")
    
    print("\nSending request...")
    response = requests.post(
        f"{BASE_URL}/api/start_optimization",
        json=payload,
        timeout=60
    )
    
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"\nResponse Content (first 500 chars):")
    print(response.text[:500])
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"\n✅ Success! Response keys: {list(data.keys())}")
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON Decode Error: {e}")
            print("Server returned non-JSON content")
    else:
        print(f"\n❌ HTTP Error {response.status_code}")
        
except requests.exceptions.Timeout:
    print("❌ Request timeout (server too slow)")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Diagnostic complete!")
print("="*70)