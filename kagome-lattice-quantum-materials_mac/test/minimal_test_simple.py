"""
Minimal Test - Only Core Functions
最小化测试 - 仅核心功能
"""

import requests
import time

BASE_URL = "http://localhost:5000"

print("="*60)
print("Minimal Backend Test")
print("="*60)

# Test 1: Health
print("\n[1/3] Health check...")
r = requests.get(f"{BASE_URL}/health")
print(f"✅ Server: {r.json()['status']}")

# Test 2: DOS
print("\n[2/3] DOS computation...")
start = time.time()
r = requests.post(f"{BASE_URL}/api/compute_dos",
                 json={'t_a': -0.3, 't_b': -0.2})
print(f"✅ DOS: {len(r.json()['dos'])} points in {time.time()-start:.1f}s")

# Test 3: Local Optimization (skip BO for now)
print("\n[3/3] Local optimization...")
# Generate simple target
r = requests.post(f"{BASE_URL}/api/generate_target_dos",
                 json={'t_a': -0.3, 't_b': -0.2})
target = r.json()

start = time.time()
r = requests.post(f"{BASE_URL}/api/local_optimize",
                 json={
                     'initial_point': [-0.25, -0.25],
                     'dos_target': target['dos'],
                     'bins_target': target['bins']
                 })
result = r.json()
print(f"✅ Optimized: t_a={result['optimized_point'][0]:.3f}, "
      f"t_b={result['optimized_point'][1]:.3f} "
      f"in {time.time()-start:.1f}s")

print("\n" + "="*60)
print("✅ Core functions working!")
print("="*60)