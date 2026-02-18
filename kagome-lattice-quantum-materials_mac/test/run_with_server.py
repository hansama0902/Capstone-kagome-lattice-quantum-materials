"""
VSCode Integrated Runner
自动化运行服务器和测试
"""

import subprocess
import time
import sys
import os
import requests

print("="*70)
print("Kagome Lattice - Automated Test Runner")
print("="*70)

# Start server in background
print("\n[1/4] Starting backend server...")
server_process = subprocess.Popen(
    [sys.executable, "app_pytorch.py"],
    cwd=os.path.dirname(__file__) or ".",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Wait for server to be ready
print("[2/4] Waiting for server to be ready...")
max_wait = 30
waited = 0
server_ready = False

while waited < max_wait:
    try:
        response = requests.get('http://localhost:5000/health', timeout=1)
        if response.status_code == 200:
            server_ready = True
            print(f"✅ Server ready after {waited} seconds")
            break
    except:
        pass
    time.sleep(1)
    waited += 1
    if waited % 5 == 0:
        print(f"   Still waiting... ({waited}s)")

if not server_ready:
    print("❌ Server failed to start within 30 seconds")
    server_process.kill()
    sys.exit(1)

# Run tests
print("\n[3/4] Running tests...")
print("="*70)
try:
    result = subprocess.run(
        [sys.executable, "minimal_test_simple.py"],
        cwd=os.path.dirname(__file__) or ".",
        capture_output=False,
        text=True
    )
    test_success = result.returncode == 0
except KeyboardInterrupt:
    print("\n\n⚠️  Tests interrupted by user")
    test_success = False

# Cleanup
print("\n[4/4] Cleaning up...")
print("Stopping server...")
server_process.terminate()
try:
    server_process.wait(timeout=5)
    print("✅ Server stopped")
except:
    server_process.kill()
    print("⚠️  Server force killed")

print("\n" + "="*70)
if test_success:
    print("✅ All tests completed successfully!")
else:
    print("⚠️  Tests completed with issues")
print("="*70)

sys.exit(0 if test_success else 1)