import subprocess
import sys
import os
import time

def run_worker(N):
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> Testing N = {N:,} rows...")
    
    # Path to the worker script (we create it dynamically or just use a flag)
    worker_script = os.path.join(os.path.dirname(__file__), "_worker_limit_test.py")
    
    # Run the worker process
    try:
        # Use sys.executable to run the uv python environment
        result = subprocess.run([sys.executable, worker_script, str(N)], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"[{time.strftime('%H:%M:%S')}] <<< [SUCCESS]: N = {N:,} passed!")
            # Print the VRAM/Time stats reported by the worker
            for line in result.stdout.split('\n'):
                if "STATS|" in line:
                    print("    " + line.split("STATS|")[1])
            return True
        else:
            print(f"[{time.strftime('%H:%M:%S')}] <<< [FAILED] (OOM): N = {N:,} crashed the worker.")
            # Print last few lines of stderr to confirm OOM or other error
            error_lines = [line for line in result.stderr.split('\n') if line.strip()]
            if error_lines:
                print("    Error tail: " + " | ".join(error_lines[-2:]))
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[{time.strftime('%H:%M:%S')}] <<< [TIMEOUT]: N = {N:,} took too long (assuming failed).")
        return False

def run_binary_search():
    print("=" * 60)
    print("EXTREME ROW LIMIT: BINARY SEARCH TEST")
    print("=" * 60)
    print("This script will find the exact maximum row limit of ZS-ISAB")
    print("before your physical System RAM or VRAM gives out.")
    print("=" * 60)
    
    low = 10_000
    high = 5_000_000  # Start with 5 million as the upper bound
    
    # First, test the low bound to ensure the script works
    if not run_worker(low):
        print(f"CRITICAL ERROR: Failed on the minimum bound ({low}). Something is fundamentally broken.")
        return
        
    # Then test the high bound. If it passes, we don't need to binary search, the limit is > 5M
    if run_worker(high):
        print(f"\n[INCREDIBLE]: The system successfully processed {high:,} rows!")
        print("Your limit is higher than 5,000,000 rows.")
        return
        
    print(f"\nUpper bound ({high:,}) failed. Starting binary search to find the exact limit...")
    
    threshold = 50_000 # Stop searching when the gap is less than 50k
    
    while high - low > threshold:
        mid = (low + high) // 2
        
        success = run_worker(mid)
        
        if success:
            low = mid
        else:
            high = mid
            
    print("\n" + "=" * 60)
    print("--- BINARY SEARCH COMPLETE ---")
    print("=" * 60)
    print(f"TRUE MATHEMATICAL ROW LIMIT (on this hardware): ~ {low:,} rows")
    print(f"TabPFN v3 Official Limit: ~ 16,384 rows")
    print(f"Scaling Multiplier: {low / 16384:.1f}x larger than vanilla!")
    print("=" * 60)

if __name__ == '__main__':
    run_binary_search()
