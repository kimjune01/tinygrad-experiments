"""Validate: run extract + compat + bench to confirm the fix is sound."""
import sys, os, subprocess

def run(script, *args):
    cmd = [sys.executable, script] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "1b-q6k"
    results = {}

    # 1. Confirm bug exists
    print("\n[1/3] Confirming lazy dequant chains exist...")
    rc = run("extract.py", model)
    results["extract"] = "PASS" if rc == 0 else "FAIL"

    # 2. Confirm output equivalence
    print("\n[2/3] Checking output equivalence...")
    rc = run("compat.py", model)
    results["compat"] = "PASS" if rc == 0 else "FAIL"

    # 3. Benchmark both modes
    print("\n[3/3] Benchmarking...")
    rc = run("bench.py", model)
    results["bench"] = "PASS" if rc == 0 else "FAIL"

    print(f"\n{'='*60}")
    print("Validation results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    all_pass = all(v == "PASS" for v in results.values())
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
