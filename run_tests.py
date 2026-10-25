import importlib.util, inspect, sys, traceback, pathlib
root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("t", root / "tests" / "test_book_code.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ok = bad = 0
for n, f in sorted(vars(m).items()):
    if n.startswith("test_") and callable(f):
        try: f(); ok += 1; print("PASS", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
print(f"passed={ok} failed={bad}"); sys.exit(1 if bad else 0)
