"""Compile generated C code to a shared library and load via ctypes."""
import subprocess, ctypes, tempfile, hashlib, os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


class UOpFlat(ctypes.Structure):
    _fields_ = [
        ("op", ctypes.c_int),
        ("n_src", ctypes.c_int),
        ("src_ops", ctypes.c_int * 8),
        ("dtype_id", ctypes.c_int64),
        ("arg_int", ctypes.c_int64),
        ("arg_is_int", ctypes.c_int),
    ]


def flatten_uop(uop) -> UOpFlat:
    flat = UOpFlat()
    flat.op = int(uop.op)
    flat.n_src = len(uop.src)
    for i, s in enumerate(uop.src[:8]):
        flat.src_ops[i] = int(s.op)
    flat.dtype_id = id(uop.dtype)
    flat.arg_int = uop.arg if isinstance(uop.arg, int) else 0
    flat.arg_is_int = 1 if isinstance(uop.arg, int) else 0
    return flat


def compile_c(c_source: str) -> ctypes.CDLL:
    """Compile C source to a shared library and return the loaded CDLL."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    source_hash = hashlib.md5(c_source.encode()).hexdigest()[:12]
    so_path = os.path.join(CACHE_DIR, f"turbo_{source_hash}.so")

    if not os.path.exists(so_path):
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(c_source)
            c_path = f.name
        try:
            subprocess.check_output(
                ["clang", "-O2", "-shared", "-fPIC", "-o", so_path, c_path],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed:\n{e.output.decode()}")
            raise
        finally:
            os.unlink(c_path)

    return ctypes.CDLL(so_path)


def bind_function(lib: ctypes.CDLL, fn_name: str):
    """Bind a match function from the compiled library.

    Returns a callable: fn(UOpFlat*) -> int
    """
    fn = getattr(lib, fn_name)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(UOpFlat)]
    return fn
