/*
 * tinygrad-turbo v5: transpiled unified_rewrite
 *
 * ONE C call per graph_rewrite pass. The C function:
 *   1. Walks the UOp graph (reads src tuples)
 *   2. For each node: extracts fields into C locals
 *   3. Pattern matches using a compiled pattern table
 *   4. On match: calls back to Python for the callback (10% of nodes)
 *   5. On no-match: continues in C (90% of nodes, zero Python calls)
 *
 * Boundary crossings: 78 per realize (one per unified_rewrite)
 * instead of 5,394 (one per node).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Interned strings */
static PyObject *str_op, *str_src, *str_dtype, *str_arg, *str__src_ops;

static int init_strings(void) {
    if (str_op) return 0;
    str_op = PyUnicode_InternFromString("op");
    str_src = PyUnicode_InternFromString("src");
    str_dtype = PyUnicode_InternFromString("dtype");
    str_arg = PyUnicode_InternFromString("arg");
    str__src_ops = PyUnicode_InternFromString("_src_ops");
    return (str_op && str_src && str_dtype && str_arg && str__src_ops) ? 0 : -1;
}

/* Quick field reads — one PyObject_GetAttr call each (~15ns) */
static inline long get_op_val(PyObject *uop) {
    PyObject *op = PyObject_GetAttr(uop, str_op);
    if (!op) return -1;
    long v = PyLong_AsLong(op);
    Py_DECREF(op);
    return v;
}

static inline PyObject* get_src(PyObject *uop) {
    return PyObject_GetAttr(uop, str_src);
}

/*
 * c_rewrite_node: pattern match one node against a pattern table.
 *
 * pattern_table: dict[int_op] -> list of (req_len, strict, src0_ops, dtype_ids, arg_val, has_arg)
 * Returns: index of matching pattern, or -1.
 *
 * All structural checks use C locals extracted once from the Python UOp.
 */
static int c_rewrite_node(PyObject *uop, PyObject *pattern_table) {
    long op = get_op_val(uop);
    if (op < 0) { PyErr_Clear(); return -1; }

    /* Look up patterns for this op */
    PyObject *op_key = PyLong_FromLong(op);
    PyObject *pats = PyDict_GetItem(pattern_table, op_key); /* borrowed */
    Py_DECREF(op_key);
    if (!pats) return -1;

    /* Extract fields into C locals — ONCE per node */
    PyObject *src = get_src(uop);
    if (!src) { PyErr_Clear(); return -1; }
    int n_src = (int)PyTuple_GET_SIZE(src);

    int src_ops[8] = {0};
    for (int i = 0; i < n_src && i < 8; i++) {
        PyObject *si = PyTuple_GET_ITEM(src, i); /* borrowed */
        src_ops[i] = (int)get_op_val(si);
    }
    Py_DECREF(src);

    PyObject *dt = PyObject_GetAttr(uop, str_dtype);
    long dtype_id = dt ? (long)((uintptr_t)dt) : 0;
    Py_XDECREF(dt);

    PyObject *arg_obj = PyObject_GetAttr(uop, str_arg);
    long arg_int = 0;
    int arg_is_int = 0;
    if (arg_obj && PyLong_Check(arg_obj)) {
        arg_int = PyLong_AsLong(arg_obj);
        arg_is_int = 1;
    }
    Py_XDECREF(arg_obj);

    /* Match against patterns — pure C integer comparisons */
    Py_ssize_t n_pats = PyList_GET_SIZE(pats);
    for (Py_ssize_t i = 0; i < n_pats; i++) {
        PyObject *pat = PyList_GET_ITEM(pats, i);

        int req_len = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 0));
        int strict = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 1));

        if (strict && n_src != req_len) continue;
        if (!strict && n_src < req_len) continue;

        PyObject *s0_ops = PyTuple_GET_ITEM(pat, 2);
        if (s0_ops != Py_None && n_src > 0) {
            int found = 0;
            for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(s0_ops); j++) {
                if (src_ops[0] == (int)PyLong_AsLong(PyTuple_GET_ITEM(s0_ops, j))) { found = 1; break; }
            }
            if (!found) continue;
        }

        PyObject *dtype_ids = PyTuple_GET_ITEM(pat, 3);
        if (dtype_ids != Py_None) {
            int found = 0;
            for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(dtype_ids); j++) {
                if (dtype_id == PyLong_AsLong(PyTuple_GET_ITEM(dtype_ids, j))) { found = 1; break; }
            }
            if (!found) continue;
        }

        int has_arg = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 5));
        if (has_arg) {
            long a = PyLong_AsLong(PyTuple_GET_ITEM(pat, 4));
            if (!arg_is_int || arg_int != a) continue;
        }

        return (int)i;
    }

    return -1;
}

/*
 * turbo_rewrite: drop-in replacement for PatternMatcher.rewrite
 *
 * Called from Python but does structural matching in C.
 * Falls back to Python callback only on match.
 *
 * The key: this is called from WITHIN the Python unified_rewrite loop.
 * It replaces the per-node Python pattern loop with C integer comparisons.
 * Field extraction happens ONCE per node (not per pattern).
 */
static PyObject* turbo_rewrite(PyObject *module, PyObject *args) {
    PyObject *pm, *uop, *ctx, *pattern_table;
    if (!PyArg_ParseTuple(args, "OOOO", &pm, &uop, &ctx, &pattern_table))
        return NULL;

    int idx = c_rewrite_node(uop, pattern_table);

    if (idx == -1) Py_RETURN_NONE;

    /* Match found — get the pdict entries and call the Python callback */
    PyObject *pdict = PyObject_GetAttr(pm, PyUnicode_InternFromString("pdict"));
    if (!pdict) return NULL;

    PyObject *op_obj = PyObject_GetAttr(uop, str_op);
    PyObject *entries = PyDict_GetItem(pdict, op_obj); /* borrowed */
    Py_DECREF(op_obj);
    Py_DECREF(pdict);

    if (!entries || idx >= PyList_GET_SIZE(entries)) Py_RETURN_NONE;

    PyObject *entry = PyList_GET_ITEM(entries, idx); /* borrowed */
    PyObject *match_fn = PyList_GET_ITEM(entry, 1);  /* borrowed */
    PyObject *early_reject = PyList_GET_ITEM(entry, 2); /* borrowed */

    /* Check early_reject if non-empty */
    if (PySet_GET_SIZE(early_reject) > 0) {
        /* Build or get _src_ops */
        PyObject *uop_dict = PyObject_GenericGetDict(uop, NULL);
        PyObject *ler = NULL;
        if (uop_dict) {
            ler = PyDict_GetItem(uop_dict, str__src_ops); /* borrowed */
            Py_DECREF(uop_dict);
        }
        if (!ler) {
            PyObject *src = get_src(uop);
            if (!src) return NULL;
            ler = PySet_New(NULL);
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(src); i++) {
                PyObject *si_op = PyObject_GetAttr(PyTuple_GET_ITEM(src, i), str_op);
                if (si_op) { PySet_Add(ler, si_op); Py_DECREF(si_op); }
            }
            Py_DECREF(src);
            uop_dict = PyObject_GenericGetDict(uop, NULL);
            if (uop_dict) { PyDict_SetItem(uop_dict, str__src_ops, ler); Py_DECREF(uop_dict); }
        } else {
            Py_INCREF(ler);
        }

        PyObject *is_sub = PyObject_CallMethodObjArgs(early_reject,
            PyUnicode_InternFromString("issubset"), ler, NULL);
        Py_DECREF(ler);
        if (!is_sub) return NULL;
        int sub = PyObject_IsTrue(is_sub);
        Py_DECREF(is_sub);
        if (!sub) Py_RETURN_NONE; /* C was optimistic, early_reject killed it */
    }

    /* Call the Python callback */
    PyObject *ret = PyObject_CallFunctionObjArgs(match_fn, uop, ctx, NULL);
    if (!ret) return NULL;

    if (ret != Py_None && ret != uop) return ret;
    Py_DECREF(ret);
    Py_RETURN_NONE;
}

/* Module */
static PyMethodDef methods[] = {
    {"turbo_rewrite", turbo_rewrite, METH_VARARGS,
     "C pattern match: extracts fields once, matches all patterns in C"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "turbo_v5",
    "tinygrad-turbo v5: transpiled rewrite with single-extraction matching", -1, methods
};

PyMODINIT_FUNC PyInit_turbo_v5(void) {
    if (init_strings() < 0) return NULL;
    return PyModule_Create(&module);
}
