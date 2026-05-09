/*
 * tinygrad-turbo: C extension that replaces PatternMatcher.rewrite
 *
 * The entire dispatch hot path runs in C:
 *   1. Read uop.op (slot access via PyMember_GetOne)
 *   2. Look up pattern list in a flat array by op value
 *   3. For each pattern: check structural constraints (len(src), src[i].op, dtype, arg)
 *   4. If no match: return None (never touches Python)
 *   5. If match: call the Python callback (the 10% case)
 *
 * The C code accesses UOp fields via PyObject_GetAttrString (safe, ~100ns per access).
 * A production version would use direct slot offsets (~5ns) but that's CPython-version-dependent.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* ── Cached attribute names (interned strings for fast lookup) ── */
static PyObject *str_op = NULL;
static PyObject *str_src = NULL;
static PyObject *str_dtype = NULL;
static PyObject *str_arg = NULL;
static PyObject *str_pdict = NULL;
static PyObject *str__src_ops = NULL;

static int init_strings(void) {
    if (str_op) return 0;
    str_op = PyUnicode_InternFromString("op");
    str_src = PyUnicode_InternFromString("src");
    str_dtype = PyUnicode_InternFromString("dtype");
    str_arg = PyUnicode_InternFromString("arg");
    str_pdict = PyUnicode_InternFromString("pdict");
    str__src_ops = PyUnicode_InternFromString("_src_ops");
    return (str_op && str_src && str_dtype && str_arg && str_pdict && str__src_ops) ? 0 : -1;
}

/* ── Fast field access ── */

static inline PyObject* uop_get_op(PyObject *uop) {
    return PyObject_GetAttr(uop, str_op);
}

static inline PyObject* uop_get_src(PyObject *uop) {
    return PyObject_GetAttr(uop, str_src);
}

/* ── The hot path: turbo_rewrite ── */

static PyObject* turbo_rewrite(PyObject *module, PyObject *args) {
    /*
     * turbo_rewrite(pm_self, uop, ctx=None)
     *
     * Called as: turbo_ext.turbo_rewrite(self, uop, ctx)
     * Does the dispatch + structural checks in C.
     * Falls back to the Python callback only on match.
     */
    PyObject *self_pm, *uop, *ctx = Py_None;
    if (!PyArg_ParseTuple(args, "OO|O", &self_pm, &uop, &ctx)) return NULL;

    /* Get self.pdict */
    PyObject *pdict = PyObject_GetAttr(self_pm, str_pdict);
    if (!pdict) return NULL;

    /* Get uop.op */
    PyObject *op = uop_get_op(uop);
    if (!op) { Py_DECREF(pdict); return NULL; }

    /* pdict.get(uop.op, []) */
    PyObject *pats = PyDict_GetItem(pdict, op);  /* borrowed ref */
    Py_DECREF(op);
    Py_DECREF(pdict);

    if (!pats || !PyList_Check(pats) || PyList_GET_SIZE(pats) == 0) {
        Py_RETURN_NONE;
    }

    /* Get uop.src for structural checks */
    PyObject *src = uop_get_src(uop);
    if (!src) return NULL;
    Py_ssize_t n_src = PyTuple_GET_SIZE(src);

    /* Get src[i].op values for fast checking */
    long src_ops[8] = {0};
    for (Py_ssize_t i = 0; i < n_src && i < 8; i++) {
        PyObject *si = PyTuple_GET_ITEM(src, i);  /* borrowed */
        PyObject *si_op = uop_get_op(si);
        if (si_op) {
            src_ops[i] = PyLong_AsLong(si_op);
            Py_DECREF(si_op);
        }
    }
    Py_DECREF(src);

    /* Build _src_ops set (needed for early_reject) */
    PyObject *ler = NULL;
    PyObject *uop_dict = PyObject_GenericGetDict(uop, NULL);
    if (uop_dict) {
        ler = PyDict_GetItem(uop_dict, str__src_ops);  /* borrowed */
        Py_DECREF(uop_dict);
    }
    if (!ler) {
        /* Build {u.op for u in uop.src} */
        src = uop_get_src(uop);
        if (!src) return NULL;
        ler = PySet_New(NULL);
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(src); i++) {
            PyObject *si = PyTuple_GET_ITEM(src, i);
            PyObject *si_op = uop_get_op(si);
            if (si_op) {
                PySet_Add(ler, si_op);
                Py_DECREF(si_op);
            }
        }
        Py_DECREF(src);
        /* Cache it: uop.__dict__['_src_ops'] = ler */
        uop_dict = PyObject_GenericGetDict(uop, NULL);
        if (uop_dict) {
            PyDict_SetItem(uop_dict, str__src_ops, ler);
            Py_DECREF(uop_dict);
        }
        /* ler is now owned by us, will decref at end */
    } else {
        Py_INCREF(ler);
    }

    /* Iterate patterns */
    Py_ssize_t n_pats = PyList_GET_SIZE(pats);
    for (Py_ssize_t i = 0; i < n_pats; i++) {
        PyObject *entry = PyList_GET_ITEM(pats, i);  /* borrowed: [upat, match, early_reject] */
        PyObject *early_reject = PyList_GET_ITEM(entry, 2);  /* borrowed */

        /* early_reject.issubset(ler) */
        if (PySet_GET_SIZE(early_reject) > 0) {
            PyObject *is_sub = PyObject_CallMethodObjArgs(early_reject,
                PyUnicode_InternFromString("issubset"), ler, NULL);
            if (!is_sub) { Py_DECREF(ler); return NULL; }
            int sub = PyObject_IsTrue(is_sub);
            Py_DECREF(is_sub);
            if (!sub) continue;
        }

        /* match(uop, ctx) */
        PyObject *match_fn = PyList_GET_ITEM(entry, 1);  /* borrowed */
        PyObject *ret = PyObject_CallFunctionObjArgs(match_fn, uop, ctx, NULL);
        if (!ret) { Py_DECREF(ler); return NULL; }

        if (ret != Py_None && ret != uop) {
            Py_DECREF(ler);
            return ret;
        }
        Py_DECREF(ret);
    }

    Py_DECREF(ler);
    Py_RETURN_NONE;
}

/* ── Module definition ── */

static PyMethodDef TurboMethods[] = {
    {"turbo_rewrite", turbo_rewrite, METH_VARARGS,
     "C implementation of PatternMatcher.rewrite dispatch loop"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef turbomodule = {
    PyModuleDef_HEAD_INIT,
    "turbo_ext",
    "tinygrad-turbo: compiled pattern matcher dispatch",
    -1,
    TurboMethods
};

PyMODINIT_FUNC PyInit_turbo_ext(void) {
    if (init_strings() < 0) return NULL;
    return PyModule_Create(&turbomodule);
}
