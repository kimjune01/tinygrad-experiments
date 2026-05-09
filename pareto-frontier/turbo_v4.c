/*
 * tinygrad-turbo v4: C-native UOp type + pattern matching
 *
 * Defines CUOp as a CPython type with C-level struct fields.
 * Field access from C is a raw struct dereference (1ns), not PyMember_GetOne (15ns).
 *
 * Architecture:
 *   1. batch_flatten(list_of_uops) -> list of CUOps (one-time conversion)
 *   2. crewrite(cuop, pattern_table) -> int (pattern index, all in C)
 *   3. Python calls the callback only for the 10% that match
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

/* ── CUOp: C-native UOp with raw struct fields ── */

typedef struct {
    PyObject_HEAD
    int op;                /* Ops enum value (int) */
    int n_src;             /* len(src) */
    int src_ops[8];        /* src[i].op values (depth 1) */
    long dtype_id;         /* id(dtype) for equality comparison */
    long arg_int;          /* integer arg value */
    int arg_is_int;        /* whether arg is int */
    PyObject *py_uop;      /* back-pointer to original UOp (borrowed ref for callbacks) */
} CUOp;

static PyTypeObject CUOpType;

/* ── CUOp methods ── */

static void CUOp_dealloc(CUOp *self) {
    Py_XDECREF(self->py_uop);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyMemberDef CUOp_members[] = {
    {"op", T_INT, offsetof(CUOp, op), READONLY, "op value"},
    {"n_src", T_INT, offsetof(CUOp, n_src), READONLY, "number of sources"},
    {"dtype_id", T_LONG, offsetof(CUOp, dtype_id), READONLY, "dtype id"},
    {"arg_int", T_LONG, offsetof(CUOp, arg_int), READONLY, "int arg"},
    {"arg_is_int", T_INT, offsetof(CUOp, arg_is_int), READONLY, "is arg int"},
    {NULL}
};

static PyTypeObject CUOpType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "turbo_v4.CUOp",
    .tp_basicsize = sizeof(CUOp),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)CUOp_dealloc,
    .tp_members = CUOp_members,
};

/* ── Interned attribute names ── */

static PyObject *str_op = NULL, *str_src = NULL, *str_dtype = NULL, *str_arg = NULL;

static int init_strings(void) {
    if (str_op) return 0;
    str_op = PyUnicode_InternFromString("op");
    str_src = PyUnicode_InternFromString("src");
    str_dtype = PyUnicode_InternFromString("dtype");
    str_arg = PyUnicode_InternFromString("arg");
    return (str_op && str_src && str_dtype && str_arg) ? 0 : -1;
}

/* ── flatten_one: convert a single Python UOp to CUOp ── */

static CUOp* flatten_one(PyObject *uop) {
    CUOp *c = PyObject_New(CUOp, &CUOpType);
    if (!c) return NULL;

    /* uop.op -> int */
    PyObject *op = PyObject_GetAttr(uop, str_op);
    if (!op) { Py_DECREF(c); return NULL; }
    c->op = (int)PyLong_AsLong(op);
    Py_DECREF(op);

    /* uop.src -> extract ops */
    PyObject *src = PyObject_GetAttr(uop, str_src);
    if (!src) { Py_DECREF(c); return NULL; }
    c->n_src = (int)PyTuple_GET_SIZE(src);
    for (int i = 0; i < 8; i++) c->src_ops[i] = 0;
    for (int i = 0; i < c->n_src && i < 8; i++) {
        PyObject *si = PyTuple_GET_ITEM(src, i);
        PyObject *si_op = PyObject_GetAttr(si, str_op);
        if (si_op) {
            c->src_ops[i] = (int)PyLong_AsLong(si_op);
            Py_DECREF(si_op);
        }
    }
    Py_DECREF(src);

    /* dtype id */
    PyObject *dt = PyObject_GetAttr(uop, str_dtype);
    c->dtype_id = dt ? (long)((uintptr_t)dt) : 0;
    Py_XDECREF(dt);

    /* arg */
    PyObject *arg = PyObject_GetAttr(uop, str_arg);
    if (arg && PyLong_Check(arg)) {
        c->arg_int = PyLong_AsLong(arg);
        c->arg_is_int = 1;
    } else {
        c->arg_int = 0;
        c->arg_is_int = 0;
    }
    Py_XDECREF(arg);

    /* back-pointer */
    Py_INCREF(uop);
    c->py_uop = uop;

    return c;
}

/* ── batch_flatten: convert a list of UOps to CUOps ── */

static PyObject* batch_flatten(PyObject *module, PyObject *uop_list) {
    if (!PyList_Check(uop_list)) {
        PyErr_SetString(PyExc_TypeError, "expected list of UOps");
        return NULL;
    }

    Py_ssize_t n = PyList_GET_SIZE(uop_list);
    PyObject *result = PyList_New(n);
    if (!result) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *uop = PyList_GET_ITEM(uop_list, i);
        CUOp *c = flatten_one(uop);
        if (!c) { Py_DECREF(result); return NULL; }
        PyList_SET_ITEM(result, i, (PyObject*)c);
    }

    return result;
}

/* ── crewrite: pattern match on CUOp (all in C, no object protocol) ── */

/*
 * Match a CUOp against a pattern table.
 * pattern_table is a Python list of (required_len, strict, src0_ops_tuple, dtype_ids_tuple, arg_int, has_arg).
 * Returns the index of the first match, or -1.
 *
 * This is the HOT LOOP — all struct field access, no Python object protocol.
 */
static PyObject* crewrite(PyObject *module, PyObject *args) {
    CUOp *cuop;
    PyObject *pattern_list;

    if (!PyArg_ParseTuple(args, "O!O", &CUOpType, &cuop, &pattern_list)) return NULL;
    if (!PyList_Check(pattern_list)) {
        PyErr_SetString(PyExc_TypeError, "pattern_list must be a list");
        return NULL;
    }

    Py_ssize_t n_pats = PyList_GET_SIZE(pattern_list);

    for (Py_ssize_t i = 0; i < n_pats; i++) {
        PyObject *pat = PyList_GET_ITEM(pattern_list, i);  /* borrowed */
        /* pat = (required_len, strict, src0_ops, dtype_ids, arg_val, has_arg_check) */

        int req_len = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 0));
        int strict = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 1));

        /* len(src) check — raw struct access, ~1ns */
        if (strict && cuop->n_src != req_len) continue;
        if (!strict && cuop->n_src < req_len) continue;

        /* src[0].op check — raw struct access */
        PyObject *src0_ops = PyTuple_GET_ITEM(pat, 2);
        if (src0_ops != Py_None) {
            int found = 0;
            Py_ssize_t n_s0 = PyTuple_GET_SIZE(src0_ops);
            for (Py_ssize_t j = 0; j < n_s0; j++) {
                if (cuop->src_ops[0] == (int)PyLong_AsLong(PyTuple_GET_ITEM(src0_ops, j))) {
                    found = 1;
                    break;
                }
            }
            if (!found) continue;
        }

        /* dtype check — raw struct access */
        PyObject *dtype_ids = PyTuple_GET_ITEM(pat, 3);
        if (dtype_ids != Py_None) {
            int found = 0;
            Py_ssize_t n_dt = PyTuple_GET_SIZE(dtype_ids);
            for (Py_ssize_t j = 0; j < n_dt; j++) {
                if (cuop->dtype_id == PyLong_AsLong(PyTuple_GET_ITEM(dtype_ids, j))) {
                    found = 1;
                    break;
                }
            }
            if (!found) continue;
        }

        /* arg check — raw struct access */
        int has_arg = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 5));
        if (has_arg) {
            long arg_val = PyLong_AsLong(PyTuple_GET_ITEM(pat, 4));
            if (!cuop->arg_is_int || cuop->arg_int != arg_val) continue;
        }

        /* All checks passed — return this pattern index */
        return PyLong_FromLong(i);
    }

    return PyLong_FromLong(-1);
}

/* ── flatten_one exposed to Python ── */

static PyObject* py_flatten_one(PyObject *module, PyObject *uop) {
    CUOp *c = flatten_one(uop);
    if (!c) return NULL;
    return (PyObject*)c;
}

/* ── rewrite_one: flatten + match in a single C call ── */
/*
 * rewrite_one(uop, pattern_table) -> int
 *
 * Extracts fields from Python UOp, matches against pattern table,
 * returns index of first match or -1. ONE call, no intermediate CUOp.
 */
static PyObject* rewrite_one(PyObject *module, PyObject *args) {
    PyObject *uop, *pattern_list;
    if (!PyArg_ParseTuple(args, "OO", &uop, &pattern_list)) return NULL;

    /* Extract fields inline — no CUOp allocation */
    PyObject *op_obj = PyObject_GetAttr(uop, str_op);
    if (!op_obj) return NULL;
    int op = (int)PyLong_AsLong(op_obj);
    Py_DECREF(op_obj);

    PyObject *src = PyObject_GetAttr(uop, str_src);
    if (!src) return NULL;
    int n_src = (int)PyTuple_GET_SIZE(src);

    int src_ops[8] = {0};
    for (int i = 0; i < n_src && i < 8; i++) {
        PyObject *si = PyTuple_GET_ITEM(src, i);
        PyObject *si_op = PyObject_GetAttr(si, str_op);
        if (si_op) {
            src_ops[i] = (int)PyLong_AsLong(si_op);
            Py_DECREF(si_op);
        }
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

    /* Now match — all local variables, no object protocol */
    Py_ssize_t n_pats = PyList_GET_SIZE(pattern_list);
    for (Py_ssize_t i = 0; i < n_pats; i++) {
        PyObject *pat = PyList_GET_ITEM(pattern_list, i);

        int req_len = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 0));
        int strict = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 1));

        if (strict && n_src != req_len) continue;
        if (!strict && n_src < req_len) continue;

        PyObject *s0_ops = PyTuple_GET_ITEM(pat, 2);
        if (s0_ops != Py_None) {
            int found = 0;
            Py_ssize_t n_s0 = PyTuple_GET_SIZE(s0_ops);
            for (Py_ssize_t j = 0; j < n_s0; j++) {
                if (src_ops[0] == (int)PyLong_AsLong(PyTuple_GET_ITEM(s0_ops, j))) {
                    found = 1; break;
                }
            }
            if (!found) continue;
        }

        PyObject *dtype_ids = PyTuple_GET_ITEM(pat, 3);
        if (dtype_ids != Py_None) {
            int found = 0;
            Py_ssize_t n_dt = PyTuple_GET_SIZE(dtype_ids);
            for (Py_ssize_t j = 0; j < n_dt; j++) {
                if (dtype_id == PyLong_AsLong(PyTuple_GET_ITEM(dtype_ids, j))) {
                    found = 1; break;
                }
            }
            if (!found) continue;
        }

        int has_arg = (int)PyLong_AsLong(PyTuple_GET_ITEM(pat, 5));
        if (has_arg) {
            long a = PyLong_AsLong(PyTuple_GET_ITEM(pat, 4));
            if (!arg_is_int || arg_int != a) continue;
        }

        return PyLong_FromLong(i);
    }

    return PyLong_FromLong(-1);
}

/* ── Module definition ── */

static PyMethodDef methods[] = {
    {"batch_flatten", batch_flatten, METH_O, "Convert list of UOps to CUOps"},
    {"flatten_one", py_flatten_one, METH_O, "Convert single UOp to CUOp"},
    {"crewrite", crewrite, METH_VARARGS, "Pattern match on CUOp, returns index or -1"},
    {"rewrite_one", rewrite_one, METH_VARARGS, "Flatten + match in one call, returns index or -1"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "turbo_v4",
    "tinygrad-turbo v4: C-native UOp pattern matching", -1, methods
};

PyMODINIT_FUNC PyInit_turbo_v4(void) {
    if (init_strings() < 0) return NULL;
    if (PyType_Ready(&CUOpType) < 0) return NULL;

    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;

    Py_INCREF(&CUOpType);
    PyModule_AddObject(m, "CUOp", (PyObject*)&CUOpType);
    return m;
}
