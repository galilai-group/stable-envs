/* Diagnostic only: CPython 3.12.9/Linux. Never part of the production adapter.
 * Read the actual atomic GIL locked flag without acquiring the GIL. The native
 * sampler lives on a separate allocated core. No ptrace or security changes.
 */
#define _GNU_SOURCE
#define Py_BUILD_CORE
#include <Python.h>
#include "internal/pycore_interp.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

static pthread_t thread;
static struct _gil_runtime_state *gil;
static atomic_int running, active, ready, affinity_error;
static atomic_ulong total, held, invalid;
static int sampler_cpu, interval_us;
static uint64_t rng;
static pthread_mutex_t counters_mutex = PTHREAD_MUTEX_INITIALIZER;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *sample_loop(void *unused) {
    (void)unused;
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(sampler_cpu, &cpus);
    atomic_store(&affinity_error, pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus));
    atomic_store(&ready, 1);
    while (atomic_load(&running)) {
        /* Jitter avoids synchronizing the sampler with the task's step cycle. */
        rng = rng * UINT64_C(6364136223846793005) + 1;
        long us = interval_us / 2 + (long)((rng >> 32) % (unsigned)interval_us);
        struct timespec delay = {0, us * 1000};
        nanosleep(&delay, NULL);
        if (atomic_load(&active)) {
            pthread_mutex_lock(&counters_mutex);
            int locked = _Py_atomic_load_relaxed(&gil->locked);
            if (atomic_load(&active)) {
                if (locked == 0 || locked == 1) {
                    atomic_fetch_add(&total, 1);
                    if (locked) atomic_fetch_add(&held, 1);
                } else {
                    atomic_fetch_add(&invalid, 1);
                }
            }
            pthread_mutex_unlock(&counters_mutex);
        }
    }
    return NULL;
}

static PyObject *init_probe(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long seed;
    if (!PyArg_ParseTuple(args, "iik", &sampler_cpu, &interval_us, &seed)) return NULL;
    if (strncmp(Py_GetVersion(), "3.12.9 ", 7) != 0 || interval_us < 100 || interval_us > 100000
        || sampler_cpu < 0 || sampler_cpu >= CPU_SETSIZE || atomic_load(&running)) {
        PyErr_SetString(PyExc_ValueError, "Requires CPython 3.12.9, valid core/rate, and no existing probe");
        return NULL;
    }
    gil = PyThreadState_Get()->interp->ceval.gil;
    if (!gil || _Py_atomic_load_relaxed(&gil->locked) != 1) {
        PyErr_SetString(PyExc_RuntimeError, "GIL must be held while initializing probe");
        return NULL;
    }
    rng = seed;
    atomic_store(&total, 0); atomic_store(&held, 0); atomic_store(&invalid, 0);
    atomic_store(&active, 0); atomic_store(&ready, 0); atomic_store(&running, 1);
    int err = pthread_create(&thread, NULL, sample_loop, NULL);
    if (err) {
        atomic_store(&running, 0);
        PyErr_SetString(PyExc_RuntimeError, "pthread_create failed"); return NULL;
    }
    while (!atomic_load(&ready)) sched_yield();
    if (atomic_load(&affinity_error)) {
        atomic_store(&running, 0); pthread_join(thread, NULL);
        PyErr_SetString(PyExc_RuntimeError, "sampler CPU affinity failed"); return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *window(PyObject *self, PyObject *args) {
    (void)self;
    int enabled;
    if (!PyArg_ParseTuple(args, "p", &enabled)) return NULL;
    pthread_mutex_lock(&counters_mutex);
    atomic_store(&active, enabled);
    unsigned long n = atomic_load(&total), h = atomic_load(&held), bad = atomic_load(&invalid);
    pthread_mutex_unlock(&counters_mutex);
    return Py_BuildValue("kkk", n, h, bad);
}

static PyObject *close_probe(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    atomic_store(&active, 0);
    if (atomic_exchange(&running, 0)) {
        Py_BEGIN_ALLOW_THREADS
        pthread_join(thread, NULL);
        Py_END_ALLOW_THREADS
    }
    Py_RETURN_NONE;
}

static PyObject *control_work(PyObject *self, PyObject *args) {
    (void)self;
    double seconds; int release;
    if (!PyArg_ParseTuple(args, "dp", &seconds, &release)) return NULL;
    PyThreadState *saved = release ? PyEval_SaveThread() : NULL;
    double end = now() + seconds;
    while (now() < end) { /* Native CPU work with a known GIL state. */ }
    if (release) PyEval_RestoreThread(saved);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"init", init_probe, METH_VARARGS, "Start sampler on allocated observer CPU."},
    {"window", window, METH_VARARGS, "Enable/disable task window and return cumulative counters."},
    {"close", close_probe, METH_NOARGS, "Join sampler before interpreter shutdown."},
    {"control_work", control_work, METH_VARARGS, "Validation workload with explicit GIL state."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "gil_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit_gil_probe(void) { return PyModule_Create(&module); }
