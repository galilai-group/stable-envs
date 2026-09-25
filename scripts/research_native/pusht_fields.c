/* Research-only numeric accessors for pinned Munk2D ade7ed72849e60289eefb7a41e79ae6322fefaf3.
 * Compiles against official headers; no copied integration/collision/solver logic.
 * Caller pointers exist only at this private FFI boundary, never in stored snapshots.
 */
#include "chipmunk/chipmunk_private.h"
#include <stdint.h>

int se_body_fields(cpBody *body, double *values, int write, int bias_only) {
    int n = 0;
#define VALUE(field) do { if (write) { if (!bias_only) body->field = values[n]; } else values[n] = body->field; ++n; } while (0)
    VALUE(cog.x); VALUE(cog.y); VALUE(p.x); VALUE(p.y);
    VALUE(v.x); VALUE(v.y); VALUE(f.x); VALUE(f.y); VALUE(a); VALUE(w); VALUE(t);
    VALUE(transform.a); VALUE(transform.b); VALUE(transform.c); VALUE(transform.d);
    VALUE(transform.tx); VALUE(transform.ty);
    VALUE(sleeping.idleTime);
#undef VALUE
#define VALUE(field) do { if (write) body->field = values[n]; else values[n] = body->field; ++n; } while (0)
    VALUE(v_bias.x); VALUE(v_bias.y); VALUE(w_bias);
#undef VALUE
    return n;
}

int se_active_arbiters(cpSpace *space, uint64_t *pairs, int capacity) {
    int n = space->arbiters->num;
    if (n > capacity) return -1;
    for (int i=0; i<n; ++i) {
        cpArbiter *arb = space->arbiters->arr[i];
        pairs[2*i] = arb->a->hashid;
        pairs[2*i+1] = arb->b->hashid;
    }
    return n;
}

int se_restore_active_order(cpSpace *space, const uint64_t *pairs, int count) {
    if (space->arbiters->num != count || count > 64) return -1;
    cpArbiter *ordered[64];
    for (int i=0; i<count; ++i) {
        ordered[i] = NULL;
        for (int j=0; j<count; ++j) {
            cpArbiter *arb = space->arbiters->arr[j];
            if (arb->a->hashid == pairs[2*i] && arb->b->hashid == pairs[2*i+1]) ordered[i] = arb;
        }
        if (!ordered[i]) return -2;
    }
    for (int i=0; i<count; ++i) space->arbiters->arr[i] = ordered[i];
    return 0;
}
