from pymunk._chipmunk_cffi import ffi

for name in ("cpBody", "cpSpace", "cpArbiter", "cpContact", "cpCollisionInfo"):
    try:
        schema = ffi.typeof(name)
        print(
            name,
            "kind",
            schema.kind,
            "fields",
            [(field_name, str(field.type)) for field_name, field in (schema.fields or [])],
        )
    except Exception as exc:
        print(name, type(exc).__name__, str(exc))
