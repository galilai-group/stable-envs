import sys, os, ctypes, warnings
import pyglet
pyglet.options["headless"] = True
# Force pyglet down its "no EGL device" fallback, which uses eglGetDisplay(EGL_DEFAULT_DISPLAY).
# Under EGL_PLATFORM=surfaceless + LIBGL_ALWAYS_SOFTWARE=1 that is Mesa's llvmpipe, which is the
# only way to get a SECOND conforming GL implementation on this node: eglQueryDevicesEXT under the
# Mesa vendor lists only DRM hardware devices (card6-8) and we are not in the `video` group.
from pyglet.libs.egl import eglext
_orig = eglext.eglQueryDevicesEXT
def fake(max_devices, devices, num_devices):
    num_devices._obj.value = 0 if hasattr(num_devices, "_obj") else 0
    return 1
eglext.eglQueryDevicesEXT = fake
warnings.simplefilter("ignore")
try:
    from pyglet.gl import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
    w = pyglet.window.Window(width=1, height=1, visible=False)
    g = lambda k: ctypes.cast(glGetString(k), ctypes.c_char_p).value.decode()
    print("OK", g(GL_VENDOR), "|", g(GL_RENDERER), "|", g(GL_VERSION))
except Exception as e:
    print("FAIL", type(e).__name__, e)
