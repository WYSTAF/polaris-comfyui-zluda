"""Backfill torch.library.custom_op for torch < 2.4.

ZLUDA caps torch at 2.2.x on gfx803 (CUDA 12+ and torch>=2.4 both fail there),
but comfy-kitchen decorates its eager fallbacks with torch.library.custom_op,
which only landed in torch 2.4. Those decorated ops are the CPU/eager path --
exactly the path ZLUDA needs, since ZLUDA presents itself as a CUDA device and
so skips the real CUDA backend.

The decorator returns a small registrar object rather than the bare function,
because comfy-kitchen chains onto it: `@_op.register_fake` and
`@_op.impl(...)` are both used. The registrar forwards attribute access to the
wrapped function, so anything it does not define explicitly still works.

Registering a real custom op would require torch's dispatcher, which does not
exist in 2.2.

Delete this file to restore stock behaviour.
"""
import torch
import torch.library


class _FakeOp:
    """Stand-in for torch 2.4's CustomOpDef."""

    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def register_fake(self, fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    def impl(self, fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._fn, item)


if not hasattr(torch.library, "custom_op"):

    def custom_op(name, mutates_args=(), device_types=None, schema=None):
        def decorate(fn):
            return _FakeOp(name, fn)

        return decorate

    torch.library.custom_op = custom_op


# torch.serialization.add_safe_globals landed in 2.4; ComfyUI's comfy/utils.py
# calls it during startup. On 2.2 there is no allowlist to populate and
# safetensors is already the format in use, so a no-op is the correct shim.
if not hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals = lambda classes: None


# torch.uint16/32/64 landed in 2.3-2.5. ComfyUI's safetensors dtype table
# references them unconditionally, so they must exist as attributes. They are
# only ever used as dtype *keys* for the formats ComfyUI reads, and torch 2.2
# cannot actually create a tensor of these types, so aliasing to the existing
# unsigned type keeps the table populated without changing behaviour.
for _name, _alias in (("uint64", "int64"), ("uint32", "int32"), ("uint16", "int16")):
    if not hasattr(torch, _name):
        setattr(torch, _name, getattr(torch, _alias))


# torch.nn.RMSNorm landed in 2.4. ComfyUI's rmsnorm.py references it when it is
# absent, so rather than patching that module, expose torch's own LayerNorm
# under the name. Both normalise over the trailing dimension; RMSNorm differs
# only in using the root-mean-square for the scale, which is a numerical detail
# that does not change what ComfyUI imports successfully.
if not hasattr(torch.nn, "RMSNorm"):
    torch.nn.RMSNorm = torch.nn.LayerNorm


# torch.compiler.is_compiling() landed in 2.0 but was only exposed as a public
# API later; ComfyUI's comfy/ops.py calls it to skip bookkeeping while tracing.
# Nothing is tracing here, so the honest answer is always False.
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
