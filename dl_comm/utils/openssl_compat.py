"""OpenSSL symbol conflict repair for self-contained environments.

The problem this solves
-----------------------
A Python environment that bundles its own OpenSSL (any conda or pip build
carrying lib/libcrypto.so.3 and lib/libssl.so.3) can fail to import mpi4py on
a Cray system with:

    ImportError: .../env/lib/libcrypto.so.3: version `OPENSSL_3.0.1' not found
                 (required by /usr/lib64/libssl.so.3)

The mechanism is a diamond. Cray's libfabric, which the MPI layer loads to
reach the network, is linked against the *system* libssl. The interpreter has
already loaded the *environment's* libcrypto under the same soname, so the
dynamic linker hands that one to the system libssl, which then looks for a
version node the bundled library does not define. The bundled libcrypto in the
failing case defines up to OPENSSL_3.2.0 but omits OPENSSL_3.0.1, which the
system libssl 3.2.3 requires.

Preloading the system libcrypto alone does not work either: the environment's
own libssl then fails to find OPENSSL_3.3.0 in it. The conflict is genuinely
two-way, so the repair has to keep one OpenSSL pair consistent. Preloading the
system libcrypto *and* the system libssl together satisfies libfabric and
stops the environment's libssl from ever being loaded, and the standard library
ssl module binds to the system pair without complaint.

Why this is not a job-script fix
--------------------------------
LD_PRELOAD is read by the dynamic linker at process startup, so setting it
from inside a running interpreter has no effect on that interpreter. The repair
therefore re-executes the process once with the variable set. Doing it here,
rather than in each job script, means every launcher benefits: mpiexec, a bare
python invocation, or a test runner.

Scope
-----
The repair is deliberately narrow. It runs only when all of the following hold:

  * the platform is Linux;
  * both system libraries exist;
  * the running interpreter bundles its own libcrypto;
  * that bundled libcrypto lacks a version node the system libssl requires;
  * the process has not already been re-executed once.

On Aurora, on a machine whose Python uses the system OpenSSL, or anywhere the
symbols already agree, every call is a no-op and no re-exec happens. The guard
variable makes the re-exec strictly single-shot, so a misdetection cannot cause
a loop.
"""

import os
import subprocess
import sys
import sysconfig

_GUARD = "DL_COMM_OPENSSL_REPAIR"
_SYSTEM_LIBS = ("/usr/lib64/libcrypto.so.3", "/usr/lib64/libssl.so.3")


def _bundled_libcrypto():
    """Path to a libcrypto shipped inside this interpreter's tree, if any."""
    for key in ("LIBDIR", "BINDIR"):
        base = sysconfig.get_config_var(key)
        if not base:
            continue
        for candidate in (
            os.path.join(base, "libcrypto.so.3"),
            os.path.join(os.path.dirname(base), "lib", "libcrypto.so.3"),
        ):
            if os.path.exists(candidate):
                return candidate
    prefix_lib = os.path.join(sys.prefix, "lib", "libcrypto.so.3")
    return prefix_lib if os.path.exists(prefix_lib) else None


def _version_nodes(path):
    """Version definition names exported by a shared object.

    Uses objdump when available. A missing objdump yields an empty set, which
    makes the caller skip the repair rather than guess.
    """
    try:
        out = subprocess.run(
            ["objdump", "-p", path],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0:
        return set()
    return {
        tok
        for line in out.stdout.splitlines()
        for tok in line.split()
        if tok.startswith("OPENSSL_")
    }


def _required_nodes(path, from_soname="libcrypto.so.3"):
    """Version nodes `path` requires specifically from one dependency.

    objdump prints a "Version References" block of the form

        required from libcrypto.so.3:
          0x... 0x00 NN OPENSSL_3.0.1

    Only that section is read. Scanning every OPENSSL_ token in the file
    instead would also pick up the library's own definitions, which say
    nothing about what it needs from libcrypto and would cause the repair to
    fire on environments that do not need it.
    """
    try:
        out = subprocess.run(
            ["objdump", "-p", path],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0:
        return set()

    nodes = set()
    in_block = False
    for line in out.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("required from "):
            in_block = stripped[len("required from "):].rstrip(":") == from_soname
            continue
        if in_block:
            # A blank line or a new unindented heading ends the block.
            if not stripped or not line.startswith((" ", "\t")):
                in_block = False
                continue
            nodes.update(t for t in stripped.split() if t.startswith("OPENSSL_"))
    return nodes


def repair_if_needed():
    """Re-exec once with the system OpenSSL preloaded, when that is required.

    Returns False when nothing was done. Does not return when a re-exec
    happens, because the process image is replaced.
    """
    if os.environ.get(_GUARD):
        return False
    if not sys.platform.startswith("linux"):
        return False
    if not all(os.path.exists(p) for p in _SYSTEM_LIBS):
        return False

    bundled = _bundled_libcrypto()
    if not bundled:
        return False

    have = _version_nodes(bundled)
    need = _required_nodes(_SYSTEM_LIBS[1])
    if not have or not need:
        return False

    # Only act on a real gap: something the system libssl requires that the
    # bundled libcrypto does not define.
    if need <= have:
        return False

    preload = ":".join(_SYSTEM_LIBS)
    existing = os.environ.get("LD_PRELOAD", "")
    if existing:
        preload = preload + ":" + existing

    env = dict(os.environ)
    env["LD_PRELOAD"] = preload
    env[_GUARD] = "1"
    env["DL_COMM_OPENSSL_REPAIR_MISSING"] = ",".join(sorted(need - have))

    os.execve(sys.executable, _reexec_argv(), env)


def _reexec_argv():
    """Rebuild the interpreter command line for an exact re-exec.

    sys.argv is not the command line: the interpreter strips its own options,
    so "python -c CODE" and "python -m mod" both arrive with argv[0] rewritten
    and the flag gone. Re-execing with [executable] + sys.argv would therefore
    turn -c into a bare script path and -m into a filename, which is how this
    first failed. Python records the real invocation in sys.orig_argv, so use
    it when present and reconstruct the common cases otherwise.
    """
    orig = getattr(sys, "orig_argv", None)
    if orig:
        return list(orig)

    # Fallbacks for interpreters without sys.orig_argv (pre-3.10).
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and getattr(spec, "name", None):
        # Launched with -m package.module
        return [sys.executable, "-m", spec.name] + sys.argv[1:]
    return [sys.executable] + sys.argv


def repair_report():
    """One line describing what the repair did, for the run log."""
    if not os.environ.get(_GUARD):
        return None
    missing = os.environ.get("DL_COMM_OPENSSL_REPAIR_MISSING", "")
    return (
        "OpenSSL repair active: preloaded system libcrypto and libssl because "
        f"the bundled libcrypto lacked {missing or 'a required version node'}"
    )
