#!/usr/bin/env python3
"""demangle.py: how agg.py and layer.py find a C++ demangler on a lane.

`c++filt` is on the local lane's PATH; iks has no c++filt at all, only the ROCm SDK's `llvm-cxxfilt`
next to the clang++ the tree was built with (build/CMakeCache.txt). Nothing else differs per lane."""
import os, re, shutil

def path():
    """An executable that demangles, or None."""
    for name in ("c++filt", "llvm-cxxfilt"):
        p = shutil.which(name)
        if p: return p
    repo = os.environ.get("RAD_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        cache = open(os.path.join(repo, "build", "CMakeCache.txt")).read()
    except OSError:
        return None
    m = re.search(r"^CMAKE_(?:HIP|CXX)_COMPILER:[^=\n]*=(.+)$", cache, re.M)
    if not m: return None
    p = os.path.join(os.path.dirname(m.group(1).strip()), "llvm-cxxfilt")
    return p if os.access(p, os.X_OK) else None
