"""Module registry and per-module virtualenv management.

Kept out of ``scripts/`` because this package manages the *installation* of
analysis modules rather than serving requests. It was called ``platform/``
until it was renamed: that name shadows the standard library module of the
same name, which forced every call site to load it by file path.
"""
