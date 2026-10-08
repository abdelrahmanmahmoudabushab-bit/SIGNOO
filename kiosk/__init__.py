"""SIGNOO kiosk runtime (B1-B10) — see docs/SYSTEM_DESIGN.md §2.

Everything here reads and writes `core.contracts` dataclasses at block boundaries,
per the house rule in `.claude/skills/signoo-engineering`: no camera handles, no
model objects, no sessions crossing a boundary.
"""
