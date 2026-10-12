"""The kiosk test suite runs against the synthetic-trained `liu` checkpoint — there is
no real signer data yet — so it opts in to the setting a deployment leaves off
(kiosk/config.py::allow_synthetic). Set before kiosk.config is first imported, because
CONFIG is built at import time."""
import os

os.environ.setdefault("SIGNOO_ALLOW_SYNTHETIC", "1")
