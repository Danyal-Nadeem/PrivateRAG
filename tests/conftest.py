"""Session-wide environment defaults for the test suite.

These MUST be set before any src/huggingface import happens, regardless of
which test file pytest collects first:
- HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are captured by the libraries at
  import time, so setting them inside one test module is too late when a
  different module imports transformers first (order-dependent failures).
- TESTING=1 skips the embedder warm-up in the FastAPI lifespan.
- ALLOW_DEMO_ACCOUNTS=1 opts the suite into the shipped demo user store
  (data/users.json is gitignored and usually absent in CI/dev checkouts).

conftest.py is imported before any test module, which makes these flags
independent of file collection order.
"""
import os

os.environ.setdefault("TESTING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ALLOW_DEMO_ACCOUNTS", "1")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-0123456789")
