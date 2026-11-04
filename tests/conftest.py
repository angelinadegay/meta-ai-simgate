import os

# Tests run offline in mock mode even when .env holds a real key; SIMGATE_MOCK_MODE=0 runs them against the API.
os.environ.setdefault("SIMGATE_MOCK_MODE", "1")
