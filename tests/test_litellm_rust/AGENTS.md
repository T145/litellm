# Scope

- This directory tests the Rust path end to end through the Python public API, and nothing else
- A test belongs here only when it cannot be written in Rust, meaning it needs
  - the Python public API
  - Python-owned state such as `litellm.cache` and callbacks
  - the process boundary (fork, threads, event loops)
- Anything checkable from the Rust side goes in the crate's own tests
- A test of the rollout switch itself (env, `litellm.rust(...)`, catalog rules) goes in `tests/unit/rust_bridge/`
- The wire is the local recording server. No test here reaches a real provider

# Rust only

- Treat every route's Python implementation as already removed
  - never call it on purpose
  - never run one request through both engines
  - never compare Python and Rust outputs, headers, cache keys or logs
- A test that would still pass with Python serving the call does not belong here
- Every test asserts native serving with `support.routing.assert_served_natively`

# Selecting the Rust path

- Use `support.routing.native_route(...)`, which prepends `RUST_REQUIRED` rules to `catalog.RULES` and restores them on exit
  - route folders apply it to every test through the autouse fixture in their `conftest.py`
  - a test outside a route folder wraps its whole body in one `native_route(...)` block
  - a child interpreter script assigns `catalog.RULES` once at its top
- Open it once per test, never per call. It swaps a module global, so overlapping blocks (one per task in an `asyncio.gather`) restore out of order and leak rules into later tests
- Never select a backend through `LITELLM_RUST`, `litellm.rust(...)` or any other rebinding of `catalog.RULES`
- Tests pass whatever the shipped rollout in `litellm/rust_bridge/catalog.py` says, including for a `PYTHON_ONLY` route

# Layout

- One folder per route (`ocr/`, `messages/`, `responses/`, `chat_completions/`), all using the same file names
  - `test_dispatch.py` the call reaches the native route
  - `test_requests.py` Python inputs that reach the wire
  - `test_failures.py` errors raised to the caller and rejection before provider IO
  - `test_settings.py` globals, env, named credentials and their precedence
  - `test_streaming.py` stream relay, early close and stream-time logging
  - `test_callbacks.py` the logger hook contract
  - `test_body_hooks.py` handler body hooks such as interceptors and agentic loops
  - `test_lifecycle.py` cancellation, concurrency, context and nested calls
  - `test_cache.py` the configured cache seen from this route
- `<route>/data.py` holds the route's request data and its `call`
- `<route>/conftest.py` owns the `server` fixture and the autouse `native` fixture
- `support/` holds only route-neutral helpers
- Never parametrize one test across routes
- Call `_native` directly only for a case the public API cannot reach
