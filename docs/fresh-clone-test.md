# Fresh-clone test (R1, steps 7–8)

Run on 2026-11-04, Windows 11, Python 3.10, from `git clone https://github.com/angelinadegay/meta-ai-simgate.git`
into a new temporary folder for every case. Each server was started with `python server.py` and no `LLM_*` or
`SIMGATE_*` variables in the environment, so only the clone's own `.env` (or its absence) chose the mode.
The clones reused the machine's installed Python packages; the Docker build below installs them from scratch.

## Startup in each mode

| Case | Clone | Mode | Startup message (log and `/health`) | Header |
| --- | --- | --- | --- | --- |
| a) no `.env` | `1d14b98`, again at `f56bf5f` | mock | `Key missing/invalid: LLM_API_KEY is not set. Running in mock mode.` | Mock mode: rules, no model |
| b) `SIMGATE_MOCK_MODE=1` | `1d14b98` | mock | `SIMGATE_MOCK_MODE=1. Running in mock mode.` | Mock mode: rules, no model |
| c) valid key | `1d14b98` | model | `Connected to Meta Model API (muse-spark-1.3, muse-wonder-1.0)` | Meta models: muse-spark-1.3 · muse-wonder-1.0 |
| d) invalid key | `1d14b98` | mock | `Key missing/invalid: the Meta Model API rejected the key (401 Unauthorized). Running in mock mode.` | Mock mode: rules, no model |

No case crashed. Startup took 4–10 s, and no page errors appeared in the browser. In (d) the bad key was checked once
at startup; after that the app ran in mock mode normally.

![Header in mock mode](screenshots/fresh-clone/header-mock.png)
![Header in model mode](screenshots/fresh-clone/header-model.png)

## The three prompts, end to end

| Prompt | Mock mode (a) | Model mode (c) |
| --- | --- | --- |
| Drive a six-wheel rover to the summit on Gusev | Rules built the scene and its checks passed; after "run" the rover **reached the summit in 34.2 s**. 1 run, 0 model calls. | Muse Spark read it as a route question: the safe route reached the summit in 39.3 s, and driving straight there also worked (34.2 s). 16 runs, 1 model call. |
| In the apartment, can the rover get from the kitchen to the bedroom? | **Yes, by the ramp (41.7 s).** The stairs are too high: 0.17 m, and the rover handles about 0.06 m. 17 runs, 0 model calls. | Same answer and map. 17 runs, 1 model call. |
| Design a robot under 12 kg that can climb out of Victoria Crater | First run (`1d14b98`): **stopped in generation 3** with `ValueError: No unique bounded fallback mutation remains`, which is now fixed (below). Rerun in a new clone at `f56bf5f`: **all 5 generations ran, no errors**, 20 runs (the cap), 0 model calls. No design escaped under the rule-based Designer; the closest, G3-C, got 46.5% of the way before flipping at a 0.02 m step. The other two prompts and all three demos gave the same results as the first run. | All 5 generations ran: 55 runs, 9 model calls, no errors. No design passed verification: the closest, G2-B, got 100% of the way, but each apparent winner passed only 1 of 5 repeat runs, so the stop rule rejected it. The recorded demo found a winner; this live run didn't. |

Mock mode made 0 model calls in every case. In model mode the cost line counted the Meta calls. Planning used
`muse-spark-1.3`, and the Designer is configured to use `muse-wonder-1.0`. The step cards record the model by name,
not which child came from which call.

## Saved demo runs

All three replayed in both modes with 0 simulations and 0 model calls:
- Gusev safe route: watched the safe route, which reached the goal in 40.6 s.
- Apartment: watched the ramp route, 41.7 s.
- Crater evolution: served compressed to the browser, 5 generations.

## Errors found and fixed

| Error | Fix | Commit |
| --- | --- | --- |
| A Meta-model check raised `ModelUnavailable` before deciding whether the model was allowed | Reordered the check | `b34461b` |
| The GLB route test read the real InSight file | It now patches `server.INSIGHT_GLB` | `b34461b` |
| On Windows, the script runner's worker processes re-ran setup and overwrote the uploads folder | Added a `__mp_main__` guard; cleanup runs only in the parent | `b34461b` |
| Choosing another world in the Library panel kept the old terrain (e.g. Jezero's ground with Moon gravity) | A world with different ground builds a fresh scene | `1d14b98` |
| Mock-mode Victoria design stopped in generation 3: under the 12 kg and 0.8 × 0.5 m limits, the rule-based Designer ran out of unique children | After every failure-guided change, it tries small single-parameter steps inside the design space and the limits | `f56bf5f` |

Also seen, but harmless: when the headless browser closed, the server logged one `ConnectionResetError
[WinError 10054]`. This is Windows asyncio noise from a closed socket, and the app kept running.

## Tests

- Full suite on `f56bf5f`: **542 passed, 1 skipped** (17 min 51 s).
- The cases asked for are covered in `tests/test_run_modes.py` and `tests/test_meta_compliance.py` (36 passed):
  - mode selection with no key, the flag, a valid key and an invalid key;
  - the startup line and header label;
  - mock mode making zero API calls;
  - a missing InSight model using the fallback shape.
- New regression tests:
  - `test_library_selection_of_another_world_replaces_the_old_ground`;
  - `test_rule_mutations_still_fill_a_generation_when_parents_sit_at_the_design_limits`, which fails on the old code
    with the same `ValueError`.

## Terrain presets (shipped, no raw data needed)

| Preset | Label | Body | Patch |
| --- | --- | --- | --- |
| apollo15-hadley-rille | Apollo 15 / Hadley Rille edge | Moon | 160 m |
| apollo17-taurus-littrow | Apollo 17 / Taurus-Littrow valley floor | Moon | 160 m |
| arabia-northeast-mesas | Craters and Mesas in Northeast Arabia Terra | Mars | 81 m |
| ares3 | Acidalia Planitia (Ares 3 site) | Mars | 40 m |
| gale | Gale Crater | Mars | 40 m |
| gusev-columbia-hills | Gusev / Columbia Hills area (Spirit), U01 uncontrolled | Mars | 80 m |
| ius-chasma-floor | Light-Toned Deposits along Ius Chasma Floor | Mars | 81 m |
| jezero | Jezero Crater | Mars | 40 m |
| juventae-kieserite | Juventae Chasma Layered Kieserite Deposit | Mars | 80 m |
| meteor-crater-floor | Meteor Crater floor | Earth | 80 m |
| meteor-crater-rim | Meteor Crater rim | Earth | 80 m |
| victoria-opportunity | Victoria Crater (Opportunity), U01 uncontrolled | Mars | 80 m |

Plus the generated worlds and the indoor apartment.

## Assets

- **Terrain heightmaps:** ship in `simgate/terrains/` (305 KB). Every preset loaded in a clone with no raw NASA
  files.
- **G1 robot model:** ships in `third_party/`, MJCF and meshes only, with no trained policies.
- **NASA InSight lander:** not in git.
  - Without it, (a), (b) and (d) logged a warning and Mars landers used the simple fallback shape.
  - In (c), `python scripts/fetch_assets.py` downloaded it: 22.4 MB in 4 s, SHA-256 checked.
  - The Docker image fetches it during the build.
- **Demo runs:** `demos/`, 4.4 MB.
- **Home-page thumbnails:** `static/thumbnails/`, 0.13 MB.
- **`.env`:** git-ignored. In (c) the key was copied into the temporary clone, and `git status` stayed clean.

## Repo size

A fresh clone is 159 MB of files plus 51 MB of git history, and the repo has 313 tracked files.

## Docker

`docker compose build --no-cache` from a fresh clone: **233 s** (3.9 min), image 1.8 GB. Other test runs were using
the CPU at the same time, so a quiet machine should be a little faster. `docker compose up` with no `.env` started
in mock mode with the expected startup message; the InSight warning did not appear, since the image already has the
model. Docker Desktop's engine first returned HTTP 500 for every request and worked after one restart of Docker
Desktop. That was a local Docker problem, not SimGate.
