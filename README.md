# SimGate

**A plain-language gateway to physical simulation.** Describe a scenario in plain words, for example
*"Can a robot dog walk to the top of a hill on Mars?"* A team of AI agents then picks the right simulator,
builds the world, checks the physics against reference values, runs it live in your browser and explains
what happened.

SimGate is for people who aren't simulation engineers: students, researchers, hobbyists and engineers
from other fields. They can test an idea against real physics without learning a simulator first.

| Domain | Simulator | Good for |
| --- | --- | --- |
| `robot_terrain` | MuJoCo | Wheeled and legged robots on generated or real terrain: traction, tipping, climbing, reaching a goal, gravity on other worlds |
| `structure` | OpenSees | Building frames in earthquakes: sway, story drift, resonance, damage level |

---

## Project status

This is a **prototype**. It was built before the hackathon building period started, to test the idea and
learn the tools.

| | Date |
| --- | --- |
| Prototype started | ____ |
| Prototype state described in this README | ____ |
| Hackathon building period | ____ to ____ |
| Submission | ____ |

---

## What we built so far

### 1. Agent pipeline

```
prompt ─► Planner ─► SceneSpec ─► Router ─► Builder ─► Checker ─► (you approve) ─► Simulator ─► Explainer
          (model)    (shared      (picks     (writes     (reference   "Does this      (streams      (plain-language
                      format)     simulator)  sim code)   values)      look right?")   frames)       explanation)
```

- **Planner** turns a prompt into a structured scene spec. It also handles follow-up edits such as
  *"make the slope steeper"* or *"use six legs"*. Any value the user doesn't give is filled from reference
  data and listed as an assumption, so nothing is hidden.
- **Router** reads the spec and picks the simulator: MuJoCo for robots, OpenSees for buildings.
- **Builder** writes the simulation code. By default it uses a verified template. In `llm` mode the model writes
  the MuJoCo XML itself. If MuJoCo rejects the code, the error goes back to the model to fix, and after a few
  failed tries the Builder falls back to the template.
- **Checker** verifies the scene before anything runs (see [Physics checks](#physics-checks)).
- **Checkpoint:** the user approves the scene ("Does this look right?") before it runs. One click gives exactly
  one run, and an explicit "run again" repeats it.
- **Simulator** streams frames live to the browser.
- **Explainer** describes what happened in plain language, using only measured evidence. It marks anything it
  infers with "likely" and suggests one thing to try next.
- **Mock mode:** with no model configured, simple rules stand in for every agent, so the whole pipeline works
  offline.

### 2. MuJoCo backend (robots on terrain)

- **Worlds:** Earth, Moon, Mars, Titan, Europa and Enceladus, with reference gravity values. Saturn and Jupiter
  have no solid surface, so prompts about them map to Titan and Europa.
- **Surfaces:** rock, regolith, gravel, sand, mud and ice, each with a reference friction value.
- **Terrain:** flat, hills, rocky, slope (any angle), craters and **real terrain** (see section 4). Small bumps
  fade out around the start point so the robot lands cleanly, without making the terrain artificially steep there.
- **Robots:**
  - Rover with 4 or 6 wheels.
  - Walker with 4 legs ("robot dog", trot gait) or 6 legs ("spider", tripod gait). The gait settings were
    tuned with a parameter sweep.
  - Wheels and legs touch the ground but not each other. This prevents a crash in the physics we found with
    the 6-wheel rover.
- **Diagnostics** that explain a failure:
  - When a fall started, which way the robot fell, and when it was fully over.
  - Slope where it fell: how steep, uphill or downhill ahead, the sideways tilt, and which side was downhill.
  - Steepest slope crossed on the path.
  - Foot and wheel slip: the share of contact time spent sliding, overall and in the last second before a fall.
  - A trace of the robot's tilt over time.
  - Flags for physics blow-ups and for robots that went off the terrain edge.

### 3. OpenSees backend (buildings in earthquakes)

- A 2D steel moment frame with a chosen number of stories, bays and member sizes (light, medium or heavy),
  shaken by an earthquake with a set peak ground acceleration and frequency.
- Natural period, resonance with the shaking, roof sway, story drift and a damage estimate based on drift.
- The 3D viewer exaggerates the sway so it's visible and colors the columns by drift.

### 4. Real terrain from NASA Mars data

- `scripts/hirise_to_simgate.py` converts NASA HiRISE elevation models into SimGate terrain. These are about
  1 m per pixel and public domain (CC0 1.0), hosted on AWS Open Data. The script reads only the window it needs
  over the internet, so no download of the full file and no AWS account are needed.
- It automatically picks the most varied patch a robot could still cross (typical slope up to 22°), fills
  small gaps in the data and keeps north and south the right way round.
- Presets tested so far, each a 40 m square:

  | Preset | Place | Relief | Typical slope |
  | --- | --- | --- | --- |
  | `jezero` | Jezero Crater (Perseverance landing area) | 15.5 m | 18.2° |
  | `gale` | Gale Crater (Curiosity's area) | 17.5 m | 21.5° |
  | `ares3` | Acidalia Planitia (the *Ares 3* site from *The Martian*) | 13.3 m | 19.3° |

- Real terrain sets the world and surface from the data, credits the source, and is checked against the
  source data before the robot runs.
- `scripts/make_test_dtm.py` makes a **fake** file in the same format for offline testing. It's always
  labelled as synthetic.

### 5. Start and goal missions

- Give a robot somewhere to go:
  - In words: *"walk to the summit"* or *"drive from the west to the crater floor"*.
  - With coordinates: *"goal at (5, 3)"*.
  - By clicking: press **Set start** or **Set goal** and click the terrain. This doesn't call the model.
- Named places (lowest, highest, center, north, south, east, west) are found on the actual terrain, so
  the model never guesses coordinates.
- The robot starts facing the goal and steers straight at it. The rover turns by spinning its wheels at
  different speeds; the walker takes shorter steps on the inside of the turn. It does not plan a route around
  steep ground.
- Results show whether it reached the goal and how the run ended (reached, fell, stuck, out of time or went off
  the edge), plus time to the goal, closest and final distance, path length and path efficiency.
- The path is drawn on the terrain: green if the robot arrived, red if it didn't. A run stops 1.5 s after the
  robot arrives.

### 6. Obstacle courses

- Describe a course in words: *"10 small rocks, a big step, stairs and a narrow trench"*. Eight kinds are
  available: rock, boulder, step, stairs, gap (trench), ramp, wall and log. Each comes in small, medium or
  large, or an exact height.
- **Placement is done in code, never by the model.** Obstacles go on the line from start to goal in the order
  asked for, with the start and goal kept clear. They can also be scattered or put at a named place. Boulders are
  each placed separately; rocks form a field. A course gets a finish line automatically, and the terrain grows
  if the course doesn't fit.
- Rocks, steps, stairs, ramps, walls and logs are solid shapes resting on the terrain. Gaps are cut into the
  terrain itself and marked with yellow edges. Each gap's real opening is measured on the terrain grid and
  reported, not just the size asked for.
- **Checks:** obstacles are clear of the start and goal, and there's a warning when an obstacle in the way is
  beyond this robot's measured limits.
- **Results:** obstacles passed ("2 of 3"), which ones the robot touched and when, and which one stopped it. That
  obstacle glows red in the 3D view, and the badge says, for example, "Missed by 4.8 m · stuck at the gap".
- **Measured robot limits** (default sizes, flat Earth ground). The explainer compares obstacles with these:

  | Robot | Climbs steps up to | Crosses gaps |
  | --- | --- | --- |
  | 4-wheel rover | ~0.09 m | ~0.23 m opening, not 0.48 m |
  | 6-wheel rover | ~0.13 m reliably, single steps up to ~0.2 m (rear wheels push the front up) | same as 4-wheel |
  | 6-legged walker | ~0.05 m | no (feet drop in) |
  | 4-legged walker | ~0.03 m | no |

- The terrain grid was made finer (160 × 160, 0.13 m cells on a 20 m terrain) so gaps and edges are sharp
  enough. The earlier stability matrix was rerun on the finer grid (84 runs) with the same result.

### 7. Browser UI

- A chat timeline shows each agent working live, with example prompts and one-tap chips for real terrain.
- A three.js 3D view shows live playback, a follow camera, a play/pause button and a timeline scrubber.
- Cards show the physics checks and the results, with plain-language labels.
- **Show code** shows the simulation code the Builder wrote.

### 8. Stress test (robots)

- With a robot scene open, click **Stress test** or ask *"stress test"*, *"how robust?"*,
  *"test it under different conditions"*, or *"will it work on Mars?"*. The unchanged scene runs first
  as **Baseline**, followed by up to 23 variations in parallel. Each run has a 120-second timeout;
  failed or timed-out runs appear as errors instead of aborting the batch.
- The planner can vary gravity, surface, friction, added payload, slope angle (slope terrain only),
  obstacle scale (when obstacles exist), lateral push and terrain seed. Surface changes reset friction
  to the surface preset. Gravity changes the gravity value directly and does not change the body/world,
  including for real terrain. Payload mass is added to the robot's mass; a push lasts 0.3 seconds near
  the middle of the run.
- A run passes if it reaches its goal. Without a goal, it passes if it stays upright and makes at least
  1 m of forward progress. The report card lists each outcome and measured progress or goal distance,
  maximum tilt and slip, followed by a summary of the pattern across runs.
- The ghost view overlays the baseline and same-geometry variations on one timeline; clicking an
  available report row replays that run alone. Robot poses are recorded at 15 fps, rounded to three
  decimals and compressed, with a 4 MiB ghost-track budget (the complete batch payload stays below
  about 5 MB). Slope, obstacle-scale and seed variations change geometry, so they remain in the report
  but are marked **not in ghost view**. The follow camera tracks the baseline.

### 9. Model support

The prototype works with any OpenAI-compatible endpoint, configured in `.env`:

| Provider | Base URL | Models |
| --- | --- | --- |
| Meta Model API | `https://api.meta.ai/v1` | Muse Spark for planning and explaining, Muse Wonder for building |
| Claude (Anthropic) | `https://api.anthropic.com/v1/` | Any current Claude model |
| OpenAI, OpenRouter or local Ollama | see `.env.example` | Any chat model |

Without a key, it runs in mock mode.

### 10. Tests

`python tests/test_pipeline.py` runs 21 offline tests:

| Test | Checks that |
| --- | --- |
| `rover_on_mars` | A rover on Mars hills passes all checks and makes progress |
| `icy_slope_physics` | A rover can't climb a 25° ice slope (friction below tan 25°) |
| `walkers_walk_without_falling` | 4- and 6-legged walkers walk on Mars hills without falling |
| `building_textbook_and_resonance` | OpenSees matches the textbook cantilever, and a softer building sways more |
| `real_terrain_pipeline` | Fake HiRISE file → converter → preset → simulation, with orientation preserved |
| `fall_diagnostics_on_steep_slope` | A walker falling on a 25° slope gets correct fall direction, slope and slip evidence |
| `slope_geometry_and_no_fall_fields` | Slope ahead and sideways are measured correctly, with no false steep edges |
| `run_once_per_approval` | A double click on Run doesn't run twice; "run again" does |
| `rover_steers_to_goal_off_its_heading` | A rover turns 45°, reaches the goal and ends the run early |
| `walker_reaches_goal_on_hills` | A 6-legged walker reaches a goal on Mars hills |
| `goal_up_an_icy_slope_is_missed` | An impossible goal is reported as missed, with the reason |
| `named_places_and_mock_missions` | "From the west to the crater floor" resolves to real places on the terrain |
| `click_to_set_goal` | A clicked goal updates the scene without a model call |
| `step_height_decides_pass_or_stop` | A 0.07 m step is passed and a 0.15 m step stops the 4-wheel rover, but the 6-wheel rover climbs it |
| `walker_stopped_by_gap` | A walker is stopped by a small gap, and the summary names the gap and its measured width |
| `course_layout` | Six obstacle kinds are placed in order, clear of start and goal; the terrain grows to fit |
| `mock_course_parsing` | "10 small rocks, a big step, stairs and a narrow trench" parses correctly; edits add and remove obstacles |
| `planner_routes_by_domain` | Buildings go to OpenSees, robots go to MuJoCo |
| `batch_stress_conditions_and_ghosts` | Baseline ordering, gravity/surface changes, icy-slope failure, push effect, invalid-knob errors, ghost eligibility and payload budget |
| `stress_trigger_requires_robot_scene` | Stress-test phrases start a batch only when a robot scene is already active |
| `stress_planner_falls_back_when_conditions_do_not_apply` | A repeated model suggestion for an inapplicable slope variation falls back to compatible conditions |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_pipeline.py                          # offline tests
uvicorn server:app --reload                            # then open http://localhost:8000
```

To use a model, copy `.env.example` to `.env` and fill in a base URL, key and model names. Then open
`http://localhost:8000/health` to confirm which mode is active.

To import real terrain:

```bash
pip install rasterio scipy
python scripts/hirise_to_simgate.py --name jezero --label "Jezero Crater, Mars" --size-m 40 \
  --input https://astrogeo-ard.s3.us-west-2.amazonaws.com/mars/mro/hirise/controlled/dtm/ESP_045994_1985_ESP_046060_1985/DTEEC_045994_1985_046060_1985_U01.tif
```

Things to try:
- *"A robot dog walking on real terrain: Jezero Crater, Mars"* → **Run** → *"use six legs"* → **Run**
- *"Rover on the Moon, drive from the west to the crater floor"* → **Run**
- *"Rover obstacle course: 10 small rocks, a step, stairs and a gap"* → **Run** → *"use six wheels"* → **Run**
- *"How robust is this rover?"* → **Stress test** → compare the report card and ghost view
- *"A 5-story building in an earthquake"* → **Run** → *"make it stiffer"* → **Run**

## Physics checks

- **MuJoCo:**
  - The model compiles.
  - Gravity matches the NASA reference for the chosen world.
  - Friction and slope are plausible.
  - Real terrain matches the source data.
  - Start and goal are on the terrain and a real distance apart, with a warning if the straight path looks
    too steep.
  - A 1 s settle test stays stable, and the robot actually rests on the ground.
- **OpenSees:**
  - The solver reproduces the textbook cantilever deflection PL³/3EI.
  - The building's natural period is in a realistic range of the ASCE 7 estimate (Ta = 0.0724·h^0.8).

## Code layout

| Path | What it does |
| --- | --- |
| `simgate/spec.py` | The shared scene spec, reference defaults and missions |
| `simgate/presets.py` | Reference gravity and friction values |
| `simgate/agents.py` | Planner, Router, Builder and Explainer, each with a model or mock rules |
| `simgate/batch.py` | Parallel robot-condition runs and compressed ghost tracks |
| `simgate/orchestrator.py` | Runs a session and streams events to the browser |
| `simgate/backends/` | One file per simulator (`mujoco_backend.py`, `opensees_backend.py`) |
| `simgate/terrain_data.py` | Loads real-terrain presets |
| `scripts/` | HiRISE converter and fake test-data generator |
| `server.py`, `static/vendor/index.html` | FastAPI WebSocket server and three.js viewer |
| `tests/test_pipeline.py` | Offline test suite |

To add a simulator: add a domain to `SceneSpec`, subclass `Simulator` in `backends/`, register it in
`backends/__init__.py`, and add a viewer branch in `static/index.html`.

## Known limits

- Robots steer straight at the goal and go over obstacles; there's no route planning around them yet.
- Gap widths snap to the terrain grid (about 0.13 m steps on a 20 m terrain); the measured width is reported.
- The walkers use a fixed rhythmic gait, not a trained policy. The 4-legged walker falls on side slopes
  of about 10° or more.
- Orbital terrain can't show rocks smaller than about 1 m. Small random bumps stand in for them.
- The OpenSees model is linear-elastic (no yielding, no P-delta), so damage levels are rough estimates.
- The mock planner understands only simple keywords; a real model handles open-ended prompts.

## Planned next

- Steering around obstacles (route planning)
- Upload your own robot (MJCF, URDF) and a trained ONNX policy, wired up by inspector and adapter agents
- A robot gallery from MuJoCo Menagerie, with mesh rendering
- Quick-tune swarm, and voice editing if time allows

## Credits and licenses

- Mars elevation data: NASA/JPL/University of Arizona HiRISE; DTMs via USGS on AWS Open Data (CC0 1.0).
- three.js is vendored in `static/vendor/` (MIT license, see `three-LICENSE`).
- MuJoCo (Apache-2.0) and OpenSeesPy are installed from PyPI.
