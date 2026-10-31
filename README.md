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
prompt ─► Director ─► fixed workflow ─► Builder/Checker ─► (you approve) ─► Simulator ─► Explainer
          (one plan)    (code runs the     (bounded repair)   "Run it?"      (streams      (measured
                         planned steps)                                      frames)       evidence)
```

- The **Director** makes one planning call for a request. It returns a validated Plan; code expands robot counts
  and executes the selected workflow in a fixed order. The model never chooses arbitrary physical values or
  object placements.
- **Builder** produces the simulation from a verified template by default. In `llm` mode it can write MuJoCo
  XML; invalid output is retried and falls back to the template if needed. The constrained `build_custom` tool
  only appends body/geom fragments and validates them without allowing robot joints or actuators to be changed.
- **Checker** verifies the scene before anything runs (see [Physics checks](#physics-checks)). Failed checks
  enter a visible repair loop capped at three attempts. Template mode applies only safe, registered preset
  corrections; LLM Builder mode receives the exact failed checks. Remaining failures are shown and block a run.
  The two advisory warnings (a steep straight path, a course beyond the robot's measured limits) are shown but
  never trigger a repair or block a run.
- **Approval:** the Director asks "Run it?" unless the user explicitly requested a run. One click gives exactly
  one run, and an explicit "run again" repeats it. A scene whose checks still fail is never run by "run" or "run again"; the reply
  names the failing checks, and "run anyway" is the explicit override.
- **Simulator** streams frames live to the browser.
- **Explainer** describes what happened in plain language, using only measured evidence. It marks anything it
  infers with "likely" and suggests one thing to try next.
- **Mock mode:** with no model configured, simple rules stand in for every agent, so the whole pipeline works
  offline.

### How the Director works

The Director's model call produces one complete JSON Plan: workflow, world, robot count and choices, obstacles,
goals, variants, and whether the user explicitly asked to run. The plan is checked against registered robot,
policy, world, and terrain IDs. Invalid output gets one correction request; if that is still invalid, the rules
planner is used and the user is told.

After planning, deterministic code runs the fixed workflow: build/race sets up the world and robots, places
obstacles and goals, then builds and checks before asking for approval or running. Compare and stress workflows
use the existing batch runner. Edit changes only named fields; answer plans skip simulation. An edit is applied to
a copy of the scene while planning, so an invalid edit, or one that changes nothing, gets the one correction
request. Setting a slope angle on non-slope terrain switches it to a slope, and "steeper" without a number adds 10°.

Code then checks the model's plan against the request's own words, and each correction appears as an assumption
the user sees: a world named in the request ("Mars", "lunar") or a real place ("Gale Crater") wins over the plan;
a real-terrain preset the user didn't ask for is dropped in favour of generated terrain; course obstacles go along
the path in the order given unless the request says to scatter them or names a place; and a request ending in
"go", "start", "run it" or "race them" runs; a race with no goal finishes on the far side, and "to the summit/top/bottom/far side" sets a goal the plan left out. There is no
one-tool-per-model-turn loop or 12-call limit.

After a failed run, "why did it fail?" (or "prove it", "what caused it") plans the **diagnose** workflow and
"fix it", "make it pass" or "what would work" plans **redesign** (sections 11 and 12). If the model picks another
workflow for such a request after a failed run, a code guard switches to the right one and says so. "use this
design" applies the last redesign's winner through the typed edit path without a model call.

A typical race uses **one Director planning model call**. If it runs, a separate Explainer call may follow; a
failed check can add up to three bounded Builder repair attempts. Mock mode uses rules and makes no model calls.

The registered tools include `set_world`, `spawn_robot`, `remove_robot`, `list_library`, `add_obstacles`,
`clear_obstacles`, `set_start`, `set_goal`, `build_and_check`, `run`, `compare`, `build_custom`, `ask_user` and
`explain`; workflows invoke them in code rather than asking the Director to select one at a time.

### Several robots

A world can contain up to six robots sharing the same terrain and obstacles. They collide through the shared
MuJoCo physics, while each robot keeps its own namespaced bodies, controls, diagnostics and mission. When a
start is omitted, robots are placed in separated lanes across the route; one shared goal can be used for a race.
The older single `robot`/`mission` format is still accepted and becomes the first robot entry automatically.
Multi-robot results include per-robot measurements and an arrival/progress leaderboard; the browser shows
individual colored trails and labels, and leaderboard rows move the follow camera to that robot.

### Seeing the agents work

The chat timeline shows the Director's planning thought and each workflow/agent step; expand a row to inspect
its arguments and result. The **Agents** panel shows Director, Builder, Checker, Simulator and Explainer with
their model names and run counts. Checker-to-Builder repair hand-offs and the follow-up check appear as separate
steps in the event tree.

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
- Cards show physics checks, measured results and a multi-robot leaderboard. Robot camera chips, colored paths
  and floating labels distinguish robots in shared scenes; clicking a leaderboard row follows that robot.
- **Show code** shows the simulation code the Builder wrote.

### 8. Robot and policy library

- Open **Library** to browse the shipped robot, policy and world cards. Choose a robot, one of its
  compatible policies, and a world, then apply the selection before describing or rebuilding a robot
  scene. The panel displays each card's license and credit. Upload your own terrains, robots and policies from the
  same panel (section 9).
- The library includes the G1 humanoid and its trained walking policy, alongside the built-in rover and
  walkers. Prompts such as *"G1 humanoid on flat ground"* select the registered G1 robot and
  `g1_walking` policy; the planner is given the installed IDs and must not invent library entries.
- The G1 model and policy are sourced from Unitree Robotics' `unitree_rl_gym` repository under
  BSD-3-Clause. The SimGate copy, source file list and modifications are recorded in
  [`third_party/unitree_rl_gym/NOTICE.md`](third_party/unitree_rl_gym/NOTICE.md). The policy was
  trained for flat ground and Earth gravity; results on other surfaces, gravity levels and obstacle
  courses are transfers to validate, not guaranteed behavior.
- To add a policy, put its model under a project-relative asset directory and add a JSON
  `PolicyCard` in `simgate/library/presets/policies/` (or `simgate/library/uploads/policies/`).
  Set `robot_id`, format, model file, control interval, ordered joint names, observation blocks and
  scales, action scale, default angles, gains, command axes and ranges, training notes, license and
  credit. For a robot with a different ID, compatibility is checked by matching all policy joints
  against the robot card's joint names and count. The current G1 card records the exact 12-joint
  deployment order. Robot cards and any MJCF/mesh assets belong in the corresponding robots folder;
  asset paths must remain inside the project.
- G1 inference uses the shipped ONNX model and named MuJoCo joints/actuators, not fixed qpos offsets.
  A 0.5 s default-angle settle precedes policy control; the Navigator supplies walking commands. The
  browser sends shared mesh assets once and draws the G1 meshes as BufferGeometry, positioned from
  MuJoCo geom poses so asset offsets are not applied a second time.

### 9. Bring your own: terrains, robots and policies

Upload a **world** (terrain), a **robot** (its body) or a **policy** (a trained brain for a Library robot). Agents
inspect it, write its Library card, ask when they're unsure, test it, and then it appears in the Library to use like
any built-in item: in plans, races, stress tests, diagnose and redesign.

**How to upload.** Open **Library**, pick the tab and press **Upload robot**, **Upload policy** or **Upload world**.
Drop files (or one `.zip`) on the form or choose them, and fill in a name, a credit and a license (all three are
required and shown on the card). A policy needs the Library robot it drives; a world asks which world it's on and
what the ground is like. The chat shows the work as agent steps: **Upload** → **Inspector** or **Adapter** (the
FACTS it measured and the draft card) → **Checker** (each check). Questions come as quick-reply chips, and an answer
updates the card without a re-upload. A listed item gets an *uploaded* badge, its credit and license, and a **Use**
button; one whose checks failed is shown as *not ready*, with the reasons. Behind the panel: `POST /upload`,
`POST /upload/{robot|policy}/{id}/answers` and `GET /library`.

| Upload | Files | What happens |
| --- | --- | --- |
| World | A GeoTIFF elevation model in a metric projection, or a square greyscale PNG/JPG heightmap | A GeoTIFF goes through the HiRISE converter's path (patch picking, hole filling, north up). A heightmap needs its width and height range in metres; black is the lowest point and the image's top edge is north. A scene can resize it from the chat ("make it wider and not as tall", "make the hill 30 m wide"); the upload itself is unchanged, and real elevation data keeps its measured scale. |
| Robot | MJCF (`.xml`/`.mjcf`) or URDF, with its meshes (`.stl`, `.obj`, `.msh`) | Compiled with MuJoCo from upload files only (a URDF is converted with `mj_saveLastXML`), then inspected and tested. |
| Policy | An ONNX model plus a yaml/json config and/or a README | Read in code, adapted to the chosen robot's joints by name, and tested on that robot. |

**Safety rules.** Nothing uploaded is ever run.
- At most 50 MB per upload; a zip holds at most 500 files and may unpack to at most 200 MB, counted while unpacking.
- Allowed file types only: `.xml .urdf .mjcf .stl .obj .msh .png .jpg .tif .tiff .onnx .yaml .yml .json .md .txt`.
  Code and pickles (`.py`, `.so`, executables, `.pt`/`.pth`) are refused.
- No absolute paths, no `..` in stored file names, no drive letters, no symbolic links; a rejected upload leaves
  nothing on disk.
- YAML is read with `yaml.safe_load`; model XML may not declare a DOCTYPE or entities; MJCF `<plugin>` (native code)
  is refused; every `file`, `filename`, `meshdir`, `texturedir` and `assetdir` must stay inside the upload, counted
  from the model's own folder (`meshdir="../meshes"` from `urdf/` is fine).
- Policies must be ONNX: a TorchScript `.pt` file can run arbitrary code when loaded. Export it on your own machine:
  ```python
  import torch
  policy = torch.jit.load("policy.pt", map_location="cpu").eval()   # run this on your own machine
  observation = torch.zeros(1, NUM_OBSERVATIONS)                      # the size your policy expects
  torch.onnx.export(policy, observation, "policy.onnx", input_names=["observation"],
                    output_names=["action"], opset_version=17)
  ```
- Uploads are stored under `simgate/library/uploads/<type>/<id>/` (the id is the name plus a content hash), stay on
  the machine they were made on (git-ignored), and are listed with source `upload`. `SIMGATE_UPLOADS_DIR` moves the
  folder.

**How the agents inspect it.** In each case code measures the FACTS and an agent (the planner model; rules in mock
mode) only drafts what needs judgment. Its draft is checked against the FACTS in code, with one retry that shows it
the exact errors, and the rules draft is used if it is still wrong.
- **Terrain:** the Inspector reports size, relief and typical (95th percentile) and maximum slope, and warns when the
  ground is steeper than walkers (20°) or rovers (25°) usually climb or smaller than 10 m across.
- **Robot:** SimGate stores it with one root body and a free joint, turned so forward is +x, every joint named, and
  motors added (with the URDF's effort limits) if it has no actuators. FACTS: bodies, joints (type, range, axis),
  actuators (type, gear, ranges), mass, bounding box (a robot under 5 cm or over 5 m tall gets a "mm or inches?"
  warning), meshes, candidate feet and wheels, and name hints. The Inspector drafts its kind, forward axis, wheels or
  legs and which controller fits; code measures wheel sides, radii and spin, and each leg's gait phase. Checks before
  it is listed: it compiles, mass and size are plausible, it stands 1 s, a wheeled robot drives 0.5 m with the wheel
  controller, and a legged robot's simple gait is tried and labelled *baseline gait* (it may walk badly). A robot
  without a trained policy gets a built-in controller: the wheel controller, the baseline gait, or holding its pose.
- **Policy:** FACTS are the ONNX inputs and outputs (and an LSTM's state), the config's keys and values, and the
  robot's joints and default pose. The Adapter drafts the card in a fixed vocabulary of observation blocks
  (`base_ang_vel`, `base_lin_vel`, `projected_gravity`, `command`, `joint_pos`, `joint_vel`, `last_action`, `clock`;
  `height_scan` is refused as unsupported), with scales, action scale, default angles, gains, control step, joint
  order and commands. Anything the files don't give (the clock period, missing default angles) becomes a question.
  Checks: the blocks add up to the ONNX input, the actions match the output and the joints mapped by name, the policy
  loads, it stands 2 s at command 0 and walks 0.5 m forward in 4 s. A failure says what is likely wrong, for example
  "the observation order or scales (the robot falls immediately)".

**The golden test.** The G1's own `motion.onnx` and `g1.yaml`, uploaded as a user would for robot `g1`, must give a
card equal to the shipped G1 card field by field, and the G1 must walk with it. `g1.yaml` doesn't contain the gait
clock's period (Unitree's `deploy_mujoco.py` hard-codes 0.8 s), so the Adapter asks for it; after the answer "0.8"
every field matches and the G1 walks about 1.9 m in the 4 s check. This shows the Adapter works on a real policy.
With gpt-4o-mini as the planner model, the model's own drafts were rejected by the checks in our runs, so the rules
draft was used; the checks are what keep a wrong draft out of the Library.

### 10. Stress test (robots)

- With a robot scene open, click **Stress test** or ask *"stress test"*,
  *"test it under different conditions"*, or *"will it work on Mars?"* (*"how robust?"* finds where it breaks
  instead, section 14). The unchanged scene runs first
  as **Baseline**, followed by up to 23 variations in parallel. Each run has a 120-second timeout;
  failed or timed-out runs appear as errors instead of aborting the batch.
- **Repeat ×3** (the chip after a stress test, or *"repeat ×3"*, *"how reliable is it?"*, *"stress test, repeat"*)
  runs every row on 2 more seeds and shows each row's pass rate (see section 13).
- The planner can vary gravity, surface, friction, added payload, slope angle (slope terrain only),
  obstacle scale (when obstacles exist), lateral push and terrain seed. Surface changes reset friction
  to the surface preset. Gravity changes the gravity value directly and does not change the body/world,
  including for real terrain. Payload mass is added to the robot's mass (not for library robots such as the G1, whose trained policy
  expects its exact body); a push lasts 0.3 seconds near the middle of the run.
- A run passes if it reaches its goal. Without a goal, it passes if it stays upright and makes at least
  1 m of forward progress. The report card lists each outcome and measured progress or goal distance,
  maximum tilt and slip, followed by a summary of the pattern across runs.
- The ghost view overlays the baseline and same-geometry variations on one timeline; clicking an
  available report row replays that run alone. Robot poses are recorded at 15 fps, rounded to three
  decimals and compressed, with a 4 MiB ghost-track budget (the complete batch payload stays below
  about 5 MB). Slope, obstacle-scale and seed variations change geometry, so they remain in the report
  but are marked **not in ghost view**. The follow camera tracks the baseline.

### 11. Prove why it failed

After a failed run, ask *"Why did it fail?"* (or click the chip). The agents test their own explanation instead
of guessing:

1. **Reproduce:** the unchanged scene runs again, on the original seed and on a second one (section 13). If it
   passes on the original seed, the answer is "the failure didn't reproduce" and nothing else runs. If it fails only
   on the original seed, the timeline says the failure itself is borderline.
2. **Hypotheses:** the Explainer proposes 2–4 causes, each tested by changing exactly **one** thing. Only these
   experiments are allowed: slipping → a rock surface (or friction +50%); low gravity → 9.81 m/s²; steep ground →
   10° less slope (slope terrain) or the same scene on flat ground (generated hills, rocks or craters); an obstacle
   → remove that obstacle, or halve every obstacle; a push → no push; too fast → speed × 0.6; tipping on a side
   slope → face straight uphill. The model's reply is checked against this list: anything else is dropped with a
   visible reason, the value of each change is the mapping's (a different value is replaced, visibly), and a bad reply gets one retry that shows the model its own reply and the exact errors. In
   mock mode, rules read the evidence (slip above 40%, the obstacle that stopped it, a fall on ground steeper than
   12°, gravity below 5 m/s², a large sideways slope). There is always at least one **control**: a cause the
   evidence says is *not* responsible, so the results can rule something out.
3. **Experiments:** all of them run in parallel through the batch runner, on the same seed and terrain, with the
   original run's goal and time limit written in (so removing a course's only obstacle doesn't also remove its
   finish line).
4. **Verdicts:** *confirmed* ✓ = passes with only this change; *partly* ~ = still fails but clearly better
   (30% less distance left, or a fall avoided); *ruled out* ✕ = no clear improvement. Less slip alone isn't
   progress: it is shown as a side note ("slip fell from 10% to 1%, but it got no closer") and the verdict stays
   *ruled out*.
5. **Decisions:** after each batch the Explainer decides to *conclude*, *test more* (up to 2 follow-ups, e.g. the
   top two causes together or a stronger version of one) or say it *can't tell* (results contradict). At most two
   decision rounds; mock mode tests more only when the best result is "partly". A follow-up that only adds to a
   change that already passed on its own is dropped, with the reason shown.
6. **Conclusion:** the opening sentence is written in code from the verdicts ("Confirmed: X." or "No single tested
   cause explains it."); the Explainer adds 2–4 plain sentences after it, from the verdicts only. A reply that
   contradicts the opening (a cause confirmed AND "none explains it", or a cause claimed when none is confirmed) gets
   one retry with the exact problem, then the rules text is used. Unless a combination was tested, it notes that
   causes which only matter together could be missed.

**Seeds:** every experiment runs on both seeds, and each seed is judged against the unchanged scene on that seed.
When the two seeds give different verdicts the cause is marked **uncertain** (a grey-amber badge), the code-written
opening of the conclusion says so ("Uncertain: slipping (the 2 seeds disagreed)."), and a model conclusion that calls
it confirmed or ruled out is sent back.

**Budget:** at most 12 reruns in total, counting each seed: the reproduction on 2 seeds, typically 2 causes and a
control on 2 seeds each (6), and room for one follow-up round of up to 2 experiments on 2 seeds (4). In a multi-robot scene one robot is diagnosed
(the first that failed, or the one named), with the others unchanged.

### 12. Make it pass

Ask *"Make it pass"* (or *"fix it"*, *"what would work?"*), optionally with limits such as *"keep it under
15 kg"*, *"keep 4 wheels"*, *"only change the wheels"* or *"don't change the speed"*. The agents change the
**robot's design, never the world**: environment, terrain, obstacles, start, goal and conditions stay fixed.

| Robot | Knobs and hard bounds |
| --- | --- |
| Rover | wheel radius 0.08–0.35 m, 4 or 6 wheels, wheel speed 2–12 rad/s, chassis length 0.4–1.2 m, chassis width 0.3–0.9 m, mass 3–40 kg |
| Walker | 4 or 6 legs, leg length 0.15–0.35 m, gait 1.2–2.5 Hz, stride 8–22° (outside the tuned 2.0 Hz / 15° the result is labelled "untested gait") |
| Library robot with a trained policy | walking command speed 0.2–1.0 m/s and the choice of compatible policy only; the body is never edited because the policy expects that exact body |

The user's limits narrow these bounds and are never violated; out-of-range proposals are clamped and the
adjustment is shown.

- **Rounds:** up to 3 rounds × 4 candidates in parallel (at most 12 runs plus robustness checks). If there is no
  diagnosis yet, one runs first, and round 1 is aimed by it. Before each round the Designer (`BUILDER_MODEL`)
  chooses a strategy and says why: *bigger change* (nothing is close; numeric knobs are also tried at their
  limits), *refine* (something nearly passed), *try a different knob* (a knob is at its bound or isn't helping)
  or *stop*. Mock rules decide from how close the best run got. The Designer is told what each knob does and the
  robot's measured climb and gap limits; if its round has no new valid candidate (all repeats or out of bounds),
  the design rules fill that round and say so. After each round the Checker confirms the world is unchanged and
  every change is in bounds.
- **Winner:** the smallest change that passes (each numeric knob counted as a share of its range, a choice such
  as 6 wheels as 0.5; fewer changed knobs win ties).
- **Robustness:** the winner reruns on 3 nearby conditions inside the same world (terrain/obstacle seed +1 and +2,
  friction −10%) and is reported as "passes 3/3 (95% CI 44–100%)" or which failed. A design is the winner only if it passes at least
  2 of the 3; otherwise the next-best passing design is tried. If none qualifies, the result says "found a design
  that passed once but isn't reliable (1/3)" and shows it as **fragile**, not as the winner; the button then reads
  *"Use fragile design"* and still applies it.
- **No pass:** the result says so, shows the best attempt and how close it got, and names the limit that blocked
  it (e.g. "wheel radius already at the 0.35 m maximum").
- **Nothing is applied automatically.** *"Use this design"* (button or chip) applies only the winner's changes
  through the typed edit path, and the title is rebuilt as for any edit. **Show runs** replays the winning design
  with streamed frames without applying it (the ghost view can't draw a different robot body).

The browser shows a **Diagnosis** card (each cause, the single change tested, a verdict badge and its key number,
the decisions and the conclusion) and a **Redesign** card (changed knobs before and after, result, robustness,
runs tried and each round's strategy). Every decision appears in the timeline with its reason; the agent graph
shows Explainer → Batch runner → Explainer for a diagnosis and Designer → Batch runner → Checker → Designer for a
redesign.

### 13. Honest uncertainty

One run says what happened once. SimGate says how sure a result is, and what it cost to find out.

- **Seeds.** `simgate/reliability.py` runs the same condition on several seeds in parallel through the batch runner.
  Seed 0 is the condition exactly as it is (the run you saw). Each further seed changes the terrain and obstacle
  randomness (terrain seed + 1, + 2, …) and nudges each robot's start by up to ±1 cm and its heading by up to ±1°
  (`perturb_seed`; the goal and the course stay put).
- **Counting.** For each condition: passes, rate, a **Wilson 95% interval** (honest for small counts: 3 of 3 is
  44–100%, not "100%"), the mean ± spread of the key number (distance left to the goal, or forward progress without
  one), and a label: **reliable pass** (every seed passed), **reliable fail** (none did) or **borderline** (mixed).
  A crashed run is a system problem, not a result: it is left out of the count and reported.
- **Where it shows.**

  | Analysis | What changes |
  | --- | --- |
  | Prove why it failed | Every reproduction and experiment runs on 2 seeds; seeds that disagree make the verdict *uncertain* (section 11). |
  | Make it pass | The robustness check reads "passes 3/3 (95% CI 44–100%)"; a winner still needs at least 2 of 3. |
  | Stress test | Optional **Repeat ×3**: each row on 3 seeds, with its pass count, interval, label and the key number's mean ± spread; the explanation ends with how many rows are reliable passes, borderline or reliable fails. |

- **Cost.** Every request reports what it used: simulation runs, wall time and model calls, e.g.
  *"12 runs · 48 s · 3 model calls"*. It is shown on the Diagnosis, Redesign and Report cards and under the reply in
  the chat. Runs are counted when the batch runner starts them (and for each streamed run); model calls are counted
  in the model client.

### 14. Find where it breaks

Ask *"where does it break?"*, *"find failures"*, *"edge cases"*, *"how robust is it?"*, *"red team it"* or *"what are
its operating limits?"*, or click **Find where it breaks**. The agents make the **world** harsher, never the robot's
design, until one robot fails, and report where. No earlier run is needed.

- **What counts as failing:** *task* (the default) is missing the goal; *safety* (when the request mentions falling,
  tipping or safety) is falling or tilting past 60°. One robot is judged: the one named, or the first.
- **Budget:** *quick* (the default) uses at most 25 runs; *thorough* (*"thoroughly"*, *"in depth"*) at most 60.
- **Conditions** (2–4 that fit the scene, picked by rules in this order, or the ones the request names):

  | Condition | Range searched | Only when |
  | --- | --- | --- |
  | slope | the scene's angle → 30° | the ground is flat or a slope (flat is searched as a 0° slope: the same ground) |
  | obstacle height | the scene's → 2× the robot's measured climb limit | the course has obstacles it could climb |
  | friction | the scene's → 0.1 | always |
  | roughness | the scene's → 1 | the ground isn't flat |
  | sideways push (mid-run) | 0 → 80 N | always |
  | payload | 0 → 50% of the robot's mass | not a library robot (its policy expects its exact body) |
  | gravity | the scene's → 1.62 m/s² | the scene is on Earth, or the request mentions gravity (on any world) |
  | commanded speed | the scene's → +50%, and for walkers and library robots also → −50% (two half-searches of one condition) | always (rovers: faster only) |

- **Phase 1, one change at a time:** every condition is searched at once, in parallel. Each starts at its harsh end;
  if the robot copes there, the condition *never broke*. Otherwise the bracket between the last value that was OK
  and the first that failed is halved 4 times (at most 5 runs per condition, so the bracket ends 1/16 of the range
  wide). The scene as it is runs first; if it already fails there is nothing to search, and the answer says to ask
  why it failed instead.
- **Phase 2, a map:** the two conditions whose limits are closest to the scene as it is (as a share of their range)
  are combined on a 4×4 grid reaching just past the limits. The edges come from phase 1; the cells nearest the
  estimated boundary run first, 6 at a time, within the budget left. Every other cell is **inferred**, by assuming
  harsher on both never helps: it has no run id, is labelled "inferred" and faded on the card, is counted in the
  honesty line, and is never given to the Explainer, which sees only the combinations that ran. Cells left over are
  shown as not run.
- **Result:** the **Envelope** card shows each condition's value now, the last OK value and the first failing one with
  their run ids (every run is kept as evidence), how it failed, the map, the cost, and an honesty line written in
  code: *"14 runs, limits accurate to one search step (slope ±1.9°), simulation only (one seed; one robot judged; 10
  of 16 map cells inferred, not run)."*
- **Gravity:** a model plan that names gravity when the request doesn't is corrected (with a note), so another
  world keeps its own gravity unless you ask.
- **Explainer:** 3–5 sentences from FACTS (the limits from the tightest), starting with the tightest limit. Code
  checks every number it states against the envelope and that it starts with the tightest limit; one retry with the
  exact problems, then the rules text. Mock mode uses the rules text.

Found on a 4-wheel rover driving east on Earth sand: OK at a 24.4° slope and fell at 26.2° (a direct sweep in 2.5°
steps passes at 22.5° and fails from 25°); friction alone never broke down to 0.1 and a sideways push never broke up
to 80 N, but with slope and friction changed together 4 of the 6 combinations that ran failed.

### 15. Model support

The prototype works with any OpenAI-compatible endpoint, configured in `.env`:

| Provider | Base URL | Models |
| --- | --- | --- |
| Meta Model API | `https://api.meta.ai/v1` | Muse Spark for planning and explaining, Muse Wonder for building |
| Claude (Anthropic) | `https://api.anthropic.com/v1/` | Any current Claude model |
| OpenAI, OpenRouter or local Ollama | see `.env.example` | Any chat model |

Without a key, it runs in mock mode.

### 16. Tests

`python tests/test_pipeline.py` runs 136 offline tests (simulations, the agents with scripted models, uploads and
the page's wiring). rasterio is in requirements.txt, so the real-terrain integration test runs too; without it
that test reports "skip" with the reason.

| Test | Checks that |
| --- | --- |
| `rover_on_mars` | A rover on Mars hills passes all checks and makes progress |
| `three_rover_race_has_non_overlapping_spawns_and_leaderboard` | Three robots start apart, pass checks and are ranked by arrival |
| `two_rovers_collide_in_shared_world` | Two robots collide and contacts are recorded by shared physics |
| `legacy_robot_spec_becomes_single_entry` | The single-robot format remains backward compatible |
| `director_plan_schema_validates_ids_and_expanded_robot_limit` | Plan IDs are registered and expanded robot counts stay within six |
| `director_planning_uses_one_call_retries_validation_and_falls_back` | A valid plan uses one call; a validation error gets one retry that shows the model its own reply and the exact error; then rules fallback |
| `model_library_robot_entries_validate_first_time` | The model's own G1 entries (`"kind": "library"` with `robot_id`) validate on the first call with the robots, world, goal and run it asked for; a redundant kind is dropped, a missing policy is the robot's default, an unknown one becomes its only policy with a note, and kind "rover" with g1 says to omit kind |
| `robot_entry_errors_list_kinds_and_library_ids` | A bad robot entry (unknown kind, no robot, unknown robot_id, rover kind with g1) gets an error listing the built-in kinds and library robot_ids and saying to give robot_id and omit kind |
| `rules_planner_keeps_goals_and_races_have_a_finish_line` | In mock mode "G1 … walking on Mars hills to the summit" builds with the summit as its goal, "race a G1 against a rover on Jezero" gets a far-side finish line, "to the bottom" is the lowest point, and any race without a goal finishes at the far side |
| `model_plans_get_a_finish_line_and_a_named_goal` | A model's race plan without a goal (the real goal-less Jezero reply) gets the far-side finish line with the note "No finish line was given, so I put it on the far side."; a goal-less single robot asked to go "to the summit" gets the highest point |
| `race_goals_and_run_cues_in_plans` | "Race … to the summit/top/bottom/far side" sets the shared goal, a closing "go"/"start"/"run it" asks to run, and a model's "far side" maps to the east goal |
| `model_plans_are_corrected_from_the_request_words` | Model plans are corrected from the request's words (named world or real place, unrequested preset dropped, course obstacles on the path, closing run words), with a note the user sees |
| `stress_plan_requires_an_existing_robot_scene` | Stress plans without a robot scene fall back to a safe answer |
| `plan_workflow_expands_race_and_runs_only_when_requested` | A 3-rover race asks before running, and a requested run is handed to the session to stream |
| `run_now_race_streams_frames_before_metrics` | A race the user asked to run streams run_start and frames before metrics, then the leaderboard and explanation, like an approved run |
| `plan_workflow_bounds_visible_repairs_and_asks_before_running` | Repair events form a visible Checker → Builder → Checker chain with each failure reported once, stop after three attempts and never run failed scenes |
| `diagnose_confirms_the_step_and_rules_out_low_gravity` | On Mars a 0.2 m step stops the rover: diagnose reproduces it, confirms the obstacle and rules out low gravity on both seeds, within 12 reruns, and reports its cost |
| `diagnose_confirms_slipping_on_an_icy_slope` | On a 25° ice slope, a rock surface alone makes it pass, so slipping is confirmed |
| `diagnose_does_not_diagnose_a_passing_run` | A passing run gets a clear "nothing to explain" and no reruns |
| `experiment_knobs_and_per_robot_metrics` | The experiment and design knobs change only what they name; one robot's numbers are read from a multi-robot run |
| `redesign_finds_the_smallest_passing_design_and_keeps_the_world` | For a 0.15 m step the smallest passing design wins (bigger wheels or 6 wheels), the world is identical, robustness is reported; "keep 4 wheels" uses wheel radius |
| `redesign_reports_the_limit_when_nothing_passes` | A 1.0 m wall: no design passes within 12 runs, and the limit is named |
| `design_constraints_narrow_the_design_space` | "under 15 kg", "only change the wheels", "keep 4 wheels" and library robots narrow the knobs |
| `explainer_tips_use_only_settings_the_robots_have` | FACTS list what each robot can change (rover: wheels, wheel size, wheel speed; walker: legs, leg length, gait; G1: walking speed and compatible policies); a tip naming another setting, such as a gait for the G1, is replaced by the rules tip |
| `mock_explainer_tests_more_only_when_the_best_is_partly` | The mock Explainer tests more (combined or stronger) only when the best verdict is "partly" and reruns are left |
| `model_explainer_decisions_are_validated_bounded_and_visible` | Fake model: a disallowed knob is dropped, "test more" runs exactly one follow-up within budget, every decision is a step |
| `model_designer_candidates_are_clamped_and_stop_ends_early` | Fake Designer: out-of-range and unknown knobs are clamped or dropped visibly; "stop" ends the search |
| `model_loops_have_code_backstops` | A follow-up that only adds to a confirmed change is dropped; a Designer round of repeats is filled by the design rules; the Designer sees knob effects and the robot's limits |
| `director_plans_diagnose_and_redesign_in_one_call` | "why did it fail?" and "make it pass, keep it under 15 kg" plan diagnose/redesign in one call; the guard and the rules cues |
| `use_this_design_applies_only_the_winner_and_rebuilds_the_title` | "use this design" changes only the winner's knobs, with no model call, and rebuilds the title |
| `an_answer_keeps_the_built_scene` | A question about an open scene keeps its build, so "run" still runs |
| `show_runs_replays_the_winner_without_applying_it` | Show runs streams the winning design without applying it; the next run restores the scene first |
| `browser_handles_the_analysis_events` | The page renders the diagnosis and redesign events, chips and buttons |
| `full_session_why_then_make_it_pass` | Course with a big step → run → why → make it pass → use this design → run passes with frames streamed |
| `advisory_course_warning_does_not_block_or_repair` | An obstacle course beyond the robot's limits is a warning: it asks "Run it?" or runs, with no repair loop |
| `answer_first_does_not_leave_an_empty_scene` | A question asked before any scene leaves no empty scene, so the next prompt builds one |
| `new_plan_scenes_keep_the_planned_title` | New robot and structure scenes are titled from the plan, not "Untitled scene" |
| `plan_workflow_preserves_structural_scene_support` | Structural scenes still use the OpenSees build/check workflow |
| `edit_plan_changes_only_the_requested_scene_fields` | "Steeper" adds 10° (hills count as 0°) and the edit preserves the existing robot, mission, world and obstacles |
| `rules_planner_treats_short_change_requests_as_edits` | With a scene open, short change requests ("use six wheels", "ice", "add a wall", "make it stiffer") are edits; questions stay answers |
| `leg_and_wheel_counts_apply_only_to_their_part` | "six-legged" never sets a rover's wheels and "six-wheeled" never sets legs; multi-robot titles name each robot |
| `model_edit_plans_are_typed_and_checked_against_the_scene` | Model edits are typed and dry-run against the scene: a slope angle on hills becomes a slope, a no-op edit gets the one retry, "steeper" adds 10° |
| `edits_rebuild_the_scene_title` | Every edit rebuilds the title from the scene (slope angle, unusual surface, robots, course); only a name the user gives ("call it …") is kept |
| `adding_obstacles_keeps_the_existing_terrain` | Adding obstacles to an existing scene keeps its terrain (a 10° ice slope stays one); only a new course starts flat |
| `template_repair_uses_registered_physics_presets` | Safe repairs restore registered gravity, friction and slope values |
| `director_tools_validate_and_update_world` | Tools validate inputs and update the draft world |
| `set_world_accepts_a_bare_real_terrain_preset_name` | A plan naming a preset by its bare name ("jezero") sets that real terrain instead of crashing |
| `no_tool_loop_leftovers` | The removed one-tool-per-turn loop leaves no docstring or tool-catalog code behind |
| `mock_director_turns_race_into_ordered_tool_calls` | Offline planning creates a mixed race and asks before running |
| `build_custom_retries_and_emits_attempt_steps` | A rejected custom fragment is corrected and both attempts appear in the step tree |
| `plan_summaries_are_scene_names` | The planner is asked for a short scene name as summary (used as the title), and the rules planner's summaries are scene names too |
| `plan_prompt_states_world_course_and_preset_rules` | The planning prompt states the named-world, course-placement and preset rules; it shows a library robot and a mixed race by robot_id with no kind; every example validates, edit examples carry no angle, and a typed robot edit changes only its given fields |
| `plan_prompt_offers_built_in_robots_only_as_kinds` | The planning prompt lists only library robots by ID; rover and walker are offered as kinds |
| `plan_prompt_uses_compact_current_world_state` | Current-world prompts exclude large code and mesh payloads |
| `icy_slope_physics` | A rover can't climb a 25° ice slope (friction below tan 25°) |
| `walkers_walk_without_falling` | 4- and 6-legged walkers walk on Mars hills without falling |
| `building_textbook_and_resonance` | OpenSees matches the textbook cantilever, and a softer building sways more |
| `real_terrain_pipeline` | Fake HiRISE file → converter → preset → simulation, with orientation preserved |
| `script_runner_reports_skips` | The plain-script runner prints "skip" with the reason for a skipped test instead of "ok" |
| `fall_diagnostics_on_steep_slope` | A walker falling on a 25° slope gets correct fall direction, slope and slip evidence |
| `slope_geometry_and_no_fall_fields` | Slope ahead and sideways are measured correctly, with no false steep edges |
| `run_once_per_approval` | A double click on Run doesn't run twice; "run again" does |
| `a_run_that_raises_reports_an_error_instead_of_hanging` | A simulation that raises while streaming shows its error instead of leaving the session waiting forever |
| `failing_scene_runs_only_with_run_anyway` | A scene whose checks still fail isn't run by "run" or "run again"; it says which checks fail, and only "run anyway" runs it |
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
| `stress_payload_is_never_claimed_for_a_library_robot` | A G1 stress test has no payload row (its body is fixed, so the payload could not be applied); a rover's payload still adds mass |
| `g1_stands_for_two_seconds_at_zero_command` | G1 stays upright with pelvis height within 10% of its standing height |
| `g1_walks_two_metres_in_ten_seconds` | G1 makes at least 2 m of forward progress on flat Earth terrain without falling |
| `g1_reaches_diagonal_five_metre_goal` | G1's Navigator reaches a 5 m goal at 45° to its starting heading |
| `two_g1s_share_one_scene` | A race with two G1s compiles: their shared MJCF defaults go into the scene once |
| `g1_lying_on_the_ground_counts_as_fallen` | A G1 that falls on a 10° slope and lies near 90° is reported as fell, with fall time and direction (humanoids count as fallen past 60°, not the rovers' 100°) |
| `g1_named_mapping_policy_validation_and_mesh_budget` | Checker validates the policy and 1 s stand, joint-name mapping survives obstacles/goal markers, bad observation sizes are rejected, and the mesh scene stays under 5 MiB |
| `policy_with_out_of_range_targets_is_rejected_on_load` | A policy whose first targets leave the joint ranges is rejected on load with the joint named, not silently clipped |
| `g1_planner_and_library_selection_use_registered_cards` | G1 prompts and UI selections resolve to registered robot, policy and world cards |
| `library_selection_builds_and_checks_the_selected_robot` | After "Use selected setup", the next scene is built and checked as the selected G1, not built as a rover and relabelled |
| `library_selection_sets_up_one_scene_only` | A Library setup applies to one scene; a later "race three rovers on Mars" gets three rovers on Mars, not the selected G1 and world |
| `rules_planner_reads_a_unitree_robot_in_a_race_as_the_g1` | "Race a Unitree robot against a rover" plans a G1 and a rover; "Unitree G1 humanoid" is one G1 |
| `library_lists_only_each_robots_own_compatible_policies` | The rover and walker controllers, which name no joints, are offered only for their own robot (server and browser) |
| `uploads_reject_unsafe_files` | Uploads refuse code and pickles (.py, .so, .pt with the ONNX export snippet), unsafe zip paths, symlinks, more than 500 files, zip bombs, oversize uploads, MJCF plugins, file references leaving the upload, DOCTYPE/entities, unsafe YAML, a missing main file and missing name/credit/license, and leave nothing on disk |
| `uploads_are_stored_and_listed_in_the_registry` | A safe upload is stored under uploads/<type>/<id>/ with an upload.json record, the same upload twice is stored once, and the registry lists it with source "upload" |
| `upload_endpoint_stores_files_and_reports_errors` | POST /upload stores a robot and answers 400 with the reason for a rejected file or a missing credit, and 413 for an oversize upload |
| `uploaded_geotiff_and_png_terrains_become_presets` | A fake GeoTIFF (make_test_dtm.py) and a small PNG heightmap uploaded as terrains become presets that build and pass the physics checks; missing world, surface, size or height range come back as questions; the image's top is north; the Inspector reports size, relief and slopes and warns when it is too steep or too small; both appear as Library worlds with source "upload" |
| `planner_names_an_uploaded_terrain_only_by_its_whole_name` | An uploaded terrain called "My hill" is chosen for "a rover on my hill" but not for "make my rover faster" |
| `chat_resizes_an_uploaded_heightmap_but_not_real_elevation_data` | "Make it wider and not as tall" on a 2 m uploaded hill gives a 20 m wide, 1 m tall scene that passes its checks while the upload stays as it was; given numbers are used as given; "lower the gravity" isn't a resize; Jezero (real data) is refused |
| `uploaded_urdf_cart_is_inspected_checked_and_listed` | A URDF cart with a mesh is converted to MJCF from upload files only, gets motors, is measured (mass, wheels, mesh), drafted as wheeled, stands, drives 0.5 m, is listed with its wheel controller and reaches a goal in a scene; a missing mesh is refused |
| `urdf_laid_out_like_unitrees_is_accepted` | A URDF in urdf/ with its own `<mujoco>` block and `meshdir="../meshes"` is accepted and inspected; a reference that leaves the upload is still refused |
| `inspector_questions_are_answered_without_a_reupload` | A cart whose names don't say which end is the front asks "Which way is forward"; it isn't listed until the answer ("-x") turns it around, and a bad answer is refused |
| `uploaded_mjcf_quadruped_gets_a_baseline_gait` | An MJCF quadruped (scene parts dropped) gets a baseline gait in diagonal pairs, labelled as such |
| `inspector_flags_wrong_units` | A robot 8 mm tall gets the mm/inches warning and isn't listed |
| `inspector_model_draft_is_checked_against_facts` | The Inspector agent's draft is checked against FACTS with one retry showing the exact errors, falls back to the rules when still wrong, and loses questions that FACTS already settle |
| `uploaded_robots_race_with_name_prefixes_and_their_own_defaults` | An uploaded cart and dog race a G1 with prefixed names: the G1's joint defaults stay on the G1, visual meshes stay visual, and all three move |
| `golden_g1_policy_upload_matches_the_shipped_card_and_walks` | GOLDEN: the G1's own motion.onnx and g1.yaml, uploaded as a user would for robot g1, ask which robot and then the clock period (g1.yaml doesn't give it); after "0.8" the Adapter's card equals the shipped G1 card field by field, it stands 2 s, walks 0.5 m in 4 s and is listed |
| `adapter_model_draft_is_checked_against_the_config` | The Adapter agent's draft is checked against the ONNX sizes and the config's own numbers, with one retry showing the exact errors; a height scan is refused as unsupported |
| `adapter_checks_explain_a_wrong_observation_order` | A policy card with joint positions and velocities swapped fails the standing/walking checks, each with what is likely wrong, and isn't listed |
| `policy_runtime_reads_base_linear_velocity_in_the_body_frame` | The base_lin_vel block is the root's velocity in its own frame |
| `library_page_offers_uploads` | The Library's Upload buttons open a working form (drag-and-drop or file picker; name, credit and license required; robot picker for policies) and show Upload → Inspector/Adapter → Checker steps, question chips, "uploaded"/"not ready" badges and a Use button |
| `library_endpoint_shows_uploads_with_their_state` | GET /library lists uploaded cards and each upload's state: listed, needs answers (with its questions), or not ready (with the failed checks) |
| `an_uploaded_robot_is_usable_everywhere` | An uploaded cart is offered to the Director by id, planned by name, titled by its name, raced, run with its own climb limits, diagnosed, redesigned by command speed only, and stress-tested in worker processes |
| `explainer_uses_code_facts_and_retries_contradictions` | The Explainer gets a FACTS block built in code (terrain, tilt words); a reply calling hills "flat" gets one retry with the exact problem, and a 56.9° tilt called stable twice falls back to the rules text |
| `explainers_get_each_robots_kind_and_mark_untested_gaits` | Explainers see each robot's kind and only the settings it has (no gait for a rover, no wheel speed for a walker); gait numbers other than the tested 2.0 Hz / 15° are marked "untested settings" |
| `diagnose_conclusion_opening_is_code_and_contradictions_are_rejected` | The diagnosis opening ("Confirmed: X." or "No single tested cause explains it.") is written in code; a reply contradicting it gets one retry, then the rules text |
| `less_slip_without_getting_closer_is_ruled_out` | Less slip without getting closer is "ruled out" with a side note; "partly" needs 30% less distance left or an avoided fall |
| `wilson_interval_labels_and_seeded_repeats` | Wilson intervals match known values (3/3 → 44–100%), the reliable pass/borderline/reliable fail labels, a crashed run left out of the count, seed 0 unchanged and seed k adding terrain seed + k, and the start nudge staying within ±1 cm and ±1° and repeating for the same seed |
| `diagnose_marks_a_verdict_the_seeds_disagree_on_uncertain` | With faked batch results, a cause that passes on one seed and not the other is "uncertain", the conclusion opens "Confirmed: obstacle. Uncertain: slipping (the 2 seeds disagreed).", 8 runs are used (2 reproductions + 3 experiments × 2 seeds; a cause passed on its own, so no follow-up), a model conclusion calling it ruled out is rejected, and the cost is reported |
| `diagnose_still_runs_a_follow_up_after_three_causes_on_two_seeds` | With faked batch results, after 3 causes on 2 seeds each, two causes that each only partly help are tested together as a follow-up on 2 seeds and confirmed: 2 + 6 + 2 = 10 runs within the 12-rerun budget |
| `finder_slope_limit_matches_a_direct_sweep_and_the_map_agrees` | On Earth sand the finder's slope bracket overlaps a direct 2.5° sweep's (passes at 22.5°, fails from 25°), within 25 runs and 5 per condition, each run one change; the map's edges agree with phase 1; the honesty line and the rules text's numbers check out |
| `finder_budget_explainer_number_check_and_baseline_failure` | With a faked batch runner: a thorough search uses 4 conditions within 60 runs and fills the map; the slope bracket is 1/16 of its range; a scene that already fails stops after 1 run; the number check rejects "fails at 21.0°" and a reply that doesn't start with the tightest limit, retries once and falls back to the rules text |
| `finder_searches_speed_both_ways_for_legged_and_library_robots` | A walker (and a library robot) gets speed searched faster and slower as one condition with two half-searches of at most 5 runs each, finding both limits; a rover is searched faster only |
| `finder_searches_gravity_on_any_world_only_when_asked` | On Mars, "where does it break if gravity drops?" searches gravity from 3.71 m/s² and finds its limit; without the word it isn't searched; gravity named only by the model is removed with a note |
| `finder_inferred_map_cells_are_labelled_counted_and_never_explained_as_measured` | Inferred map cells have no run id while run cells point to phase-2 runs; the counts add up, the honesty line counts the inferred cells, FACTS for the Explainer state only the combinations that ran, and the card labels inferred cells |
| `director_plans_find_failures_from_the_cue_phrases` | "where does it break", "find failures", "edge cases", "how robust", "red team", "operating limits" and the chip plan find_failures (only with a robot scene); "thoroughly", "tip over" and named conditions set budget, failure and knobs; a model answer is switched by the guard; a session shows the Envelope card with the obstacle condition, its honesty line and cost |
| `stress_repeat_shows_pass_rates_and_every_card_shows_its_cost` | "Repeat ×3" after a stress test repeats it (not a rerun of the scene): every row gets 3 seeds with its pass count, interval and label, the explanation counts them, and the report and the turn report 9 runs; the page has a cost line on each analysis card and the uncertain badge |
| `robustness_decides_the_winner` | A design that passes only 1/3 robustness checks loses to the next-best that passes 3/3; if none holds up, the best is shown as fragile and can still be applied |
| `race_diagnosis_and_redesign_target_one_robot` | In a race, "why did it fail?" diagnoses the robot that failed (or the one named; a winner has nothing to explain) and redesign changes only that robot |
| `failure_chips_name_the_robot_that_failed` | After a race the chips name the robot that failed ("Why did the Walker fail?", "Make the Walker pass"); none when every robot arrived |
| `race_on_an_existing_course_builds_and_checks` | A race on an open course builds, checks starts and goals against the obstacles, and stress-tests without errors |
| `stress_test_on_a_race_reports_each_robot` | A race's stress test runs every variation on every robot, reports one cell per robot, and a payload reaches each of them |
| `race_stress_report_counts_each_robot` | The race stress report counts passes per robot ("Rover: 5/8 · Walker: 2/8") and never marks a row failed because one robot failed |
| `mock_stress_explainer_reads_each_racer` | The mock stress explanation gives each racer's pass count and its baseline outcome |
| `mixed_race_time_limit_uses_the_slowest_robot` | A rover and walker race gets the walker's longer time limit, and reruns keep it |
| `crashed_runs_are_a_system_problem_not_physics` | Crashed stress runs are reported as a system problem and left out of the analysis; if all crash, there is no Explainer call |
| `mock_tips_fit_the_scene` | Mock tips fit the scene: a course suggests wheels or legs, flat ground never "make the slope steeper", a 4-legged walker "use six legs" |
| `policy_without_default_angles_asks_for_them` | A policy config without default angles gets a question ("robot pose" or "zeros"); the answer fills them from the robot's own pose and a bad answer is refused |
| `policy_whose_onnx_sizes_dont_fit_is_refused_with_the_numbers` | A config that explains 44 of the ONNX's 47 inputs, or a 12-action policy for a 4-joint robot, is refused with those numbers and shown as not ready |
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
| `simgate/diagnose.py` | Prove why: reproduce a failed run, test single-change hypotheses on 2 seeds, verdicts and conclusion |
| `simgate/finder.py` | Find where it breaks: world conditions searched one at a time, a 4×4 map of the two tightest, the honesty line and the checked Explainer |
| `simgate/reliability.py` | Honest uncertainty: the same condition on several seeds, Wilson 95% interval, reliable pass / borderline / reliable fail |
| `simgate/cost.py` | What a request cost: runs, wall time and model calls |
| `simgate/redesign.py` | Make it pass: bounded design search, constraints, robustness checks |
| `simgate/library/` | Validated robot, policy and world cards and filesystem registry |
| `simgate/policy_runtime.py` | Named-joint policy inference and PD control |
| `simgate/simple_control.py` | Built-in controllers for uploaded robots: wheels, baseline gait, held pose |
| `simgate/library/uploads.py` | Safe upload storage: size, file type, path and content checks |
| `simgate/library/inspector.py` | The Inspector: uploaded robot → normalized MJCF, FACTS, draft card, checks |
| `simgate/library/adapter.py` | The Adapter: uploaded ONNX policy → policy card for a Library robot, checks |
| `simgate/library/terrain_upload.py`, `simgate/terrain_import.py` | Uploaded GeoTIFF/heightmap → terrain preset and the Inspector's report |
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
- Uploaded robots: the baseline gait is a simple open-loop gait and may walk badly; MJCF equality constraints,
  tendons and texture images in an upload aren't used; slip is measured only on built-in feet and wheels.
- The Adapter's rules know legged_gym / unitree_rl_gym deploy configs; other conventions rely on the planner model
  or the user's answers. A config that names no joints is mapped in the robot's own joint order (noted on the card).
- Gap widths snap to the terrain grid (about 0.13 m steps on a 20 m terrain); the measured width is reported.
- The built-in walkers use a fixed rhythmic gait. The G1 humanoid uses a trained policy but was trained
  for flat Earth conditions; the 4-legged built-in walker falls on side slopes
  of about 10° or more.
- Orbital terrain can't show rocks smaller than about 1 m. Small random bumps stand in for them.
- The OpenSees model is linear-elastic (no yielding, no P-delta), so damage levels are rough estimates.
- The mock planner understands only simple keywords; a real model handles open-ended prompts.
- Diagnose tests single changes (plus at most two follow-ups), so causes that only matter in combination can be
  missed. Payload is not a hypothesis because scenes have no payload to remove.
- Redesign searches at most 12 designs with simple strategies; it can miss a passing design that needs several
  knobs changed at once.

## Planned next

- Steering around obstacles (route planning)
- Additional robot and policy library entries
- Quick-tune swarm, and voice editing if time allows

## Credits and licenses

- Mars elevation data: NASA/JPL/University of Arizona HiRISE; DTMs via USGS on AWS Open Data (CC0 1.0).
- G1 model, meshes and walking policy: Unitree Robotics, `unitree_rl_gym` (BSD-3-Clause); see
  `third_party/unitree_rl_gym/LICENSE` and `NOTICE.md`.
- three.js is vendored in `static/vendor/` (MIT license, see `three-LICENSE`).
- MuJoCo (Apache-2.0) and OpenSeesPy are installed from PyPI.
