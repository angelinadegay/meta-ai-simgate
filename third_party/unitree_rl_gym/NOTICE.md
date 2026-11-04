# Unitree RL Gym notice

The files in this directory originate from
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), published
by Unitree Robotics under the BSD 3-Clause License. The applicable license is
included in `LICENSE`.

| SimGate files | Upstream source | Changes |
| --- | --- | --- |
| `resources/robots/g1_description/g1_12dof.xml` and the referenced files in `resources/robots/g1_description/meshes/` | `resources/robots/g1_description/g1_12dof.xml` and its referenced `meshes/` assets | Copied without modification. The unrelated `scene.xml` floor and other robot variants are not included. |

Only the robot model (MJCF and meshes) is included. Unitree's trained walking
policy is not included and is never run: in SimGate the G1 is a standing-only
humanoid held in its default pose by a plain-code PD controller.
