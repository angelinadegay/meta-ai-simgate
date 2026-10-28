# Unitree RL Gym notice

The files in this directory originate from
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), published
by Unitree Robotics under the BSD 3-Clause License. The applicable license is
included in `LICENSE`.

| SimGate files | Upstream source | Changes |
| --- | --- | --- |
| `resources/robots/g1_description/g1_12dof.xml` and the referenced files in `resources/robots/g1_description/meshes/` | `resources/robots/g1_description/g1_12dof.xml` and its referenced `meshes/` assets | Copied without modification. The unrelated `scene.xml` floor and other robot variants are not included. |
| `deploy/pre_train/g1/motion.pt` | `deploy/pre_train/g1/motion.pt` | Copied without modification; intended for conversion to ONNX for SimGate runtime. |
| `deploy/pre_train/g1/motion.onnx` | Generated from the bundled `motion.pt` with `scripts/torchscript_to_onnx.py` | Converted for ONNX Runtime. The LSTM hidden and cell states are explicit inputs and outputs so the recurrent policy remains stateful. |
| `deploy/deploy_mujoco/configs/g1.yaml` | `deploy/deploy_mujoco/configs/g1.yaml` | Copied without modification as the policy configuration reference. |
| `deploy/deploy_mujoco/deploy_mujoco.py` | `deploy/deploy_mujoco/deploy_mujoco.py` | Copied without modification as a deployment/observation-layout reference; it is not used as SimGate runtime code. |
