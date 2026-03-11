import numpy as np

from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures import FixtureType


class DualPickPlace(Kitchen):
    """
    Two-robot pick-and-place environment. Each robot is assigned its own object
    and target fixture:

      - Robot 0: pick obj_r0 from a counter and place it in a cabinet
      - Robot 1: pick obj_r1 from a counter and place it in the sink

    Per-robot success is exposed via ``_check_success_robot(robot_idx)``.
    The global ``_check_success()`` returns True only when *both* subtasks
    are completed.
    """

    def __init__(self, obj_groups="all", exclude_obj_groups=None, *args, **kwargs):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Fixture / object setup
    # ------------------------------------------------------------------

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        # Robot 0: counter → cabinet
        self.target_r0 = self.register_fixture_ref(
            "target_r0", dict(id=FixtureType.CABINET)
        )
        self.counter_r0 = self.register_fixture_ref(
            "counter_r0", dict(id=FixtureType.COUNTER, ref=self.target_r0)
        )

        # Robot 1: counter → sink
        self.target_r1 = self.register_fixture_ref(
            "target_r1", dict(id=FixtureType.SINK)
        )
        self.counter_r1 = self.register_fixture_ref(
            "counter_r1", dict(id=FixtureType.COUNTER, ref=self.target_r1)
        )

        # Compute per-robot spawn anchors so each robot starts near its own
        # target fixture instead of being placed with a fixed offset.
        self.init_robot_base_ref = self.target_r0
        if len(self.robots) >= 2:
            pos0, ori0 = EnvUtils.compute_robot_base_placement_pose(
                self, ref_fixture=self.target_r0, robot_idx=0,
            )
            pos1, ori1 = EnvUtils.compute_robot_base_placement_pose(
                self, ref_fixture=self.target_r1, robot_idx=1,
            )
            self.init_robot_base_pos_anchor = np.array([pos0, pos1])
            self.init_robot_base_ori_anchor = np.array([ori0, ori1])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang_r0 = self.get_obj_lang("obj_r0")
        obj_lang_r1 = self.get_obj_lang("obj_r1")
        lang_r0 = f"Robot 0: pick the {obj_lang_r0} from the counter and place it in the cabinet."
        lang_r1 = f"Robot 1: pick the {obj_lang_r1} from the counter and place it in the sink."
        ep_meta["lang"] = f"{lang_r0} {lang_r1}"
        ep_meta["lang_r0"] = lang_r0
        ep_meta["lang_r1"] = lang_r1
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.target_r0.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        # Object for robot 0 — placed on counter near cabinet
        cfgs.append(
            dict(
                name="obj_r0",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter_r0,
                    sample_region_kwargs=dict(ref=self.target_r0),
                    size=(0.60, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        # Object for robot 1 — placed on counter near sink
        cfgs.append(
            dict(
                name="obj_r1",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter_r1,
                    sample_region_kwargs=dict(ref=self.target_r1),
                    size=(0.60, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        return cfgs

    # ------------------------------------------------------------------
    # Success checking
    # ------------------------------------------------------------------

    def _gripper_obj_far(self, robot_idx, obj_name, th=0.25):
        obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
        gripper_pos = self.sim.data.site_xpos[
            self.robots[robot_idx].eef_site_id["right"]
        ]
        return np.linalg.norm(gripper_pos - obj_pos) > th

    def _check_success_robot(self, robot_idx):
        obj_name = f"obj_r{robot_idx}"
        if robot_idx == 0:
            obj_at_target = OU.obj_inside_of(self, obj_name, self.target_r0)
        else:
            obj_at_target = OU.check_obj_fixture_contact(
                self, obj_name, self.target_r1
            )
        return obj_at_target and self._gripper_obj_far(robot_idx, obj_name)

    def _check_success(self):
        return self._check_success_robot(0) and self._check_success_robot(1)
