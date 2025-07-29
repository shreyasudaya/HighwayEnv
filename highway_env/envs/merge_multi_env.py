from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MultiMergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "vehicles_count": 5,
                "controlled_vehicles": 2,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "screen_width": 1600,
                "screen_height": 400,
                "centering_position": [0.5, 0.5],  # Center screen on road, not ego
                "scaling": 2.5,  # Lower scaling => zoomed out (default ~5.5)
                "show_trajectories": False,
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        # Weighted combination (tweak weights as needed)
        weights = {
            "collision_reward": self.config.get("collision_reward", -1),
            "right_lane_reward": self.config.get("right_lane_reward", 0.1),
            "high_speed_reward": self.config.get("high_speed_reward", 0.2),
            "lane_change_reward": self.config.get("lane_change_reward", -0.05),
            "merging_speed_reward": self.config.get("merging_speed_reward", -0.5),
            "on_road_reward": 0.5,
        }
        reward = sum(weights[k] * rewards.get(k, 0) for k in rewards)
        return utils.lmap(reward, [-2, 2], [0, 1])  # normalize to 0–1


    def _rewards(self, action: int) -> dict[str, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )

        # Check if the ego-vehicle is on a valid lane
        in_lane = self.road.network.get_closest_lane_index(self.vehicle.position, None)[0] is not None

        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 3,  # scaled by 4 lanes (0–3)
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (v.target_speed - v.speed) / v.target_speed
                for v in self.road.vehicles
                if isinstance(v, ControlledVehicle)
            ),
            # Reward for staying inside road boundaries
            "on_road_reward": 1.0 if in_lane else -1.0,  # +1 if in a lane, -1 if off-road
        }
    
    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Build a 4-to-2 lane zipper merge:
        - 4 lanes before merge
        - Lanes 3 and 2 merge into lanes 2 and 1, respectively.
        """
        net = RoadNetwork()
        lane_w = StraightLane.DEFAULT_WIDTH

        # Segment lengths
        before = 150         # straight section (4 lanes)
        merge1 = 60          # sharper merge zone (4 → 3 lanes)
        merge2 = 60          # sharper merge zone (3 → 2 lanes)
        after = 150          # final 2 lanes

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # --- Section A->B: 4 parallel lanes ---
        for i in range(4):
            net.add_lane(
                "a", "b",
                StraightLane(
                    [0, lane_w * i], [before, lane_w * i],
                    line_types=[c, s] if i == 0 else [n, c],
                ),
            )

        # --- Section B->C: Lane 3 merges into lane 2 ---
        for i in range(3):  # lanes 0,1,2 stay
            net.add_lane(
                "b", "c",
                StraightLane(
                    [before, lane_w * i], [before + merge1, lane_w * i],
                    line_types=[c, s] if i == 0 else [n, c],
                ),
            )

        # Sharper merge for lane 3 into lane 2 (steeper sine curve)
        lane3_start = [before, lane_w * 3]
        lane3_end = [before + merge1, lane_w * 2]
        net.add_lane(
            "b", "c",
            SineLane(
                lane3_start, lane3_end,
                amplitude=lane_w,                  # stronger lateral shift
                pulsation=np.pi / merge1,          # one smooth curve
                phase=0,
                line_types=[n, c],
            ),
        )

        # --- Section C->D: Lane 2 merges into lane 1 ---
        for i in range(2):  # lanes 0,1 continue
            net.add_lane(
                "c", "d",
                StraightLane(
                    [before + merge1, lane_w * i],
                    [before + merge1 + merge2 + after, lane_w * i],
                    line_types=[c, s] if i == 0 else [n, c],
                ),
            )

        # Lane 2 merges into lane 1 (also steeper)
        lane2_start = [before + merge1, lane_w * 2]
        lane2_end = [before + merge1 + merge2, lane_w * 1]
        net.add_lane(
            "c", "d",
            SineLane(
                lane2_start, lane2_end,
                amplitude=lane_w,
                pulsation=np.pi / merge2,
                phase=0,
                line_types=[n, c],
            ),
        )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),
        )

    def _make_vehicles(self) -> None:
        road = self.road
        self.controlled_vehicles = []

        total = self.config["vehicles_count"]
        controlled = self.config["controlled_vehicles"]
        other_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Lane sections
        four_lane_section = [("a", "b", i) for i in range(4)]       # before merge
        two_lane_section = [("c", "d", i) for i in range(2)]        # after merge
        used_positions = []

        # Lane probabilities: bias toward rightmost lanes in the 4-lane part
        four_lane_weights = np.array([0.1, 0.2, 0.35, 0.35])  # outer lanes favored
        two_lane_weights = np.array([0.5, 0.5])               # balanced for after merge

        def sample_position(section="four"):
            if section == "four":
                lanes = four_lane_section
                weights = four_lane_weights
                x_range = (0, 150)  # spawn before merge
            else:
                lanes = two_lane_section
                weights = two_lane_weights
                x_range = (270, 400)  # spawn after merge (after taper)

            for _ in range(50):  # retry to avoid overlaps
                lane_idx = lanes[self.np_random.choice(len(lanes), p=weights)]
                longitudinal = self.np_random.uniform(*x_range)
                if all(
                    not (ln == lane_idx and abs(longitudinal - pos) < 12)
                    for ln, pos in used_positions
                ):
                    used_positions.append((lane_idx, longitudinal))
                    return road.network.get_lane(lane_idx).position(longitudinal, 0)

            # Fallback (crowded)
            lane_idx = lanes[self.np_random.choice(len(lanes), p=weights)]
            longitudinal = self.np_random.uniform(*x_range)
            used_positions.append((lane_idx, longitudinal))
            return road.network.get_lane(lane_idx).position(longitudinal, 0)

        # Divide vehicles: ~70% before merge, ~30% after merge
        before_count = int(total * 0.7)
        after_count = total - before_count

        # Controlled agents — always spawn in 4-lane section (before merge)
        for _ in range(controlled):
            pos = sample_position(section="four")
            speed = self.np_random.uniform(25, 35)
            v = self.action_type.vehicle_class(road, pos, speed=speed)
            self.controlled_vehicles.append(v)
            road.vehicles.append(v)

        # Other traffic — split between before and after merge
        for _ in range(before_count - controlled):
            pos = sample_position(section="four")
            speed = self.np_random.uniform(20, 35)
            road.vehicles.append(other_type(road, pos, speed=speed))

        for _ in range(after_count):
            pos = sample_position(section="after")
            speed = self.np_random.uniform(20, 35)
            road.vehicles.append(other_type(road, pos, speed=speed))
