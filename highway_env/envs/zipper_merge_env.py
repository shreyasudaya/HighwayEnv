from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle

class LaneDropMergeEnv(AbstractEnv):
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
                "vehicles_count": 15,
                "controlled_vehicles": 3,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "screen_width": 1600,
                "screen_height": 400,
                "centering_position": [0.3, 0.5],  # Center screen on road, not ego
                "scaling": 4.5,  # Lower scaling => zoomed out (default ~5.5)
                "show_trajectories": False,
                "offroad_terminal": False,
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
            "off_road_reward": self.config.get("off_road_reward", -2.0),  # strong penalty
        }
        reward = sum(weights[k] * rewards.get(k, 0) for k in rewards)
        return utils.lmap(reward, [-2, 2], [0, 1])

    def _rewards(self, action: int) -> dict[str, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )

        # Collision penalty (large negative)
        collision_penalty = -10.0 if self.vehicle.crashed else 0.0

        # Lane index scaled (assuming right lane is lane 1 before merge)
        right_lane_reward = 1.0 if self.vehicle.lane_index[2] == 1 else 0.0

        # Lane change penalty (assume 0 = keep lane, 1 or 2 = lane change)
        lane_change_penalty = -0.1 if action in [0, 2] else 0.0

        # Altruistic merging speed penalty: ego should slow down to help others merge
        merging_speed_penalty = sum(
            max(0, (v.speed - self.vehicle.speed) / v.target_speed)
            for v in self.road.vehicles
            if isinstance(v, ControlledVehicle) and v is not self.vehicle
        )

        # Off-road penalty (strong)
        off_road_penalty = -5.0 if not self.vehicle.on_road else 0.0

        return {
            "collision_reward": collision_penalty,
            "right_lane_reward": right_lane_reward,
            "high_speed_reward": scaled_speed * 0.2,
            "lane_change_reward": lane_change_penalty,
            "merging_speed_reward": -merging_speed_penalty,  # Negative if ego too fast
            "off_road_reward": off_road_penalty,
        }

    def _is_terminated(self) -> bool:
        """End if ego crashes or drives off road (if enabled)."""
        return (
            self.vehicle.crashed
            or self.config.get("offroad_terminal", False) and not self.vehicle.on_road
            or self.vehicle.position[0] > 370
        )
    
    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Build a 2-to-1 lane zipper merge:
        - 2 lanes before merge
        - Lane 1 merges into lane 0
        """
        net = RoadNetwork()
        lane_w = StraightLane.DEFAULT_WIDTH

        # Segment lengths
        before = 300
        merge = 60
        after = 300

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # --- Section A->B: 2 parallel lanes ---
        for i in range(2):
            left_line = c if i == 0 else n
            right_line = s if i == 0 else c
            net.add_lane(
                "a", "b",
                StraightLane(
                    [0, lane_w * i], [before, lane_w * i],
                    line_types=[left_line, right_line],
                )
            )

        # --- Section B->C: Lane 1 merges into lane 0 ---
        # Lane 0 continues straight
        net.add_lane(
            "b", "c",
            StraightLane(
                [before, lane_w * 0], [before + merge + after, lane_w * 0],
                line_types=[c, c],
            )
        )

        # Lane 1 merges into lane 0 using sine curve
        lane1_start = [before, lane_w * 1]
        lane1_end = [before + merge, lane_w * 0]
        net.add_lane(
            "b", "c",
            SineLane(
                lane1_start, lane1_end,
                amplitude=lane_w / 4,  # reduce for smoother transition
                pulsation=np.pi / (merge * 2),  # slower curve
                phase=0,
                line_types=[n, c],
                width=lane_w,
            )
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
        two_lane_section = [("a", "b", i) for i in range(2)]
        one_lane_section = [("b", "c", 0)]  # final merged lane is only lane 0
        used_positions = []

        # Lane probabilities
        two_lane_weights = np.array([0.5, 0.5])
        one_lane_weights = np.array([1.0])

        def sample_valid_position(section="two", vehicle_class=None, max_attempts=50):
            if section == "two":
                lanes = two_lane_section
                weights = two_lane_weights
                x_range = (0, 150)
            else:
                lanes = one_lane_section
                weights = one_lane_weights
                x_range = (270, 400)

            for _ in range(max_attempts):
                lane_idx = lanes[self.np_random.choice(len(lanes), p=weights)]
                longitudinal = self.np_random.uniform(*x_range)

                if any(
                    ln == lane_idx and abs(longitudinal - pos) < 12
                    for ln, pos in used_positions
                ):
                    continue

                lane = road.network.get_lane(lane_idx)
                position = lane.position(longitudinal, 0)
                test_vehicle = (vehicle_class or Vehicle)(road, position)
                if not test_vehicle.on_road:
                    continue

                used_positions.append((lane_idx, longitudinal))
                return lane_idx, longitudinal

            lane_idx = lanes[self.np_random.choice(len(lanes), p=weights)]
            longitudinal = self.np_random.uniform(*x_range)
            used_positions.append((lane_idx, longitudinal))
            return lane_idx, longitudinal

        before_count = int(total * 0.7)
        after_count = total - before_count

        # Controlled vehicles in two-lane section
        for _ in range(controlled):
            lane_id, long = sample_valid_position("two")
            lane = road.network.get_lane(lane_id)
            speed = self.np_random.uniform(25, 35)
            v = self.action_type.vehicle_class(road, lane.position(long, 0), speed=speed)
            road.vehicles.append(v)
            self.controlled_vehicles.append(v)

        for v in self.road.vehicles:
            if not isinstance(v, ControlledVehicle):
                lane = v.lane
                if lane is None:
                    continue
                s, d = lane.local_coordinates(v.position)
                d = np.clip(d, -lane.width / 2, lane.width / 2)
                v.position = lane.position(s, d)

        # Other vehicles in two-lane section
        for _ in range(before_count - controlled):
            lane_id, long = sample_valid_position("two")
            lane = road.network.get_lane(lane_id)
            speed = self.np_random.uniform(20, 35)
            v = other_type(road, lane.position(long, 0), speed=speed)
            v.randomize_behavior()
            road.vehicles.append(v)

        # Other vehicles in one-lane section
        for _ in range(after_count):
            lane_id, long = sample_valid_position("after")
            lane = road.network.get_lane(lane_id)
            speed = self.np_random.uniform(20, 35)
            v = other_type(road, lane.position(long, 0), speed=speed)
            v.randomize_behavior()
            road.vehicles.append(v)

        for v in road.vehicles:
            if isinstance(v, ControlledVehicle):
                continue
            lane = v.lane
            if lane is None:
                continue
            s, d = lane.local_coordinates(v.position)
            d = np.clip(d, -lane.width / 2, lane.width / 2)
            v.position = lane.position(s, d)
