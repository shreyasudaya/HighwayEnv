from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle

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

        # Check if the ego-vehicle is on a valid lane
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, None)
        in_lane = lane_index[0] is not None
        off_road = not in_lane
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
            "off_road_reward": float(off_road),
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
                amplitude=lane_w/2,                  # stronger lateral shift
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
                amplitude=lane_w/2,
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

        # Lane probabilities
        four_lane_weights = np.array([0.1, 0.2, 0.35, 0.35])
        two_lane_weights = np.array([0.5, 0.5])

        def sample_valid_position(section="four", vehicle_class=None, max_attempts=50):
            if section == "four":
                lanes = four_lane_section
                weights = four_lane_weights
                x_range = (0, 150)
            else:
                lanes = two_lane_section
                weights = two_lane_weights
                x_range = (270, 400)

            for _ in range(max_attempts):
                lane_idx = lanes[self.np_random.choice(len(lanes), p=weights)]
                longitudinal = self.np_random.uniform(*x_range)

                if any(
                    ln == lane_idx and abs(longitudinal - pos) < 12
                    for ln, pos in used_positions
                ):
                    continue  # Too close to existing vehicle

                lane = road.network.get_lane(lane_idx)
                position = lane.position(longitudinal, 0)

                # Create a dummy vehicle to test on_road status
                test_vehicle = (vehicle_class or Vehicle)(road, position)
                if not test_vehicle.on_road:
                    continue  # Skip off-road placements

                used_positions.append((lane_idx, longitudinal))
                return lane_idx, longitudinal

            # Fallback if all attempts fail
            lane_idx = lanes[self.np_random.choice(len(lanes), p=weights)]
            longitudinal = self.np_random.uniform(*x_range)
            used_positions.append((lane_idx, longitudinal))
            return lane_idx, longitudinal


        before_count = int(total * 0.7)
        after_count = total - before_count

        # Controlled vehicles — always in 4-lane section
        for _ in range(controlled):
            lane_id, long = sample_valid_position("four")
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
                # Get longitudinal and lateral coordinates of vehicle relative to lane
                s, d = lane.local_coordinates(v.position)
                # Clamp lateral offset to lane width boundaries (e.g., ±lane_width/2)
                max_lateral_offset = lane.width / 2
                if abs(d) > max_lateral_offset:
                    d = np.clip(d, -max_lateral_offset, max_lateral_offset)
                    # Set position back to the clamped point on lane centerline
                    v.position = lane.position(s, d)

        # Other vehicles before merge
        for _ in range(before_count - controlled):
            lane_id, long = sample_valid_position("four")
            lane = road.network.get_lane(lane_id)
            speed = self.np_random.uniform(20, 35)
            v = other_type(road, lane.position(long, 0), speed=speed)
            v.randomize_behavior()
            road.vehicles.append(v)

        # Other vehicles after merge
        for _ in range(after_count):
            lane_id, long = sample_valid_position("after")
            lane = road.network.get_lane(lane_id)
            speed = self.np_random.uniform(20, 35)
            v = other_type(road, lane.position(long, 0), speed=speed)
            v.randomize_behavior()
            road.vehicles.append(v)
