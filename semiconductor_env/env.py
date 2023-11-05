from collections import defaultdict, deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType


class SemiconductorEnv(gym.Env):
    # Machine states
    MACHINE_OPERATIONAL = 0
    MACHINE_BROKEN_COOLDOWN = 1
    MACHINE_MAINTENANCE = 2

    # Machine parameters
    MIN_CHIP_AREA = 0.001**2
    MAX_CHIP_AREA = 0.02**2
    WAFER_DIAMETER = 0.3
    MAX_THEORETICAL_CHIPS_PER_WAFER = np.floor(
        np.pi * (WAFER_DIAMETER / 2) ** 2 / MIN_CHIP_AREA
    )

    # Order parameters
    MAX_ORDER_SIZE = 20000
    MAX_FUTURE_ORDERS = 5

    # Industry parameters
    PRICE_PER_CM2 = 50 / 0.01**2
    BASELINE_YIELD = 0.3

    def __init__(self):
        machine_state = spaces.MultiBinary(3)

        machine_todays_production = spaces.Dict(
            {
                "yield": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
                "chips_produced": spaces.Box(
                    low=0,
                    high=self.MAX_THEORETICAL_CHIPS_PER_WAFER,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "max_temp_delta": spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float64
                ),
                "max_displacement_delta": spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float64
                ),
            }
        )

        machine_previous_state = spaces.Dict(
            {
                "days_since_last_maintenance": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),
                "days_since_last_broken": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),
            }
        )

        order_space = spaces.Dict(
            {
                "chips_left": spaces.Box(
                    low=0, high=self.MAX_ORDER_SIZE, shape=(1,), dtype=np.int64
                ),
                "price_per_chip": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "chip_area": spaces.Box(
                    low=self.MIN_CHIP_AREA,
                    high=self.MAX_CHIP_AREA,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "late_penalty_per_day": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "days_until_deadline": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),
                "days_past_deadline": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),
            }
        )

        orders = spaces.Tuple([order_space] * self.MAX_FUTURE_ORDERS)

        maintenance_space = spaces.Dict(
            {
                "scheduled": spaces.Discrete(2),
                "days_until_maintenance": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),
            }
        )

        future_maintenance = spaces.Tuple([maintenance_space] * self.MAX_FUTURE_ORDERS)

        self.observation_space = spaces.Dict(
            {
                "machine_state": machine_state,
                "machine_todays_production": machine_todays_production,
                "machine_previous_state": machine_previous_state,
                "orders": orders,
                "future_maintenance": future_maintenance,
            }
        )

        self.action_space = spaces.Discrete(5)

        self._action_lookup = {
            0: "do_nothing",
            1: "schedule_maintenance_in_4_weeks",
            2: "schedule_maintenance_in_8_weeks",
            3: "schedule_maintenance_in_12_weeks",
            4: "schedule_maintenance_in_24_weeks",
        }

    def _get_obs(self):
        future_maintenance = []
        for maintenance in self.state["future_maintenance"]:
            obs_maintenance = {
                "scheduled": maintenance["scheduled"],
                "days_until_maintenance": np.array(
                    [maintenance["days_until_maintenance"]]
                ),
            }
            future_maintenance.append(obs_maintenance)
        future_maintenance = tuple(future_maintenance)

        orders = []
        for order in self.state["orders"]:
            obs_order = {
                "chips_left": np.array([order["chips_left"]]),
                "price_per_chip": np.array([order["price_per_chip"]]),
                "chip_area": np.array([order["chip_area"]]),
                "late_penalty_per_day": np.array([order["late_penalty_per_day"]]),
                "days_until_deadline": np.array([order["days_until_deadline"]]),
                "days_past_deadline": np.array([order["days_past_deadline"]]),
            }
            orders.append(obs_order)

        orders = tuple(orders)

        obs = {
            "machine_state": self.state["machine_state"],
            "machine_todays_production": {
                "yield": np.array(
                    [self.state["machine_todays_production"]["yield"]], dtype=np.float64
                ),
                "chips_produced": np.array(
                    [self.state["machine_todays_production"]["chips_produced"]],
                    dtype=np.int64,
                ),
                "max_temp_delta": np.array(
                    [self.state["machine_todays_production"]["max_temp_delta"]],
                    dtype=np.float64,
                ),
                "max_displacement_delta": np.array(
                    [self.state["machine_todays_production"]["max_displacement_delta"]],
                    dtype=np.float64,
                ),
            },
            "machine_previous_state": {
                "days_since_last_maintenance": np.array(
                    [
                        self.state["machine_previous_state"][
                            "days_since_last_maintenance"
                        ]
                    ]
                ),
                "days_since_last_broken": np.array(
                    [self.state["machine_previous_state"]["days_since_last_broken"]]
                ),
            },
            "orders": orders,
            "future_maintenance": future_maintenance,
        }
        return obs

    def _get_reward(self):
        pass

    def _get_done(self):
        pass

    def _get_info(self):
        info = defaultdict(list)

        for order in self.state["orders"]:
            info["order_chip_sizes"].append(order["chip_area"] * 1e6)
            info["order_total_cost"].append(
                order["chips_left"] * order["price_per_chip"]
            )
            info["order_projected_yield"].append(order["projected_yield"])
            info["order_chips_per_wafer"].append(order["chips_per_wafer"])

        info["order_duration"].append(self.state["orders"][0]["days_until_deadline"])
        for i in range(1, len(self.state["orders"])):
            info["order_duration"].append(
                self.state["orders"][i]["days_until_deadline"]
                - self.state["orders"][i - 1]["days_until_deadline"]
            )

        return info

    def _chips_per_wafer(self, chip_width, chip_height):
        # we subdivide the wafer into a grid of rectangles
        # the number of chips is the number of rectangles that fit
        # inside the wafer. To calculate the number of rectangles,
        # we check every point in the grid and see if it is inside
        # the wafer. We do this by checking if the distance from
        # the point to the center of the wafer is less than the
        # radius of the wafer. Then we loop over the rectangles
        # and check if they are inside the wafer.
        x_points = np.arange(
            -self.WAFER_DIAMETER / 2, self.WAFER_DIAMETER / 2, chip_width
        )
        y_points = np.arange(
            -self.WAFER_DIAMETER / 2, self.WAFER_DIAMETER / 2, chip_height
        )
        points = np.array(np.meshgrid(x_points, y_points)).T.reshape(-1, 2)

        # calculate the distance from the center of the wafer
        # to each point in the grid
        distances = np.linalg.norm(points, axis=1)

        # now we can calculate the number of chips
        chips_per_wafer = 0
        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                rect_distances = np.array(
                    [
                        distances[i * len(y_points) + j],
                        distances[i * len(y_points) + j + 1],
                        distances[(i + 1) * len(y_points) + j],
                        distances[(i + 1) * len(y_points) + j + 1],
                    ]
                )
                if np.all(rect_distances < self.WAFER_DIAMETER / 2):
                    chips_per_wafer += 1

        return chips_per_wafer

    def _add_order(self):
        order_amount = np.random.normal(3 / 4 * self.MAX_ORDER_SIZE, 1000)
        # round to the closest multiple of 1000
        order_amount = np.round(order_amount / 1000) * 1000
        order_amount = np.clip(order_amount, 0, self.MAX_ORDER_SIZE).astype(np.int64)
        order_chip_area = (
            np.random.random() * (self.MAX_CHIP_AREA - self.MIN_CHIP_AREA)
            + self.MIN_CHIP_AREA
        )

        # assume that the aspect ratio of the chip is 3:2
        chip_width = np.sqrt(order_chip_area * 2 / 3)
        chip_height = np.sqrt(order_chip_area * 3 / 2)
        chips_per_wafer = self._chips_per_wafer(chip_width, chip_height)

        # chip price depends on chip area
        order_price_per_chip = order_chip_area * self.PRICE_PER_CM2

        # calculate projected yield to establish a sensible deadline
        # let's say that the yield is baseline + 0.5 * (1 - chip_area / MAX_CHIP_AREA)
        projected_yield = self.BASELINE_YIELD + 0.5 * (
            1 - order_chip_area / self.MAX_CHIP_AREA
        )

        # now we can establish a deadline that is possible to meet
        # let's say that the deadline is order_amount / (yield * chips_per_wafer)
        deadline = order_amount / (projected_yield * chips_per_wafer)
        # round to a full day
        deadline = np.round(deadline).astype(np.int64)

        # if there is an existing future order, this order will be right after it
        if len(self.state["orders"]) > 0:
            last_order = self.state["orders"][-1]
            deadline += last_order["days_until_deadline"]

        # construct the order dict
        order = {
            "chips_left": order_amount,
            "price_per_chip": order_price_per_chip,
            "chip_area": order_chip_area,
            "late_penalty_per_day": 5
            * chips_per_wafer
            * projected_yield
            * order_price_per_chip,
            "days_until_deadline": deadline,
            "days_past_deadline": 0,
            # these are used for info only
            "projected_yield": projected_yield,
            "chips_per_wafer": chips_per_wafer,
        }

        self.state["orders"].append(order)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.state = {"machine_state": np.zeros(3)}

        self.state["machine_state"][self.MACHINE_OPERATIONAL] = 1

        self.state["machine_todays_production"] = {
            "yield": 1,
            "chips_produced": 0,
            "max_temp_delta": 0,
            "max_displacement_delta": 0,
        }

        self.state["machine_previous_state"] = {
            "days_since_last_maintenance": 0,
            "days_since_last_broken": 0,
        }

        self.state["orders"] = deque()
        for i in range(5):
            self._add_order()

        self.state["future_maintenance"] = []
        for i in range(5):
            self.state["future_maintenance"].append(
                {"scheduled": 0, "days_until_maintenance": 0}
            )

        return self._get_obs(), self._get_info()
