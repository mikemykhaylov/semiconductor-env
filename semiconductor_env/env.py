from collections import defaultdict, deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType


class SemiconductorEnv(gym.Env):
    # Machine states
    MACHINE_OPERATIONAL = 0
    MACHINE_BROKEN = 1
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
    IDEAL_BASELINE_YIELD = 0.35
    EQUIPMENT_YIELD_MINUS_PER_DAY = 0.001
    FAILURE_PROB_PLUS_PER_UNMAINT_DAY = 0.001

    # Maintenance parameters
    MIN_TIME_UNTIL_MAINTENANCE = 7
    MAX_TIME_UNTIL_MAINTENANCE = 180

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
                # "max_temp_delta": spaces.Box(
                #     low=0, high=1, shape=(1,), dtype=np.float64
                # ),
                # "max_displacement_delta": spaces.Box(
                #     low=0, high=1, shape=(1,), dtype=np.float64
                # ),
            }
        )

        machine_previous_state = spaces.Dict(
            {
                "days_since_last_maintenance": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int64
                ),
                "days_since_last_broken": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int64
                ),
            }
        )

        max_price_per_chip = self.MAX_CHIP_AREA * self.PRICE_PER_CM2
        max_late_penalty_per_day = (
            5
            * self._chips_per_wafer(0.001, 0.001)
            * 0.8
            * self.MAX_CHIP_AREA
            * self.PRICE_PER_CM2
        )
        max_days_until_deadline = (
            5 * self.MAX_ORDER_SIZE / (0.3 * self._chips_per_wafer(0.02, 0.02))
        )

        order_space = spaces.Dict(
            {
                "chips_left": spaces.Box(
                    low=0, high=self.MAX_ORDER_SIZE, shape=(1,), dtype=np.int64
                ),
                "price_per_chip": spaces.Box(
                    low=0,
                    high=max_price_per_chip,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "chip_area": spaces.Box(
                    low=self.MIN_CHIP_AREA,
                    high=self.MAX_CHIP_AREA,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "late_penalty_per_day": spaces.Box(
                    low=0,
                    high=max_late_penalty_per_day,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "days_until_deadline": spaces.Box(
                    low=0,
                    high=max_days_until_deadline,
                    shape=(1,),
                    dtype=np.int64,
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
                    low=0, high=180, shape=(1,), dtype=np.int64
                ),
            }
        )

        # future_maintenance = spaces.Tuple([maintenance_space] * self.MAX_FUTURE_ORDERS)

        self.observation_space = spaces.Dict(
            {
                "machine_state": machine_state,
                "machine_todays_production": machine_todays_production,
                "machine_previous_state": machine_previous_state,
                "orders": orders,
                "future_maintenance": maintenance_space,
            }
        )

        self.action_space = spaces.Discrete(7)

        self._action_lookup = {
            0: "do_nothing",
            1: "schedule_maintenance_in_7_days",
            2: "schedule_maintenance_in_14_days",
            3: "schedule_maintenance_in_30_days",
            4: "schedule_maintenance_in_60_days",
            5: "schedule_maintenance_in_90_days",
            6: "deschedule_maintenance",
        }

    def _get_obs(self):
        orders = []
        for order in self.state["orders"]:
            obs_order = {
                "chips_left": np.array([order["chips_left"]], dtype=np.int64),
                "price_per_chip": np.array([order["price_per_chip"]], dtype=np.float64),
                "chip_area": np.array([order["chip_area"]], dtype=np.float64),
                "late_penalty_per_day": np.array(
                    [order["late_penalty_per_day"]], dtype=np.float64
                ),
                "days_until_deadline": np.array(
                    [order["days_until_deadline"]], dtype=np.int64
                ),
                "days_past_deadline": np.array(
                    [order["days_past_deadline"]], dtype=np.int64
                ),
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
                # "max_temp_delta": np.array(
                #     [self.state["machine_todays_production"]["max_temp_delta"]],
                #     dtype=np.float64,
                # ),
                # "max_displacement_delta": np.array(
                #     [self.state["machine_todays_production"]["max_displacement_delta"]],
                #     dtype=np.float64,
                # ),
            },
            "machine_previous_state": {
                "days_since_last_maintenance": np.array(
                    [
                        self.state["machine_previous_state"][
                            "days_since_last_maintenance"
                        ]
                    ],
                    dtype=np.int64,
                ),
                "days_since_last_broken": np.array(
                    [self.state["machine_previous_state"]["days_since_last_broken"]],
                    dtype=np.int64,
                ),
            },
            "orders": orders,
            "future_maintenance": {
                "scheduled": self.state["future_maintenance"]["scheduled"],
                "days_until_maintenance": np.array(
                    [self.state["future_maintenance"]["days_until_maintenance"]],
                    dtype=np.int64,
                ),
            },
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

    def _update_orders(self, chips_produced):
        # make all orders 1 day older
        for order in self.state["orders"]:
            order["days_past_deadline"] += 1 if order["days_until_deadline"] == 0 else 0
            order["days_until_deadline"] = max(0, order["days_until_deadline"] - 1)

        # sell chips to the first order
        current_order = self.state["orders"][0]
        if chips_produced > current_order["chips_left"]:
            chips_produced = current_order["chips_left"]

        current_order["chips_left"] -= chips_produced
        profit = (
            chips_produced * current_order["price_per_chip"]
            - current_order["late_penalty_per_day"]
            * current_order["days_past_deadline"]
        )

        # if the order is complete, remove it and add a new one
        if current_order["chips_left"] == 0:
            # if current_order["days_past_deadline"] > 0:
            #     print(
            #         f"Order completed {current_order['days_past_deadline']} days late"
            #     )
            # elif current_order["days_until_deadline"] == 0:
            #     print("Order completed on time")
            # else:
            #     print(
            #         f"Order completed {current_order['days_until_deadline']} days early"
            #     )
            self.state["orders"].popleft()
            self._add_order()

        return profit

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        # set the random seed
        if seed is not None:
            np.random.seed(seed)

        self.state = {"machine_state": np.zeros(3)}

        self.state["machine_state"][self.MACHINE_OPERATIONAL] = 1

        self.state["machine_todays_production"] = {
            "yield": 1,
            "chips_produced": 0,
            # "max_temp_delta": 0,
            # "max_displacement_delta": 0,
        }

        self.state["machine_previous_state"] = {
            "days_since_last_maintenance": 0,
            "days_since_last_broken": 0,
        }

        self.state["orders"] = deque()
        for i in range(5):
            self._add_order()

        self.state["future_maintenance"] = {"scheduled": 0, "days_until_maintenance": 0}

        return self._get_obs(), self._get_info()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, np.float64, bool, bool, dict[str, Any]]:
        self.state["machine_previous_state"]["days_since_last_maintenance"] += 1
        self.state["machine_previous_state"]["days_since_last_broken"] += 1

        if self._action_lookup[action] == "schedule_maintenance_in_7_days":
            self.state["future_maintenance"]["scheduled"] = 1
            self.state["future_maintenance"]["days_until_maintenance"] = 7
        elif self._action_lookup[action] == "schedule_maintenance_in_14_days":
            self.state["future_maintenance"]["scheduled"] = 1
            self.state["future_maintenance"]["days_until_maintenance"] = 14
        elif self._action_lookup[action] == "schedule_maintenance_in_30_days":
            self.state["future_maintenance"]["scheduled"] = 1
            self.state["future_maintenance"]["days_until_maintenance"] = 30
        elif self._action_lookup[action] == "schedule_maintenance_in_60_days":
            self.state["future_maintenance"]["scheduled"] = 1
            self.state["future_maintenance"]["days_until_maintenance"] = 60
        elif self._action_lookup[action] == "schedule_maintenance_in_90_days":
            self.state["future_maintenance"]["scheduled"] = 1
            self.state["future_maintenance"]["days_until_maintenance"] = 90
        elif self._action_lookup[action] == "deschedule_maintenance":
            self.state["future_maintenance"]["scheduled"] = 0
            self.state["future_maintenance"]["days_until_maintenance"] = 0

        # remove maintenance if machine was maintained yesterday
        if self.state["machine_state"][self.MACHINE_MAINTENANCE] == 1:
            self.state["machine_state"][self.MACHINE_MAINTENANCE] = 0
            self.state["machine_state"][self.MACHINE_OPERATIONAL] = 1
            self.state["machine_previous_state"]["days_since_last_maintenance"] = 0

        # advance the maintenance schedule if necessary
        if self.state["future_maintenance"]["scheduled"] == 1:
            self.state["future_maintenance"]["days_until_maintenance"] -= 1
            if self.state["future_maintenance"]["days_until_maintenance"] == 0:
                self.state["future_maintenance"]["scheduled"] = 0
                # if the machine is currently broken, maintenance can't be performed
                if self.state["machine_state"][self.MACHINE_BROKEN] != 1:
                    self.state["machine_state"][self.MACHINE_OPERATIONAL] = 0
                    self.state["machine_state"][self.MACHINE_MAINTENANCE] = 1
                    self.state["future_maintenance"]["scheduled"] = 0

        # check that machine is not in two states at once
        if np.sum(self.state["machine_state"]) > 1:
            print(self.state)
            raise ValueError("Machine is in two states at once")

        # process the current machine state
        if self.state["machine_state"][self.MACHINE_OPERATIONAL] == 1:
            # calculate probability of machine breaking based on days since last maintenance
            # and days since last broken
            prob_break = min(
                1.0,
                self.FAILURE_PROB_PLUS_PER_UNMAINT_DAY
                * self.state["machine_previous_state"]["days_since_last_maintenance"],
            )
            if np.random.random() < prob_break:
                self.state["machine_state"][self.MACHINE_OPERATIONAL] = 0
                self.state["machine_state"][self.MACHINE_BROKEN] = 1
                self.state["machine_previous_state"]["days_since_last_broken"] = 0
        elif self.state["machine_state"][self.MACHINE_BROKEN] == 1:
            if self.state["machine_previous_state"]["days_since_last_broken"] >= 3:
                self.state["machine_state"][self.MACHINE_BROKEN] = 0
                self.state["machine_state"][self.MACHINE_MAINTENANCE] = 1
        elif self.state["machine_state"][self.MACHINE_MAINTENANCE] != 1:
            raise ValueError("Invalid machine state")

        current_order = self.state["orders"][0]

        # if the machine is operational, produce chips
        chip_yield = max(
            self.IDEAL_BASELINE_YIELD
            + 0.5 * (1 - current_order["chip_area"] / self.MAX_CHIP_AREA)
            - self.EQUIPMENT_YIELD_MINUS_PER_DAY
            * self.state["machine_previous_state"]["days_since_last_maintenance"],
            self.IDEAL_BASELINE_YIELD,
        )

        # multiply by operational state to get 0 if machine is broken or in maintenance
        chips_produced = (
            int(chip_yield * current_order["chips_per_wafer"])
            * self.state["machine_state"][self.MACHINE_OPERATIONAL]
        )

        profit = self._update_orders(chips_produced)

        # update state
        self.state["machine_todays_production"]["yield"] = (
            chip_yield
            if self.state["machine_state"][self.MACHINE_OPERATIONAL] == 1
            else 0
        )
        self.state["machine_todays_production"]["chips_produced"] = chips_produced

        if self.state["machine_state"][self.MACHINE_OPERATIONAL] == 1:
            reward = profit
        elif self.state["machine_state"][self.MACHINE_BROKEN] == 1:
            reward = -5000
        elif self.state["machine_state"][self.MACHINE_MAINTENANCE] == 1:
            reward = -1000

        return self._get_obs(), reward, False, False, self._get_info()
