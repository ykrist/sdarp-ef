import random
import math
import textwrap
import dataclasses
from typing import Tuple, List
from hypothesis import given, strategies as st, note, settings

@dataclasses.dataclass(frozen=True)
class LocData:
    coords: Tuple[float, float]
    tw: Tuple[float, float]


@dataclasses.dataclass(frozen=True)
class RequestInfo:
    size: int
    srv_time: float
    pickup: LocData
    delivery: LocData

@dataclasses.dataclass(frozen=True)
class Instance:
    n_vehicles: int
    vehicle_cap: int
    ride_time: float
    time_horizon: float
    requests: List[RequestInfo]

    def to_riedler_fmt(self) -> str:
        s = textwrap.dedent(f"""\
        |N|: {len(self.requests)}
        |K|: {self.n_vehicles}
        L: {self.ride_time}
        Depot: 0 0 0 {self.time_horizon}
        Vehicles
        """)

        for _ in range(self.n_vehicles):
            s += f"{self.vehicle_cap} {self.time_horizon}\n"

        def req_line(r: RequestInfo) -> str:
            return f"{r.pickup.coords[0]:.7f} {r.pickup.coords[1]:.7f} {r.pickup.tw[0]} {r.pickup.tw[1]} " \
                   f"{r.delivery.coords[0]:.7f} {r.delivery.coords[1]:.7f} {r.delivery.tw[0]} {r.delivery.tw[1]} " \
                   f"{r.size} {r.srv_time}\n"


        s += "Requests\n"
        for r in self.requests:
            s += req_line(r)

        return s

def travel_time(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class InstanceGenerator:
    def __init__(self, n_req, n_vehicles, vehicle_cap, tw_size, ride_time, time_horizon):
        self.n_req = n_req
        self.n_vehicles = n_vehicles
        self.tw_size = tw_size
        self.ride_time = ride_time
        self.time_horizon = time_horizon
        self.vehicle_cap = vehicle_cap


    def run(self, seed) -> Instance:
        rng = random.Random(seed)

        def coord():
            nonlocal rng
            x = rng.random()*20 - 10
            y = rng.random()*20 - 10
            return (x,y)


        pickups = [coord() for _ in range(self.n_req)]
        deliveries = [coord() for _ in range(self.n_req)]
        srv_time = 3
        depot = (0, 0)

        def outbound_tw_pair(i):
            nonlocal self, rng, pickups, deliveries, srv_time, depot
            a =  travel_time(depot, pickups[i]) + srv_time + travel_time(pickups[i], deliveries[i])
            a = math.ceil(a)
            b = self.time_horizon - travel_time(deliveries[i], depot) - srv_time - self.tw_size
            b = math.floor(b)
            delivery_lb = rng.uniform(a, b)
            delivery_ub = delivery_lb + self.tw_size

            pickup_lb = max(delivery_lb - srv_time - self.ride_time, travel_time(depot, pickups[i]))
            pickup_ub = delivery_ub - travel_time(pickups[i], deliveries[i]) - srv_time

            return (round(pickup_lb), round(pickup_ub)), (round(delivery_lb), round(delivery_ub))

        def inbound_tw_pair(i):
            nonlocal self, rng, pickups, deliveries, srv_time, depot
            a =  travel_time(depot, pickups[i])
            a = math.ceil(a)
            b = self.time_horizon - travel_time(deliveries[i], depot) - srv_time - travel_time(pickups[i], deliveries[i]) - srv_time - self.tw_size
            b = math.floor(b)
            pickup_lb = rng.uniform(a, b)
            pickup_ub = pickup_lb + self.tw_size

            delivery_lb = pickup_lb + srv_time + travel_time(pickups[i], deliveries[i])
            delivery_ub = min(pickup_ub + srv_time + self.ride_time, self.time_horizon - travel_time(deliveries[i], depot))

            return (round(pickup_lb), round(pickup_ub)), (round(delivery_lb), round(delivery_ub))

        # First n/2 requests are outbound requests
        n_outbound = self.n_req // 2
        requests = []
        tw_pairs = [outbound_tw_pair(i) for i in range(n_outbound)] + [inbound_tw_pair(i) for i in range(n_outbound, self.n_req)]

        for (i, (ptw, dtw)) in enumerate(tw_pairs):
            requests.append(RequestInfo(
                size=1,
                srv_time=srv_time,
                pickup=LocData(pickups[i], ptw),
                delivery=LocData(deliveries[i], dtw),
            ))

        return Instance(
            n_vehicles=self.n_vehicles,
            ride_time= self.ride_time,
            time_horizon= self.time_horizon,
            vehicle_cap=self.vehicle_cap,
            requests=requests,
        )


@given(st.integers())
@settings(max_examples=10_000)
def test_times_are_within_range(i):
    data = InstanceGenerator(
        n_req=30,
        n_vehicles=5,
        vehicle_cap=3,
        tw_size=30,
        ride_time=60,
        time_horizon=240,
    ).run(i)

    for i, r in enumerate(data.requests):
        note(r)
        note(i)

        assert r.pickup.tw[0] >= 0
        assert r.pickup.tw[1] >= 0
        assert r.delivery.tw[0] >= 0
        assert r.delivery.tw[1] >= 0

        assert r.pickup.tw[0] <= data.time_horizon
        assert r.pickup.tw[1] <= data.time_horizon
        assert r.delivery.tw[0] <= data.time_horizon
        assert r.delivery.tw[1] <= data.time_horizon


def iter_parameter_combinations(params: dict, fixed=None):
    if fixed is None:
        fixed = {}

    if not params:
        yield fixed
        return

    params = params.copy()
    key, values = params.popitem()

    for val in values:
        f = fixed.copy()
        f[key] = val
        yield from iter_parameter_combinations(params, f)


if __name__ == '__main__':
    from pathlib import Path
    dest = Path("data/new-instances")
    dest.mkdir(exist_ok=True, parents=True)

    instance_sizes = [30, 45, 60]
    default_parameters = dict(
        n_vehicles=3,
        vehicle_cap=3,
        tw_size=15,
        ride_time=30,
        time_horizon=240,
    )
    parameter_changes = [
        {"tw_size" : [22, 30], "n_req": instance_sizes },
        {"ride_time": [45], "n_req": instance_sizes },
        {"vehicle_cap" : [100], "n_req": instance_sizes},
    ]

    def instance_name(n_req, n_vehicles, tw, ride_time, vehicle_cap):
        # nonlocal default_parameters
        name = f"{n_req}N_{n_vehicles}K"
        if tw != default_parameters["tw_size"]:
            name += f"_TW{tw}"
        if ride_time != default_parameters["ride_time"]:
            name += f"_L{ride_time}"
        if vehicle_cap != default_parameters['vehicle_cap']:
            name += f"_Q{vehicle_cap}"
        return name

    generators = []

    for p in parameter_changes:
        for param in iter_parameter_combinations(p, default_parameters):
            generators.append(InstanceGenerator(**param))

    generators.sort(key=lambda g: (g.n_req, g.tw_size, g.ride_time))



    for gen in generators:
        for (seed, l) in zip([26996, 49243, 63405], 'abcdefghijklmnopqrst'):
            data = gen.run(seed)
            name = instance_name(gen.n_req, gen.n_vehicles, gen.tw_size, gen.ride_time, gen.vehicle_cap)
            filename = name + "_" + l.upper() + ".dat"
            with open(dest/filename, "w") as fp:
                fp.write(data.to_riedler_fmt())
