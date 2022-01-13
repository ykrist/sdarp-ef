from fraglibpy import *
import utils.data
from oru import frozendict

def unwrap(data):
    if isinstance(data, apvrp.ApvrpDataWrapper):
        assert data.n_loc == 2*(data.n_passive + data.n_req) + 1
        return utils.data.APVRP_Data(
            id=data.id,
            n_req=data.n_req,
            n_loc=data.n_loc,
            n_passive=data.n_passive,
            n_active=data.n_active,
            tmax=data.tmax,
            tw_start=frozendict(data.start_time),
            tw_end=frozendict(data.end_time),
            srv_time=frozendict(data.srv_time),
            compat_req_passive=frozendict({k: frozenset(v) for k,v in data.compat_req_passive.items()}),
            compat_passive_req=frozendict({k: frozenset(v) for k,v in data.compat_passive_req.items()}),
            compat_passive_active=frozendict({k: frozenset(v) for k,v in data.compat_passive_active.items()}),
            travel_cost=frozendict(data.travel_cost),
            travel_time=frozendict(data.travel_time),
        )

    elif isinstance(data, sdarp.SdarpDataWrapper):
        n = data.num_req
        k = data.num_vehicles
        return utils.data.SDARP_Data(
            id=data.id,
            travel_time=frozendict(data.travel_time),
            demand=frozendict(data.demand),
            tw_start=frozendict(data.tw_start),
            tw_end=frozendict(data.tw_end),
            max_ride_time=frozendict(data.max_ride_time),
            capacity=data.capacity,
            o_depot=0,
            d_depot=2 * n + 1,
            n=n,
            P=range(1, n+1),
            D=range(n+1, 2*n + 1),
            K = range(k),
            N=range(2*n + 2),
        )
    else:
        raise NotImplementedError

