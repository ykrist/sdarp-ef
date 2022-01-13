from sdarp import *
from utils.data import get_named_instance_SDARP, get_index_by_name
import subprocess
import io
from oru.grb import BinVarDict, IntVar, CtsVar
from typing import Iterable, TypeVar
from oru import TablePrinter
from utils.graph import decompose_paths_and_cycles
import bisect
import fraglib

TIME_SCALE = 10 ** 5

PRINT_HLINE_WIDTH = 120

@frozen_dataclass
class TNode(SerialisableFrozenSlottedDataclass):
    __slots__ = ['loc', 't']
    loc: int
    t: int

    def __str__(self):
        return f"({self.loc}, t={self.t:,d})"


@frozen_dataclass
class TExFrag(LazyHashFrozenDataclass, SerialisableFrozenSlottedDataclass):
    __slots__ = ['start', 'end', 'path', 'tef', 'tls', 'ttt']
    start: TNode
    end: TNode
    path: Tuple[int]
    tef: int
    tls: int
    ttt: int

    def __str__(self):
        return str(self.path)

    def __hash__(self):
        return LazyHashFrozenDataclass.__hash__(self) # i think MRO is fucked with dataclasses

@frozen_dataclass
class ExFrag(LazyHashFrozenDataclass, SerialisableFrozenSlottedDataclass):
    __slots__ = ['start', 'end', 'path', 'tef', 'tls', 'ttt']
    start: int
    end: int
    path: Tuple[int]
    tef: int
    tls: int
    ttt: int

    def __str__(self):
        return str(self.path)

    def __hash__(self):
        return LazyHashFrozenDataclass.__hash__(self)


class Network:
    data: SDARP_Data

    def __init__(self, data: SDARP_Data, time_delta, preprocess=True, domination=True):
        self.o_depot = TNode(loc=data.o_depot, t=data.tw_start[data.o_depot])
        self.d_depot = TNode(loc=data.d_depot, t=data.tw_end[data.o_depot])
        self.data = data
        self.preprocess = preprocess
        self.domination = domination

        self.ef = []
        self.ef_by_start = {}
        self.ef_by_end = {}
        self.ef_by_loc_inc_start = {}
        self.ef_by_loc_inc_end = {}
        self.start_ef = {}
        self.end_ef = {}
        self.ef_by_start_loc = {}
        self.ef_by_end_loc = {}
        self.delta = round(time_delta * TIME_SCALE)
        self.nodes_by_loc = {}
        self.wait_arcs_by_end = {}
        self.wait_arcs_by_start = {}
        self.wait_arcs = []

        self._rust_size_info = None

        self.ef_by_path = {} # maps path to a list of TExFrag, sorted by ascending time order
        self.paths_by_locset = {}

    def loc_triple(self, i, j, k):
        a = set(self.ef_by_loc_inc_start[i])
        b = set(self.ef_by_loc_inc_start[j])
        c = set(self.ef_by_loc_inc_start[k])
        return (a & b) | (a & c) | (b & c)


    def size_info(self):
        size_info = {
            "timed_ef" : len(self.ef),
            "nodes" : sum(map(len, self.nodes_by_loc.values())) + 2,
            "arcs" : len(self.wait_arcs)
        }

        if self._rust_size_info is not None:
            size_info['fragments'] = self._rust_size_info['undominated_fragments']
            size_info['dominated_fragments'] = self._rust_size_info['fragments'] - size_info['fragments']
            size_info['ef'] = self._rust_size_info['undominated_ef']
            size_info['dominated_ef'] = self._rust_size_info['ef'] - size_info['ef']

        return size_info

    def _discretise_time_window(self, loc: int) -> List[TNode]:
        tw_start = self.data.tw_start[loc]
        tw_end = self.data.tw_end[loc]
        times = range(tw_start, tw_end + 1, self.delta)
        return [TNode(loc=loc, t=t) for t in times]

    def _round_time(self, loc: int, t: int) -> int:
        s = self.data.tw_start[loc]
        assert t >= s
        assert t <= self.data.tw_end[loc]
        return s + ((t - s) // self.delta) * self.delta


    def build(self, data, cpus : int, travel_time_obj : bool):
        def to_exfrag(vals: List):
            p, tef, tls, ttt = vals
            return ExFrag(start=p[0], end=p[-1], path=tuple(p),
                          tef=tef, tls=tls, ttt=ttt)

        if self.domination:
            if travel_time_obj:
                domination = "tt"
            else:
                domination = "cover"
        else:
            domination = None

        rust_erf, info = fraglib.sdarp.extended_restricted_fragments(data, domination, cpus=cpus)
        self._rust_size_info = info

        num_tnodes = 0
        for p in self.data.P:
            tnodes = self._discretise_time_window(p)
            self.nodes_by_loc[p] = tnodes
            num_tnodes += len(tnodes)

        print(f"{num_tnodes:,d} initial nodes")

        self.untimed_ef = list(map(to_exfrag, rust_erf))
        print(f"{len(self.untimed_ef):,d} untimed extended fragments")

        for efid, f in enumerate(self.untimed_ef):
            if f.start == self.data.o_depot:
                end_node = TNode(f.end, self._round_time(f.end, f.tef))
                assert end_node in self.nodes_by_loc[end_node.loc]
                tf = TExFrag(start=self.o_depot, end=end_node, path=f.path,tef=f.tef,tls=f.tls,ttt=f.ttt)
                self.start_ef[f.end] = tf
                group = [tf]

            elif f.end == self.data.d_depot:
                start_node = TNode(f.start, self._round_time(f.start, f.tls))
                assert start_node in self.nodes_by_loc[start_node.loc]
                tf = TExFrag(start=start_node, end=self.d_depot,path=f.path,tef=f.tef,tls=f.tls,ttt=f.ttt)
                if len(f.path) == 3:
                    self.end_ef[f.start] = tf
                group = [tf]

            else:
                group = []
                end_times = []
                start_nodes = []
                for start_node in self.nodes_by_loc[f.start]:
                    if start_node.t > f.tls:
                        break

                    t = max(f.tef, start_node.t + f.ttt)
                    t = self._round_time(f.end, t)

                    if len(end_times) == 0 or end_times[-1] != t:
                        start_nodes.append(start_node)
                        end_times.append(t)
                    else:
                        start_nodes[-1] = start_node  # replace start node, would lead to dominated fragment.

                for start_node, t in zip(start_nodes, end_times):
                    end_node = TNode(f.end, t)
                    tf = TExFrag(start=start_node, end=end_node, tef=f.tef, tls=f.tls, ttt=f.ttt, path=f.path)
                    group.append(tf)

            self.ef.extend(group)
            self.ef_by_path[f.path] = group
            # Important: Only one timed fragment for each untimed fragment is stored here:
            self.ef_by_end_loc.setdefault(f.end, []).append(group[0])
            self.ef_by_start_loc.setdefault(f.start, []).append(group[-1])
            self.paths_by_locset.setdefault(frozenset(f.path), set()).add(f.path)

        for f in self.ef:
            self.ef_by_start.setdefault(f.start, set()).add(f)
            self.ef_by_end.setdefault(f.end, set()).add(f)

        print(f"{len(self.ef):,d} timed extended fragments")
        self._remove_excess_nodes()
        self._add_wait_arcs()
        print(f"{len(self.wait_arcs):,d} waiting arcs")
        self._build_lookups()

    def _remove_excess_nodes(self):
        # ASSUMPTION: the earliest timed node is accessible from the o-depot, which is true if TW have been tightened
        #   likewise, there should be a fragment leaving the last node should also be accessible
        num_removed = 0
        for i in self.data.P:
            assert self.nodes_by_loc[i][0] in self.ef_by_end
            assert self.nodes_by_loc[i][-1] in self.ef_by_start
            nodes = self.nodes_by_loc[i]
            num = len(nodes)
            nodes = [n for n in nodes if n in self.ef_by_start or n in self.ef_by_end]
            num_removed += num - len(nodes)
            self.nodes_by_loc[i] = nodes
        print(f"removed {num_removed:,d} detached nodes.")

        return

    def _add_wait_arcs(self):
        for p in self.data.P:
            nodes = self.nodes_by_loc[p]

            for arc in zip(nodes, nodes[1:]):
                self.wait_arcs.append(arc)
                assert arc not in self.wait_arcs_by_start
                self.wait_arcs_by_start[arc[0]] = arc
                assert arc not in self.wait_arcs_by_end
                self.wait_arcs_by_end[arc[1]] = arc

    def _build_lookups(self):
        for f in self.ef:
            self.ef_by_start.setdefault(f.start, set()).add(f)
            self.ef_by_end.setdefault(f.end, set()).add(f)

        for path, frags in self.ef_by_path.items():
            for i in path[:-1]:
                self.ef_by_loc_inc_start.setdefault(i, set()).update(frags)

            for i in path[1:]:
                self.ef_by_loc_inc_end.setdefault(i, set()).update(frags)

    @memoise
    def legal_after_timed(self, f : TExFrag):
        assert f.end != self.d_depot and f.start != self.o_depot
        legal = []
        exact_end_t = max(f.start.t + f.ttt, f.tef)
        locset = set(f.path[:-1])
        for g in self.ef_by_start_loc[f.end.loc]:
            if exact_end_t > g.tls: # g.tls implicitly includes the delivery and d_depot if the last loc of g is a pickup
                continue

            if not locset.isdisjoint(g.path): # include last loc here
                continue

            legal.extend(g for g in self.ef_by_path[g.path] if g.start.t >= f.end.t)

        assert len(legal) > 0
        return legal


    @memoise
    def legal_before_timed(self, f: TExFrag):
        assert f.start != self.o_depot
        legal = []

        locset = set(f.path) # include last loc, accounts for `h` if it was appended
        tnodes = self.nodes_by_loc[f.start.loc]
        tnodes = tnodes[:tnodes.index(f.start)+1]

        legal_paths = set()
        illegal_paths = set()
        for tnode in reversed(tnodes): # descending time order
            for g in self.ef_by_end.get(tnode, []):
                if g.path in legal_paths:
                    legal.append(g)
                elif g.path in illegal_paths:
                    continue
                else:
                    if locset.isdisjoint(g.path[:-1]):
                        if g.start == self.o_depot or max(g.start.t + g.ttt, g.tef) <= f.tls:
                            legal.append(g)
                            legal_paths.add(g.path)
                    else:
                        illegal_paths.add(g.path)

        assert len(legal) > 0
        return legal

    def legal_after_untimed(self, chain: Tuple[TExFrag, ...]) -> List[TExFrag]:
        candidates = []
        assert chain[-1].end != self.d_depot
        for g in self.ef_by_start_loc[chain[-1].end.loc]:
            if self.is_legal(chain + (g,), check_cover=True):
                candidates.extend(self.ef_by_path[g.path])

        assert len(candidates) > 0
        return candidates


    def legal_before_untimed(self, chain: Tuple[TExFrag, ...]) -> List[TExFrag]:
        candidates = []

        assert chain[0].start != self.o_depot

        for g in self.ef_by_end_loc[chain[0].start.loc]:
            if self.is_legal((g,) + chain, check_cover=True):
                candidates.extend(self.ef_by_path[g.path])

        assert len(candidates) > 0
        return candidates


    def is_legal(self, chain: Tuple[ExFrag], check_cover=False):
        return self.minimal_illegal_index(chain, check_cover) is None

    def minimal_illegal_index(self, chain: Tuple[TExFrag], check_cover=False) -> Union[None, int]:
        """
        Checks a chain for fragments for legality.  Checks the true timings, ignores times at nodes.
        If legal, returns None, otherwise returns the smallest index `k` such that chain[:k] is illegal.
        """
        assert len(chain) > 1
        if chain[0].start != self.o_depot:
            chain = (self.start_ef[chain[0].start.loc],) + chain
            idx_start = -1
        else:
            idx_start = 0
        if chain[-1].end != self.d_depot:
            chain = chain + (self.end_ef[chain[-1].end.loc],)

        idx = idx_start
        t = chain[0].tef
        for f in chain[1:-1]:
            idx += 1
            if t > f.tls:
                return idx + 1
            t = max(f.tef, t + f.ttt)
        if t > chain[-1].tls:
            return len(chain)

        if check_cover:
            idx = idx_start
            locs = set(chain[0].path[:-1])
            for f in chain[1:]:
                idx += 1
                for i in f.path[:-1]:
                    if i in locs:
                        return idx + 1
                    locs.add(i)
            if self.data.d_depot in locs:
                return len(chain)

        return None


@dataclasses.dataclass(frozen=True)
class SDARP_EF_Model_Params:
    repair_heuristic: bool
    cpus: int
    ff_cuts: bool
    ssr_cuts: bool
    cut_violate: float
    cuts_min_global_gap: float
    cuts_max_local_gap: float
    cuts_max_nodes: int
    cuts_root_node: bool

    def __post_init__(self):
        super().__setattr__(
            'mipnode_cuts',
            (self.ff_cuts | self.ssr_cuts) & (self.cuts_min_global_gap != float('inf'))
        )
        super().__setattr__(
            'lp_cuts',
            (self.ff_cuts | self.ssr_cuts) & self.cuts_root_node
        )

    @classmethod
    def from_experiment(cls, experiment):
        kwargs = {f.name: experiment.parameters[f.name] for f in dataclasses.fields(cls)}
        return cls(**kwargs)


class SDARP_EF_Model(BaseModel):
    X: BinVarDict  # fragments
    Y: BinVarDict  # waiting arcs
    W: BinVarDict  # is a customer served?
    Z: IntVar  # Number of customers served

    Xv: dict
    Yv: dict
    Wv: dict
    Zv: float

    heuristic_solution: Union[None, Tuple[int, Tuple[ExFrag]]]
    # is_restricted: bool
    lc_best_obj: float
    parameters: SDARP_EF_Model_Params
    _IDX = 0

    VI_CUTS = set()
    LC_CUTS = set()
    CUT_FUNC_LOOKUP = {}

    def __init__(self, network: Network, parameters: SDARP_EF_Model_Params):
        super().__init__(cpus=parameters.cpus)
        self.network = network
        self.heuristic_solution = None
        self.parameters = parameters
        self.lc_best_obj = float('inf')
        self.log = {
        }
        self.model_id = self.__class__._IDX
        self.__class__._IDX += 1
        self.solve_secondary_obj = False

        if __debug__:
            X = {f: self.addVar(name=f"X[{f.start.t}|{f!s}]", vtype=GRB.BINARY) for f in network.ef}
            Y = {a: self.addVar(name=f"Y{a!s}", vtype=GRB.BINARY) for a in network.wait_arcs}
        else:
            X = {f: self.addVar(vtype=GRB.BINARY) for f in network.ef}
            Y = {a: self.addVar(vtype=GRB.BINARY) for a in network.wait_arcs}

        W = {p: self.addVar(name=f"W{p}", vtype=GRB.BINARY) for p in network.data.P}
        Z = self.addVar(vtype=GRB.INTEGER, ub=len(network.data.P))

        self.setAttr('ModelSense', GRB.MINIMIZE)

        self.set_vars_attrs(X=X, Y=Y, W=W, Z=Z)
        self.setObjective(Z, GRB.MAXIMIZE)

        self.cons['obj'] = self.addLConstr(Z == quicksum(W.values()))
        self.cons['cover'] = {
            i: self.addLConstr(quicksum(X[f] for f in self.network.ef_by_loc_inc_start[i]) == W[i])
            for i in self.network.data.P
        }
        self.cons['flow'] = {}

        for p in self.network.data.P:
            nodes = self.network.nodes_by_loc[p]

            for n in nodes:
                if n in self.network.ef_by_start:
                    lhs = quicksum(X[f] for f in self.network.ef_by_start[n])
                    if n != nodes[-1]:
                        lhs += Y[self.network.wait_arcs_by_start[n]]
                else:
                    lhs = Y[self.network.wait_arcs_by_start[n]]

                if n in self.network.ef_by_end:
                    rhs = quicksum(X[f] for f in self.network.ef_by_end[n])
                    if n != nodes[0]:
                        rhs += Y[self.network.wait_arcs_by_end[n]]
                else:
                    rhs = Y[self.network.wait_arcs_by_end[n]]

                self.cons['flow'][n] = self.addLConstr(lhs == rhs)

        self.cons['num_vehicles'] = self.addLConstr(
            quicksum(X[f] for f in self.network.ef_by_start[self.network.o_depot]) <= len(self.network.data.K)
        )

    @classmethod
    def define_cut(cls, name: str, valid_inequality: bool):
        def decorator(func):
            assert (name not in cls.CUT_FUNC_LOOKUP), f"{name} is already defined."
            cls.CUT_FUNC_LOOKUP[name] = func
            if valid_inequality:
                cls.VI_CUTS.add(name)
            else:
                cls.LC_CUTS.add(name)
            return func

        return decorator

    def add_valid_inequalities(self, where, cutdict, name):
        if where is None:
            if name not in self.cons:
                self.cons[name] = dict()
            for key, cut in cutdict.items():
                assert key not in self.cons[name]
                self.cons[name][key] = self.addConstr(cut)

        elif where == GRB.Callback.MIPNODE:
            for key, cut in cutdict.items():
                self.cbCut(cut, name, key)

        else:
            raise NotImplementedError

    def add_lazy_cuts(self, where, cutdict, name):
        if where is None:
            if name not in self.cons:
                self.cons[name] = dict()
            for key, cut in cutdict.items():
                assert key not in self.cons[name]
                self.cons[name][key] = self.addConstr(cut)

        elif where == GRB.Callback.MIPNODE or where == GRB.Callback.MIPSOL:
            for key, cut in cutdict.items():
                self.cbLazy(cut, name, key)

        else:
            raise NotImplementedError

def _coerce_sorted_intlist(val : Union[str, Iterable[int]]) -> List[int]:
    if isinstance(val, str):
        if len(val) == 0:
            return []
        val = list(map(int, val.split(",")))
    return sorted(val)

class SDARPFragmentsExperiment(SDARP_Experiment):
    ROOT_PATH = SDARP_Experiment.ROOT_PATH / 'ef'
    INPUTS = {
        "hobj" : {
            "type": "boolean",
            "default" : False,
            "help": "Solve a hierarchical objective, maximising requests served first, then minimising travel time."
        },
        "unserved_penalty" : {
            "type" : "float",
            "default" : 0.0,
            "min" : 0.0,
            "help": "Penalty factor for unserved requests in the secondary objective (has no effect if --hobj is not "
                    "given. ).  Penalty of unserved request r is FACTOR * TRAVEL_TIME(PICKUP[r], DELIVERY[r])"
        },
        **SDARP_Experiment.INPUTS,
    }
    PARAMETERS = {
        "domination": {"type": "boolean", "default": True},
        "time_delta" : {"type" : "float", "default" : 1, "min" : 0.01},
        "preprocess": {"type": "boolean", "default": True},
        "ff_cuts" : {"type": "boolean", "default": True},
        "ssr_cuts" : {"type": "boolean", "default": True},
        "cut_violate": {"type": "float", "min": 0, "default": 0.01},
        "cuts_root_node": {"type": "boolean", "default": True},
        "cuts_max_nodes": {"type": "integer", "min": 0, "default": 10},
        "cuts_min_global_gap": {"type": "float", "min": 0, "default": 0.999, "coerce": float,
                                "help": "Minimum absolute global objective gap required to add cuts at MIP nodes.  Set to"
                                        " `inf` to disable adding cuts at MIP nodes"},
        "cuts_max_local_gap": {"type": "float", "min": 0, "default": 3},
        "repair_heuristic": {"type": "boolean", "default": True},
        "restricted_mips": {"type": "list", "schema" : {"type" : "integer"},  "default": [2, 4, 6],
                            "coerce" : _coerce_sorted_intlist,
                              "help": "Run a series of smaller MIPs with a reduced number of fragments.  The parameter"
                                      " is a list of integers n, where each n provided will run Branch-and-Cut with "
                                      " only the fragments who serve n requests or fewer.  These solutions will allow "
                                      "for variables to be fixed by reduced cost. Set to the empty string to disable"},
        "min_len_fix_frac" : {"type": "float", "min": 0, "max": 1, "default": 0.05,
                              "help" : "Minimum proportion of EF that can be fixed to 0 because of length.  If fewer"
                                       " than this proportion would be fixed, then no EF are fixed to 0 because of "
                                       "length.  EF may still be fixed due to RC"
                              },
        "tt_heuristic": {"type": "boolean", "default": False,
                   "help": "TT Heuristic - before solving the main travel-time MIP, fix the requests covered and solve"
                           " the smaller MIP to try to improve the starting solving (expensive).  Only affects"
                           " the secondary travel time objective."},
        **SDARP_Experiment.PARAMETERS
    }

    @property
    def resource_memory(self) -> str:
        if self.inputs['index'] in {0, 1, 2, 3, 10, 11, 12, 15, 20, 21, 23, 27}:
            mem = 5

        elif self.inputs['index'] in {4, 5, 13, 16, 22, 25, 26}:
            mem = 15

        elif self.inputs['index'] in {6, 8, 9, 14, 17, 19, 24, 28}\
                or self.inputs['index'] in NAMED_DATASET_INDICES['loose_easy']\
                or self.inputs['index'] in NAMED_DATASET_INDICES['loose_cap']:
            mem = 30

        elif self.inputs['index'] == 18:
            mem = 80

        elif self.inputs['index'] in NAMED_DATASET_INDICES['loose'] - NAMED_DATASET_INDICES['loose_easy']:
            mem = 200

        else:
            assert self.inputs['index'] in {7, 18, 29}
            mem = 60

        if self.parameters['time_delta'] < 0.2:
            if self.inputs['index'] == 18:
                mem = 200
            else:
                mem += 20

        return f"{mem}GB"

    @property
    def resource_time(self) -> str:
        if self.inputs['index'] in NAMED_DATASET_INDICES['medium']:
            time = 600
        elif self.inputs['index'] in NAMED_DATASET_INDICES['loose_cap'] & NAMED_DATASET_INDICES['loose_easy']:
            time = 300
        elif self.inputs['index'] in {2, 12, 15, 20, 27, 28, 29, 19}\
                or self.inputs['index'] in NAMED_DATASET_INDICES['loose_easy']\
                or self.inputs['index'] in NAMED_DATASET_INDICES['loose_cap']:
            time = 3600
        else:
            time = self.parameters['timelimit'] + 900

        # special cases
        if self.parameters['time_delta'] < 0.2:
            time = self.parameters['timelimit'] + 3600
        elif self.parameters['time_delta'] > 100:
            time = self.parameters['timelimit'] + 60*15
        elif math.isclose(self.parameters['time_delta'], 0.25, abs_tol=0.0001) and self.inputs['index'] == 14:
            time = 3600

        return slurm_format_time(time)

    @property
    def input_string(self):
        s = "EF{index:d}-{instance}".format_map(self.inputs)
        if self.inputs["hobj"]:
            s += "_TT"
            if self.inputs['unserved_penalty'] > 0.0:
                u = round(self.inputs['unserved_penalty'] * 1000)
                s += f'{u:04d}'

        return s


def _subtract_fragment(F, f, val):
    new_val = F[f.start][f] - val
    if new_val < EPS:
        del F[f.start][f]
        if len(F[f.start]) == 0:
            del F[f.start]
    else:
        F[f.start][f] = new_val


def build_chains(model):
    eps = 1e-5
    F = {}
    for f, val in model.Xv.items():
        if val > eps:
            F.setdefault(f.start, {})[f] = val
    A = {}
    for a, val in model.Yv.items():
        if val > eps:
            A.setdefault(a[0], {})[a] = val

    graph = [F, A]

    def find_start(g):
        frags, _ = g

        if model.network.o_depot in frags:
            return model.network.o_depot

        return take(frags)

    def outgoing_arc(g, n):
        frags, arcs = g
        try:
            f, val = take(frags[n].items())
            return f, val, f.end
        except KeyError:
            pass

        a, val = take(arcs[n].items())
        return a, val, a[1]

    def subtract_arc(g, e, val):
        if isinstance(e, TExFrag):
            idx = 0
            n = e.start
        else:
            idx = 1
            n = e[0]
        new_val = g[idx][n][e] - val
        if math.isclose(new_val, 0, abs_tol=eps):
            del g[idx][n][e]
            if len(g[idx][n]) == 0:
                del g[idx][n]
        else:
            assert new_val > 0
            g[idx][n][e] = new_val

    sinks = {model.network.d_depot}
    return decompose_paths_and_cycles(graph, sinks, find_start, outgoing_arc, subtract_arc)


def separate_legality_cuts(model: SDARP_EF_Model, chains_cycles=None):
    cuts = {}
    if chains_cycles is None:
        chains, cycles = build_chains(model)
    else:
        chains, cycles = chains_cycles

    for chain_with_arcs, _ in chains:
        chain = tuple(f for f in chain_with_arcs if isinstance(f, TExFrag))
        end_idx = model.network.minimal_illegal_index(chain, check_cover=False)
        if end_idx is None:
            continue
        # Note, the end fragments may not be minimal, but start fragments are.
        assert end_idx < len(chain) or len(chain[-1].path) > 3

        for k in range(2, end_idx - 1):  # idx 0 is start frag, idx 1 would just implicitly add the start frag back on
            start_idx = k - 1  # the one just before was illegal
            if model.network.is_legal(chain[k:end_idx], check_cover=False):
                break
        else:
            start_idx = end_idx - 2
            assert not model.network.is_legal(chain[start_idx:end_idx], check_cover=False)

        chain = chain[start_idx:end_idx]
        assert len(chain) > 1
        print("illegal:", tuple(f.path for f in chain))

        cuts[chain, 'before'] = (
                quicksum(model.X[g] for f in chain[1:] for g in model.network.ef_by_path[f.path])
                <=
                len(chain) - 2 + quicksum(model.X[f] for f in model.network.legal_before_untimed(chain[1:]))
        )
        cuts[chain, 'after'] = (
                quicksum(model.X[g] for f in chain[:-1] for g in model.network.ef_by_path[f.path])
                <=
                len(chain) - 2 + quicksum(model.X[f] for f in model.network.legal_after_untimed(chain[:-1]))
        )

    for cycle, _ in cycles:
        print("cycle:", tuple(f.path for f in cycle))
        cuts[cycle, 'cycle'] = quicksum(model.X[f] for f in cycle) <= len(cycle) - 1

    return cuts

@SDARP_EF_Model.define_cut('ssr_cuts', True)
def separate_ssr_cuts(model: SDARP_EF_Model):
    cuts = {}

    loc_triples = defaultdict(lambda : 0)
    loc_pairs = defaultdict(lambda : 0)
    visited = set()
    for f,v in model.Xv.items():
        if f.start != model.network.o_depot and len(f.path) > 3:
            pickups = sorted(p for p in f.path[:-1] if p in model.network.data.P)
            visited.update(pickups)
            for pair in itertools.combinations(pickups, 2):
                loc_pairs[pair] += v
            for triple in itertools.combinations(pickups, 3):
                loc_triples[triple] += v

    visited = sorted(visited)

    for (i,j), v_ij in loc_pairs.items():
        j_idx = bisect.bisect_left(visited, j)
        for k in visited[j_idx+1:]:
            # i < j < k
            v_ik = loc_pairs.get((i,k), 0)
            v_jk = loc_pairs.get((j,k), 0)
            v_ijk = loc_triples.get((i,j,k), 0)
            if v_ij + v_ik + v_jk - 2*v_ijk > 1 + model.parameters.cut_violate:
                # TODO add cut
                cuts[i,j,k] = quicksum(model.X[f] for f in model.network.loc_triple(i,j,k)) <= 1

    return cuts

@SDARP_EF_Model.define_cut('ff_cuts', False)
def separate_ff_cuts(model : SDARP_EF_Model):
    cuts = {}
    frags_by_start_loc = defaultdict(list)
    frags_by_end_loc = defaultdict(list)

    for f, val in model.Xv.items():
        frags_by_end_loc[f.end.loc].append((f,val))
        frags_by_start_loc[f.start.loc].append((f,val))

    frags_by_start_loc.default_factory = None
    frags_by_end_loc.default_factory = None

    # Don't need to check cover here - implied by flow and cover constraints, and acyclic network
    for f, lhs in model.Xv.items():
        # legal after
        if lhs > model.parameters.cut_violate and f.start != model.network.o_depot:
            f_locset = set(f.path)

            if f.end != model.network.d_depot:
                rhs = 0
                for g, v in frags_by_start_loc[f.end.loc]:
                    if f.end.t <= g.start.t and f.start.t + f.ttt <= g.tls and f_locset.isdisjoint(g.path[1:]):
                        rhs += v
                        if lhs <= rhs + model.parameters.cut_violate:
                            break
                else:
                    # cut must be violated.
                    cuts[f, 'after'] = model.X[f] <= quicksum(model.X[g] for g in model.network.legal_after_timed(f))

            # legal before
            rhs = 0
            for g, v in frags_by_end_loc[f.start.loc]:
                if g.start == model.network.o_depot or (
                    g.end.t <= f.start.t and
                    max(g.start.t + g.ttt, g.tef) <= f.tls and
                    f_locset.isdisjoint(g.path[:-1])
                ):
                    rhs += v
                    if lhs <= rhs + model.parameters.cut_violate:
                        break
            else:
                cuts[f, 'before'] = model.X[f] <= quicksum(model.X[g] for g in model.network.legal_before_timed(f))

    return cuts


def grb_callback(model: SDARP_EF_Model, where):
    if where == GRB.Callback.MIPSOL:
        model.update_var_values(where, EPS)
        chains, cycles = build_chains(model)
        cuts = separate_legality_cuts(model, chains_cycles=(chains, cycles))

        for key, c in cuts.items():
            model.cbLazy(c, 'legality', key)

        if len(cuts) > 0:
            bound = round(model.cbGet(GRB.Callback.MIPSOL_OBJBND))
            assert model.solve_secondary_obj or math.isclose(bound, model.cbGet(GRB.Callback.MIPSOL_OBJBND))
            if model.parameters.repair_heuristic and not model.solve_secondary_obj:
                current_heur_obj, _ = model.heuristic_solution or (0, None)
                best = max(model.lc_best_obj, current_heur_obj)
                max_drop = round(model.Zv - best) - 1 # need an improvement of at least 1

                if len(cycles) == 0 and max_drop >= 1:
                    frags = []
                    print("attempting solution repair")
                    for route, _ in chains:
                        route_ef = tuple(f for f in route if isinstance(f, TExFrag))
                        if model.network.is_legal(route_ef, check_cover=False):
                            frags.extend(route_ef)
                        else:
                            heur_ef, dropped = yeet_requests(model.network, route_ef, max_drop)
                            if heur_ef is None:
                                print("repair failed")
                                break
                            max_drop -= dropped
                            frags.extend(heur_ef)
                    else:
                        obj = sum((len(f.path) - 1) // 2 for f in frags if f.start != model.network.o_depot)
                        assert obj > best
                        print(f"found heuristic solution {best} -> {obj}")
                        model.heuristic_solution = (obj, frags)

        else:
            model.lc_best_obj = round(model.Zv)

    elif where == GRB.Callback.MIPNODE:
        best_obj = model.lc_best_obj

        if model.heuristic_solution is not None:
            heur_obj, frags = model.heuristic_solution

            if heur_obj > best_obj:
                arcs = []
                for f, g in zip(frags, frags[1:]):
                    if g.start == model.network.o_depot:
                        continue

                    n = f.end
                    while n != g.start:
                        a = model.network.wait_arcs_by_start[n]
                        n = a[1]
                        arcs.append(a)

                print(f'post solution: {heur_obj:.3f} ', end='')
                model.cbSetSolution([model.X[f] for f in frags], [1] * len(frags))
                obj = model.cbUseSolution()
                print(f'({obj:.3f})' if obj < GRB.INFINITY else '(infeasible)')
                # assert round(obj) >= heur_obj # if there's an empty route, Gurobi may trivially improve the objective
            else:
                model.heuristic_solution = None

        elif model.parameters.mipnode_cuts:
            if (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
                global_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                global_gap = global_bound - best_obj
                if global_gap > model.parameters.cuts_min_global_gap:
                    local_bound = model.cbGetNodeRel(model.Z)
                    local_gap = global_bound - local_bound  # global bound >= local_bound >= best_objective

                    if ((local_gap < model.parameters.cuts_max_local_gap) or
                            model.cbGet(GRB.Callback.MIPNODE_NODCNT) <= model.parameters.cuts_max_nodes):
                        model.update_var_values(where)
                        for cutname, sep_func in model.CUT_FUNC_LOOKUP.items():
                            if getattr(model.parameters, cutname):
                                cuts = sep_func(model)
                                if len(cuts) > 0:
                                    print(cutname.rjust(20) + ":", len(cuts))
                                    if cutname in model.VI_CUTS:
                                        model.add_valid_inequalities(where, cuts, cutname)
                                    else:
                                        model.add_lazy_cuts(where, cuts, cutname)


def route_to_ef_paths(network: Network, route: Tuple[int]) -> List[Tuple[int]]:
    assert route[0] == network.data.o_depot and route[-1] == network.data.d_depot
    frags = [(network.data.o_depot, route[1])]
    q = 0
    start_k = 1
    for k, i in enumerate(route[1:-1]):
        k = k + 1
        q += network.data.demand[i]
        if q == 0 or k == 0:
            f = route[start_k:k + 2]  # need pickup after k
            assert f[-1] in network.data.P or f[-1] == network.data.d_depot
            start_k = k + 1
            frags.append(f)
    assert q == 0

    return frags


def _find_efs(network : Network, path) -> TExFrag:
    locset = frozenset(path)
    paths = network.paths_by_locset.get(locset, [])
    return [network.ef_by_path[p][0] for p in paths if p[0] == path[0] and p[-1] == path[-1]]

def _fix_up_tnodes(network: Network, chain : Tuple[TExFrag,...]) -> Tuple[TExFrag]:
    out = [chain[0]]
    for uf in chain[1:]:
        for f in network.ef_by_path[uf.path]:
            if f.start.t >= out[-1].end.t:
                out.append(f)
                break
        else:
            raise Exception("bug: chain should be legal")

    return tuple(out)

def yeet_requests(network: Network, bad_chain: Tuple[TExFrag, ...], max_drop: int):
    if max_drop <= 0:
        return None, None

    bad_route = tuple(i for f in bad_chain for i in f.path[:-1]) + (network.data.d_depot,)
    drop = (p for p in bad_route if p in network.data.P)

    # Phase 1, try to drop just a single customer.
    for p in drop:
        route = tuple(i for i in bad_route if i != p and i != p + network.data.n)
        paths = route_to_ef_paths(network, route)
        chain = [network.ef_by_path[paths[0]][0]]
        t = bad_chain[0].tef
        for path in paths[1:]:
            frags = _find_efs(network, path)
            frags = [f for f in frags if t <= f.tls]
            if len(frags) == 0: # may happen because of preprocessing removing arcs between locations
                break
            t, f = min(((max(t + f.ttt, f.tef), f) for f in frags), key=lambda x : x[0])
            chain.append(f)
        else:
            return _fix_up_tnodes(network, chain), 1

        assert (not all(a in network.data.travel_time for a in zip(route, route[1:])) or
                get_early_schedule(route[1:-1], network.data, 0) is None)

    # Phase 2: drop an entire fragment (prioritise short ones)
    # Dump the shortest fragment, modify the (k-1)th fragment to fix the ends
    # We need to drop fragments with two or more requests, otherwise the previous step would have succeeded
    drop_frags = sorted((len(bad_chain[k].path), k) for k in range(1, len(bad_chain)) if len(bad_chain[k].path) > 3)

    failed_chains = []
    for length, k in drop_frags:
        dropped = (length-1)//2
        if dropped > max_drop:
            break

        if k == len(bad_chain) - 1:
            previous_path = bad_chain[k - 1].path[:-1] + (network.data.d_depot,)
            try:
                # if the path has been dominated then this will result in a key error
                f = network.ef_by_path[previous_path][-1]
            except KeyError:
                # _find_efs should never return be empty, since EF must always be part of a route
                f = max(_find_efs(network, previous_path), key=lambda f: f.tls)
        else:
            previous_path = bad_chain[k - 1].path[:-1] + (bad_chain[k + 1].start,)
            frags = _find_efs(network, previous_path)
            if len(frags) == 0:
                continue
            f = max(frags, key=lambda f : f.ttt)

        chain = bad_chain[:k - 1] + (f,) + bad_chain[k + 1:]
        if network.is_legal(chain, check_cover=False):
            return _fix_up_tnodes(network, chain), dropped
        else:
            failed_chains.append((dropped, chain))

    for dropped, chain in failed_chains:
        chain, additional_dropped = yeet_requests(network, chain, max_drop - dropped)  # must yeet more fellows
        if chain is not None:
            return chain, dropped + additional_dropped

    return None, None

def lp_cut_loop(model: SDARP_EF_Model):
    if not model.parameters.lp_cuts:
        model.optimize()
        return model.ObjVal, model.ObjVal

    enabled_cuts = [cutname for cutname in model.CUT_FUNC_LOOKUP if getattr(model.parameters, cutname)]
    model.update()
    assert model.IsMIP == 0 and model.IsQP == 0 and model.IsQCP == 0
    assert len(enabled_cuts) > 0
    output = TablePrinter(enabled_cuts + ["Objective", "Sep. time", "Solve time"])

    t = time.time()
    with model.temp_params(OutputFlag=0):
        model.optimize()
        starting_bound = model.ObjVal
        output.print_line(model.ObjVal, "", time.time()-t, pad_left=True)


        while True:
            model.update_var_values()
            cuts_added_this_iter = 0
            output_row = []
            t = time.time()
            for cutname in enabled_cuts:
                separate_cuts = model.CUT_FUNC_LOOKUP[cutname]
                cuts = separate_cuts(model)
                for key, cut in cuts.items():
                    consdict = model.cons.setdefault(cutname, {})
                    assert key not in consdict, "duplicate cut"
                    consdict[key] = model.addLConstr(cut)
                ncuts = len(cuts)
                cuts_added_this_iter += ncuts
                output_row.append(ncuts if ncuts > 0 else "")
            sep_time = time.time() - t
            t = time.time()
            model.optimize()
            soln_time = time.time() - t
            output_row.extend([model.ObjVal, sep_time, soln_time])
            output.print_line(*output_row)
            if cuts_added_this_iter == 0:
                break

    return starting_bound, model.ObjVal

def initial_heuristic(model: SDARP_EF_Model, max_req_per_fragment: int, timelimit: float):
    k = max_req_per_fragment * 2 + 1
    for f, var in model.X.items():
        if len(f.path) > k:
            var.ub = 0

    with model.temp_params(TimeLimit=timelimit):
        model.optimize(grb_callback)
    objective = model.ObjVal
    for var in model.X.values():
        var.ub = 1

    nconstraints = model.flush_cut_cache()
    model.update_var_values()
    print(f"added {nconstraints:,d} cuts as hard constraints.")
    return objective

def get_model_size(model : SDARP_EF_Model):
    return {
        'vars' : {
            'X' : len(model.X),
            'Y' : len(model.Y),
            'W' : len(model.W),
            'Z' : 1
        },
        'cons' : model.cons_size
    }

def get_tt_obj(f: TExFrag, network: Network) -> float:
    # if f.start == network.o_depot:
    #     ttt = network.data.travel_time[f.start.loc, f.end.loc]
    # elif f.end == network.d_depot:
    #     ttt = sum(network.data.travel_time[i, j] for i, j in zip(f.path, f.path[1:]))
    # else:
    #     ttt = f.ttt
    #
    assert f.ttt < 2**32 - 1

    return f.ttt / TIME_SCALE

def solve_secondary_obj(exp: SDARP_Experiment, model: SDARP_EF_Model, stopwatch: Stopwatch):
    model.setObjective(0, GRB.MINIMIZE)
    t_rem = exp.parameters['timelimit'] - stopwatch.time
    info = {}

    p = model.parameters
    model.parameters = SDARP_EF_Model_Params(
        repair_heuristic = False,
        cpus = p.cpus,
        ff_cuts = True, #p.ff_cuts,
        ssr_cuts = p.ssr_cuts,
        cut_violate = p.cut_violate,
        cuts_min_global_gap = p.cuts_min_global_gap,
        cuts_max_local_gap = p.cuts_max_local_gap,
        cuts_max_nodes = p.cuts_max_nodes,
        cuts_root_node = p.cuts_root_node,
    )
    model.solve_secondary_obj = True

    covered_req = {i for f in model.Xv for i in f.path[:-1] if i in model.network.data.P }

    model.Z.ub = len(covered_req)
    model.Z.lb = len(covered_req)

    for (f, var) in model.X.items():
        var.obj = get_tt_obj(f, model.network)
        var.ub = 1

    penalty_factor =  exp.inputs['unserved_penalty']
    obj = sum(get_tt_obj(f, model.network) for f in model.Xv.keys())

    if penalty_factor > 0.0:
        obj_constant = 0
        for p, w in model.W.items():
            penalty = penalty_factor * model.network.data.travel_time[p, p + model.network.data.n] / TIME_SCALE
            obj_constant += penalty
            w.obj = -penalty
            if w.x < .1:
                obj += penalty

        model.ObjCon = obj_constant


    # assert all(math.isclose(v.lb, 0) for v in model.getVars())


    model.set_variables_continuous()
    start_bnd, bnd = lp_cut_loop(model)

    info["root_lp"] = start_bnd
    info["lifted_lp"] = bnd
    info["initial"] = obj

    if t_rem <= 1:
        model.Xv = None
        model.Yv = None
        return False, info

    nfixed = 0
    for (f, var) in model.X.items():
        if f not in model.Xv and bnd + var.rc > obj - 0.01/TIME_SCALE:
            var.ub = 0
            nfixed += 1
    ntotal = len(model.X)

    print(f"Fixed {nfixed:,d}/{ntotal:,d} ({100*nfixed/ntotal:.2f}%) variables")
    model.set_variables_integer()

    if exp.parameters['tt_heuristic'] and len(covered_req) == model.network.data.n:
        print("Skipping travel time heuristic since all requests are covered")
    elif exp.parameters['tt_heuristic']:
        print("Running travel time heuristic (fixing requests covered)")

        for i, var in model.W.items():
            if i in covered_req:
                var.lb = 1

        with model.temp_params(MIPFocus=1, Heuristics=.30, TimeLimit=min(t_rem / 3, 300), NodeLimit=1500):
            model.optimize(grb_callback)

        model.update_var_values()
        info["heuristic_obj"] = model.ObjVal
        obj = model.ObjVal

        for i, var in model.W.items():
            var.lb = 0

        for (f, var) in model.X.items():
            var.ub = 1
        model.flush_cut_cache()

        model.set_variables_continuous()
        bnd1, bnd2 = lp_cut_loop(model)

        for (f, var) in model.X.items():
            if f in model.Xv:
                continue
            if bnd + var.rc > obj - 0.01/TIME_SCALE:
                var.ub = 0
                nfixed += 1

        model.set_variables_integer()
        print(f"Fixed {nfixed:,d}/{ntotal:,d} ({100 * nfixed / ntotal:.2f}%) variables")

        info["final_lp_pre_cuts"] = bnd1
        info["final_lp_post_cuts"] = bnd2

    # sol = SDARPSolution.from_json_file("/home/yannik/phd/src/sdarp/scrap/EF0-30N_4K_A_TT_soln.json")
    # model.cons['debug'] = {}
    # for route in sol.routes:
    #     for path in route_to_ef_paths(model.network, route):
    #         for f in model.network.ef_by_path[path]:
    #             model.X[f].lb = 0
    #             model.X[f].ub = 1
    #         model.cons['debug'][path] = model.addLConstr(quicksum(model.X[f] for f in model.network.ef_by_path[path]) == 1)

    with model.temp_params(TimeLimit=t_rem):
        model.optimize(grb_callback)

    stopwatch.stop("tt_mip")
    info["final_obj"] = model.ObjVal
    info["final_bound"] = model.ObjBound
    info["status"] = model.Status

    model.update_var_values()
    model.flush_cut_cache()
    return True, info



def get_solution(model, best_obj, bound):
    chains, cycles = build_chains(model)
    assert len(cycles) == 0
    soln = SDARPSolution(best_obj, bound)
    for chain_with_arcs, _ in chains:
        chain = tuple(f for f in chain_with_arcs if isinstance(f, TExFrag))
        r = tuple(i for f in chain for i in f.path[:-1])[1:]
        pprint_path(r, model.network.data, add_depots=True)
        assert (model.network.is_legal(chain, check_cover=False))
        soln.add_route((model.network.data.o_depot, *r, model.network.data.d_depot))

    ef_per_route = [sum(1 if isinstance(f, TExFrag) else 0 for f in chain) for chain, _ in chains]
    req_per_route = [len(r) // 2 for r in soln.routes]
    soln_info = {
        'ef_per_route': ef_per_route,
        'mean_ef_per_route': sum(ef_per_route) / len(ef_per_route),
        'req_per_route': req_per_route,
    }

    return soln, soln_info




def main():
    stopwatch = Stopwatch()
    exp = SDARPFragmentsExperiment.from_cl_args()

    exp.print_summary_table()
    exp.write_index_file()

    stopwatch.start()
    rust_data = fraglib.sdarp.instance.load(exp.inputs['index'])
    if exp.parameters['preprocess']:
        fraglib.sdarp.preprocess_data(rust_data)
    data = fraglib.unwrap(rust_data)
    # ASSUMPTION: maximum route time is built into the depot time windows.
    assert data.tw_end[data.d_depot] - data.tw_start[data.o_depot] <= data.max_ride_time[data.o_depot]

    bounds = {}
    misc_info = {}

    network = Network(data,exp.parameters['time_delta'],  exp.parameters['preprocess'], exp.parameters['domination'])
    network.build(rust_data, exp.parameters['cpus'], exp.inputs['hobj'])
    stopwatch.lap('network')

    model_params = SDARP_EF_Model_Params.from_experiment(exp)
    model = SDARP_EF_Model(network, model_params)
    model_size = {'initial' : get_model_size(model)}
    model.Z.BranchPriority = 10
    model.setParam('GURO_PAR_MINBPFORBID', 1)
    for pname, val in exp.parameters["gurobi"].items():
        model.setParam(pname, val)
    model.set_variables_continuous()


    start_bound, lifted_bound = lp_cut_loop(model)
    bounds['bnd_root_lp'] = start_bound
    model_size['lifted_lp'] = get_model_size(model)
    bounds['bnd_lifted_lp'] = lifted_bound

    stopwatch.lap('root_node')
    model.setParam('LazyConstraints', 1)

    lp_bnd = model.ObjVal
    best_obj = None
    best_soln_frags = None
    best_soln_arcs = None

    fragments_max_reqs = exp.parameters["restricted_mips"].copy()
    best_obj_guess = len(fragments_max_reqs) > 0
    fragments_max_reqs.append(data.n)
    min_num_len_fixed = len(model.X) * exp.parameters["min_len_fix_frac"]

    for max_req_per_frag in fragments_max_reqs:
        t_rem = exp.parameters["timelimit"] - stopwatch.time
        if t_rem <= EPS:
            print("hit time limit")
            break

        print(f" MIP: MAX {max_req_per_frag:d} REQ PER FRAGMENT ".center(PRINT_HLINE_WIDTH, '-'))
        n_fixed_len = 0
        n_fixed_rc = 0
        len_fixed_frags = []

        if best_obj_guess:
            fake_obj = round(math.floor(round(lp_bnd, 6)))
            print(f"First restricted MIP: targeting objective of {fake_obj}")
            for f, var in model.X.items():
                if round(math.floor(lp_bnd + var.rc)) < fake_obj:
                    n_fixed_rc += 1
                    var.ub = 0
                elif (len(f.path) - 1) // 2 > max_req_per_frag:
                    n_fixed_len += 1
                    var.ub = 0
                    if n_fixed_len < min_num_len_fixed:
                        len_fixed_frags.append(f)

        else:
            for f, var in model.X.items():
                if best_soln_frags is not None and round(math.floor(lp_bnd + var.rc)) <= best_obj and f not in best_soln_frags:
                    # Without the check for the current best solution, it is possible that we cut off the current solution,
                    # since we only allow variables which can lead to a strictly better solution (one with an objective of
                    # `best_obj + 1` or better).  Also note that reduced costs are negative here
                    n_fixed_rc += 1
                    var.ub = 0
                elif (len(f.path) - 1)//2 > max_req_per_frag:
                    n_fixed_len += 1
                    var.ub = 0
                    if n_fixed_len < min_num_len_fixed:
                        len_fixed_frags.append(f)
                else:
                    var.ub = 1

        print(f"RC-fixing fixed {n_fixed_rc:,d} variables ({100 * n_fixed_rc / len(model.X):.2f}%)")
        print(f"Length-fixing fixed {n_fixed_len:,d} variables ({100 * n_fixed_len / len(model.X):.2f}%)")
        if 0 < n_fixed_len < min_num_len_fixed:
            print(f"Less than {exp.parameters['min_len_fix_frac']*100:.1f}% of fragments are length-fixed, unfixing these...")
            for f in len_fixed_frags:
                model.X[f].ub = 1
            n_fixed_len = 0

        model.set_variables_integer()
        with model.temp_params(TimeLimit=t_rem):
            model.optimize(grb_callback)

        model.flush_cut_cache()
        model.update_var_values()
        best_obj = round(model.ObjVal)
        best_soln_frags = model.Xv.copy()
        best_soln_arcs = model.Yv.copy()

        if best_obj_guess:
            best_obj_guess = False
            if best_obj == fake_obj:
                print("Found solution equal to LB rounded down: solution must be optimal")
                break

            elif n_fixed_len == 0 and model.Status == GRB.OPTIMAL:
                if best_obj == fake_obj - 1:
                    print(f"No fixed-len vars, moving Z UB to {fake_obj - 1} would prove optimality:"
                          f" solution must be optimal")
                    lp_bnd = fake_obj-1
                    break
                else:
                    model.Z.ub = fake_obj - 1
                    print(f"No fixed-len vars, moved Z UB from {data.n} to {fake_obj-1}")
        else:
            if n_fixed_len == 0:
                if model.Status == GRB.OPTIMAL:
                    print("no more length-fixed vars: solution must be optimal")
                elif model.Status == GRB.TIME_LIMIT:
                    print("time limit reached")
                else:
                    raise Exception("bug")
                lp_bnd = model.ObjBound
                break  # at this point the solution is optimal

            if lp_bnd - best_obj < 1-EPS:
                print("gap less than 1: solution must be optimal")
                break

        if model.Status == GRB.TIME_LIMIT:
            print("time limit reached")
            break

        model.set_variables_continuous()
        for var in model.X.values():
            var.ub = 1
        if model.parameters.lp_cuts:
            _, lp_bnd = lp_cut_loop(model)
        else:
            model.optimize()
            lp_bnd = model.ObjVal

        if lp_bnd - best_obj < 1-EPS:
            print("gap less than 1, solution must be optimal")
            break

    lp_bnd = round(math.floor(lp_bnd))

    model.Xv = best_soln_frags
    model.Yv = best_soln_arcs
    bounds['obj_final'] = best_obj
    bounds['bnd_final'] = lp_bnd
    model_size['final'] = get_model_size(model)

    if exp.inputs['hobj']:
        soln, soln_info = get_solution(model, best_obj, lp_bnd)
        misc_info['cover_soln'] = soln_info
        soln.to_json_file(exp.get_output_path(".start_soln.json"))
        stopwatch.lap("mip")
        model.flush_cut_cache()
        soln_found, tt_solve_info = solve_secondary_obj(exp, model, stopwatch)
        misc_info['hobj'] = tt_solve_info
        if soln_found:
            soln, soln_info = get_solution(model, tt_solve_info['final_obj'], tt_solve_info['final_bound'])
            soln.to_json_file(exp.outputs['solution'])
        else:
            soln_info = None
    else:
        stopwatch.stop('mip')
        model.flush_cut_cache()
        soln, soln_info = get_solution(model, best_obj, lp_bnd)
        soln.to_json_file(exp.outputs['solution'])

    info = {
        'time': stopwatch.times,
        'bounds': bounds,
        'network_size' : network.size_info(),
        'model' : dataclasses.asdict(model.get_gurobi_model_information()),
        'model_size' : model_size,
        'cons' : model.cons_size,
        'soln': soln_info,
        **misc_info
    }

    with open(exp.outputs["info"], 'w') as fp:
        json.dump(info, fp, indent='  ')


if __name__ == '__main__':
    main()
