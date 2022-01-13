from oru import *
from oru.slurm import slurm_format_time
from typing import FrozenSet, Tuple
from utils import *
from utils.data import SDARP_Data,SDARP_Skeleton_Data,  get_named_instance_skeleton_SDARP, modify, indices, get_name_by_index
import yaml
from gurobi import *
import json
from pathlib import Path
import dataclasses
import fraglib
import darp

frozen_dataclass = dataclasses.dataclass(frozen=True, eq=True)


def locstring(i, num_req):
    return darp.locstring(i, num_req)

def get_early_schedule(path, data: SDARP_Data, start_time=0, check_illegal=True):
    return darp.get_early_schedule(path, data, start_time, check_illegal)

def get_late_schedule(path, data : SDARP_Data, end_time=float('inf')):
    return darp.get_late_schedule(path, data, end_time)


def pformat_path(path, data : SDARP_Data, color=True, add_depots=False):
    schedule = get_early_schedule(path, data, check_illegal=False)
    if add_depots:
        O = 0
        D = data.n*2+1
        schedule = (schedule[0] - data.travel_time[O,path[0]],) + schedule + \
                   (schedule[-1] + data.travel_time[path[-1],D], )
        path = (O, ) + path + (D, )

    return pformat_schedule(schedule, path, data, color=color)


def pformat_schedule(schedule, path, data : SDARP_Data, color=True):
    output = ''
    cell_width = 12
    h_sep = ' '
    times = schedule

    path_str = map(lambda i: locstring(i, data.n).center(cell_width), path)
    if color:
        colors = [TTYCOLORS.CYAN if i in data.P else (TTYCOLORS.MAGENTA if i in data.D else None) for i in path]
        path_str = [colortext(s, c) for s, c in zip(path_str, colors)]
    output += h_sep.join(path_str) + '\n'

    tw_start = []
    tw_end = []
    times_str = []
    time_fmt_str = f"{{:{cell_width:d},d}}"
    format_time = lambda v: time_fmt_str.format(v)
    delivery_deadlines = {}
    for i, t in zip(path, times):
        if i in data.P:
            delivery_deadlines[i + data.n] = t + data.max_ride_time[i]
        elif i in delivery_deadlines:
            if t <= delivery_deadlines[i] + EPS:
                del delivery_deadlines[i]

    for i, t in zip(path, times):
        ts = format_time(t)
        es = format_time(data.tw_start[i])
        ls = format_time(data.tw_end[i])
        if color:
            if t < data.tw_start[i] - EPS:
                es = colortext(es, TTYCOLORS.RED)
            if t > data.tw_end[i] + EPS:
                ls = colortext(ls, TTYCOLORS.RED)
            if i + data.n in delivery_deadlines or i in delivery_deadlines:
                ts = colortext(ts, TTYCOLORS.RED)
        tw_start.append(es)
        tw_end.append(ls)
        times_str.append(ts)

    output += h_sep.join(tw_start) + '\n'
    output += h_sep.join(times_str) + '\n'
    output += h_sep.join(tw_end)

    return output


def pprint_path(path, data: SDARP_Data, add_depots=False):
    print(pformat_path(path, data, add_depots=add_depots))

def pprint_schedule(schedule, path, data : SDARP_Data):
    print(pformat_schedule(schedule, path, data))


def preprocess(data : SDARP_Data) -> SDARP_Data:
    return darp.remove_arcs(darp.tighten_time_windows(data))


def _read_yaml_file(filename):
    if isinstance(filename, dict):
        return filename
    with open(filename, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CFullLoader)


class SDARP_Experiment(BaseExperiment):
    ROOT_PATH = BaseExperiment.ROOT_PATH / 'sdarp'
    INPUTS = {
        "index": {
            "type": "integer",
            "coerce": int,
            "min": 0,
            # "max": 29,
            "help" : "Data index."
        },
        "instance": {
            "type": "string",
            "derived": True,
        },
        **BaseExperiment.INPUTS
    }
    PARAMETERS = {
        "param_name": {
            "type": "string",
            "default": "",
            "help": "Parameter set name, will be generated automatically from a hash of parameters if left blank"
        },
        "timelimit": {
            "type": "float",
            "min": 0,
            "coerce": float,
            "default": 7200
        },
        "gurobi": {
            "type": "dict",
            "default": {},
            "keysrules": {
                "type": "string",
            },
            "valuesrules": {
                "type": ["number", "string"]
            },
            'coerce' : _read_yaml_file,
            'help' : 'Path to JSON/YAML file containing Gurobi parameters'
        },
        "cpus": {
            "type": "integer",
            "min": 1,
            "default": 4
        },
        **BaseExperiment.PARAMETERS
    }
    OUTPUTS = {
        "info": {"type": "string", "derived": True, 'coerce' : str},
        "solution": {"type": "string", "derived": True, 'coerce' : str},
        **BaseExperiment.OUTPUTS
    }

    @property
    def parameter_string(self):
        if len(self.parameters['param_name']) > 0:
            return self.parameters['param_name']
        return super().parameter_string

    def __init__(self, profile, inputs, outputs, parameters=None):
        super().__init__(profile, inputs, outputs, parameters)
        self._data = None

    def define_derived(self):
        self.inputs["instance"] = fraglib.sdarp.instance.index_to_name(self.inputs['index'])
        self.outputs["info"] = self.get_output_path("info.json")
        self.outputs["solution"] = self.get_output_path("soln.json")
        super(SDARP_Experiment, self).define_derived()

    def write_index_file(self):
        index = {k: os.path.basename(v) for k, v in self.outputs.items() if k != 'indexfile'}
        index.update(self.inputs)
        index['data_id'] = self.data.id
        with open(self.outputs['indexfile'], 'w') as fp:
            json.dump(index, fp, indent='\t')
        return index

    @classmethod
    def get_parser_arguments(cls):
        clargs = super().get_parser_arguments()
        args, kwargs = clargs['gurobi']
        kwargs["metavar"] = "FILE"
        clargs['gurobi'] = (args, kwargs)
        return clargs

    @property
    def data(self) -> SDARP_Skeleton_Data:
        if self._data is None:
            self._data = get_named_instance_skeleton_SDARP(self.inputs['instance'])
        return self._data

    @property
    def input_string(self):
        s = "{index:d}-{instance}".format_map(self.inputs)
        return s

    @property
    def resource_mail_user(self) -> str:
        return 'yanni555rist@gmail.com'

    @property
    def resource_job_name(self) -> str:
        return self.parameter_string

    @property
    def resource_name(self) -> str:
        return f'{self.parameter_string}/{self.input_string}'

    @property
    def resource_constraints(self) -> str:
        return 'R640'

    @property
    def resource_cpus(self) -> str:
        return str(self.parameters['cpus'])

    @property
    def resource_time(self) -> str:
        return slurm_format_time(self.parameters['timelimit'] + 300)


class SDARPSolution:
    def __init__(self, best, bound):
        self.best = best
        self.bound = bound
        self.routes = set()

    def add_route(self, path: Tuple):
        self.routes.add(path)

    def to_json_file(self, filename):
        with open(filename, 'w') as fp:
            json.dump({
                'best' : self.best,
                'bound' : self.bound,
                'routes' : sorted(self.routes),
            }, fp, indent='\t')


    @classmethod
    def from_json_file(cls, filename):
        with open(filename, 'r') as fp:
            d = json.load(fp)
        soln = cls(d['best'], d['bound'])
        for r in d['routes']:
            soln.add_route(tuple(r))
        return soln

    def __eq__(self, other):
        if not isinstance(other, SDARPSolution):
            raise NotImplementedError
        return self.best == other.best and self.bound == other.bound and self.routes == other.routes

    def pprint(self, data : SDARP_Data):
        print(self.pformat(data))

    def pformat(self, data):
        s = ''
        for k,r in enumerate(self.routes):
            s += f"Vehicle {k:d}\n"
            s += pformat_path(r, data) + '\n'
        return s[:-1]



NAMED_DATASET_INDICES = {
    "easy" : frozenset([0,1,3,10,11,12,21,23]),
    "medium" : frozenset([0,1,2,3,4,5,6,8,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27]),
    "very_hard" : frozenset([7, 18, 29]),
    "riedler": frozenset(range(30)),
    "loose_easy": frozenset([30, 31, 32, 33, 34, 35, 36, 37, 38, 43, 44, 45, 57, 58, 59, 60, 61, 62]),
    "loose": frozenset(range(30, fraglib.sdarp.instance.len())),
    "loose_impossible": frozenset({48, 49, 50}),
    "loose_cap": frozenset(range(57, 66)),
    "all" : frozenset(range(fraglib.sdarp.instance.len())),
}
NAMED_DATASET_INDICES["hard"] = NAMED_DATASET_INDICES["all"] - NAMED_DATASET_INDICES["easy"]
