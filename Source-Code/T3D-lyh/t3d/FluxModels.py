from abc import ABC, abstractmethod
from collections import OrderedDict

from t3d.Logbook import info


class FluxModelDict():
    '''
    This library contains model functions for fluxes.

    + there is an analytic ReLU model called Flux_model
    + there is an anlaytic model from Barnes' thesis
    + there is the GX flux model
    '''

    def __init__(self, inputs, grid, time):
        from t3d.flux_models.GX import GX_FluxModel
        from t3d.flux_models.GS2_QL import GS2_QL_FluxModel
        from t3d.flux_models.ReLU import ReLU_FluxModel
        from t3d.flux_models.ChangHintonNeoclassical import ChangHintonNeo_FluxModel
        from t3d.flux_models.KNOSOS import KNOSOS_FluxModel
        from t3d.flux_models.neo import neo_FluxModel
        from t3d.flux_models.RR import RechesterRosenbluth
        from t3d.flux_models.SFINCS import SFINCS_FluxModel
        from t3d.flux_models.PowerBalance import PowerBalance_FluxModel

        config = inputs.get('model', {})
        if isinstance(config,dict):
            config = [config]
        if not isinstance(config,list) or any(not isinstance(pars,dict) for pars in config):
            raise TypeError("model must be a table or an array of tables")
        self.models_dict = OrderedDict()

        models = {
            "GX": GX_FluxModel,
            "gx": GX_FluxModel,
            "GS2-QL": GS2_QL_FluxModel,
            "gs2-ql": GS2_QL_FluxModel,
            "ReLU": ReLU_FluxModel,
            "relu": ReLU_FluxModel,
            "ChangHinton": ChangHintonNeo_FluxModel,
            "KNOSOS": KNOSOS_FluxModel,
            "knosos": KNOSOS_FluxModel,
            "NEO": neo_FluxModel,
            "neo": neo_FluxModel,
            "RR": RechesterRosenbluth,
            "SFINCS": SFINCS_FluxModel,
            "sfincs": SFINCS_FluxModel,
            "PowerBalance": PowerBalance_FluxModel
        }

        for pars in config:
            key = pars.get('model','GX')
            if key not in models:
                raise ValueError(f"Unknown flux model {key!r}")
            label = pars.get('label', {'gx':'GX','GS2-QL':'GS2','gs2-ql':'GS2','relu':'ReLU',
                                       'knosos':'KNOSOS','neo':'NEO','sfincs':'SFINCS',
                                       'ChangHinton':'Chang-Hinton'}.get(key,key))
            if not isinstance(label,str) or not label or '/' in label:
                raise ValueError(f"Invalid flux model label {label!r}")
            model = models[key](pars, grid, time)
            label = model.label
            if label in self.models_dict:
                raise ValueError(f"Duplicate flux model label {label!r}; assign unique labels.")
            self.models_dict[label] = model
        self.N_models = len(self.models_dict)

        # check for compatibility of sqrt(2) v_th conventions
        self.vt_sqrt_2 = None
        for model in self.get_models_list():
            if model.vt_sqrt_2 is not None and self.vt_sqrt_2 is None:
                self.vt_sqrt_2 = model.vt_sqrt_2
            elif model.vt_sqrt_2 is not None and self.vt_sqrt_2 is not None:
                if model.vt_sqrt_2 != self.vt_sqrt_2:
                    raise ValueError("Flux models must use the same thermal-speed normalization")

        # Print logfile for flux models
        info('\n  Flux model log files')
        for model in self.get_models_list():
            info(f'    {model.label}: {model.logfile_info()}')

        if self.vt_sqrt_2 is None:
            self.vt_sqrt_2 = False

    def __getitem__(self, s):
        '''
        accessor that allows a particular element (FluxModel) of the models_dict dictionary to be accessed via models[s] == models.models_dict[s]
        '''
        return self.models_dict[s]

    def get_models_list(self):
        '''
        return list of all elements of models_dict. each element is a FluxModel (sub)class member
        '''
        return self.models_dict.values()


class FluxModel(ABC):

    # base class constructor, should be called in derived class
    # constructor by super().__init__()
    def __init__(self, inputs, grid, time):
        self.N_fluxtubes = len(grid.midpoints)
        self.rho = grid.midpoints
        self.grid = grid
        self.time = time

        self.vt_sqrt_2 = None

    @abstractmethod
    def compute_fluxes(self, species, geo, norms):
        pass

    def logfile_info(self) -> str:
        return ''

    def final_info(self):
        pass

    def collect_results(self, species, geo, norms):
        pass
