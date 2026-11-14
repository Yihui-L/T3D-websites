from collections import OrderedDict

from t3d.Logbook import info


class SourceModelDict():
    '''
    Container for models of particle & energy sources
    '''

    def __init__(self, inputs, grid, time):
        from t3d.source_models.BEAMS3D import BEAMS3D_SourceModel

        self._sources = OrderedDict()

        if 'source' in inputs:
            sources = self._normalize_input(inputs['source'])
        else:
            info('\n  No source models found in inputs.')
            return

        _registered_sources = {
            "BEAMS3D": BEAMS3D_SourceModel,
        }

        for pars in sources:
            key = pars['source']
            label = pars.get("label", key)
            if key not in _registered_sources:
                raise ValueError(f"Unknown source model {key!r}")
            if label in self._sources or label in {"aux", "alpha", "rad", "coll", "heating"}:
                raise ValueError(f"Duplicate or reserved source label {label!r}")
            self._sources[label] = _registered_sources[key](pars, grid, time)

        # Print logfile for source models
        info('\n  Source model log files')
        for model in self.get_sources_list():
            info(f'    {model.label}: {model.logfile_info()}')

    def _normalize_input(self, sources):
        """Ensure the input is always a list-of-dict"""
        if type(sources) is dict:
            return [sources]
        elif type(sources) is list:
            return sources
        else:
            raise TypeError("SourceModelDict: source must be a single dict or list-of-dict")

    def __getitem__(self, s):
        '''
        Access a particular source via brackets [ ]
        '''
        return self._sources[s]

    def get_sources_list(self):
        '''
        Return a list of SourceModel (sub)class instances
        '''

        return self._sources.values()


# base class
class SourceModel():

    # base class constructor, should be called in derived class
    # constructor by super().__init__()
    def __init__(self, inputs, grid, time):
        self.N_radial = len(grid.rho)
        self.rho = grid.rho
        self.grid = grid
        self.time = time

    def compute_sources(self, species, geo, norms):
        assert False, "Error: no compute_sources function for this source model."

    def final_info(self):
        pass

    def collect_results(self, species, geo, norms):
        pass
