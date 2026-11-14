import numpy as np
from copy import copy
from pathlib import Path
from tempfile import mkdtemp
from t3d.Logbook import info


class DescHandler():

    def __init__(self, desc_file, nr=100, use_vmec_format=None, device="auto"):

        from desc import set_device

        # Configure DESC before importing its backend. Merely retrying the
        # import after GPU initialization fails leaves DESC configured for GPU.
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("desc_device must be auto, cpu or gpu")
        if device == "auto":
            import jax
            try:
                device = "gpu" if jax.devices("gpu") else "cpu"
            except RuntimeError:
                device = "cpu"
        set_device(device)
        import desc.io
        info(f"  DESC device: {device}")

        from desc.grid import LinearGrid
        from desc.vmec import VMECIO

        self.rho = np.linspace(0, 1, nr)
        self.desc_grid = LinearGrid(rho=self.rho)

        # load desc output
        # check if we are using VMEC format or DESC format
        if desc_file[-3:] == '.nc':
            if use_vmec_format is not None:
                self.use_vmec_format = use_vmec_format
            else:
                self.use_vmec_format = True
            eq = VMECIO.load(desc_file, L=10, M=10, N=10)
            self.out_vmec = desc_file
        else:
            if use_vmec_format is not None:
                self.use_vmec_format = use_vmec_format
            else:
                self.use_vmec_format = False
            eq = desc.io.load(desc_file)
            if hasattr(eq, '__len__'):
                eq = eq[-1]
            self.out_vmec = None

        self.get_geometry(eq)

        # store for later
        self.desc_eq = eq

    def get_geometry(self, eq):
        from desc.grid import LinearGrid

        # gets area and grad rho from DESC for Trinity
        if hasattr(eq, '__len__'):
            eq = eq[-1]

        grid = self.desc_grid

        # area = dVdrho * <|grad rho|>
        self.area = eq.compute('S(r)', grid=grid)['S(r)']
        self.dVdrho = eq.compute('V_r(r)', grid=grid)['V_r(r)']
        self.grad_rho = np.divide(np.asarray(self.area), np.asarray(self.dVdrho),
                                  out=np.zeros_like(np.asarray(self.area)),
                                  where=np.asarray(self.dVdrho) != 0)

        self.B_ref = eq.compute('|B|', grid=grid)['|B|'][0]
        self.B0 = eq.compute('<|B|>', grid=LinearGrid(rho=0.0))['<|B|>'][0]
        self.iota23 = eq.compute('iota', grid=LinearGrid(rho=0.66667))['iota'][0]
        self.aminor = eq.compute('a')['a']
        self.Rmajor = eq.compute('R0')['R0']

    @staticmethod
    def pressure_ratio(old, new):
        old, new = np.asarray(old), np.asarray(new)
        if (old.shape != new.shape or not np.all(np.isfinite(old)) or not np.all(np.isfinite(new))
                or np.any(old < 0) or np.any(new < 0)):
            raise ValueError("DESC requires finite nonnegative pressure on matching grids")
        floor = max(float(np.max(old)), float(np.max(new)), 1.)*1e-12
        old, new = np.maximum(old, floor), np.maximum(new, floor)
        return float(max(np.max(old/new), np.max(new/old)))

    def update_desc_eq(self, geometry, species, time):
        axis = geometry.grid.rho
        p_trin = np.asarray(species.get_pressure_SI())
        if hasattr(self, "transport_pressure_target"):
            p_desc = self.transport_pressure_target
        else:
            p_desc = self.get_pressure_SI(axis)
        delta = self.pressure_ratio(p_desc, p_trin)
        if delta <= geometry.desc_pressure_threshold:
            return None
        root = Path(geometry.geo_outputs).resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(prefix=f"desc-t{time.step_idx:03d}-p{time.iter_idx}-", dir=root))
        logfile = directory / "desc.log"

        # Hold the given outer pressure constant between the last transport node
        # and LCFS rather than invent a zero edge pressure.
        if axis[-1] < 1:
            axis = np.r_[axis, 1.]
            p_trin = np.r_[p_trin, p_trin[-1]]
        from contextlib import redirect_stdout
        from desc.profiles import PowerSeriesProfile
        pressure = PowerSeriesProfile.from_values(axis, p_trin, sym=True, order=min(6, 2*(axis.size-1)))
        if np.any(np.asarray(pressure(np.linspace(0, 1, 100))) < 0):
            raise ValueError("Fitted pressure is negative between transport grid points")
        eq = self.desc_eq.copy()
        from t3d.Logbook import warn
        with logfile.open("w", buffering=1) as f, redirect_stdout(f):
            if delta < geometry.desc_pressure_perturb:
                try:
                    # Use the same pressure coefficient ordering for perturbation.
                    warm_pressure = pressure.copy()
                    warm_pressure.change_resolution(eq.pressure.basis.L)
                    if warm_pressure.params.shape == eq.pressure.params.shape:
                        eq = eq.perturb(deltas={"p_l": warm_pressure.params-eq.pressure.params}, verbose=3)
                except (ValueError, RuntimeError, AttributeError, np.linalg.LinAlgError):
                    warn("DESC pressure perturbation failed; attempting full solve")
                    eq = self.desc_eq.copy()
            eq.pressure = pressure
            eq, result = eq.solve(verbose=3)
            if not result.success:
                raise RuntimeError(f"DESC equilibrium did not converge; see {logfile}")

            candidate = copy(self)
            candidate.get_geometry(eq)
            candidate.desc_eq = eq
            candidate.transport_pressure_target = np.asarray(species.get_pressure_SI()).copy()
            if self.use_vmec_format:
                from desc.vmec import VMECIO
                output = directory / "wout_desc.nc"
                VMECIO.save(eq, str(output), surfs=512)
                candidate.out_vmec = str(output)
            else:
                output = directory / "desc.h5"
                eq.save(str(output))
                candidate.out_vmec = None
            candidate.output_file = str(output)
        return candidate

    def get_pressure_SI(self,axis):
        from desc.grid import LinearGrid
        grid = LinearGrid(rho=axis)
        p_SI = self.desc_eq.compute("p", grid=grid)['p']

        return p_SI


DescReader = DescHandler  # backwards compatibility
