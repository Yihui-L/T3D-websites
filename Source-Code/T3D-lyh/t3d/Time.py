from math import isfinite, nan
from t3d.Logbook import info


class Time():

    def __init__(self, inputs):

        # read time parameters from input file

        time_parameters = inputs.get('time', {})
        self.max_newton_iter = time_parameters.get('max_newton_iter', 4)       # maximum number of newton iterations
        self.newton_threshold = time_parameters.get('newton_threshold', 0.02)  # stop newton iteration when err < newton_threshold
        self.newton_tolerance = time_parameters.get('newton_tolerance', 0.1)   # allow advance to next timestep if err < newton_tolerance after max_newton_iter
        self.alpha = time_parameters.get('alpha', 1)                           # implicitness parameter (1.0 -> fully implicit, 0.0 -> fully explicit)
        self.dtau = time_parameters.get('dtau', 0.5)                           # initial timestep
        self.dtau_max = time_parameters.get('dtau_max', 10.0)                  # maximum allowed timestep
        self.dtau_adjust = time_parameters.get('dtau_adjust', 2.0)             # adjustment factor when decreasing timestep
        self.dtau_increase_threshold = time_parameters.get('dtau_increase_threshold', self.newton_threshold/4.)  # increase timestep if err < dtau_increase_threshold
        self.N_steps = time_parameters.get('N_steps', 1000)                    # number of timesteps
        self.t_max = time_parameters.get('t_max', 1000)                        # end time
        self.use_SI = time_parameters.get('use_SI', False)                     # interpret t_max and dtau time inputs as SI
        self.rms_converged = time_parameters.get('rms_converged', 1e-16)       # rms converged halting condition
        self.max_timestep_retries = time_parameters.get('max_timestep_retries', 20)
        if (not isfinite(self.dtau) or self.dtau <= 0
                or not isfinite(self.dtau_max) or self.dtau_max <= 0
                or not isfinite(self.dtau_adjust) or self.dtau_adjust <= 1):
            raise ValueError("dtau and dtau_max must be finite and positive; dtau_adjust must exceed 1")
        iteration_limits = ((self.N_steps, 0), (self.max_newton_iter, 1), (self.max_timestep_retries, 0))
        if (not isfinite(self.t_max) or self.t_max < 0
                or any(type(n) is not int or n < minimum for n, minimum in iteration_limits)):
            raise ValueError("invalid time limit, step limit, Newton count, or timestep retry limit")

        if any(not isfinite(x) or x < 0 for x in (self.newton_threshold, self.newton_tolerance,
                                                    self.dtau_increase_threshold, self.rms_converged)):
            raise ValueError("RMS thresholds must be finite and nonnegative")

        if not isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and between 0 and 1")
        self.dtau = min(self.dtau, self.dtau_max)

        # init some variables
        self.time = 0
        self.step_idx = 0
        self.iter_idx = 0
        self.retry_generation = 0
        self.gx_idx = 0
        self.prev_p_id = 0
        self.dtau_old = self.dtau  # legacy flux-backend retry cache hint
        self.dtau_previous_accepted = None  # BDF2 history, updated only when a step advances time
        self.dtau_success = self.dtau
        self.prev_step_success = False
        self.newton_mode = False
        self.rms = nan
        self.steady_rms = nan
        self.implicit_rms = nan
        self.diagnostics_current = False

    def normalize_inputs(self, t_ref):
        """Function to compute normalized time inputs if specified in SI"""

        if not isfinite(t_ref) or t_ref <= 0:
            raise ValueError("Reference time must be finite and positive")
        if self.use_SI:
            # interpret Trinity time inputs as SI units
            info("  Time inputs set in SI units, normalizing")
            info(f"    t_max: {self.t_max} s, {self.t_max/t_ref:.3f} in normalized units")
            info(f"    dtau: {self.dtau} s, {self.dtau/t_ref:.3f} in normalized units\n")

            self.t_max = self.t_max / t_ref
            self.dtau = self.dtau / t_ref
            self.dtau_max = self.dtau_max / t_ref
            # The shrink factor and RMS threshold are dimensionless.

            self.dtau_old = self.dtau
            self.dtau_success = self.dtau

        else:
            info("  Normalized time inputs specified")
            info(f"    t_max: {self.t_max:.3f} which corresponds to {self.t_max*t_ref:.3e} s")
            info(f"    dtau: {self.dtau:.3f} which corresponds to {self.dtau*t_ref:.3e} s\n")

    def d1(self, step_idx):
        if self.alpha == 1 and step_idx > 0 and self.dtau_previous_accepted is not None:
            previous = self.dtau_previous_accepted
            return (2.*self.dtau + previous)/(self.dtau + previous)/self.dtau
        else:
            return 1/self.dtau

    def d0(self, step_idx):
        if self.alpha == 1 and step_idx > 0 and self.dtau_previous_accepted is not None:
            previous = self.dtau_previous_accepted
            return -(self.dtau + previous)/previous/self.dtau
        else:
            return -1/self.dtau

    def dm1(self, step_idx):
        if self.alpha == 1 and step_idx > 0 and self.dtau_previous_accepted is not None:
            previous = self.dtau_previous_accepted
            return self.dtau / (previous * (self.dtau + previous))
        else:
            return 0
