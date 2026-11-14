import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from t3d.flux_models.GS2_QL import GS2_QL_FluxModel
from t3d.flux_models.GX import GX_FluxModel
from t3d.flux_models.KNOSOS import KNOSOS_FluxModel
from t3d.flux_models.SFINCS import SFINCS_FluxModel
from t3d.flux_models.neo import neo_FluxModel
from t3d.flux_models._external import (
    require_outputs, run_workspace, trial_tag, wait_checked, worker_count,
)


class TestExternalTransportJobs(TestCase):
    def test_unique_run_namespace_and_retry_identity(self):
        with TemporaryDirectory() as root:
            one = Path(run_workspace(root))
            two = Path(run_workspace(root))
            self.assertNotEqual(one, two)
            self.assertEqual(one.parent, two.parent)
            time = SimpleNamespace(step_idx=2, iter_idx=1, retry_generation=0)
            first = trial_tag(time)
            time.retry_generation += 1
            self.assertNotEqual(first, trial_tag(time))

    def test_output_validation_rejects_missing_radius_and_zero_length(self):
        with TemporaryDirectory() as root:
            files = [Path(root, str(i)) for i in range(2)]
            files[0].write_text("flux")
            with self.assertRaisesRegex(RuntimeError, "KNOSOS.*complete output"):
                require_outputs(files, "KNOSOS")
            files[1].touch()
            with self.assertRaisesRegex(RuntimeError, "KNOSOS.*complete output"):
                require_outputs(files, "KNOSOS")
            files[1].write_text("flux")
            require_outputs(files, "KNOSOS")

    def test_invalid_capacity_and_exit_status(self):
        self.assertEqual(worker_count(10, 3), 3)
        with self.assertRaises(ValueError):
            worker_count(2, 3)
        process = SimpleNamespace(args=["solver"], wait=lambda: 7)
        with self.assertRaisesRegex(RuntimeError, "SFINCS exited with status 7"):
            wait_checked(process, "SFINCS")

    def test_gs2_srun_status_is_not_hidden_by_shell(self):
        backend = GS2_QL_FluxModel.__new__(GS2_QL_FluxModel)
        backend.mpiexec = "srun"
        backend.cpus_per_gs2 = 2
        backend.workers = 1
        backend.gs2_app = "/some path/gs2"
        with TemporaryDirectory() as root:
            case = str(Path(root, "my case.in"))
            with patch("t3d.flux_models.GS2_QL.subprocess.run",
                       side_effect=subprocess.CalledProcessError(5, ["srun"])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    backend.submit_gs2_job([case])
            self.assertEqual(run.call_args.args[0],
                             ["srun", "-n", "2", "--exact", "/some path/gs2", case])
            self.assertTrue(run.call_args.kwargs["check"])

    def test_neo_parser_failure_does_not_launch_solver(self):
        backend = neo_FluxModel.__new__(neo_FluxModel)
        backend.cpu = 2
        backend.mpiexec = "mpirun"
        backend.neo_app = "/some path/neo"
        with TemporaryDirectory() as root:
            with patch("t3d.flux_models.neo.subprocess.run",
                       side_effect=subprocess.CalledProcessError(2, ["neo_parse"])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    backend.run_neo_single(root)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], [sys.executable, "/some path/neo/bin/neo_parse.py"])
            self.assertEqual(run.call_args.kwargs["cwd"], root)
            self.assertEqual(run.call_args.kwargs["env"]["OMP_NUM_THREADS"], "2")

    def test_sfincs_equilibrium_copy_is_ready_before_submission(self):
        with TemporaryDirectory() as root:
            equilibrium = Path(root, "wout space.nc")
            equilibrium.write_text("mock equilibrium")
            state = SimpleNamespace(step_idx=0, iter_idx=0, retry_generation=3)
            profiles = np.ones((1, 1))
            species = SimpleNamespace(get_profiles_on_flux_grid=lambda **kw: (profiles,)*3)
            geo = SimpleNamespace(a_minor=1, geo_file=str(equilibrium))
            seen = []
            def submit(path):
                seen.append(Path(path, equilibrium.name).read_text())
                return "mock"
            backend = SimpleNamespace(time=state, out_dir=root, N_fluxtubes=1,
                                      namelist={"speciesparameters": {}, "geometryparameters": {}},
                                      write_sfincs_input=Mock(), submit_sfincs_job=submit)
            SFINCS_FluxModel.run_sfincs(backend, [0.5], profiles, profiles, species, geo, 0)
            self.assertEqual(seen, ["mock equilibrium"])
            self.assertTrue(Path(root, "t00-p0-a3-j0", "rho00").is_dir())

    def test_sfincs_honors_ranks_threads_and_waits_before_reusing_slot(self):
        process = Mock()
        process.wait.return_value = 0
        backend = SFINCS_FluxModel.__new__(SFINCS_FluxModel)
        backend.log = Mock()
        backend.__dict__.update(
            processes=[process], workers=1, label="SFINCS", SFINCS_PATH="/space root",
            mpis_per_sfincs=3, cpus_per_sfincs=2, run_command=None, sfincs_index=0,
            info=Mock(), print_time=Mock(),
        )
        with patch.dict(os.environ, {"GK_SYSTEM": "local"}):
            with patch("t3d.flux_models.SFINCS.start_job") as start:
                SFINCS_FluxModel.submit_sfincs_job(backend, "/case space")
        process.wait.assert_called_once()
        self.assertEqual(start.call_args.args[0][:4],
                         ["mpirun", "-np", "3", "/space root/sfincs"])
        self.assertEqual(start.call_args.kwargs["env"]["OMP_NUM_THREADS"], "2")
        self.assertEqual(start.call_args.kwargs["cwd"], "/case space")

    def test_knosos_does_not_skip_a_partially_populated_case(self):
        with TemporaryDirectory() as root:
            Path(root, "knosos_for_predictive.txt.00").write_text("stale")
            backend = SimpleNamespace(
                processes=[], workers=1, label="KNOSOS", KNOSOS_PATH="/space root",
                N_fluxtubes=2, run_command=None, info=Mock(), print_time=Mock(),
            )
            with patch.dict(os.environ, {"GK_SYSTEM": "local"}):
                with patch("t3d.flux_models.KNOSOS.start_job") as start:
                    paths = KNOSOS_FluxModel.submit_knosos_job(backend, root)
            self.assertEqual(start.call_args.args[0], ["mpirun", "-n", "2", "/space root/knosos.x"])
            self.assertEqual(len(paths), 2)
            with self.assertRaisesRegex(RuntimeError, "KNOSOS.*complete output"):
                require_outputs(paths, "KNOSOS")

    def test_local_gx_uses_mpi_ranks_and_disjoint_device_groups(self):
        launcher = GX_FluxModel.__new__(GX_FluxModel)
        launcher.dry_run = False
        launcher.run_command = None
        launcher.gpus_per_gx = 2
        launcher.gx_max_concurrent = None
        with patch.dict(os.environ, {"GX_PATH": "/path with space", "GK_SYSTEM": "local",
                                    "CUDA_VISIBLE_DEVICES": "2,3,5,7"}):
            launcher._configure_devices()
            self.assertEqual(launcher.workers, 2)
            self.assertEqual(launcher.gpu_groups, [(None, "2,3"), (None, "5,7")])
            launcher.gx_app = "/path with space/gx"
            launcher.processes = []
            launcher.logs = []
            launcher._progress = {}
            launcher.call_counter = 0
            launcher.gx_index = 0
            launcher.info = Mock()
            launcher.log = Mock()
            launcher.print_time = Mock()
            with patch("t3d.flux_models.GX.start_job") as start:
                launcher.submit_gx_job("a", "/my case")
                launcher.submit_gx_job("b", "/my case")
            ref = ["mpiexec", "-n", "2", "/path with space/gx"]
            calls = start.call_args_list
            self.assertEqual([c.args[0] for c in calls],
                             [ref + ["/my case/a.in"], ref + ["/my case/b.in"]])
            self.assertEqual([c.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for c in calls], ["2,3", "5,7"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3,5,7")

    def test_polaris_gpu_group_assigns_one_three_gpu_case_per_node(self):
        launcher = GX_FluxModel.__new__(GX_FluxModel)
        launcher.log = Mock()
        launcher.dry_run = False
        launcher.run_command = None
        launcher.gpus_per_gx = 3
        launcher.gx_max_concurrent = None
        with TemporaryDirectory() as root:
            launcher.out_dir = root
            nodefile = Path(root, "allocated_hosts")
            nodefile.write_text("first\nfirst\nsecond\n")
            with patch.dict(os.environ, {"GK_SYSTEM": "polaris", "PBS_NODEFILE": str(nodefile)}):
                launcher._configure_devices()
            self.assertEqual(launcher.workers, 2)
            self.assertEqual([mask for _, mask in launcher.gpu_groups], ["0,1,2", "0,1,2"])
            self.assertEqual([Path(p).read_text().strip() for p, _ in launcher.gpu_groups],
                             ["first", "second"])

    def test_gx_wait_uses_exit_status_without_text_sentinel(self):
        process = Mock()
        process.poll.return_value = 0
        launcher = SimpleNamespace(check_stalls=True, label="GX", _progress={})
        GX_FluxModel._wait_one_gx(launcher, process, "/absent/no_end_time")
        process.poll.assert_called_once()
        process.poll.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "GX exited with status 7"):
            GX_FluxModel._wait_one_gx(launcher, process, "/absent/no_end_time")

    def test_gx_stall_raises_instead_of_rerunning_invalid_result(self):
        process = Mock()
        process.poll.return_value = None
        launcher = SimpleNamespace(check_stalls=True, label="GX", monitor_time=10,
                                   stall_abort_count=2, _progress={"run.log": (0, 0)})
        with patch("t3d.flux_models.GX.os.path.getsize", return_value=0):
            with patch("t3d.flux_models.GX._time.monotonic", return_value=20):
                with self.assertRaisesRegex(RuntimeError, "stopped writing progress"):
                    GX_FluxModel._wait_one_gx(launcher, process, "run.log")

    def test_gx_cache_does_not_accept_incomplete_angle_or_surface_set(self):
        with TemporaryDirectory() as root:
            lhs = str(Path(root, "angle0"))
            rhs = str(Path(root, "angle1"))
            Path(lhs + ".out.nc").write_text("mock")
            launcher = SimpleNamespace(label="GX")
            with self.assertRaisesRegex(RuntimeError, "angle1"):
                GX_FluxModel._require_gx_outputs(launcher, [lhs, rhs])
            Path(rhs + ".nc").write_text("mock")
            GX_FluxModel._require_gx_outputs(launcher, [lhs, rhs])

    def test_gx_refreshes_surrogate_on_retry_and_changed_equilibrium(self):
        model = GX_FluxModel.__new__(GX_FluxModel)
        model.log = Mock()
        model.build_ReLU = True
        model.nsteps_ReLU = 10
        model._restart_geometry = "original"
        model._geometry_token = Mock(return_value="original")
        model.compute_fluxes_ReLU = Mock()
        model.compute_fluxes_gx = Mock()
        model.time = SimpleNamespace(iter_idx=0, step_idx=1, retry_generation=0)

        model.use_ReLU = True
        model.compute_fluxes(None, None, None)
        model.compute_fluxes_ReLU.assert_called_once()

        model.time.retry_generation = 1
        model.compute_fluxes(None, None, None)
        model.compute_fluxes_gx.assert_called_once()

        model.time.retry_generation = 0
        model._geometry_token.return_value = "updated"
        model.use_ReLU = True
        model.compute_fluxes(None, None, None)
        self.assertEqual(model.compute_fluxes_gx.call_count, 2)
