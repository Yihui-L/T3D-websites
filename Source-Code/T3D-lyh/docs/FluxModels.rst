.. _flux_models:

Flux Models
===========

In non-trivial use cases, T3D requires interfacing with one or more external codes that compute transport fluxes. The available interfaces and instructions for using them are provided below. The available interfaces can also be found in ``t3d/flux_models``. If you would like to add a new flux model interface, see :ref:`Adding New Flux Models <add_model>`.

The model(s) used in the calculation are controlled by the ``[[model]]`` group in the T3D input file, via

.. code-block:: toml

  [[model]]
    model = "..."

The possible values of this parameter are given in the headings below. For each model, there are additional ``[[model]]`` parameters that can/should be specified, as detailed below.

.. _gx_model:

model = "GX"
------------

`GX <https://gx.rtfd.io/en/latest/>`_ is a gyrokinetic turbulence code that runs on NVIDIA GPUs. After `installing <https://gx.readthedocs.io/en/latest/Install.html>`_ GX on your target system, set the ``GX_PATH`` environment variable via::

  export GX_PATH=[/path/to/gx_repo]

so that the ``gx`` executable can be found at ``$GX_PATH/gx``. If T3D cannot find the ``gx`` executable it will throw an error::

  Error: gx executable not found! Make sure the GX_PATH environment variable is set.

Additionally, T3D requires a template GX input file, which will be used to set parameters such as the numerical resolution for the GX calculation. The (relative) path to the template file should be passed via the ``gx_template`` parameter in ``[[model]]``. This need not be a full GX input file (although it can be), as T3D will automatically modify and/or add many parameters. An example GX template is given in ``tests/regression/gx_template.in``. The most critical tunable parameters are in ``[Dimensions]`` (numerical resolution), ``[Domain]`` (box dimensions), and ``[Time]`` (length of GX calculation).

In addition to setting ``model = "GX"``, the following parameters in the ``[model]`` group can/should be specified in the T3D input file.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[model]``
     - ``gx_template``
     - Path to template input file for GX.
     - "tests/regression/gx_template.in"
   * - ``[model]``
     - ``gx_outputs``
     - Root directory for GX files. Each model instance writes its own ``t3d-run-*`` subdirectory.
     - "gx/"
   * - ``[model]``
     - ``overwrite``
     - Legacy option. GX does not reuse flux outputs from any previous run or rejected trial regardless of this value.
     - False
   * - ``[model]``
     - ``gpus_per_gx``
     - GPU/MPI rank count for each GX case. GX species and Hermite grids must support this decomposition.
     - 1
   * - ``[model]``
     - ``gx_max_concurrent``
     - Limit the number of simultaneous GX cases for this model. The local launcher partitions visible GPUs into disjoint groups. Built-in Slurm and custom launchers default to one case at a time; set an explicit cap compatible with the allocation if desired.
     - Auto
   * - ``[model]``
     - ``check_stalls``
     - Abort a running case if its log stops growing for ``monitor_time * stall_abort_count`` seconds. Enable only when an allocated job will report progress reliably; a queued scheduler step may produce no output.
     - False
   * - ``[model]``
     - ``electromagnetic``
     - If False, beta will be set to :math:`10^{-4}` in the GX input file to effectively force the GX calculation into the electrostatic limit.
     - True
 

.. note::

   External GX, GS2, SFINCS, KNOSOS, and NEO runs use fresh
   ``t3d-run-*`` work directories beneath their configured output roots.
   Time step, Newton iteration, and retry generation isolate each trial.
   Raw output files from older invocations are not replayed automatically.
   GX may use a successful checkpoint from the current invocation to seed a
   new calculation on unchanged magnetic geometry. This is not flux reuse.
   With several models sharing one hardware allocation, coordinate per-model
   capacity limits or give each a disjoint allocation.
   The SFINCS ``cpus_total`` and KNOSOS ``cpus_total`` options bound their
   CPU concurrency; GS2 and NEO use ``nproc`` as the CPU budget.

model = "KNOSOS"
----------------

.. `KNOSOS <https://github.com/joseluisvelasco/KNOSOS>`_ is a neoclassical transport code for stellarators. After `installing <https://github.com/joseluisvelasco/KNOSOS/blob/master/MANUAL/KNOSOSManual.pdf>`_ KNOSOS on your target system, set the ``KNOSOS_PATH`` environment variable via::
.. 
..   export KNOSOS_PATH=[/path/to/KNOSOS/SOURCES/]
.. 
.. so that the ``knosos.x`` executable can be found at ``$KNOSOS_PATH/knosos.x``. If T3D cannot find the ``knosos.x`` executable it will throw an error::
.. 
..   Error: knosos.x executable not found! Make sure the KNOSOS_PATH environment variable is set.

model = "ReLU"
--------------

.. _add_model:

Adding New Flux Models
----------------------
