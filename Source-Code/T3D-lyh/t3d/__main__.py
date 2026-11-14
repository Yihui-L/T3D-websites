from t3d.trinity_lib import TrinityEngine

import argparse
from importlib import metadata
from pathlib import Path
# import time
from t3d.Logbook import info, bold, log


def print_time(dt):
    ''' Print elapsed time '''
    h = int(dt // 3600)
    m = int((dt - 3600 * h) // 60)
    s = dt - 3600 * h - 60 * m
    info(f"Total time: {h:d}h {m:d}m {s:.1f}s")


def main():

    # Set-up command line arguments
    description = "Trinity3D profile prediction for stellarators and tokamaks"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file",
                        help="Name of input file",
                        default=None)
    parser.add_argument("-l", "--log",
                        help="Specify the log file (optional)",
                        default=None)

    # Parse command line arguments
    args = parser.parse_args()

    fin = args.input_file
    if Path(fin).suffix != ".in":
        parser.error("T3D input file must have the .in extension")

    # Create the log file
    if args.log:
        log_file = args.log
    else:
        log_file = args.input_file[:-3] + ".out"

    log.set_handlers(term_stream=True,
                     file_stream=True,
                     file_handler=log_file)

    bold("Running T3D calculation.")
    try:
        version = metadata.version('t3d')
    except metadata.PackageNotFoundError:
        version = "source checkout"
    info(f"Version = {version}")

    info("\n  Loading input file: " + fin)
    info("  Logging output file: " + log_file)

    # start_time = time.time()

    try:
        engine = TrinityEngine(fin)
        engine.evolve_profiles()
        bold('\nT3D Complete, exiting normally.')
    finally:
        log.finalize()


if __name__ == "__main__":

    main()
