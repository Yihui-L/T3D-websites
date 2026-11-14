import os
import argparse
from netCDF4 import Dataset
import adios2


def main():

    description = "Utility to create a T3D input file from an output NetCDF4 or ADIOS2 output file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("output_file",
                        help="Specify plot file ")
    parser.add_argument("-i", "--input-file",
                        help="Specify the name of the input file to create",
                        default=None)

    args = parser.parse_args()

    input_file = os.path.splitext(args.output_file)[0] + ".in"
    if args.input_file:
        if not args.input_file.endswith(".in"):
            raise ValueError("The specified input must end with '.in'")
        else:
            input_file = args.input_file

    if args.output_file.endswith("nc"):
        data = Dataset(args.output_file, "r", format="NETCDF4")
        input_file_str = data.getncattr("inputs")
        data.close()

    elif args.output_file.endswith("bp") or args.output_file.endswith("bp/"):
        data = adios2.FileReader(args.output_file)
        input_file_str = data.read_attribute("inputs")
        data.close()

    else:
        raise ValueError("The output file format is not recognized, must be .nc or .bp")

    with open(input_file, "w") as f:
        f.write(input_file_str)


if __name__ == '__main__':
    main()
