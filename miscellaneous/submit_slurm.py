#!struct/sachse/software/springbox_2017-dev/bin/python

from spring.csinfrastr.csproductivity import OpenMpi
import sys

OpenMpi().check_if_mpi_works_and_launch_command('springenv python {0}'.format(" ".join(sys.argv[1:])), 10)
