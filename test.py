from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['cpi.py'],
                           maxprocs=5)
rank = comm.Get_rank()

N = np.ones(10, dtype=np.int64) * np.array(range(10))
M = np.ones(10, dtype=np.int64) * 25
recv_buffer = None
print("scattering    ", N, "from rank", rank)
#comm.scatter([N, M], root=MPI.ROOT)
#comm.Scatter(N, None, root=MPI.ROOT)
comm.Scatterv([N, 2, 2, MPI.LONG], None, root=MPI.ROOT)
"""
N = numpy.array(100, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
PI = numpy.array(0.0, 'd')
comm.Reduce(None, [PI, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
print(PI)
"""
comm.Disconnect()
