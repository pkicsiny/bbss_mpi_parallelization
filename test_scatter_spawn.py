from mpi4py import MPI
import sys


comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['src/scatter_src.py'],
                           maxprocs=4)
rank = comm.Get_rank()

data = [(i+1)**2 for i in range(4)]
print("Root: {}, rank {}, sending {} to 4 processes".format(MPI.ROOT,rank, data))
data = comm.scatter(data, root=MPI.ROOT)

comm.Disconnect()
