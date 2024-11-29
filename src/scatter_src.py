from mpi4py import MPI
import sys
comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()

data = None
data = comm.scatter(data, root=0)
print("Rank {}, data: {}".format(rank, data))
assert data == (rank+1)**2

comm.Disconnect()


