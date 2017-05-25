'''
Math574 - Computationa Topology - Spring 2016
---------------------------------------------

Homework6 on Smith Normal Form(SNF).

*Tasks:*
1. Given a boundary matrix, compute SNF over Z/2
   Refer to a function called SNF()
   - Used recursive matrix reduction algorithm
        Algorithm steps are commented in the function
        This algorithm does reduction recursively and for each recursion
        it works on smaller submatrix. The submatrix sizes decrease as follows.
        [mxn],[(m-1)x(n-1)], [(m-2)x(n-2)], ... ,[1,1].
        It interrupts if there is no nonzero element at the indices
        greater or equal to current iteration. On the other words, if the current
        submatrix is zero matrix, it stops.
   - Data structure - Sparse dok_matrix which support index lookup,
     assignment and simple operations such as multiplication, addition.
     Besides, it supports dot product. Though it doesn't support 
     module(%) operation. Hence, +/2 is done separately on nonzero elements
     after every addition of row or column. Since the number of nonzeros are small,
     it doesn't increase the complexity of the algorithm much.
2. Find all relevant Betti numbers of the triangulation of the spine given
    Triangles.txt, Vertices.txt
3.Plot the surface

*Data:*
Triagles.txt - List of triangles(mesh) which approxiates the surface
Vertices.txt - Coordinates of the points used in the trianglular mesh
'''
import os
import time
import numpy as np
from scipy.sparse import dok_matrix, find, identity
from scipy.sparse.linalg import inv
import scipy.sparse as sparse

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys
sys.setrecursionlimit(10**6)

N = None
U = None
V = None

def get_subsimplices(simplices):
    simplices = np.sort(simplices, axis=1)
    subsimplices = set()
    for j in np.arange(simplices.shape[1]):
        idx = list(range(simplices.shape[1]))
        idx.pop(j)
        subsimplices = subsimplices.union(set(tuple(sub) for sub in simplices.take(idx, axis=1)))
    subsimplices = np.array([ sub for sub in subsimplices], dtype=int)
    return subsimplices
    
def boundary_matrix(simplices, subsimplices, is_sparse=True, format='coo'):
    simplex_dim  = simplices.shape[1] 
    n_simplices = simplices.shape[0]
    m_subsimplices = subsimplices.shape[0] 
    if is_sparse:
        boundary_matrix = dok_matrix((m_subsimplices, n_simplices), dtype=int) 
    else:
        boundary_matrix = np.array((m_subsimplices, n_simplices), dtype=int) 
    val = 1
    simplices = np.sort(simplices, axis=1)
    subsimplices = np.sort(subsimplices, axis=1)
    for i, simplex in enumerate(simplices):
        for j in np.arange(simplex_dim):
            idx = list(range(simplex_dim))
            idx.pop(j)
            subsimplex = simplex.take(idx)
            # to check the membership of subsimplex in subsimplices
            subsimplex_idx = np.argwhere((subsimplices==subsimplex).all(axis=1) == True)
            if subsimplex_idx.size == 0:
                sys.stderr.write("Unable to find subsimplex! Make sure subsimplices contains all boundary subsimplices\n")
                exit()
            subsimplex_idx = subsimplex_idx[0][0]
            boundary_matrix[subsimplex_idx, i] = val
    if is_sparse:
        return boundary_matrix.asformat(format)
    else:
        return boundary_matrix

def SNF(x, uv=False):
    #print x
    global N
    global U
    global V
    #Get indices of nonzeros
    tmp = find(N)
    nonzero = tmp[0].reshape(-1,1)
    nonzero = np.hstack((nonzero, tmp[1].reshape(-1,1)))
    if np.amax(nonzero) < x:
        return
    #Get indices of nonzeros which is greater or equal to x
    idx = np.argwhere(np.all(nonzero>=x, axis=1) == True).reshape(-1,)
    idx = idx[idx.argsort()]
    if len(idx) !=0:
        if N[x,x] == 0:
            #Choose the least indices of nonzeros which is greater or equal to x
            k, l = nonzero[idx[0]]
            if k != x:
                #Exchanging xth row wih kth row
                x_nonzero = N[x, :].nonzero()[1]
                k_nonzero = N[k,:].nonzero()[1]
                for j in x_nonzero:
                    N[x,j] = 0
                for j in k_nonzero:
                    N[x,j] = 1
                    N[k,j] = 0
                for j in x_nonzero:
                    N[k,j] = 1
                if uv:
                    # Update U
                    Utmp = dok_matrix(identity(N.shape[0], dtype=int), dtype=int)
                    Utmp[k,x] = 1
                    Utmp[x,k] = 1
                    Utmp[k,k] = 0
                    Utmp[x,x] = 0
                    U = (Utmp*U).asformat("dok").astype(int)
                    del Utmp
                    for key,val in U.items():
                        U[key[0], key[1]] = val%2
            if l != x:
                #Exchanging xth column wih lth column
                x_nonzero = N[:, x].nonzero()[0]
                l_nonzero = N[:, l].nonzero()[0]
                for i in x_nonzero:
                    N[i, x] = 0
                for i in l_nonzero:
                    N[i,x] = 1
                    N[i,l] = 0
                for i in x_nonzero:
                    N[i,l] = 1
                if uv:
                    # Update V
                    Vtmp = dok_matrix(identity(N.shape[1], dtype=int), dtype=int)
                    Vtmp[l,x] = 1
                    Vtmp[x,l] = 1
                    Vtmp[l,l] = 0
                    Vtmp[x,x] = 0
                    V = (V*Vtmp).asformat("dok").astype(int)
                    del Vtmp
                    for key,val in V.items():
                        V[key[0], key[1]] = val%2
        #Iterate over rows of N down xth row
        x_col = N[:,x].nonzero()[0]
        x_col_i = x_col[np.argwhere(x_col >= x+1).reshape(-1,)]
        x_col_i = np.sort(x_col_i)
        x_row = N[x,:].nonzero()[0]
        x_row_j = x_row[np.argwhere(x_row >= x+1).reshape(-1,)]
        x_row_j = np.sort(x_row_j)
        for i in x_col_i:
            # Add row x to row i
            N[i,:] = N[i,:] + N[x,:]
            for key, val in N[i,:].items():
                N[i, key[1]] = val%2
            if uv:
                # Update U
                Utmp = dok_matrix(identity(N.shape[0], dtype=int), dtype=int)
                Utmp[i,x] = 1
                U = (Utmp*U).asformat("dok").astype(int)
                del Utmp
                for key,val in U.items():
                    U[key[0], key[1]] = val%2
        #Iterate over colmns of N right to col x
        x_row = N[x,:].nonzero()[1]
        x_row_j = x_row[np.argwhere(x_row >= x+1).reshape(-1,)]
        x_row_j = np.sort(x_row_j)
        for j in x_row_j:
            # Add col x to col j
            N[:,j] = N[:,j] + N[:,x]
            for key, val in N[:,j].items():
                N[key[0], j] = val%2
            if uv:
                # Update V
                Vtmp = dok_matrix(identity(N.shape[1], dtype=int), dtype=int)
                Vtmp[x,j] = 1
                V = (V*Vtmp).asformat("dok").astype(int)
                del Vtmp
                for key,val in V.items():
                    V[key[0], key[1]] = val%2
        SNF(x+1)

# Saves sparse matrix as text. if the input is not sparse, set is_sparse argument to False.
def sparse_savetxt(fname, matrix, fmt='%d', include_dim=False):
    if sparse.issparse(matrix):
        if matrix.getformat() !='coo':
            matrix = matrix.asformat('coo')
    else:
        matrix = sparse.coo_matrix(matrix, dtype=matrix.dtype)
    with open(fname, 'w') as f:
        if include_dim:
            f.write("%d %d \n" % (matrix.shape[0], matrix.shape[1]))
        for i in range(len(matrix.row)):
            fmt_str = "%d %d " + fmt + "\n"
            f.write(fmt_str % (matrix.row[i], matrix.col[i], matrix.data[i]))
#Loads SNF matrix from text file and returns in dok_matrix format
def load_N(fname):
    N = None
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    data = line.split()
                    N = sparse.dok_matrix((int(data[0]), int(data[1])), dtype=int)
                else:
                    data = line.split()
                    N[int(data[0]), int(data[1])] = int(data[2])
    else:
        print "Can't load <N>. <%s> doesn't exist"%fname
    return N

def spinedemo():
    global N
    global U
    global V

    start = time.time()
    triangles = np.loadtxt("Triangles.txt")
    triangles = triangles -1
    triangles = np.vstack((triangles, np.array([498,499,501]).reshape(1,-1)))
    triangles = np.vstack((triangles, np.array([341,348,1256]).reshape(1,-1)))
    triangles = np.vstack((triangles, np.array([90,94,263]).reshape(1,-1)))
    points = np.loadtxt("Vertices.txt")
    edges = get_subsimplices(triangles)
    vertices = get_subsimplices(edges)
    b_matrix2 = boundary_matrix(triangles, edges, format="dok")
    b_matrix1 = boundary_matrix(edges, np.arange(0, len(points)).reshape(-1,1), format="dok")

    B = b_matrix2
    N = dok_matrix(B, dtype=int)
    U = dok_matrix(identity(N.shape[0], dtype=int), dtype=int)
    V = dok_matrix(identity(N.shape[1], dtype=int), dtype=int)
    # Smith Normal Form
    #SNF(0, uv=False)
    #sparse_savetxt("spineSNF2.txt", N, fmt="%d", include_dim=True)
    N = load_N("spineSNF2.txt")

    # Boundary and cycle ranks
    b2 = 0
    b1 = np.amax(N.nonzero()[1])+1
    z2 = N.shape[1] - b1

    B = b_matrix1
    N = dok_matrix(B, dtype=int)
    U = dok_matrix(identity(N.shape[0], dtype=int), dtype=int)
    V = dok_matrix(identity(N.shape[1], dtype=int), dtype=int)
    # Smith Normal Form
    #SNF(0, uv=False)
    #sparse_savetxt("spineSNF1.txt", N, fmt="%d", include_dim=True)
    N = load_N("spineSNF1.txt")
   
   # Boundary and cycle ranks
    b0 = np.amax(N.nonzero()[1])+1
    z1 = N.shape[1] - b0
    # There are 4 isolated points in K which should be ignored.
    z0 = N.shape[0] -4 #Number of vertices 
    print "=========================================================="
    print "Spine"
    print "=========================================================="
    print "Boundary matrix dimension(EDGESxTRIANGLES):%dx%d"%(b_matrix2.shape)
    print "Boundary matrix dimension(VERTICESxEDGES):%dx%d"%(b_matrix1.shape[0]-4, b_matrix1.shape[1])
    print r"Rank of B2: ", b2
    print r"Rank of Z2: ", z2
    print r"Rank of B1: ", b1
    print r"Rank of Z1: ", z1
    print r"Rank of B0: ", b0
    print r"Rank of Z0: ", z0
    print "Betti 0 = (z0-b0): ", z0 - b0
    print "Betti 1 = (z1-b1): ", z1-b1
    print "Betti 2 = (z2-b2):", z2-b2
    print "Betti i, i>2 is trivial since there is no tetrahedra in K"
    elapsed = time.time() - start
    print "Time(min) spent: %f" % (elapsed/60)
    ''''
    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=triangles, cmap=plt.cm.Spectral)
    plt.show()
    '''
def kleinbottledemo():
    '''
    Klein bottle simplices and boundary matrices
    '''
    triangles = np.array([[0,1,4],
                            [0,1,7],
                            [0,2,4],
                            [0,2,6],
                            [0,3,6],
                            [0,3,7],
                            [1,2,5],
                            [1,2,8],
                            [1,4,5],
                            [1,7,8],
                            [2,4,8],
                            [2,5,6],
                            [3,4,5],
                            [3,4,8],
                            [3,5,7],
                            [3,6,8],
                            [5,6,7],
                            [6,7,8]
                            ], dtype=int)
    edges = get_subsimplices(triangles)
    points = np.arange(0,9).reshape(-1,1)
    b_matrix2 = boundary_matrix(triangles, edges, format="dok")
    b_matrix1 = boundary_matrix(edges, points, format="dok")
    print "Triangles in K"
    print triangles
    print "Edges in K"
    print edges
    return b_matrix2, b_matrix1
def demo():
    '''
    # Boundary matrices for debugging
    b_matrix2 = np.matrix([[1,0],
                            [1,0],
                            [1,0],
                            [0,1],
                            [0,1],
                            [0,0],
                            [0,1]])
    b_matrix1 = np.matrix([[1,1,0,0,0,0,0],
                            [1,0,1,1,1,0,0],
                            [0,1,1,0,0,1,0],
                            [0,0,0,1,0,1,1],
                            [0,0,0,0,1,0,1]])
    '''
    print "=========================================================="
    print "Klein bottle"
    print "=========================================================="
    b_matrix2, b_matrix1 = kleinbottledemo()
    global N
    global U
    global V

    start = time.time()
    B = b_matrix2
    N = dok_matrix(B, dtype=int)
    U = dok_matrix(identity(N.shape[0], dtype=int), dtype=int)
    V = dok_matrix(identity(N.shape[1], dtype=int), dtype=int)
    #Smith Normal Form
    print "SNF running:"
    SNF(0, uv=True)
    sparse_savetxt("N2.txt", N, fmt="%d", include_dim=True)
    print "Boundary matrix, B2:\n", B.todense()
    print "SNF2:\n",N.todense()

    # Boundary and cycle ranks
    b2 = 0
    b1 = np.amax(N.nonzero()[1])+1
    z2 = N.shape[1] - b1

    B = b_matrix1
    N = dok_matrix(B, dtype=int)
    U = dok_matrix(identity(N.shape[0], dtype=int), dtype=int)
    V = dok_matrix(identity(N.shape[1], dtype=int), dtype=int)
    print "SNF running:"
    # Smith Normal Form
    SNF(0, uv=True)
    sparse_savetxt("N1.txt", N, fmt="%d", include_dim=True)
    print "Boundary matrix, B1:\n", B.todense()
    print "SNF1:\n",N.todense()
   
   # Boundary and cycle ranks
    b0 = np.amax(N.nonzero()[1])+1
    z1 = N.shape[1] - b0
    z0 = N.shape[0] #Number of vertices 
    print r"Rank of B1: ", b1
    print r"Rank of Z2: ", z2
    print r"Rank of B2: ", b2
    print r"Rank of B0: ", b0
    print r"Rank of Z1: ", z1
    print r"Rank of Z0: ", z0
    print "Betti 0 = (z0-b0): ", z0 - b0
    print "Betti 1 = (z1-b1): ", z1-b1
    print "Betti 2 = (z2-b2):", z2-b2
    print "Betti i, i>2 is trivial since there is no tetrahedra in K"

    '''
    print "U:\n", U.todense()
    print "U inverse:\n", inv(U).todense()
    print "V:\n",V.todense()
    print "U*B*V:"
    tmp = (U*dok_matrix(B, dtype=int)).asformat("dok").astype(int)
    for key, val in tmp.items():
        tmp[key[0], key[1]] = val%2
    tmp = (tmp*V).asformat("dok").astype(int)
    for key, val in tmp.items():
        tmp[key[0], key[1]] = val%2
    print tmp.todense()
    '''

    elapsed = time.time() - start
    print "Time(min) spent: %f" % (elapsed/60)
    
if __name__ == "__main__":
    demo()
    spinedemo();
