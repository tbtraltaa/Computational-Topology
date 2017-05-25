'''
Math574 - Computationa Topology - Spring 2016
---------------------------------------------

Homework4 on different types of complex construction.

*Tasks:*
1.Q4 - Generating Cech and VR complexes
a) Given randomly generated 10 points, when r=2, Cech is a strict subcomplex of VR.
    Cech complex, K 
    - Constructed a pairwise distance matrix, M = [V_{i,j}](10x10)
    - Turned M into upper triangular matrix with 0 entries below the diagonal
      which help to generate edges in lexicographical ordering also.
    - By filtering the distance matrix, constructed intersection matrix i_matrix
      where the corresponding entry is 1 if the dist(v_{i}, v_{j}) is less or equal to 2*r.
                                       0 if greater
    - Generated edges or 1-simplex of Cech by taking indices of nonzero entries in the intersection
      matrix.
    - For n-simplices wheren n>=2, used miniball recursive algorithm.
      Generated all possible combinations of the vertices in the complex
      and checked if the miniball containing the simplex has the radius less than or equal to the given.
      if true, include the n-simplex in n-skeleton of K.
    - Highes number of nonzero entries in intersection matrix row tells 
      the highest order of the complex, hence no need to check upto n-simplex
      where n = number of points, 10.
    VR complex, K:
    - the methods are same till 1-skeleton of K.
    - for n>=2, 
    -   while (n-1) is not empty:
            for all combinations of n-simplex,
            check if its all faces are in (n-1)-skeleton of VR complex K
            If true, add the n-simplex to (n+1)-skeleton of K
b) When r=2.5, VR had more than 2 tetrahedras.
2. Q5 - Generating Delaunay complex and filter it by Beta complex
a)  - Choose 10 points in a unit cube randomly since it illustrates the concepts better.
      Commented out the part where unform point generating. To use it uncomment it.
    - Used scipy.spatial.Delaugnay
    - Used matplotlib and distmesh for visualization
b) Similar filteration as used in VR construction.
    The difference is we already had n-skeletons of the Delaunay complex
    - Get upto 1-skeleton from pairwise distance matrix and intersection matrix
    - Iterate 2-simplices of Del and check if its faces are in Beta.
      if True, add it to Beta
    - Repeated till 3-skeleton, since we know the structure of Del
c) Generated Beta complex for different radiuses.
    - r = 1/2 - Beta is a strict subcomplex of Del
    - r = 1/sqrt(2) - Beta is a strict subcomplex of Del
    - r = sqrt(3)/2 - Beta is equal to Del
'''

import itertools as it

import numpy as np
from scipy.sparse import dok_matrix
from scipy.spatial import distance
from scipy.misc import comb
from scipy.spatial import Delaunay

from mpl_toolkits.mplot3d import Axes3D
from distmesh.plotting import axes_simpplot3d

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def is_included(B, u):
    if ((u[0]-B[0][0])**2 + (u[1]-B[0][1])**2) < B[1]**2:
        return True
    return False

def miniball(t, v, points):
    '''
    Refer to Computational Topology[Herbert]
    The method to choose minifall of a fewer number of points
    is not mentioned in the algorithm.
    I used the following method.
    For miniball of small number points(2,3):
        Choose mean a center, and 
        the max distance from to one of the points as a radius.
    '''
    B = None
    if len(t) == 0:
        if len(v) == 0:
            B = [[0,0], 0]
        if len(v) == 1:
            B = [points[v[0]], 0]
        if len(v) == 2:
            p1 = points[v[0]]
            p2 = points[v[1]]
            B = [(p1 + p2)*1.0/2, float(distance.cdist(p1.reshape(1,-1),p2.reshape(1,-1)))/2]
        if len(v) >= 3:
            tri_points = points[v]
            center = np.sum(tri_points, axis=0)/len(v)
            r = np.amax(distance.cdist(points[v].reshape(-1,2), center.reshape(1,2)))
            B = [center, r]
    else:
        u = 0
        B = miniball(np.delete(t,u, axis=0), v, points)
        if not is_included(B, points[u]):
            v.append(t[u])
            v = np.unique(v).tolist()
            t = np.delete(t, u, axis=0)
            B = miniball(t, v, points)
    return B

def Cech(points, r):
    '''
    Cech complex.
    points - an nd-array of points
    r - radius of the balls centered at the points
    - Constructed a pairwise distance matrix, M = [V_{i,j}](10x10)
    - Turned M into upper triangular matrix with 0 entries below the diagonal
      which help to generate edges in lexicographical ordering also.
    - By filtering the distance matrix, constructed intersection matrix i_matrix
      where the corresponding entry is 1 if the dist(v_{i}, v_{j}) is less or equal to 2*r.
                                       0 if greater
    - Generated edges or 1-simplex of Cech by taking indices of nonzero entries in the intersection
      matrix.
    - For n-simplices wheren n>=2, used miniball recursive algorithm.
      Generated all possible combinations of the vertices in the complex
      and checked if the miniball containing the simplex has the radius less than or equal to the given.
      if true, include the n-simplex in n-skeleton of K.
    - Highes number of nonzero entries in intersection matrix row tells 
      the highest order of the complex, hence no need to check upto n-simplex
      where n = number of points, 10.
    '''
    n = len(points)
    point_indices = np.arange(n)
    dist_matrix = distance.cdist(points, points)
    i_matrix = np.triu(np.array(dist_matrix <= 2*r, dtype=int), k=1)
    i_max = np.amax(np.sum(i_matrix, axis=1))
    print i_matrix
    edges = np.argwhere(i_matrix==1)
    K_skeleton = {idx: [] for idx in point_indices}
    K_skeleton[0] = point_indices
    K_skeleton[1] = edges
    for i in np.arange(2,i_max+1):
        for i_points in it.combinations(point_indices, i+1):
            B = miniball(list(i_points), list(), points)
            if B[1] <= r:
                K_skeleton[i].append(i_points)
    return K_skeleton

def VR(points, r):
    '''
    VR complex.
    points - an nd-array of points
    r - radius of the balls centered at the points
    VR complex, K:
    - the methods are same till 1-skeleton of K.
    - for n>=2, 
    -   while (n-1) is not empty:
            for all combinations of n-simplex,
            check if its all faces are in (n-1)-skeleton of VR complex K
            If true, add the n-simplex to (n+1)-skeleton of K
    '''
    n = len(points)
    point_indices = np.arange(n)
    #intersection_matrix = dok_matrix((n,n), dtype = int)
    dist_matrix = distance.cdist(points, points)
    i_matrix = np.triu(np.array(dist_matrix <= 2*r, dtype=int), k=1)
    print i_matrix
    edges = np.argwhere(i_matrix==1)
    K_skeleton = {idx: [] for idx in point_indices}
    K_skeleton[0] = point_indices
    K_skeleton[1] = edges
    for i in np.arange(2, n):
        if len(K_skeleton[i-1]) != 0:
            for i_points in it.combinations(point_indices, i+1):
                cnt = 0
                comb_cnt = 0
                for subsimplex in it.combinations(i_points, i):
                    comb_cnt +=1
                    if np.any(np.all(np.array(K_skeleton[i-1])==np.sort(subsimplex), axis=1)):
                        cnt +=1
                if cnt == comb_cnt:
                    K_skeleton[i].append(i_points)
        else:
            break
    return K_skeleton

def Beta(Del, r):
    '''
    b) Similar filteration as used in VR construction.
       The difference is we already had n-skeletons of the Delaunay complex
       - Get upto 1-skeleton from pairwise distance matrix and intersection matrix
       - Iterate 2-simplices of Del and check if its faces are in Beta.
         if True, add it to Beta
       - Repeated till 3-skeleton, since we know the structure of Del
    '''
    point_indices = np.arange(Del.points.shape[0])
    K_skeleton = {idx: [] for idx in np.arange(4)}
    K_skeleton[0] = point_indices
    tris = get_subsimplices(Del.simplices)
    edges = get_subsimplices(tris)
    dist = np.array([distance.cdist(Del.points[e[0]].reshape(1,3), Del.points[e[1]].reshape(1,3)) for e in edges]).reshape(-1,)
    b_edges = edges[np.where(dist<=2*r), :].reshape(-1,2)
    K_skeleton[1] = b_edges
    nsimp = {2:tris, 3:Del.simplices}
    for i in np.arange(2, 4):
        if len(K_skeleton[i-1]) != 0:
            for simplex in nsimp[i]:
                cnt = 0
                comb_cnt = 0
                for subsimplex in it.combinations(simplex, i):
                    comb_cnt +=1
                    if np.any(np.all(np.array(K_skeleton[i-1])==np.sort(subsimplex), axis=1)):
                        cnt +=1
                if cnt == comb_cnt:
                    K_skeleton[i].append(simplex)
    return K_skeleton

def plot_balls(points, r):
    '''
    Plotting 3D balls around points
    Not used.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r* np.outer(np.cos(u), np.sin(v))
    y = r* np.outer(np.sin(u), np.sin(v))
    z = r* np.outer(np.ones(np.size(u)), np.cos(v))
    for p in points :
        ax.plot_surface(p[0] + x, p[1] + y, p[2]+z, color="r", rstride=10, cstride=10)
    plt.show()

def plot_circles(points, r):
    '''
    Plotting disks around points given.
    '''
    patches = []
    ax = plt.gca()
    plt.scatter(points[:,0], points[:,1], s=10)
    for i, point in enumerate(points):
            ax.annotate(i, (point[0], point[1]+0.5), horizontalalignment='center', verticalalignment='top')
    for p in points :
        # add a circle
        circle = mpatches.Circle((p[0], p[1]), r, ec="none")
        patches.append(circle)

    colors = np.linspace(0, 1, len(patches))
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    plt.axis('equal')
    #plt.axis('off')
    #plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.title("Radius=%.1f"%r)
    plt.show()

def plotComplex(K, points, title=""):
    '''
    Given a skeleton, K of a complex and vertex coordinates, plots the complex
    '''
    if points.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])
        edges_in_tri = []
        tri_in_tetra = []
        if len(K[2]) != 0:
            edges_in_tri = get_subsimplices(np.array(K[2])).reshape(-1,2)
        for e in K[1]:
            if len(edges_in_tri) != 0:
                if not np.any(np.all(edges_in_tri==e, axis=1)):
                    edge = points[e]
                    ax.plot(edge[:,0], edge[:,1], edge[:,2], color="k")
            else:
                edge = points[e]
                ax.plot(edge[:,0], edge[:,1], edge[:,2])
        if len(K[3]) != 0:
            tri_in_tetra = get_subsimplices(K[3]).reshape(-1,3)
            axes_simpplot3d(ax, points, np.array(K[3]))
        for t in K[2]:
            if len(tri_in_tetra) != 0:
                if not np.any(np.all(tri_in_tetra==t, axis=1)):
                    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=[t])
                #else:
                    #ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=[t], color="m")
            else:
                ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=K[2])

        plt.axis('equal')
        plt.title(title)
        plt.show()

    if points.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(points[:,0], points[:,1])
        for i, point in enumerate(points):
                ax.annotate(i, (point[0], point[1]+0.5), horizontalalignment='center', verticalalignment='top')
        edges_in_tri = get_subsimplices(np.array(K[2]).reshape(-1,2))
        for e in K[1]:
            if not np.any(np.all(edges_in_tri==e, axis=1)):
                edge = points[e]
                ax.plot(edge[:,0], edge[:,1])
        if len(K[2]) !=0:
            plt.triplot(points[:,0], points[:,1], K[2])
        plt.axis('equal')
        plt.title(title)
        plt.show()

def get_subsimplices(simplices):
    simplices = np.sort(simplices, axis=1)
    subsimplices = set()
    for j in np.arange(simplices.shape[1]):
        idx = list(range(simplices.shape[1]))
        idx.pop(j)
        subsimplices = subsimplices.union(set(tuple(sub) for sub in simplices.take(idx, axis=1)))
    subsimplices = np.array([ sub for sub in subsimplices], dtype=int)
    return subsimplices

def Q4():
    '''
    Q4 - Generating Cech and VR complexes
    a) Given randomly generated 10 points, when r=2, Cech is a strict subcomplex of VR.
        Cech complex, K 
        - Constructed a pairwise distance matrix, M = [V_{i,j}](10x10)
        - Turned M into upper triangular matrix with 0 entries below the diagonal
          which help to generate edges in lexicographical ordering also.
        - By filtering the distance matrix, constructed intersection matrix i_matrix
          where the corresponding entry is 1 if the dist(v_{i}, v_{j}) is less or equal to 2*r.
                                           0 if greater
        - Generated edges or 1-simplex of Cech by taking indices of nonzero entries in the intersection
          matrix.
        - For n-simplices wheren n>=2, used miniball recursive algorithm.
          Generated all possible combinations of the vertices in the complex
          and checked if the miniball containing the simplex has the radius less than or equal to the given.
          if true, include the n-simplex in n-skeleton of K.
        - Highes number of nonzero entries in intersection matrix row tells 
          the highest order of the complex, hence no need to check upto n-simplex
          where n = number of points, 10.
        VR complex, K:
        - the methods are same till 1-skeleton of K.
        - for n>=2, 
        -   while (n-1) is not empty:
                for all combinations of n-simplex,
                check if its all faces are in (n-1)-skeleton of VR complex K
                If true, add the n-simplex to (n+1)-skeleton of K
    b) When r=2.5, VR had more than 2 tetrahedras.
    '''
    #points = np.random.randint(0,10, size=(10,2))
    #np.savetxt("points.txt", points)
    #np.savetxt("points3d.txt", points)
    #points = np.random.randint(0,10, size=(10,3))
    points = np.loadtxt("points.txt")
    points[1] = [2, 1+np.sqrt(12)]
    points[3] = [4,1]
    points[7] = [6, 1+np.sqrt(12)]
    r = 2.5
    plot_circles(points, r)
    K1 = Cech(points, r)
    K2 = VR(points, r)
    
    print K1
    print K2
    title = "Cech Complex - K1 - Radius=%.1f (Magenta- included in tetrahedra)"%r
    plotComplex(K1, np.hstack((points, np.arange(1,points.shape[0]+1).reshape(-1,1))), title)
    title = "VR Complex - K2 - Radius=%.1f (Magenta - included in tetrahedra)"%r
    plotComplex(K2, np.hstack((points, np.arange(1,points.shape[0]+1).reshape(-1,1))), title)
    
def Q5():
    '''
    Tasks:
    a) To tetrahedralize a unit cube using corners and 10 other interior points and plot the result
    b) Defining Beta complex
    c) Generate Beta complex with r = 1/2, 1/sqrt(2), sqrt(3)/2
    Generating Delaunay complex and filter it by Beta complex
    a)  - Choose 10 points in a unit cube randomly since it illustrates the concepts better.
          Commented out the part where unform point generating. To use it uncomment it.
        - Used scipy.spatial.Delaugnay
        - Used matplotlib and distmesh for visualization
    b) Similar filteration as used in VR construction.
        The difference is we already had n-skeletons of the Delaunay complex
        - Get upto 1-skeleton from pairwise distance matrix and intersection matrix
        - Iterate 2-simplices of Del and check if its faces are in Beta.
          if True, add it to Beta
        - Repeated till 3-skeleton, since we know the structure of Del
    c) Generated Beta complex for different radiuses.
        - r = 1/2 - Beta is a strict subcomplex of Del
        - r = 1/sqrt(2) - Beta is a strict subcomplex of Del
        - r = sqrt(3)/2 - Beta is equal to Del
    '''
    #a)
    corners = [[0,0,0], [0,0,1], [0,1,0], [1,0,0], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
    #interiors = np.random.random(size=(10,3)) 
    #np.savetxt("interiors.txt", interiors)
    interiors = np.loadtxt("interiors.txt")

    # Remove the comment to generate uniformly scattered points
    '''
    bbox = [0,0,0,1,1,1]
    interiors = np.mgrid[tuple(slice(min+0.2, max, 0.6) for min, max in np.array(bbox).reshape(2,-1).T)]
    interiors = interiors.reshape(3, -1).T
    interiors = np.vstack((interiors, [0.5,0.5,0.5]))
    '''
    print "Interior points:"
    print interiors
    points = np.vstack((np.array(corners).reshape(-1,3),interiors))
    ax = plt.gca(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.title("Points")
    plt.show()
    Del  = Delaunay(points)
    ax = plt.gca(projection='3d')
    axes_simpplot3d(ax, Del.points, Del.simplices)
    #tris = get_subsimplices(Del.simplices)
    #ax.plot_trisurf(Del.points[:,0], Del.points[:,1], Del.points[:,2], triangles=tris, color="y")
    plt.title("Delaunay tetrahedralization")
    plt.show()
    #b,c)
    rs = [1.0/2, 1.0/np.sqrt(2), np.sqrt(3)/2]
    for r in rs:
        beta = Beta(Del, r)
        title = "Beta Complex-Radius=%.1f (Light blue- tetrahedra,\ndark blue - triangles not in tetrahedra)"%r
        #print title
        #print beta
        plotComplex(beta, Del.points, title)
    
if __name__ == "__main__":
    Q4()
    Q5()
