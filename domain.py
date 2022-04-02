# Python File Containing Useful Classes for my Implementation of the Discontinuous Galerkin Method
from numpy import argmin, argmax, setdiff1d, array, append, reshape, cross, allclose, arctan2, pi

from numpy.linalg import norm

class Mesh: # Mesh class, an arrangement of cells
    def __init__(self, dim, cells):
        # cells: a list of cell objects
        
        self.dim = dim
        self.cells = cells
        self.retrieve_x() # centers of each cell
        self.compute_adjacent_cells() # gives cells a list of adjacent cells in the mesh
        self.boundary() # array of cells on the boundary
        self.interior() # array of cells on the interior
    
    def retrieve_x(self): # returns array of cell centers
        x = array([])
        for cell in self.cells:
            x = append(x,cell.x0)
        self.x = reshape(x, (len(self.cells),-1))
    
    def retrieve_u(self): # returns array of solution values for each cell
        u = array([])
        for cell in self.cells:
            u = append(u,cell.u)
        self.u = reshape(u, (len(self.cells),-1))
        
    def boundary(self): # Returns an array of the cells that are on the boundary of the mesh
        boundary = array([])
        if self.dim == 1:
            boundary = array([self.cells[argmin(self.x)],self.cells[argmax(self.x)]])
        elif self.dim == 2:
            for cell in self.cells:
                if len(cell.nbrs) < len(cell.edges):
                    boundary = append(boundary, cell)
        self.boundary = boundary
    
    def interior(self): # Returns an array of the cells that are on the interior of the mesh
        self.interior = setdiff1d(self.cells, self.boundary, assume_unique=True)
        
    def impose_IC(self, IC): # sets the u values for each cell in the mesh according to IC
        for cell in self.cells:
            cell.u = IC(cell.x0)
        self.retrieve_u()
    
    def impose_periodic_BC(self): # sets the ghost cell values which impose the BC
        if self.dim == 1:
            self.boundary[-1].u = self.interior[0].u
            self.boundary[0].u = self.interior[-1].u
        self.retrieve_u()
        
    def impose_noflux_BC(self):
        if self.dim == 1:
            self.boundary[0].u = self.interior[0].u
            self.boundary[-1].u = self.interior[-1].u
        self.retrieve_u()
            
    def compute_adjacent_cells(self):
        for cell in self.cells:
            for nbr in self.cells:
                shared_points = []
                for X in cell.X:
                    for nbrX in nbr.X:
                        if allclose(X,nbrX):
                            shared_points.append(X)
                    
                if len( shared_points ) == 2 and nbr != cell and self.dim == 2:
                    edge_idx = [ [tuple(xi) for xi in cell.X].index(tuple(X)) for X in shared_points]
                    edge_idx.sort()
                    if edge_idx[1] == len(cell.X)-1 and edge_idx[0] == 0:
                        edge = cell.X[edge_idx[0]]-cell.X[edge_idx[1]]
                    else:
                        edge = cell.X[edge_idx[1]]-cell.X[edge_idx[0]]
                    edge_normal = cross(append(edge,0.),array([0.,0.,1.]))
                    edge_normal = edge_normal / norm(edge_normal)
                    cell.nbrs.append(nbr)
                    cell.nbrnormal.append(edge_normal)
                if len( shared_points) == 1 and nbr != cell and self.dim == 1:
                    cell.nbrs.append(nbr)


class Cell1d: # 1D Cell class, containing spatial parameters, solution value, fluxes, and dudt values
    def __init__(self, x, h):
        # x: the cell center
        # h: the cell diameter
        # f: flux function for PDE associated 
        
        self.h = h
        self.x0 = x
        self.X = [[x-h/2], [x+h/2]] # cell edges/boundary points
        
        self.nbrs = []
        
    def set_flux(self, f): # sets the flux function associated with the Cell
        # f: flux function for associated PDE
        self.flux = f
        
    def flux(self, u): # evaluates the flux function
        return self.flux(u)
        
    def set_numflux(self, numf): # sets the numerical flux function associated with the cell
        # numf: numerical flux function
        self.numflux = numf
        
    def numflux(self, nbr, u): # evaluates the numflux function
        return self.numflux(self, nbr, u)
    
class Cell2d:
    def __init__(self, x0, X):
        # x0: cell center
        # X: list of cell vertices in anti-clockwise order
        self.x0 = x0
        self.X = X # 3 or 4 length
        if len(self.X) == 3:
            self.edges = array([self.X[1]-self.X[0], self.X[2]-self.X[1], self.X[0]-self.X[2]])
            self.size = cross(self.edges[0], -self.edges[2])/2
        elif len(self.X) == 4:
            self.edges = array([self.X[1]-self.X[0], self.X[2]-self.X[1], self.X[3]-self.X[2], self.X[0]-self.X[3]])
            self.size = cross(self.edges[0], self.edges[1])
        self.edgelen = norm(self.edges, axis=1)
        
        self.nbrs = []
        self.nbrnormal = []
        