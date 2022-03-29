# Python File Containing Useful Classes for my Implementation of the Discontinuous Galerkin Method
from numpy import argmin, argmax, setdiff1d

class Mesh: # Mesh class, an arrangement of cells
    def __init__(self, dim, cells):
        # cells: a list of cell objects
        
        self.dim = dim
        self.cells = cells
        self.retrieve_x() # centers of each cell
        self.retrieve_u() # solution values of each cell
        self.boundary() # array of cells on the boundary
        self.interior() # array of cells on the interior
    
    def retrieve_x(self): # returns array of cell centers
        x = []
        for cell in self.cells:
            x.append(cell.x)
        self.x = x
    
    def retrieve_u(self): # returns array of solution values for each cell
        u = []
        for cell in self.cells:
            u.append(cell.u)
        self.u = u
        
    def boundary(self): # Returns an array of the cells that are on the boundary of the mesh
        boundary = []
        if self.dim == 1:
            boundary.append(self.cells[argmin(self.x)])
            boundary.append(self.cells[argmax(self.x)])
        self.boundary = boundary
    
    def interior(self): # Returns an array of the cells that are on the interior of the mesh
        self.interior = setdiff1d(self.cells, self.boundary, assume_unique=True)
        
    def impose_IC(self, IC): # sets the u values for each cell in the mesh according to IC
        for cell in self.cells:
            cell.u = IC(cell.x)
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


class Cell1d: # 1D Cell class, containing spatial parameters, solution value, fluxes, and dudt values
    def __init__(self, x, h, f):
        # x: the cell center
        # h: the cell diameter
        # f: flux function for PDE associated 
        
        self.h = h
        self.x = x
        self.e = [x-h/2, x+h/2] # cell edges/boundary points
        
        self.u = 0 # Solution value at t0
        
        self.f = f # flux function
        
        self.L0 = 0 # dudt value at t0
        self.L1 = 0 # dudt value at t1
        
    def flux(self):
        return self.f(self.x)
    
    def numflux(self, nbr, J): # Numerical flux function
        # nbr: a neighbouring cell
        
        alpha = max(abs(J(self.u)),abs(J(nbr.u))) # maximum wave speed
        return 0.5*(self.f(nbr.u)+self.f(self.u)-alpha*(nbr.u-self.u)) # Roe-split flux