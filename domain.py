# Python File Containing Useful Classes for my Implementation of the Discontinuous Galerkin Method
from numpy import argmin, argmax, setdiff1d, array, append, reshape

class Mesh: # Mesh class, an arrangement of cells
    def __init__(self, dim, cells):
        # cells: a list of cell objects
        
        self.dim = dim
        self.cells = cells
        self.retrieve_x() # centers of each cell
        # self.boundary() # array of cells on the boundary
        # self.interior() # array of cells on the interior
    
    def retrieve_x(self): # returns array of cell centers
        x = array([])
        for cell in self.cells:
            x = append(x,cell.x)
        self.x = reshape(x, (len(self.cells),-1))
    
    def retrieve_u(self): # returns array of solution values for each cell
        u = array([])
        for cell in self.cells:
            u = append(u,cell.u)
        self.u = reshape(u, (len(self.cells),-1))
        
    def boundary(self): # Returns an array of the cells that are on the boundary of the mesh
        boundary = array([])
        if self.dim == 1:
            boundary = np.array([self.cells[argmin(self.x)],self.cells[argmax(self.x)]])
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
        
    def set_flux(self, f): # sets each cells' flux to f
        for cell in self.cells:
            cell.set_flux(f)
    
    def set_numflux(self, numf): # sets each cells' numflux to numf
        for cell in self.cells:
            cell.set_numflux(numf)


class Cell1d: # 1D Cell class, containing spatial parameters, solution value, fluxes, and dudt values
    def __init__(self, x, h):
        # x: the cell center
        # h: the cell diameter
        # f: flux function for PDE associated 
        
        self.h = h
        self.x = x
        self.e = [x-h/2, x+h/2] # cell edges/boundary points
        
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