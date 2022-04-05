### Some Important classes for the MHD simulations that aren't nice to look at
import numpy as np

# mesh class, in standard PET format


class Mesh:
    def __init__(self, xtuple, Nx):
        self.Nx = Nx
        x = np.linspace(xtuple[0],xtuple[1],Nx+1)
        
        self.P = x
        I = np.zeros((2,Nx), dtype=int)
        x0 = np.zeros((Nx,1))
        for j in range(Nx):
            I[:,j] = (j,j+1)
            x0[j] = np.sum(x[np.ravel(I[:,j])])/2
        self.x0 = x0
        
        
        self.T = I
        self.NT = len(self.T.T)
        spans = np.zeros((self.NT,1))
        for j in range(self.NT):
            spans[j] = x[I[1,j]]-x[I[0,j]]
        self.spans = spans
        
        
        self.sln = Solution(self)


class Solution:
    def __init__(self, mesh):
        # the mesh that the solution is associated with
        self.msh = mesh
        self.F1 = np.zeros((self.msh.NT,8))
        self.F0 = self.F1
    
    def set_u(self, u0):
        self.U = np.array([u0(x) for x in self.msh.x0])
        
    def impose_BC(self, BCpnts, BCvals):
        self.U[BCpnts] = BCvals
    
    def get_prim(self, gamma):
        self.W = np.array([prim(row, gamma) for row in self.U])


# Convert from Primitive to Conserved Variables
def consv(W, gamma=5/3):
    E = W[7]/(gamma-1) + 0.5*W[0]*(W[1]**2+W[2]**2+W[3]**2) + 0.5*(W[4]**2+W[5]**2+W[6]**2)
    return np.array([W[0], W[0]*W[1], W[0]*W[2], W[0]*W[3], W[4], W[5], W[6], E])

def prim(U,gamma=5/3): # returns the primitive variable W, corresponding to the conserved variable U
    return np.array([U[0],U[1]/U[0],U[2]/U[0],U[3]/U[0],U[4],U[5],U[6], p(U,gamma) ])

# Pressure from Conserved Variables
def p(U, gamma=5/3):
    return (gamma-1)*(U[7] - 0.5*(U[1]**2+U[2]**2+U[3]**2)/U[0] - 0.5*(U[4]**2+U[5]**2+U[6]**2))

 # Fast Magneto-acoustic wave speed
def cf(U, gamma=5/3):
    alpha = (gamma*p(U, gamma)+(U[4]**2+U[5]**2+U[6]**2))/U[0]
    return np.sqrt(alpha/2 + np.sqrt( alpha**2 - 4*gamma*p(U, gamma)*U[4]**2/(U[0]**2) ) )

 # Single Second order Adams-Bashforth step
def ab2(U,F1,F0,dt):
    return U + 0.5*dt*(3*F1-F0)

 # Single Forward Euler step
def fe(U,F0,dt):
    return U + dt*F0