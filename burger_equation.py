import numpy as np 
import matplotlib.pyplot as plt 

def initialBell(x):
    return np.where(x%1. < 0.75, np.power(np.sin(2*x*np.pi), 2), 0)
def conservative_form():
    print('conservative')
    
def main():
    ### solver of burguers equation, forward in time backward in space
    u0=10
    nt=1000
    nx=100
    x=np.linspace(0,2*np.pi,nx+1)
    alternative_shape=np.exp(-1*(x-np.pi)**2)# - (y-np.pi)^2)
    u=1-np.cos(x)#power(np.sin(2*(x)*np.pi),2)#uold[0]-0.5*c*uold[j]*(uold[1]-uold[nx-1])
    c=0.2
    u=alternative_shape
    unew=u.copy()
    uold=u.copy()
    for nt in range(1,nt):
        for j in range(1,nx):
            unew[j]=u[j]-c*u[j]*(u[j]-u[j-1])   
        unew[0]=uold[0]-c*u[0]*(u[0]-u[nx-1])
        unew[nx]=unew[0]
        uold=u.copy()
        u=unew.copy()
        un=1.
        dx=2*np.pi/nx
        dt=c*dx/un
        t=nt*dt
        plt.plot(x,u,'b',label=r'$t=$'+str(nt))
        plt.ylabel('$u$',fontsize=14)
        plt.ylim([0,1])
        plt.axhline(0,linestyle='--',color='black')
        plt.legend(fancybox=True,fontsize=15)
        plt.savefig('sine'+str(nt)+'.png')
        plt.close()
    print(u)
def main_2d():
    ### solver of burguers equation, forward in time backward in space
    u0=10
    nt=290
    nx=100
    x=np.linspace(0,2*np.pi,nx+1)
    y=np.linspace(0,2*np.pi,nx+1)
    x,y=np.meshgrid(x,y)
    alternative_shape=np.exp(-1*(x-np.pi)**2 - (y-np.pi)**2)
    u=1-np.cos(x)#power(np.sin(2*(x)*np.pi),2)#uold[0]-0.5*c*uold[j]*(uold[1]-uold[nx-1])
    c=0.2
    u=alternative_shape
    print(u.shape)
    unew=u.copy()
    uold=u.copy()
    for nt in range(1,nt):
        for j in range(1,nx):
            for k in range(1,nx):
                unew[j,k]=uold[j,k]-c*u[j,k]*(u[j,k]-u[j-1,k-1])   
        unew[0,0]=uold[0,0]-c*u[0,0]*(u[0,0]-u[nx-1,nx-1])
        unew[nx,nx]=unew[0,0]
        uold=u.copy()
        u=unew.copy()
        un=1.
        dx=2*np.pi/nx
        dt=c*dx/un
        t=nt*dt
        plt.plot(x,u,'b',label=r'$t=$'+str(nt))
        plt.ylabel('$u$',fontsize=14)
        plt.ylim([0,1])
        plt.axhline(0,linestyle='--',color='black')
        plt.legend(fancybox=True,fontsize=15)
        plt.savefig('sine'+str(nt)+'.png')
        plt.close()
    print(u)

# Put everything inside a main function to avoid global variables
def main2():
    # Setup space, initial phi profile and Courant number
    nx = 40                 # number of points in space
    c = 0.2                 # The Courant number
    # Spatial variable going from zero to one inclusive
    x = np.linspace(0.0, 1.0, nx+1)
    # Three time levels of the dependent variable, phi
    phi = initialBell(x)
    phiNew = phi.copy()
    phiOld = phi.copy()

    # FTCS for the first time-step, looping over space
    for j in range(1,nx):
        phi[j] = phiOld[j] - 0.5*c*(phiOld[j+1] - phiOld[j-1])
    # apply periodic boundary conditions
    phi[0] = phiOld[0] - 0.5*c*(phiOld[1] - phiOld[nx-1])
    phi[nx] = phi[0]
    plt.plot(x,phi)
    plt.savefig('phi.png')
    # Loop over remaining time-steps (nt) using CTCS
    nt = 40
    for n in range(1,nt):
        # loop over space
        for j in range(1,nx):
            phiNew[j] = phiOld[j] - c*(phi[j+1] - phi[j-1])
        # apply periodic boundary conditions
        phiNew[0] = phiOld[0] - c*(phi[1] - phi[nx-1])
        phiNew[nx] = phiNew[0]
        #update phi for the next time-step
        phi = phiNew.copy()
        phiOld = phi.copy()
        u = 1.
        dx = 1./nx
        dt = c*dx/u
        t = nt*dt
        plt.plot(x, initialBell(x - u*t), 'k', label='analytic')
        plt.plot(x, phi, 'b', label='CTCS')
        plt.legend(loc='best')
        plt.ylabel('$\phi$')
        plt.axhline(0, linestyle=':', color='black')
        plt.savefig(str(n)+'.png')


    # derived quantities


    # Plot the solution in comparison to the analytic solution

main()
