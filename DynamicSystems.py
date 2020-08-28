import numpy as np
from scipy.integrate import solve_ivp
import autograd
from autograd.builtins import tuple
import autograd.numpy as np
import matplotlib.pyplot as plt

class SystemDynamics:
"""The class System_Dynamics contains the attributes necessary to realise and compute
the evolution of an ODE dynamic system.

Parameters
----------
    Private:
        name : {'String'} default 'System'
            System identifier
        t0 : float
            System initial time.
        tf : float
            System final time.
        d_t : float
            Systems time between points.
        f_x_y : callable
            Right hand side of the equation. It should be
            of the form:
            
                dy/dt = f(t,x,y)
                
        t : ndarray of shape (n_points)
            Time points array. The array will be empty
            at first, only when the system has been evolved,
            the time array will appeared complete.
        Y : ndarray of shape (n_points)
            Solution array. The array will be empty
            at first, only when the system has been evolved,
            the time array will appeared complete.
        

Attributes:
-----------

    X_var :
        (n-dim array): Storage array for the independent variable.
    Y_var :
        (n-dim array): Storage array for the dependent variable.
    t_var :
        (1-dim array): a 1-dimensional array for storing a time dependent variable
    name  :
        (String): Systems Name
    
    
Methods:
    
    evolve_sys():



"""
def __init__(self, name = "System"):
    
    self.__name = name
    
        
def getname(self):
    '''getname returns the system dynamics name for identification purposes'''
    return self.__name

def gety(self):
    '''gety returns the system dynamics results'''
    return self.__Y

def gett(self):
    '''getname returns the system dynamics times'''
    return self.__t

def getf_x_y(self):
    '''getname returns the system dynamics times'''
    return self.__f_x_y

    
    
def system_init(self, t0, tf, d_t, f_x_y):
    '''The system_init method instanciates the dynamic system properties.
        The system initialiser expects a system with the following structure.
            
                    dy_t = f(x,t,... )
    
        
        system_init(t0, tf, d_t, f_x_y)
        
        Parameters
        ----------
        
        t0 : float
            System initial time.
        tf : float
            System final time.
        d_t : float
            Systems time between points.
        f_x_y : callable
            Right hand side of the equation. It should be
            in the form:
            
                dy/dt = f(t,x,y)
                 
        t : ndarray of shape (n_points)
            Time points array. The array will be empty
            at first, only when the system has been evolved,
            the time array will appeared complete.
        Y : ndarray of shape (n_points)
            Solution array. The array will be empty
            at first, only when the system has been evolved,
            the time array will appeared complete.
        
                
        '''
    self.__t0 = t0
    self.__tf = tf
    self.__d_t = d_t
    self.__f_x_y = f_x_y
    self.__t = []
    self.__Y = []
    
    
def system_properties(self):
    '''The system_properties method prints the information regarding the parameters contained by the system
        
        It prints the data related to the initial time, t0; final time, tf; time delta, d_t; name and other
        parameters.
        '''
    print('- System\'s Name: {}'.format(self.__name))
    print('\t- Initial time: {}s'.format(self.__t0))
    print('\t- Final time: {}s'.format(self.__tf))
    print('\t- Time step: {}s'.format(self.__d_t))
    
    if not self.__Y:
        print('Ready to evolve...')
    else:
        print('\t- Time data points: {}'.format(len(self.__t)))
        print('\t- Y data points: {} Y vars{}'.format(*self.__Y.shape))
    
    

def system_evolve(self, Y0, method = 'RK45', params = None, system = None):
    '''The system_evolve method computes the time evolution of the define ordinary differential system.
        The method receives the inital conditions and integrates the solution with a defined numerical method.
        Leading to a form:
            
                    dy/dt = F(x,t,params)
    
        
        system_evolve(Y0, method = 'RK45', params = None, system = None)
        
        Parameters
        ----------
        
            Y0 : ndarray of shape (n,)
                Initial conditions for the system.
            params : ndarray of size (n,)
                Extra arguments the system might need. eg: constants, parameters.
            method : {'RK45', 'RK23', 'DOP853', 'BDF', 'LSODA'} auto : 'RK45'
                Numeric approximation method. Other solver methods should be
                available. Refer to the scipy.integrate library.
            system : callable, auto : None
                Replacement system object. Additional function to be time-evolved.
                Used for expanding solutions or state space dynamic systems.


        '''
    
    
    points = int(np.floor((self.__tf-self.__t0)/self.__d_t))
    
    self.__t = np.linspace(self.__t0, self.__tf, points)
    
    t_span = (self.__t0, self.__tf)
    
    
    if system is not None:
        
        evolved_system = system
    else:
        evolved_system = self.__f_x_y
        
    
    
    print("Time evolving using parameters = {}".format(params))
    
    
    """Time evolution solve_ivp syntax should include a tuple([params]) to generate a tuple of an array and
        empty so it can be unpacked by the solve_ivp function as a single array and interpreted as a single
        list to the user defined function.
        
            params = np.array([1,2,3])
        
            tuple([params]) = (array([1,2, ...]),)
        
        
        otherwise, multiple values will be sent, instead on only one. List should be unpacked at the user
        defined function."""
    time_evolution = solve_ivp(evolved_system,
                               t_span= t_span,
                               y0 = Y0,
                               method = method,
                               t_eval= self.__t,
                               args= tuple([params]))
    
    self.__Y = time_evolution.y
    
    return time_evolution
    
    
def plot_dynamics(self, labels, colours, xlab = 'time [s]', ylab = 'dyn [adm]'):
    '''The plot_dynamics method plots the system dynamics of the predefined parameters
    within the SystemDynamics object. If the systen has not been evolved, then the
    function return an empty array, otherwise, a plot will be displayed.
            
    
        
        plot_dynamics(self, labels, colours)
        
        Parameters
        ----------
        
            labels : ndarray of shape (n,)
                Label strings. Contains the names of the plotted curves in
                descending order.
            colours : ndarray of shape (n,)
                Hex colour codes. Contains the colour codes for the curves
                in descending order
        '''
    
    
    if not self.__Y.any():
        print("Nothing to print")
        return []

    else:
        
        t = self.__t
        dyn = self.__Y

        for idx, var in enumerate(labels):
            plt.plot(t,dyn[idx,:], label= var, color= colours[idx], linewidth= 5)


        plt.legend()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        
    
    
    
