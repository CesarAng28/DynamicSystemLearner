import numpy as np
from scipy.integrate import solve_ivp
import autograd
from autograd.builtins import tuple
import autograd.numpy as np
import matplotlib.pyplot as plt
import DynamicSystems

class DynamicSystemLearner(SystemDynamics):
   
   
   def __init__(self, name = 'Learned System'):
       super().__init__(name)
       
       
   def system_init(self, t0, tf, d_t, f_x_y = None):
       
       '''Superclass SystemDynamics initialisation
       
           system_init(self, t0, tf, d_t, f_x_y)
           
           @Params
           
            t0 = (float) Inital time of the observation dynamics
            tf = (float) Final time of the observed dynamics
            d_t = (float) time step resolution for the system to be observed
            f_x_y = (n-dim array) = function array referent to f(x,t,... )
           
           
       
       '''
       super().system_init(t0, tf, d_t, f_x_y)
       
   
   def learner_init(self, n_vars, n_params, cost_fun = None):
         
       '''DynamicSystemLearner initialisation
       
           learner_init(n_vars, n_params, cost_fun)
           
           @Params
           
            n_vars = (float) Number of variables of the System
            n_params = (float) Number of parameters to be learnt
            cost_fun = (numpy-function) cost function to be minimised
           
       
       '''
       
       self.__n_vars = n_vars
       self.__n_params = n_params
       self.__cost_fun = cost_fun
       self.__theta = []
       self.__gradY_t = n_vars
       self.__gradY_theta = n_vars*n_params
       self.__Y = super().gety()
       self.__t = super().gett()
       self.__f_x_y = super().getf_x_y()
       self.__err = np.inf
       

   def cost_nnorm(self, y_obs, n_order= 2):
       def cost(Y):
           "Squared error loss"
           
           n = y_obs[0].shape

           err = (np.linalg.norm(y_obs-Y, n_order, axis=0)/n).sum()
           print('{}-norm total error: {}'.format(n_order, err))
           print((np.linalg.norm(y_obs-Y, n_order, axis=0)/n))
           if err < self.__err:
               self.__err = err
           return err
       return cost


   def gradient_setter(self, fun, argnum):
                 
       '''The gradient_setter function defines the computational graph of
       a function, fun, with respect to the argument, argnum, defined on
       definition. The function, fun, must contain valid operations defined
       on the autograd API.
       
       The mathematical formula of the computational graph, corresponds to the
       symbolic gradient of the function, fun(x, t, ... , y_n) with respect to
       the argument enumerated from 0 to M-1.
       
           gradient : d fun(t,x0,..., xi , ... y_n ) / d xi
           
       Parameters
       ----------
           
            fun : callable
                Function for which the gradient will be evaluated. The function
                operations should be compatible with autograd API graphs.
            argnum : int
                Index of the parameter with respect to fun will be
                differenciated.

       
       '''
       return autograd.jacobian(fun, argnum)
   
   def jacobian_gen(self, fun, param_idx = [1, 2]):
                   
       '''The jacobian_gen function defines the computational graph of
       a function, fun, with respect to the arguments, param_idx, defined on
       definition to return a jacobian functions array.
       
       The function, fun, must contain valid operations defined
       on the autograd API.
       
       
            jacobian_gen(self, fun, param_idx = [1, 2])
           
       Parameters
       ----------
           
            fun : callable
                 Function for which the jacobian will be computed. The operations should be defined by the
                 autograd API for the correct generation of the mathematica computation graph.
            param_idx : array_like, shape (n,), optional
                 parameters with respect to fun will be differenciated
                 default = 0 -> wrt Y and 2 -> wrt parameters

       
       '''
       
       grad_array = [ autograd.jacobian(fun, argnum=n) for n in param_idx ]
       
       self.J_fs = grad_array
       return grad_array
       
   def sensitivity_expansion(self, t, Y, theta):
       '''The sensitivity_expansion function defines the computational graph of
       a function, fun, with respect to the arguments, param_idx, defined on
       definition to return a jacobian functions array.
       
       The function, fun, must contain valid operations defined
       on the autograd API.
       
            sensitivity_expansion(t, Y, theta)
           
           Parameters
           ----------
           

            t : float
                New state time value
            Y : array_like, shape (n,)
                Previous state array of expanding function
            theta : array_like, shape (n,)
                Differential equations parameters array_like data structure

       
       '''

       # Y has 5 columns
       # Y[:0], Y[:1], Y[:2] are the ODEs SIR
       # Y[:3], Y[:4] Sensitivities wrt dS_a, dS_b .
       # Y[:5], Y[:6] Sensitivities wrt dI_a, dI_b .
       

       
       """ Evaluating differential equation """
       dy_dt = self.__f_x_y(t, Y[:self.__n_vars], theta)
 
       """ Calculating the gradient of the function wrt Ys through self.J_fs[0]"""
       df_y = self.J_fs[0](t, Y[:self.__n_vars], theta)
       
       '''
       ##### df_y ######
       # Y[-4:].reshape(-1,2) into tensor [[ds_a ds_b] [di_a di_b]]
       # Gradient array
       
       # Jacobian of F wrt Y. Y[:self.__n_vars] number of Y variables
       
       
                           Jf_y = [[dF1/dy1 dF1/dy2 .. ],
                                   [dF2/dy1 dF2/dy2 .. ]
                                 ...]
                                   
       
       '''

       dy_theta = Y[-self.__gradY_theta:].reshape(-1,self.__n_params)
         
       '''
       ##### dy_th ######
           Jacobian of F wrt th. Y[:self.__n_vars] number of Y variables
           
       
                           Jf_th = [[dY1/dth1 dY1/dth2 .. ],
                                    [dY2/dth1 dY2/dth2 .. ]
                                 ...]
                                   
       
       '''

       """ Calculating the gradient of the function wrt the parameters through self.J_fs[1]"""
       df_theta = self.J_fs[1](t, Y[:self.__n_vars], theta)
       

  
       grady_theta = df_y@dy_theta + df_theta
       

       ode = np.append(dy_dt, grady_theta.reshape(-1,self.__gradY_theta))

       return ode

   
   

   def train(self, x_train, y_train, I0, learning_rate, params, cost_n= 2, max_iter= 100,
                tol= 0.1, obs_idx= [0,1]):
        
       '''The train(x_train, y_train, I0, learning_rate, params, cost_n= 2, max_iter= 100, tol= 0.1, obs_idx= (1,2))
           method fits the user defined system dynamics F(x,y) durnig the class initialisation, with the data
           provided within the x_train and y_train arrays, parametrised by some paramter.
           
           The differential equation system or equation should be defined for the same range and domain as the
           evolution dynamics function F(x,y).
           
           
           
           The differential equation should have the following form:
               
                       Y' = F(x,y,n) parametrised by learnable parameters n:  F(x(n),y(n),n)
                       
           

          f_S_I_R(Y, t, *params)
          
           
       Parameters
       ----------
       
       x_train : array_like, shape (n,)
           Present states tensor. Array-like data tensor
           corresponding to the right hand-side of the system.
       y_train : array_like, shape (n,)
           Next state tensor. Array-like tensor corresponding
           to the time evolved state data.
       I0 : array_like, shape (m,)
           Initial state. Array like with initial conditions.
           Should have the same dimensionality as y_train.
       learning_rate : array_like, shape (m,)
           learning factor. Array-like tensor with the learning
           factor applied to the gradient descent step. Should
           have the same dimension as number of parameters in
           the system.
       params : array_like, shape (m,)
           System parameters vector. Vector corresponding to the
           total number of learnable parameters of the system.
       cost_n : float, optional
           Norm degree. Order of the distance measure to be applied
           to the cost function.
       max_iter : float, optional
           Number of epochs performed during the training process.
       tol= 0.1 : float, optional
           Minimum training error. Training error to stop the training
           process.
       obs_idx : n-tuple, optional
           Data-model corresponding index. Array-like variable that
           maps the systems variables with the training tensors so
           the training can be performed with the corresponding state
           variables.
           
       Returns
       -------
       
       Two objects referent to the trained model:
       
       theta_star : ndarray, shape (n_points,)
           Most optimal parameters found during the training procedure.
           A result might be given, even without converging. Checking
           the error gives more information about the convergence of the
           algorithm.
       final_error : ndarray, shape (n_points,)
           Error vector. Array containing the error between the observed
           variables and the prediction function. If NaN or np.inf the
           the model had no solution and diverged from the data.
           
               
                    
       '''
       print("Train")
       print("Initial parameters = {}".format(params))
       ''' Initial conditions set up and sensitivity variable expansion '''
       Y0= np.append(I0, np.array([0.]*self.__gradY_theta))
       theta_iter= params
       
       
       '''Gradients and derivatives with respect to cost and observed variables set up'''
       self.jacobian_gen(self.__f_x_y)
       cost= self.cost_nnorm(y_train, cost_n)
       
       grad_C= autograd.jacobian(cost)

       
       obs_n_vars = len(obs_idx)
       
       
       """Generation fucntion of the sensitivity indeces corresponding to the number of the parameters and variables
       within the system"""
       gradf_gen = lambda var_idx, n_params: (-self.__gradY_theta+ var_idx*self.__n_params,
                                              -self.__gradY_theta+ var_idx*self.__n_params + self.__n_params )
       
       grad_idx = np.array([gradf_gen(n,self.__n_params) for n in obs_idx])
       
       for i in range(max_iter):

           sol = super().system_evolve(Y0, method= 'RK45', params= params , system = self.sensitivity_expansion)
           
           """Seleccion of the observed variables that correspond to the y solution"""
          
           Y = np.array([sol.y[n, :] for n in obs_idx]).T

           """ Generation of gradients through the computer graph"""
           gradC_Y = grad_C(Y)
           gradC_Y= gradC_Y.reshape(-1,1,obs_n_vars)
           """Tensor formation for multiplication"""

           Sens_Y = np.array([sol.y[df_1: df_2] for df_1, df_2 in grad_idx]).T
           #Sensitivity variable seleccion that correspond with the observed data variables
           

           Sens_Y = Sens_Y.reshape(-1,obs_n_vars,self.__n_params); """Reshape for the computation of the jacobian"""




           jacF_p = (gradC_Y@Sens_Y).sum(axis = 0) #Computation of the jacobian tensor and collapsing
    
           theta_iter -= (learning_rate*jacF_p)[0] #Gradiend descent step

           if i%10==0:
               print(theta_iter)
               

       self.__trained_params = theta_iter
       
       return (self.__trained_params, self.__err)
       



       
       
       
   
class SIRSystem(DynamicSystemLearner):
  
  '''SIR system is a particular non-linear ODE system that is believed to describe the dynamics of a viral
      pandemic. It is inherited from the SystemDynamics Class so it shares their methods from the superclass
      
      It also allows acces to the SIR method which abstracts the evolution of the S-I-R simple model
      
      '''
  
  
  def __init__(self,name = "SIR"):
      super().__init__(name)
      
      
  @staticmethod
  def f_S_I_R(t, Y, params):
      
      '''The f_S_I_R(Y, t, *params) method contains the described dynamics of the SIR compartment model.
          The differential equation has the following form:
              
                      [S' I' R']= [-r SI, rSI-aI, aI]
                      
          
          Where S, I and R, define the Suceptible, Infectious and Removed populations respectively.
      
          
         f_S_I_R(Y, t, *params)
          
          Parameters
          ----------
          
              Y : Array_like, shape (n,)
                  Previous system state.
              t : float
                  Next state corresponding time
              params : 2-tuple of floats
                  Model parameters r and a coefficients. Parameter r,
                  corresponds to the contact rate and , a, to recovery
                  rate.
                  
          Returns
          -------
          
              Y_n : ndarray, shape (n_points,)
                  System next state vector
      '''

      S, I, R = Y

      r, a = params

      return np.array([-r*S*I, r*S*I-a*I, a*I])

      
      
  def system_init(self, t0, tf, d_t, f_x_y = None):
      
      '''Superclass SystemDynamics initialisation
      
          system_init(self, t0, tf, d_t, f_x_y)
          
          Parameters
          ----------
          
          t0 : float
              Inital time of the observation dynamics
          tf : float
              Final time of the observed dynamics
          d_t: float
              Time step resolution for the system to be observed
          f_x_y : callable
              function redefinition of a state-spate SIR
              variant function. Should be referent to f(x,t,... )
          
          
      
      '''
      if f_x_y is None:
          f_x_y = self.f_S_I_R
      
      super().system_init(t0, tf, d_t, f_x_y)
      
  def system_properties(self):
      ''' Superclass method SystemDynamics system_properties method.
      
          The system_properties method prints the information regarding the parameters contained by the system
          
          It prints the data related to the initial time, t0; final time, tf; time delta, d_t; name and other
          parameters.
      '''
      super().system_properties()
  
  def system_evolve(self, Y0, method = 'RK45', params= None):
      ''' Superclass method SystemDynamics system_evolve method.
      
          The system_evolve method evolves the dynamic SIR model through the superclass system_evolve method.
          
          It prints the data related to the initial time, t0; final time, tf; time delta, d_t; name and other
          parameters.
      '''
      return super().system_evolve(Y0, method, params)
  
  def plot_dynamics(self, labels= ['S','I','R'], colours= ['C0', 'C1', 'C2'], xlab = 'time [s]', ylab = 'population'):
      """The plot_dynamics function allows you to generate plots of a time-evolved system.
      
          Parameters
          ----------
              labels : Array-like, shape (n,m), default : {'S', 'I', 'R'}
                  Curve names. Names of the curves that will appear within the
                  plot.
              colours : Array-like, shape (n,m), default : {'C0', 'C1', 'C2'}
                  Color code key. Colour codes of the displayed curves.
              xlab : string, default : 'time [S]'
                  X axis label.
              ylab : string, default : 'population'
                  y axis label
                  
      """
      super().plot_dynamics(labels, colours, xlab, ylab)
      
  

  
      
  
  
  
