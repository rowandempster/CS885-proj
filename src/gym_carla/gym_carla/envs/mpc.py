import sys
import os
# sys.path.append(os.path.join(sys.path[0], 'casadi-linux-py38-v3.5.5-64bit'))
from casadi import *
import numpy as np
import time
from math import sin, cos



# ==============================================================================
# -- MPC CONTROLLER ------------------------------------------------------------
# ==============================================================================

class mpc_control(object):
    def __init__(self):
        #self._contPub = pub
        self._yawOld = 0
        self._X_State_Limit = np.inf
        self._Y_State_Limit = np.inf
        self._stop = 0
        self._Vmpc = 0
        self._xDis = None
        self._yDis = None
        self.str = 0.0
        self.rvr = 0.0 
        self.thr = 0.0
        self.brk = 0.0
        self.Initialize_MPC()

    def Initialize_MPC(self):
        # seconds = rospy.get_time()
        # while(rospy.get_time() - seconds  <1.5):
        #     print("Waiting")
        #     time.sleep(0.2)
        self._T = 0.1  # Sampling time [s]
        self._T_interpolation = 0.02 # Interpolation time [s]
        self._N = 25   # Prediction horizon [samples]
        self._L = 2.45  # Wheel Base
        self._ActPathParam = 0 
        #   St : Front wheel steerng angle (rad)
        #   v  : Vehicle speed  
        self._Errold = 0
        self._IntErrVel = 0.0
        
        self._v_max = 12             # Maximum forward speed 16 m/s
        self._v_min = -self._v_max/4        # Maximum backward speed 4 m/s 
        self._St_max = 1.1344               # Maximum steering angle 65 degree
        self._St_min = -self._St_max        # Maximum steering angle 65 degree
        self._a_max = 5                     # Maximum acceleration 4 m/s**2
        self._a_min = self._a_max * -1.5    # Maximum decceleration 6 m/s**2
        self._dSt_max = np.pi/3                # Maximum change of steering angle 
        self._dSt_min = -self._dSt_max      # Maximum change of steering angle

        # States
        self._x = SX.sym('x')           # Vehicle position in x
        self._y = SX.sym('y')           # Vehicle position in y
        self._theta = SX.sym('theta')   # Vehicle orientation angle 
        self._Vel = SX.sym('v')         # Vehicle speed
        self._Steer = SX.sym('Steer')   # Vehicle steering angle (virtual central wheel)

        self._states = vertcat(self._x, self._y, self._theta, self._Vel, self._Steer)
        self._n_states = SX.size(self._states)
        self._StAct = 0
        self._SpAct = 0
        self._Vact = 0
        self._Vdes = 4

        # Sytem Model
        self._VelCMD = SX.sym('VelCMD') # System Inputs - Car Speed
        self._St = SX.sym('St')         # System Inputs - Steering angle
        self._controls = vertcat(self._VelCMD, self._St)    # System Control Inputs
        self._n_control = SX.size(self._controls)

        beta = np.arctan2(np.tan(self._St),2)
        
        self._Vx = self._Vel  * np.cos(beta + self._theta)
        self._Vy = self._Vel  * np.sin(beta + self._theta)
        self._W = self._Vel  * np.cos(beta) * np.tan(self._St) / self._L

        self._rhs = vertcat( self._Vx,
        self._Vy,
        self._W,
        self._VelCMD,
        self._St)

        self._f = Function('f',[self._states,self._controls],[self._rhs]) # Nonlinear mapping Function f(x,u)
        
        self._U = SX.sym('U', self._n_control[0], self._N)         # Decision variables (controls)
        self._P = SX.sym('P', self._n_states[0] + 4)                # Parameters (initial state of the car) + reference state
        self._X = SX.sym('X', self._n_states[0], (self._N+1))      # A matrix that represents the states over the optimization problem
        
        self._obj = 0 # Objective function
        self._g = SX([]) # Constraints vector

        # Weighing matrices (States)
        self._Q = DM([[0,0,0],[0,5,0],[0,0,10]])

        # Weighing matrices (Controls)
        self._R = DM([[2,0],[0,200]])
        # self._R = [[self._P[-3], self._P[-1]], [self._P[-1],self._P[-2]]]

        self._con_old = DM.zeros(self._n_control[0],1)

        self._st = self._X[:,0]   # initial state
        self._g = vertcat(self._g, self._st[:] - self._P[0:5])     # initial condition constraints

        position = SX.sym('Position',2)
        '''
            P[0]: Current X            P[5]: Ref X                             
            P[1]: Current Y            P[6]: Ref Y                                           
            P[2]: Current Yaw          P[7]: Ref Heading  
            P[3]: Current Velocity     P[8]: Ref Velocity            
            P[4]: Current Steering          
        '''
        # Compute Objective
        for k in range(self._N):
            self._st = self._X[:,k]
            self._con = self._U[:,k]

            yawRef = self._P[7]

            ErrX = self._P[5] - self._st[0]
            ErrY = self._P[6] - self._st[1]
            ErrLateral = -ErrX*np.sin(yawRef) + ErrY*np.cos(yawRef)

            objT1 = ErrLateral * ErrLateral * self._Q[0,0]
            objT2 = self._Q[1,1] * (1-np.sin(self._st[2])*np.sin(yawRef) - np.cos(self._st[2])*np.cos(yawRef))
            objT3 = (self._P[8] - self._st[3])**2 * self._Q[2,2]

            objT4 = mtimes(self._con[0:2].T-self._con_old[0:2].T,mtimes(self._R, self._con[0:2]-self._con_old[0:2])) 

            self._obj = self._obj + objT1 + objT2 + objT3 + objT4
            self._con_old = self._con

            self._st_next = self._X[:,k+1]
        
            k1 = self._f(self._st, self._con)
            k2 = self._f(self._st + k1*self._T/2, self._con)
            k3 = self._f(self._st + k2*self._T/2, self._con)
            k4 = self._f(self._st + k3*self._T, self._con)

            gradientRK4 = (k1 + 2*k2 + 2*k3 + k4) / 6
            st_next_RK4 = self._st[0:3] + self._T * gradientRK4[0:3]
            st_next_RK4 = vertcat(st_next_RK4, k1[3:5])

            self._g = vertcat(self._g, self._st_next - st_next_RK4) # compute constraints

        # Compute Constraints
        for k in range(self._N):
            dV = self._X[3,k+1] - self._X[3,k]
            self._g = vertcat(self._g, dV)
        
        for k in range(self._N):
            dU = self._U[1,k] - self._X[4,k]
            self._g = vertcat(self._g, dU)

        # print(self._g)
        # Make the decision variables on column vector 
        self._OPT_variables = reshape(self._X, self._n_states[0] * (self._N+1), 1)
        self._OPT_variables = vertcat(self._OPT_variables, reshape(self._U, self._n_control[0] * self._N, 1))
        
        self._nlp_prob = {'f':self._obj, 'x':self._OPT_variables, 'g':self._g, 'p':self._P}

        # Pick an NLP solver
        self._MySolver = 'ipopt'

        # Solver options
        self._opts = {'ipopt.max_iter':100, 'ipopt.print_level':0,'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

        self._solver = nlpsol('solver',self._MySolver,self._nlp_prob,self._opts)

        # Constraints initialization:
        self._lbx = DM.zeros(self._n_states[0] * (self._N+1) + self._n_control[0] * self._N,1)
        self._ubx = DM.zeros(self._n_states[0] * (self._N+1) + self._n_control[0] * self._N,1)

        self._lbg = DM.zeros(self._n_states[0] * (self._N+1) + 2 * self._N,1) 
        self._ubg = DM.zeros(self._n_states[0] * (self._N+1) + 2 * self._N,1)

        # inequality constraints (state constraints)

        # constraint on X position (0->a,0)
        a = self._n_states[0]*(self._N+1)
        self._lbx[0:a:self._n_states[0],0] = DM(-self._X_State_Limit)
        self._ubx[0:a:self._n_states[0],0] = DM(self._X_State_Limit)

        # constraint on Y position (0->a,1)
        self._lbx[1:a:self._n_states[0],0] = DM(-self._Y_State_Limit)
        self._ubx[1:a:self._n_states[0],0] = DM(self._Y_State_Limit)
        # constraint on yaw angle (0->a,2)
        self._lbx[2:a:self._n_states[0],0] = DM(-np.inf)
        self._ubx[2:a:self._n_states[0],0] = DM(np.inf)

        # constraint on velocity (state) (0->a,3)
        self._lbx[3:a:self._n_states[0],0] = DM(self._v_min)
        self._ubx[3:a:self._n_states[0],0] = DM(self._v_max)

        # constraint on steering angle (state) (0->a,4)
        self._lbx[4:a:self._n_states[0],0] = DM(self._St_min)
        self._ubx[4:a:self._n_states[0],0] = DM(self._St_max)

        # constraint on velocity input (a->end)
        self._lbx[a:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._v_min)
        self._ubx[a:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._v_max)

        # constraint on steering input (a->end)
        self._lbx[a + 1:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._St_min)
        self._ubx[a + 1:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._St_max)

        # equality constraints
        self._lbg[0:self._n_states[0]*(self._N+1),0] = DM(0)
        self._ubg[0:self._n_states[0]*(self._N+1),0] = DM(0)

        # constraint on vehicle acceleration
        self._lbg[self._n_states[0]*(self._N+1): self._n_states[0]*(self._N+1) + self._N,0] = DM(self._a_min * self._T)
        self._ubg[self._n_states[0]*(self._N+1): self._n_states[0]*(self._N+1) + self._N,0] = DM(self._a_max * self._T)

        # constraint on steering rate
        self._lbg[self._n_states[0]*(self._N+1) + self._N: self._n_states[0]*(self._N+1) + 2 * self._N,0] = DM(self._dSt_min * self._T)
        self._ubg[self._n_states[0]*(self._N+1) + self._N: self._n_states[0]*(self._N+1) + 2 * self._N,0] = DM(self._dSt_max * self._T)

        self._args = {}
        self._args['lbg'] = self._lbg       # dU and States constraints
        self._args['ubg'] = self._ubg       # dU and States constraints
        self._args['lbx'] = self._lbx       #  input constraints
        self._args['ubx'] = self._ubx       #  input constraints
        self._u0  = DM.zeros(self._N ,self._n_control[0])    # Control inputs
        self._x0  = DM.ones(self._N+1 ,self._n_states[0])   # Initialization of the states decision variables
        self._T_old = 0
        self._T_old_interpolation = 0
        self._CompTime = 0
        self._yaw = 0
        print("MPC Controller Initialized")

    def feedbackCallback(self,xEgo,yEgo,yawEgo,vEgo,stEgo,xRef,yRef,yawRef,vRef):
        self._yawOld = self._yaw
        self._VactOld = self._Vact 
        self._xDis = xEgo
        self._yDis = yEgo
        self._yaw = yawEgo
        self._Vact = vEgo
        self._StAct = stEgo
        self._Xs = [xRef,yRef,yawRef,vRef]
        self.runMPCstep()
        
    def runMPCstep(self):
        seconds = time.time()
        if abs(self._yawOld - self._yaw) > 6:
            for n in range(DM.size(self._x0)[0]):
                self._x0[n,2] = -1* self._x0[n,2]
        if((seconds - self._T_old) > self._T):
            self._T_old = seconds
            # self._Xs = [5,5,0]
            self._ActPathParam = self._x0[0,3].__float__()
            
            ErrX = self._Xs[0] - self._xDis
            ErrY = self._Xs[1] - self._yDis
            yawErr = 1 - (np.sin(self._yaw)*np.sin(self._Xs[2]) + np.cos(self._yaw)*np.cos(self._Xs[2]))

            Err = ErrX**2 + ErrY**2 
            LongErr = ErrX*np.cos(self._Xs[2]) + ErrY*np.sin(self._Xs[2])  
            LatErr = -ErrX*np.sin(self._Xs[2]) + ErrY*np.cos(self._Xs[2])  
            print("ErrX: " + str(ErrX))
            print("ErrY: " + str(ErrY))
            print("yaw: " + str(self._Xs[2]))
            print("ErrYaw: " + str(yawErr))
            print("LongErr: " + str(LongErr))
            print("LateralErr: " + str(LatErr))
            print("===========================")
            if(True):
                '''
                P[0]: Current X            P[6]: Ref X                             
                P[1]: Current Y            P[7]: Ref Y                                           
                P[2]: Current Yaw          P[8]: Ref Heading  
                P[4]: Current Velocity     P[9]: Ref Velocity             
                P[5]: Current Steering          
                '''
                R = [self._xDis, self._yDis, self._yaw, self._Vact, self._StAct] + self._Xs
                # R = [10, 10, 0.5, 0, 0, 10, 10, 0.5, 1]
                R = np.array(R)
                R = np.reshape(R, (R.size, 1))
                self._args['p'] = R
                self._args['x0'] = vertcat(reshape(self._x0.T, self._n_states[0]*(self._N+1), 1), reshape(self._u0.T, self._n_control[0] * self._N, 1))   # initial condition for optimization variable                   
                self.sol = self._solver(
                x0 = self._args['x0'],
                lbx = self._args['lbx'],
                ubx = self._args['ubx'],
                lbg = self._args['lbg'],
                ubg = self._args['ubg'],
                p = self._args['p'])

                self._u = reshape(self.sol['x'][self._n_states[0]*(self._N+1):].T,self._n_control[0],(self._N)).T
                self._Vmpc = self._u[0,0].__float__()
                self._Stdes = self._u[0,1].__float__()
                self.Vehicle_ContMsg()

                self._xStates = reshape(self.sol['x'][0:self._n_states[0]*(self._N+1)].T,self._n_states[0],(self._N+1)).T  
                self._x0[0:self._N,:] = self._xStates[1:self._N+1,:]
                self._x0[self._N,:] = self._xStates[self._N,:]
                self._u0[0:self._N-1,:] = self._u[1:self._N,:]
                self._u0[self._N-1,:] = self._u[self._N-1,:]

                self._CompTime = time.time() - seconds
                print("Comp Time: " + str(self._CompTime))
            
            else:
                print ('ARRIVED')
                self.Vehicle_ContMsg(True)

    def runInterpolationStep(self):
        seconds = time.time()
        if(self._T_old != 0):
            if((seconds - self._T_old_interpolation) > self._T_interpolation):
                self._T_old_interpolation = seconds
                w1 = (self._T_old_interpolation - self._T_old)/self._T
                w0 = 1 - w1
                # w0 = w0/self._T
                # w1 = w1/self._T
                # self._Vmpc = w0 * self._u[0,0].__float__() + w1 * self._u[1,0].__float__()
                # self._Stdes = w0 * self._u[0,1].__float__() + w1 * self._u[1,1].__float__()
                self.Vehicle_ContMsg()

            
    def Vehicle_ContMsg(self,Arrived = False):       
        if(not Arrived):
            ErrVel = self._Vmpc - self._Vact
            self._IntErrVel = self._IntErrVel + ErrVel * self._T

            # if(self._Errold * (ErrVel+2) < 0):
            #     self._IntErrVel = 0
            if (ErrVel > 0):
                AppliedPedal = 0.6 * ErrVel + 0.1 * self._IntErrVel #+ 0.1 * (ErrVel - self._Errold)/self._T
            else:
                AppliedPedal = 0.8 * ErrVel + 0.1 * self._IntErrVel
            self._Errold = ErrVel

            if(AppliedPedal>1):
                AppliedPedal = 1
            elif(AppliedPedal < 0):
                AppliedPedal = 0.1 * AppliedPedal
                if AppliedPedal < -1:
                    AppliedPedal = -1

            if(self._Vdes < 0.01 and self._Vact < 0.1):
                AppliedPedal = -0.5
            

            if(AppliedPedal > 0):
                self._throttle = AppliedPedal
                self._brake = 0
            else:
                self._throttle = 0
                self._brake = -AppliedPedal

            self._reverse = 0
            self._steer = self._Stdes
            self._hand_brake = 0
            self._reverse = 0
        else:
            self._steer = 0
            self._reverse = 0
            self._throttle = 0
            self._brake = 1
            self._hand_brake = 0
        
        if(self._stop == 1):
            self._throttle = 0
            self._brake = 1
        
        if(self._stop == 5):
            self._throttle = 0
            self._brake = 0.5 * self._brake + 0.5 
        self.str = self._steer
        self.rvr = self._reverse
        self.thr = self._throttle
        self.brk = self._brake