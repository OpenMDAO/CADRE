''' Attitude discipline for CADRE. '''

import numpy as np

from six.moves import range

from openmdao.core.component import Component

from CADRE.kinematics import computepositionrotd, computepositionrotdjacobian

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103


class Attitude_Angular(Component):
    """ Calculates angular velocity vector from the satellite's orientation
    matrix and its derivative.
    """

    def __init__(self, n=2):
        super(Attitude_Angular, self).__init__()

        self.n = n

        # Inputs
        self.add_param('O_BI', np.zeros((3, 3, n)), units="unitless",
                       desc="Rotation matrix from body-fixed frame to Earth-centered inertial frame over time")

        self.add_param('Odot_BI', np.zeros((3, 3, n)), units="unitless",
                       desc="First derivative of O_BI over time")

        # Outputs
        self.add_output('w_B', np.zeros((3, n)), units="1/s",
                        desc="Angular velocity vector in body-fixed frame over time")

        self.dw_dOdot = np.zeros((n, 3, 3, 3))
        self.dw_dO = np.zeros((n, 3, 3, 3))

    def jacobian(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        O_BI = params['O_BI']
        Odot_BI = params['Odot_BI']

        for i in range(0, self.n):
            self.dw_dOdot[i, 0, 2, :] = O_BI[1, :, i]
            self.dw_dO[i, 0, 1, :] = Odot_BI[2, :, i]

            self.dw_dOdot[i, 1, 0, :] = O_BI[2, :, i]
            self.dw_dO[i, 1, 2, :] = Odot_BI[0, :, i]

            self.dw_dOdot[i, 2, 1, :] = O_BI[0, :, i]
            self.dw_dO[i, 2, 0, :] = Odot_BI[1, :, i]

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        O_BI = params['O_BI']
        Odot_BI = params['Odot_BI']
        w_B = unknowns['w_B']

        for i in range(0, self.n):
            w_B[0, i] = np.dot(Odot_BI[2, :, i], O_BI[1, :, i])
            w_B[1, i] = np.dot(Odot_BI[0, :, i], O_BI[2, :, i])
            w_B[2, i] = np.dot(Odot_BI[1, :, i], O_BI[0, :, i])

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        if mode == 'fwd':
            w_B = dresids['w_B']

            for k in range(3):
                for i in range(3):
                    for j in range(3):
                        if 'O_BI' in dparams:
                            w_B[k, :] += self.dw_dO[:, k, i, j] * \
                                dparams['O_BI'][i, j, :]
                        if 'Odot_BI' in dparams:
                            w_B[k, :] += self.dw_dOdot[:, k, i, j] * \
                                dparams['Odot_BI'][i, j, :]

        else:
            w_B = dresids['w_B']

            for k in range(3):
                for i in range(3):
                    for j in range(3):
                        if 'O_BI' in dparams:
                            dparams['O_BI'][i, j, :] += self.dw_dO[:, k, i, j] * \
                                w_B[k, :]
                        if 'Odot_BI' in dparams:
                            dparams['Odot_BI'][i, j, :] += self.dw_dOdot[:, k, i, j] * \
                                w_B[k, :]


#class Attitude_AngularRates(Component):

    #""" Calculates time derivative of angular velocity vector.
    #"""

    #def __init__(self, n=2):
        #super(Attitude_AngularRates, self).__init__()

        #self.n = n

        ## Inputs
        #self.add('w_B', Array(np.zeros((3, n)),
                              #iotype='in',
                              #shape=(3, n),
                              #units="1/s",
                              #desc="Angular velocity vector in body-fixed frame over time"
                              #)
                 #)

        #self.add('h', Float(28.8,
                            #iotype='in',
                            #units="s",
                            #desc="Time step for RK4 integration"
                            #)
                 #)

        ## Outputs
        #self.add(
            #'wdot_B',
            #Array(
                #np.zeros((3, n)),
                #iotype='out',
                #shape=(3, n),
                #units="1/s**2",
                #desc="Time derivative of w_B over time"
            #)
        #)

    #def provideJ(self):
        #""" Calculate and save derivatives. (i.e., Jacobian) """

        ## Calculation is fairly simple, so not cached.
        #return

    #def list_deriv_vars(self):
        #input_keys = ('w_B', 'h',)
        #output_keys = ('wdot_B',)
        #return input_keys, output_keys

    #def execute(self):
        #""" Calculate output. """

        #for k in xrange(3):
            #self.wdot_B[k, 0] = self.w_B[k, 1] / self.h
            #self.wdot_B[k, 0] -= self.w_B[k, 0] / self.h
            #self.wdot_B[k, 1:-1] = self.w_B[k, 2:] / 2.0 / self.h
            #self.wdot_B[k, 1:-1] -= self.w_B[k, :-2] / 2.0 / self.h
            #self.wdot_B[k, -1] = self.w_B[k, -1] / self.h
            #self.wdot_B[k, -1] -= self.w_B[k, -2] / self.h

    #def apply_deriv(self, arg, result):
        #""" Matrix-vector product with the Jacobian. """

        #if 'w_B' in arg and 'wdot_B' in result:
            #for k in xrange(3):
                #result['wdot_B'][k, 0] += arg['w_B'][k, 1] / self.h
                #result['wdot_B'][k, 0] -= arg['w_B'][k, 0] / self.h
                #result['wdot_B'][k, 1:-1] += arg['w_B'][k, 2:] / 2.0 / self.h
                #result['wdot_B'][k, 1:-1] -= arg['w_B'][k, :-2] / 2.0 / self.h
                #result['wdot_B'][k, -1] += arg['w_B'][k, -1] / self.h
                #result['wdot_B'][k, -1] -= arg['w_B'][k, -2] / self.h

    #def apply_derivT(self, arg, result):
        #""" Matrix-vector product with the transpose of the Jacobian. """

        #if 'wdot_B' in arg and 'w_B' in result:
            #for k in xrange(3):
                #result['w_B'][k, 1] += arg['wdot_B'][k, 0] / self.h
                #result['w_B'][k, 0] -= arg['wdot_B'][k, 0] / self.h
                #result['w_B'][k, 2:] += arg['wdot_B'][k, 1:-1] / 2.0 / self.h
                #result['w_B'][k, :-2] -= arg['wdot_B'][k, 1:-1] / 2.0 / self.h
                #result['w_B'][k, -1] += arg['wdot_B'][k, -1] / self.h
                #result['w_B'][k, -2] -= arg['wdot_B'][k, -1] / self.h


#class Attitude_Attitude(Component):

    #""" Coordinate transformation from the interial plane to the rolled
    #(forward facing) plane.
    #"""
    #dvx_dv = np.zeros((3, 3, 3))
    #dvx_dv[0, :, 0] = (0., 0., 0.)
    #dvx_dv[1, :, 0] = (0., 0., -1.)
    #dvx_dv[2, :, 0] = (0., 1., 0.)

    #dvx_dv[0, :, 1] = (0., 0., 1.)
    #dvx_dv[1, :, 1] = (0., 0., 0.)
    #dvx_dv[2, :, 1] = (-1., 0., 0.)

    #dvx_dv[0, :, 2] = (0., -1., 0.)
    #dvx_dv[1, :, 2] = (1., 0., 0.)
    #dvx_dv[2, :, 2] = (0., 0., 0.)

    #def __init__(self, n=2):
        #super(Attitude_Attitude, self).__init__()

        #self.n = n

        ## Inputs
        #self.add('r_e2b_I', Array(np.zeros((6, n)),
                                  #iotype='in',
                                  #shape=(6, n),
                                  #units="unitless",
                                  #desc="Position and velocity vector from earth to satellite in Earth-centered inertial frame over time"))

        ## Outputs
        #self.add('O_RI', Array(np.zeros((3, 3, n)),
                               #iotype='out',
                               #shape=(3, 3, n),
                               #units="unitless",
                               #desc="Rotation matrix from rolled body-fixed frame to Earth-centerd inertial frame over time"))

        #self.dO_dr = np.zeros((n, 3, 3, 6))

    #def list_deriv_vars(self):
        #input_keys = ('r_e2b_I',)
        #output_keys = ('O_RI',)
        #return input_keys, output_keys

    #def provideJ(self):
        #""" Calculate and save derivatives. (i.e., Jacobian) """

        #diB_dv = np.zeros((3, 3))
        #djB_dv = np.zeros((3, 3))

        #for i in range(0, self.n):

            #r = self.r_e2b_I[0:3, i]
            #v = self.r_e2b_I[3:, i]

            #normr = np.sqrt(np.dot(r, r))
            #normv = np.sqrt(np.dot(v, v))

            ## Prevent overflow
            #if normr < 1e-10:
                #normr = 1e-10
            #if normv < 1e-10:
                #normv = 1e-10

            #r = r / normr
            #v = v / normv

            #dr_dr = np.zeros((3, 3))
            #dv_dv = np.zeros((3, 3))

            #for k in range(0, 3):
                #dr_dr[k, k] += 1.0 / normr
                #dv_dv[k, k] += 1.0 / normv
                #dr_dr[:, k] -= self.r_e2b_I[
                    #0:3, i] * self.r_e2b_I[k, i] / normr ** 3
                #dv_dv[:, k] -= self.r_e2b_I[
                    #3:, i] * self.r_e2b_I[3 + k, i] / normv ** 3

            #vx = np.zeros((3, 3))
            #vx[0, :] = (0., -v[2], v[1])
            #vx[1, :] = (v[2], 0., -v[0])
            #vx[2, :] = (-v[1], v[0], 0.)

            #iB = np.dot(vx, r)

            #diB_dr = vx
            #diB_dv[:, 0] = np.dot(self.dvx_dv[:, :, 0], r)
            #diB_dv[:, 1] = np.dot(self.dvx_dv[:, :, 1], r)
            #diB_dv[:, 2] = np.dot(self.dvx_dv[:, :, 2], r)

            #djB_diB = -vx
            #djB_dv[:, 0] = -np.dot(self.dvx_dv[:, :, 0], iB)
            #djB_dv[:, 1] = -np.dot(self.dvx_dv[:, :, 1], iB)
            #djB_dv[:, 2] = -np.dot(self.dvx_dv[:, :, 2], iB)

            #self.dO_dr[i, 0, :, 0:3] = np.dot(diB_dr, dr_dr)
            #self.dO_dr[i, 0, :, 3:] = np.dot(diB_dv, dv_dv)

            #self.dO_dr[i, 1, :, 0:3] = np.dot(np.dot(djB_diB, diB_dr), dr_dr)
            #self.dO_dr[i, 1, :, 3:] = np.dot(np.dot(djB_diB, diB_dv) + djB_dv,
                                            #dv_dv)

            #self.dO_dr[i, 2, :, 3:] = -dv_dv

    #def execute(self):
        #""" Calculate output. """

        #self.O_RI = np.zeros(self.O_RI.shape)
        #for i in range(0, self.n):

            #r = self.r_e2b_I[0:3, i]
            #v = self.r_e2b_I[3:, i]

            #normr = np.sqrt(np.dot(r, r))
            #normv = np.sqrt(np.dot(v, v))

            ## Prevent overflow
            #if normr < 1e-10:
                #normr = 1e-10
            #if normv < 1e-10:
                #normv = 1e-10

            #r = r / normr
            #v = v / normv

            #vx = np.zeros((3, 3))
            #vx[0, :] = (0., -v[2], v[1])
            #vx[1, :] = (v[2], 0., -v[0])
            #vx[2, :] = (-v[1], v[0], 0.)

            #iB = np.dot(vx, r)
            #jB = -np.dot(vx, iB)

            #self.O_RI[0, :, i] = iB
            #self.O_RI[1, :, i] = jB
            #self.O_RI[2, :, i] = -v

    #def apply_deriv(self, arg, result):
        #""" Matrix-vector product with the Jacobian. """

        #if 'r_e2b_I' in arg and 'O_RI' in result:
            #for k in xrange(3):
                #for j in xrange(3):
                    #for i in xrange(6):
                        #result['O_RI'][k, j, :] += self.dO_dr[:, k, j, i] * \
                            #arg['r_e2b_I'][i, :]

    #def apply_derivT(self, arg, result):
        #""" Matrix-vector product with the transpose of the Jacobian. """

        #if 'O_RI' in arg and 'r_e2b_I' in result:
            #for k in xrange(3):
                #for j in xrange(3):
                    #for i in xrange(6):
                        #result['r_e2b_I'][i, :] += self.dO_dr[:, k, j, i] * \
                            #arg['O_RI'][k, j, :]


#class Attitude_Roll(Component):

    #""" Calculates the body-fixed orientation matrix.
    #"""

    #def __init__(self, n=2):
        #super(Attitude_Roll, self).__init__()

        #self.n = n

        ## Inputs
        #self.add('Gamma', Array(np.zeros(n),
                                #iotype='in',
                                #shape=(n,),
                                #units="rad",
                                #desc="Satellite roll angle over time"))

        ## Outputs
        #self.add('O_BR', Array(np.zeros((3, 3, n)),
                               #iotype='out',
                               #shape=(3, 3, n),
                               #units="unitless",
                               #desc="Rotation matrix from body-fixed frame to rolled body-fixed frame over time"))

        #self.dO_dg = np.zeros((n, 3, 3))


    #def list_deriv_vars(self):
        #input_keys = ('Gamma',)
        #output_keys = ('O_BR',)
        #return input_keys, output_keys

    #def provideJ(self):
        #""" Calculate and save derivatives. (i.e., Jacobian) """

        #self.dO_dg = np.zeros((self.n, 3, 3))
        #self.dO_dg[:, 0, 0] = -np.sin(self.Gamma)
        #self.dO_dg[:, 0, 1] = np.cos(self.Gamma)
        #self.dO_dg[:, 1, 0] = -self.dO_dg[:, 0, 1]
        #self.dO_dg[:, 1, 1] = self.dO_dg[:, 0, 0]

    #def execute(self):
        #""" Calculate output. """

        #self.O_BR = np.zeros((3, 3, self.n))
        #self.O_BR[0, 0, :] = np.cos(self.Gamma)
        #self.O_BR[0, 1, :] = np.sin(self.Gamma)
        #self.O_BR[1, 0, :] = -self.O_BR[0, 1,:]
        #self.O_BR[1, 1, :] = self.O_BR[0, 0,:]
        #self.O_BR[2, 2, :] = np.ones(self.n)

    #def apply_deriv(self, arg, result):
        #""" Matrix-vector product with the Jacobian. """

        #if 'Gamma' in arg and 'O_BR' in result:
            #for k in xrange(3):
                #for j in xrange(3):
                    #result['O_BR'][k, j, :] += self.dO_dg[:, k, j] * \
                        #arg['Gamma']

    #def apply_derivT(self, arg, result):
        #""" Matrix-vector product with the transpose of the Jacobian. """

        #if 'O_BR' in arg and 'Gamma' in result:
            #for k in xrange(3):
                #for j in xrange(3):
                    #result['Gamma'] += self.dO_dg[:, k, j] * \
                        #arg['O_BR'][k, j, :]


#class Attitude_RotationMtx(Component):

    #""" Multiplies transformations to produce the orientation matrix of the
    #body frame with respect to inertial.
    #"""

    #def __init__(self, n=2):
        #super(Attitude_RotationMtx, self).__init__()

        #self.n = n

        ## Inputs
        #self.add('O_BR', Array(np.zeros((3, 3, n)),
                               #iotype='in',
                               #shape=(3, 3, n),
                               #units="unitless",
                               #desc="Rotation matrix from body-fixed frame to rolled body-fixed frame over time"
                               #))

        #self.add('O_RI', Array(np.zeros((3, 3, n)),
                               #iotype='in',
                               #shape=(3, 3, n),
                               #units="unitless",
                               #desc="Rotation matrix from rolled body-fixed frame to Earth-centered inertial frame over time"
                               #))

        ## Outputs
        #self.add('O_BI', Array(np.zeros((3, 3, n)),
                               #iotype='out',
                               #shape=(3, 3, n),
                               #units="unitless",
                               #desc="Rotation matrix from body-fixed frame to Earth-centered inertial frame over time"
                               #))

    #def list_deriv_vars(self):
        #input_keys = ('O_BR', 'O_RI',)
        #output_keys = ('O_BI',)
        #return input_keys, output_keys

    #def provideJ(self):
        #""" Calculate and save derivatives. (i.e., Jacobian) """

        ## Calculation is fairly simple, so not cached.
        #return

    #def execute(self):
        #""" Calculate output. """

        #for i in range(0, self.n):
            #self.O_BI[:, :, i] = np.dot(self.O_BR[:,:, i], self.O_RI[:,:, i])

    #def apply_deriv(self, arg, result):
        #""" Matrix-vector product with the Jacobian. """

        #if 'O_BI' in result:
            #for u in xrange(3):
                #for v in xrange(3):
                    #for k in xrange(3):
                        #if 'O_RI' in arg:
                            #result['O_BI'][u, v, :] += self.O_BR[u, k,:] * \
                                #arg['O_RI'][k, v, :]
                        #if 'O_BR' in arg:
                            #result['O_BI'][u, v, :] += arg['O_BR'][u, k,:] * \
                                #self.O_RI[k, v, :]

    #def apply_derivT(self, arg, result):
        #""" Matrix-vector product with the transpose of the Jacobian. """

        #if 'O_BI' in arg:
            #for u in xrange(3):
                #for v in xrange(3):
                    #for k in xrange(3):
                        #if 'O_RI' in result:
                            #result['O_RI'][k, v, :] += self.O_BR[u, k,:] * \
                                #arg['O_BI'][u, v, :]
                        #if 'O_BR' in result:
                            #result['O_BR'][u, k, :] += arg['O_BI'][u, v,:] * \
                                #self.O_RI[k, v, :]


#class Attitude_RotationMtxRates(Component):

    #""" Calculates time derivative of body frame orientation matrix.
    #"""

    #def __init__(self, n=2):
        #super(Attitude_RotationMtxRates, self).__init__()

        #self.n = n

        ## Inputs
        #self.add('h', Float(28.8,
                            #iotype='in',
                            #units="s",
                            #desc="Time step for RK4 integration"))

        #self.add('O_BI', Array(np.zeros((3, 3, n)),
                               #iotype='in',
                               #shape=(3, 3, n),
                               #units="unitless",
                               #desc="Rotation matrix from body-fixed frame to Earth-centered inertial frame over time"))

        ## Outputs
        #self.add('Odot_BI', Array(np.zeros((3, 3, n)),
                                  #iotype='out',
                                  #shape=(3, 3, n),
                                  #units="unitless",
                                  #desc="First derivative of O_BI over time"))


    #def list_deriv_vars(self):
        #input_keys = ('h', 'O_BI',)
        #output_keys = ('Odot_BI',)
        #return input_keys, output_keys

    #def provideJ(self):
        #""" Calculate and save derivatives. (i.e., Jacobian) """

        ## Calculation is fairly simple, so not cached.
        #return

    #def execute(self):
        #""" Calculate output. """

        #for k in range(3):
            #for j in range(3):
                #self.Odot_BI[k, j, 0] = self.O_BI[k, j, 1] / self.h
                #self.Odot_BI[k, j, 0] -= self.O_BI[k, j, 0] / self.h
                #self.Odot_BI[k, j, 1:-1] = self.O_BI[k, j, 2:] / 2.0 / self.h
                #self.Odot_BI[k, j, 1:-1] -= self.O_BI[k, j, :-2] / 2.0 / self.h
                #self.Odot_BI[k, j, -1] = self.O_BI[k, j, -1] / self.h
                #self.Odot_BI[k, j, -1] -= self.O_BI[k, j, -2] / self.h

    #def apply_deriv(self, arg, result):
        #""" Matrix-vector product with the Jacobian. """

        #if 'O_BI' in arg and 'Odot_BI' in result:
            #for k in xrange(3):
                #for j in xrange(3):
                    #result['Odot_BI'][k, j, 0] += arg['O_BI'][k, j, 1] / self.h
                    #result['Odot_BI'][k, j, 0] -= arg['O_BI'][k, j, 0] / self.h
                    #result['Odot_BI'][k, j, 1:-1] += arg['O_BI'][k, j, 2:] / \
                        #2.0 / self.h
                    #result['Odot_BI'][k, j, 1:-1] -= arg['O_BI'][k, j, :-2] / \
                        #2.0 / self.h
                    #result['Odot_BI'][k, j, -1] += arg['O_BI'][k, j, -1] / \
                        #self.h
                    #result['Odot_BI'][k, j, -1] -= arg['O_BI'][k, j, -2] / \
                        #self.h

    #def apply_derivT(self, arg, result):
        #""" Matrix-vector product with the transpose of the Jacobian. """

        #if 'Odot_BI' in arg and 'O_BI' in result:
            #for k in xrange(3):
                #for j in xrange(3):
                    #result['O_BI'][k, j, 1] += arg['Odot_BI'][k, j, 0] / self.h
                    #result['O_BI'][k, j, 0] -= arg['Odot_BI'][k, j, 0] / self.h
                    #result['O_BI'][k, j, 2:] += arg['Odot_BI'][k, j, 1:-1] / \
                        #2.0 / self.h
                    #result['O_BI'][k, j, :-2] -= arg['Odot_BI'][k, j, 1:-1] / \
                        #2.0 / self.h
                    #result['O_BI'][k, j, -1] += arg['Odot_BI'][k, j, -1] / \
                        #self.h
                    #result['O_BI'][k, j, -2] -= arg['Odot_BI'][k, j, -1] / \
                        #self.h


#class Attitude_Sideslip(Component):

    #""" Determine velocity in the body frame."""

    #def __init__(self, n=2):
        #super(Attitude_Sideslip, self).__init__()
        #self.n = n

        ## Inputs
        #self.add('r_e2b_I', Array(np.zeros((6, n)),
                                  #iotype='in',
                                  #shape=(6, n),
                                  #units="unitless",
                                  #desc="Position and velocity vector from earth to satellite in Earth-centered inertial frame over time"
                                  #))

        #self.add('O_BI', Array(np.zeros((3, 3, n)),
                               #iotype='in',
                               #shape=(3, 3, n),
                               #units="unitless",
                               #desc="Rotation matrix from body-fixed frame to Earth-centered inertial frame over time"))

        ## Outputs
        #self.add('v_e2b_B', Array(np.zeros((3, n)),
                                  #iotype='out',
                                  #shape=(3, n),
                                  #units="m/s",
                                  #desc="Velocity vector from earth to satellite in body-fixed frame over time"))


    #def list_deriv_vars(self):
        #input_keys = ('r_e2b_I', 'O_BI',)
        #output_keys = ('v_e2b_B',)
        #return input_keys, output_keys

    #def provideJ(self):
        #""" Calculate and save derivatives. (i.e., Jacobian) """

        #self.J1, self.J2 = computepositionrotdjacobian(self.n,
                                                       #self.r_e2b_I[3:, :],
                                                       #self.O_BI)

    #def execute(self):
        #""" Calculate output. """

        #self.v_e2b_B = computepositionrotd(self.n, self.r_e2b_I[3:, :],
                                           #self.O_BI)

    #def apply_deriv(self, arg, result):
        #""" Matrix-vector product with the Jacobian. """

        #if 'v_e2b_B' in result:
            #for k in xrange(3):
                #if 'O_BI' in arg:
                    #for u in xrange(3):
                        #for v in xrange(3):
                            #result['v_e2b_B'][k, :] += self.J1[:, k, u, v] * \
                                #arg['O_BI'][u, v, :]
                #if 'r_e2b_I' in arg:
                    #for j in xrange(3):
                        #result['v_e2b_B'][k, :] += self.J2[:, k, j] * \
                            #arg['r_e2b_I'][3+j, :]

    #def apply_derivT(self, arg, result):
        #""" Matrix-vector product with the transpose of the Jacobian. """

        #if 'v_e2b_B' in arg:
            #for k in xrange(3):
                #if 'O_BI' in result:
                    #for u in xrange(3):
                        #for v in xrange(3):
                            #result['O_BI'][u, v, :] += self.J1[:, k, u, v] * \
                                #arg['v_e2b_B'][k, :]
                #if 'r_e2b_I' in result:
                    #for j in xrange(3):
                        #result['r_e2b_I'][3+j, :] += self.J2[:, k, j] * \
                            #arg['v_e2b_B'][k, :]


#class Attitude_Torque(Component):

    #""" Compute the required reaction wheel tourque."""

    #J = np.zeros((3, 3))
    #J[0, :] = (0.018, 0., 0.)
    #J[1, :] = (0., 0.018, 0.)
    #J[2, :] = (0., 0., 0.006)

    #def __init__(self, n=2):
        #super(Attitude_Torque, self).__init__()

        #self.n = n

        ## Inputs
        #self.add('w_B', Array(np.zeros((3, n)),
                              #iotype='in',
                              #shape=(3, n),
                              #units="1/s",
                              #desc="Angular velocity in body-fixed frame over time"))

        #self.add('wdot_B', Array(np.zeros((3, n)),
                                 #iotype='in',
                                 #shape=(3, n),
                                 #units="1/s**2",
                                 #desc="Time derivative of w_B over time"))

        ## Outputs
        #self.add('T_tot', Array(np.zeros((3, n)),
                                #iotype='out',
                                #shape=(3, n),
                                #units="N*m",
                                #desc="Total reaction wheel torque over time"))

        #self.dT_dwdot = np.zeros((n, 3, 3))
        #self.dwx_dw = np.zeros((3, 3, 3))

        #self.dwx_dw[0, :, 0] = (0., 0., 0.)
        #self.dwx_dw[1, :, 0] = (0., 0., -1.)
        #self.dwx_dw[2, :, 0] = (0., 1., 0.)

        #self.dwx_dw[0, :, 1] = (0., 0., 1.)
        #self.dwx_dw[1, :, 1] = (0., 0., 0.)
        #self.dwx_dw[2, :, 1] = (-1., 0, 0.)

        #self.dwx_dw[0, :, 2] = (0., -1., 0)
        #self.dwx_dw[1, :, 2] = (1., 0., 0.)
        #self.dwx_dw[2, :, 2] = (0., 0., 0.)

    #def list_deriv_vars(self):
        #input_keys = ('w_B', 'wdot_B',)
        #output_keys = ('T_tot',)
        #return input_keys, output_keys

    #def provideJ(self):
        #""" Calculate and save derivatives. (i.e., Jacobian) """

        #self.dT_dw = np.zeros((self.n, 3, 3))
        #wx = np.zeros((3, 3))

        #for i in range(0, self.n):
            #wx[0, :] = (0., -self.w_B[2, i], self.w_B[1, i])
            #wx[1, :] = (self.w_B[2, i], 0., -self.w_B[0, i])
            #wx[2, :] = (-self.w_B[1, i], self.w_B[0, i], 0.)

            #self.dT_dwdot[i, :,:] = self.J
            #self.dT_dw[i, :,:] = np.dot(wx, self.J)

            #for k in range(0, 3):
                #self.dT_dw[i, :, k] += np.dot(self.dwx_dw[:,:, k],
                                             #np.dot(self.J, self.w_B[:, i]))

    #def execute(self):
        #""" Calculate output. """

        #wx = np.zeros((3, 3))
        #for i in range(0, self.n):
            #wx[0, :] = (0., -self.w_B[2, i], self.w_B[1, i])
            #wx[1, :] = (self.w_B[2, i], 0., -self.w_B[0, i])
            #wx[2, :] = (-self.w_B[1, i], self.w_B[0, i], 0.)
            #self.T_tot[:, i] = np.dot(self.J, self.wdot_B[:, i]) + \
                #np.dot(wx, np.dot(self.J, self.w_B[:, i]))

    #def apply_deriv(self, arg, result):
        #""" Matrix-vector product with the Jacobian. """

        #if 'T_tot' in result:
            #for k in xrange(3):
                #for j in xrange(3):
                    #if 'w_B' in arg:
                        #result['T_tot'][k, :] += self.dT_dw[:, k, j] * \
                            #arg['w_B'][j, :]
                    #if 'wdot_B' in arg:
                        #result['T_tot'][k, :] += self.dT_dwdot[:, k, j] * \
                            #arg['wdot_B'][j, :]

    #def apply_derivT(self, arg, result):
        #""" Matrix-vector product with the transpose of the Jacobian. """

        #if 'T_tot' in arg:
            #for k in xrange(3):
                #for j in xrange(3):
                    #if 'w_B' in result:
                        #result['w_B'][j, :] += self.dT_dw[:, k, j] * \
                            #arg['T_tot'][k, :]
                    #if 'wdot_B' in result:
                        #result['wdot_B'][j,:] += self.dT_dwdot[:, k, j] * \
                            #arg['T_tot'][k,:]
