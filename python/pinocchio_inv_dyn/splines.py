import numpy as np
from enum import Enum
from copy import copy,deepcopy

from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
from qpoases import PySolutionAnalysis as SolutionAnalysis
from qpoases import PyHessianType as HessianType # 'IDENTITY', 'INDEF', 'POSDEF', 'POSDEF_NULLSPACE', 'SEMIDEF', 'UNKNOWN', 'ZERO'
from qpoases import PySubjectToStatus as SubjectToStatus
from qpoases import PyBooleanType as BooleanType

def polyder(coeffs):
    if isinstance(coeffs,np.matrix):
        assert coeffs.shape[1] > 1
        a = np.asarray(coeffs)
        dim = a.shape[0]
        res_order = a.shape[1]-1
        res = np.matrix(np.empty([dim,res_order]))
        for k in range(a.shape[0]):
            res[k,:] = polyder(a[k,:].tolist())

        return res

    elif isinstance(coeffs,list):
        return np.polyder(coeffs[::-1])[::-1]

    else:
        assert False, "Must never happened"

class SplineNd(object):

    def __init__(self,dim,order,smin,smax):
        assert dim > 0 and isinstance(dim,int)
        assert order >= 0 and isinstance(order,int)
        assert smin < smax

        self.dim = dim
        self.order = order

        self.coeffs = np.matrix(np.empty([dim,order+1]))

        self.support = (float(smin), float(smax))

    def setWayPoint(self,point):
        assert point.shape[0] == self.dim

        self.coeffs[:,0] = point

    def setCoeffs(self,coeffs):
        assert coeffs.size == self.dim * (self.order+1)

        self.coeffs = coeffs.reshape(self.dim,self.order+1)

    def eval(self,s):

        ds = s-self.support[0]
        dsn = 1.
        res = self.coeffs[:,0].copy()
        for k in range(1,self.order+1):
            dsn *= ds
            res += self.coeffs[:,k]*dsn

        return res

    def evals(self,s,until_order):
        assert isinstance(until_order,int)
        assert  until_order >= 0 and until_order <= self.order

        res_l = []
        for k in range(until_order+1):
            res_l.append(np.matrix(self.coeffs_l[k][:,0]))

        ds = s - self.support[0]
        dsn = 1.
        for k in range(1,self.order+1):
            dsn *= ds
            for i in range(until_order+1):
                if k < self.coeffs_l[i].shape[1]:
                    res_l[i] += self.coeffs_l[i][:,k] * dsn

        return tuple(res_l)


    def diffEval(self,s,diff_order):
        assert diff_order >= 0 and diff_order <= self.order

        ds = s-self.support[0]
        dsn = 1.
        dcoeffs = self.coeffs.copy()
        for k in range(0,diff_order):
            dcoeffs = polyder(dcoeffs)

        res = np.matrix(np.zeros(self.coeffs[:,0].shape))
        for k in range(diff_order,self.order+1):
            res += dcoeffs[:,k]*dsn
            dsn *= ds

        return res

    def __compute__(self):

        self.coeffs_l = [self.coeffs.copy()]
        for k in range(1,self.order+1):
            dcoeffs = polyder(self.coeffs_l[k-1])
            self.coeffs_l.append(dcoeffs)



    def computeDifferentialConstraint(self,s,order=None):
        assert order <= self.order and order >= 0
        ds = s - self.support[0]
        coeff_basis = np.matrix(np.ones((1,self.order+1)))
        res = np.matrix(np.zeros((1,self.order+1)))
        pos = 0
        for k in range(order):
            pos += 1
            coeff_basis = polyder(coeff_basis)
        res[0,pos:pos+coeff_basis.shape[1]] = coeff_basis

        dsn = 1.
        for k in range(1,coeff_basis.shape[1]):
            dsn *= ds
            res[:,pos+k] *= dsn

        return res



class WayPointConstraint:
    ConstraintType = Enum("ConstraintType", "FREE FIXED")
    def __init__(self,vector,type):
        assert isinstance(type,self.ConstraintType)

        self.type = type
        self.vector = vector



class SplineTrajectoryNd(object):

    def __init__(self,dim,minimal_order,smin,smax):
        assert dim > 0 and isinstance(dim, int)
        assert minimal_order >= 2 and isinstance(dim, int)
        assert smin < smax

        self.dim = dim
        self.minimal_order = minimal_order

        self.support = (float(smin), float(smax))

        self.splines = []
        self.splines_support = list(self.support)
        self.way_points = [np.matrix(np.empty([self.dim,1]))]*2
        self.way_points_constraints = [([None]*(self.minimal_order-1)),([None]*(self.minimal_order-1))]

    def setInitialWayPoint(self,point):
        assert point.shape[0] == self.dim

        self.way_points[0] = np.asmatrix(point).copy()

    def setFinalWayPoint(self,point):
        assert point.shape[0] == self.dim

        self.way_points[-1] = np.asmatrix(point).copy()

    def numWayPoints(self): return len(self.way_points)

    def addWayPoint(self,point,s):
        assert s >= self.support[0] and s <= self.support[1]
        assert point.shape == (self.dim,1)

        if s == self.support[0]: self.setInitialWayPoint(point); id = 0
        elif s == self.support[1]: self.setFinalWayPoint(point); id = self.numWayPoints()-1
        elif s in self.splines_support:
            id = self.splines_support.index(s)
            self.way_points[id] = point.copy()
        else:
            indexes = [k for k, val in enumerate(self.splines_support) if val >= s]
            id = indexes[0]

            self.way_points.insert(id,point.copy())
            self.splines_support.insert(id,s)
            self.way_points_constraints.insert(id,[None]*(self.minimal_order-1))

        return id

    def addWayPointConstraint(self,way_point,way_point_constraint,diff_order):

        assert isinstance(way_point_constraint,WayPointConstraint)
        vector = way_point_constraint.vector
        assert vector.shape == (self.dim,1)
        assert diff_order >= 1

        if isinstance(way_point,int):
            id = way_point
        elif isinstance(way_point,np.matrix):
            assert way_point.shape == (self.dim,1)
            try:
                id = [np.array_equal(way_point, x) for x in self.way_points].index(True)

            except:
                raise ValueError('Invalid way_point argument')

        else:
            raise ValueError('Invalid way_point argument')

        constraints = self.way_points_constraints[id]
        constraint_id = diff_order-1
        if constraint_id < len(constraints):
            constraints[constraint_id] = copy(way_point_constraint)
        else:
            raise ValueError('This case is not handled yet')

    def __initSolver(self,arg_dim,num_in):

        self.solver = SQProblem(arg_dim, num_in);  # , HessianType.POSDEF SEMIDEF
        self.options = Options();
        self.options.setToReliable();
        self.solver.setOptions(self.options);

    def compute(self):
        self.splines = []
        num_variables = 0
        num_equalities = 0
        num_inequalities = 0

        self.spline_var_dim = []
        self.spline_var_pos = []
        self.spline_const_dim = []
        self.spline_const_pos = []


        spline_var_pos = 0

        for k in range(self.numWayPoints()-1):

            smin = self.splines_support[k]
            smax = self.splines_support[k+1]

            constraints = self.way_points_constraints[k]

            spline_position_order = 1 # The spline must pass by two points
            spline_differential_order = 0

            # We must ensure the continuity with the next spline
            if k < self.numWayPoints()-2:
                spline_differential_order += self.minimal_order - 1

            # In addition, we must add the derivative constraints which concerns the current point
            spline_additional_differential_order = 0

            for i,cons in enumerate(constraints):
                if cons is not None:
                    spline_additional_differential_order += 1
                    if cons.type == WayPointConstraint.ConstraintType.FREE:
                        num_variables += 1


            # A special care for the last way points
            if(k == self.numWayPoints()-2):
                next_constraints = self.way_points_constraints[-1]
                for i, cons in enumerate(next_constraints):
                    if cons is not None:
                        spline_additional_differential_order += 1
                        if cons.type == WayPointConstraint.ConstraintType.FREE:
                            num_variables += 1


            spline_differential_order += spline_additional_differential_order
            spline_order = spline_position_order + spline_differential_order

            spline_order = max(self.minimal_order,spline_order)


            spline_var_dim = (spline_order + 1) * self.dim
            num_variables += spline_var_dim

            spline_const_dim = (spline_order + 1) * self.dim
            num_equalities += spline_const_dim

            # Create spline
            spline = SplineNd(self.dim,spline_order,smin,smax)
            spline.setWayPoint(self.way_points[k])
            self.splines.append(spline)

            # Update spline positions and dimensions
            self.spline_var_dim.append(spline_var_dim)
            self.spline_var_pos.append(spline_var_pos)
            spline_var_pos += spline_var_dim

        # Set up the problem
        A_eq = np.matrix((np.zeros((num_equalities,num_variables))))
        self.A_eq = A_eq
        b_eq = np.matrix((np.zeros((num_equalities,1))))
        self.b_eq = b_eq

        A_in = np.matrix((np.zeros((num_inequalities,num_variables))))
        self.A_in = A_in
        b_in = np.matrix((np.zeros((num_inequalities,1))))
        self.b_in = b_in

        num_splines = len(self.splines)
        eq_const_id = 0
        for k,spline in enumerate(self.splines):
            spline_var_pos = self.spline_var_pos[k]
            spline_var_dim = self.spline_var_dim[k]

            # The spline must pass by two points
            points = (self.way_points[k],self.way_points[k+1])
            support = spline.support

            for p_id,point in enumerate(points):
                var_pos = spline_var_pos
                for i in range(self.dim):
                    slice = range(var_pos, var_pos + spline.order+1)
                    res = spline.computeDifferentialConstraint(support[p_id],0)
                    A_eq[eq_const_id,slice] = res
                    b_eq[eq_const_id] = point[i]

                    var_pos += spline.order+1
                    eq_const_id += 1


            # The spline has some continuity constraints
            if k < num_splines-1:
                next_spline = self.splines[k+1]
                next_spline_var_pos = self.spline_var_pos[k+1]
                next_spline_var_dim = self.spline_var_dim[k+1]
                s1 = support[1]

                #continuity_eq_const_id = eq_const_id
                for diff_order in range(1,self.minimal_order):
                    var_pos = spline_var_pos
                    next_var_pos = next_spline_var_pos
                    for i in range(self.dim):
                        slice = range(var_pos, var_pos + spline.order + 1)
                        res = spline.computeDifferentialConstraint(s1, diff_order)
                        A_eq[eq_const_id, slice] = res

                        next_slice = range(next_var_pos, next_var_pos + next_spline.order + 1)
                        res = next_spline.computeDifferentialConstraint(s1, diff_order)
                        A_eq[eq_const_id,next_slice] = -res

                        b_eq[eq_const_id] = 0.

                        var_pos += spline.order + 1
                        next_var_pos += next_spline.order + 1
                        eq_const_id += 1




            # The spline may have some initial constraints
            constraints = self.way_points_constraints[k]
            for cons_id,cons in enumerate(constraints):
                if cons is not None:
                    var_pos = spline_var_pos
                    vector = cons.vector
                    for i in range(self.dim):
                        slice = range(var_pos, var_pos + spline.order + 1)
                        res = spline.computeDifferentialConstraint(support[0], cons_id+1)
                        A_eq[eq_const_id, slice] = res
                        b_eq[eq_const_id] = vector[i]

                        var_pos += spline.order + 1
                        eq_const_id += 1

            # The last spline may have terminal constraints
            if k == num_splines-1:
                constraints = self.way_points_constraints[k+1]
                for cons_id, cons in enumerate(constraints):
                    if cons is not None:
                        var_pos = spline_var_pos
                        vector = cons.vector
                        for i in range(self.dim):
                            slice = range(var_pos, var_pos + spline.order + 1)
                            res = spline.computeDifferentialConstraint(support[1], cons_id + 1)
                            A_eq[eq_const_id, slice] = res
                            b_eq[eq_const_id] = vector[i]

                            var_pos += spline.order + 1
                            eq_const_id += 1

            if k < num_splines-1:
                next_spline = self.splines[k+1]
                next_spline_dim = self.spline_var_dim[k+1]
                snext = self.splines_support[k+1]
                # Add equality constraint
                for i in range(self.minimal_order+1):
                    res = spline.computeDifferentialConstraint(snext,i)




        sol_naive = np.linalg.pinv(A_eq) * b_eq
        # Solve with qpOASES

        for k, spline in enumerate(self.splines):
            spline_var_pos = self.spline_var_pos[k]
            spline_var_dim = self.spline_var_dim[k]

            spline.setCoeffs(sol_naive[spline_var_pos:spline_var_pos+spline_var_dim])
            spline.__compute__()




        self.num_variables = num_variables
        self.num_equalities = num_equalities
        self.num_inequalities = num_inequalities

    def findSpline(self,s):

        if s <= self.support[0]:
            id = 0
        elif s >= self.support[1]:
            id = len(self.splines) - 1

        else:
            indexes = [k for k, val in enumerate(self.splines_support) if s >= val]
            id = indexes[-1]

        return id

    def eval(self,s):
        id = self.findSpline(s)

        assert id >= 0

        return self.splines[id].eval(s)

    def evals(self,s,until_order):
        id = self.findSpline(s)
        assert id >= 0

        return self.splines[id].evals(s,until_order)

    def diffEval(self,s,order):
        id = self.findSpline(s)

        assert id >= 0
        assert order >= 0

        return self.splines[id].diffEval(s,order)

# Test of splines
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    dim = 2
    minimal_order = 3
    smin = 0.
    smax = 1.
    traj = SplineTrajectoryNd(dim,minimal_order,smin,smax)

    P0 = np.matrix(np.zeros((dim,1)))
    P1 = np.matrix(np.ones((dim,1)))
    P2 = 0.5*(P0+P1) + np.matrix([0.,1.]).T
    P3 = np.matrix(np.random.rand(dim,1))

    traj.setInitialWayPoint(P0)
    traj.setFinalWayPoint(P1)
    traj.addWayPoint(P2,0.5)
    #traj.addWayPoint(P3,0.75)

    normal = np.matrix([0.,1.]).T
    horizontal = np.matrix([1.,0.]).T
    constraint0_vel = WayPointConstraint(5*normal,WayPointConstraint.ConstraintType.FIXED)
    constraint0_acc = WayPointConstraint(1.*normal,WayPointConstraint.ConstraintType.FIXED)
    constraint_middle = WayPointConstraint(horizontal,WayPointConstraint.ConstraintType.FIXED)
    constraint1_vel = WayPointConstraint(-3*normal,WayPointConstraint.ConstraintType.FIXED)
    constraint1_acc = WayPointConstraint(-1.*normal,WayPointConstraint.ConstraintType.FIXED)

    traj.addWayPointConstraint(0,constraint0_vel,1)
    traj.addWayPointConstraint(0,constraint0_acc,2)

    traj.addWayPointConstraint(1,constraint_middle,1)
    traj.addWayPointConstraint(1,constraint_middle,2)

    traj.addWayPointConstraint(traj.numWayPoints()-1,constraint1_vel,1)
    traj.addWayPointConstraint(traj.numWayPoints()-1,constraint1_acc,2)

    traj.compute()

    fig = plt.figure()
    N = 100
    sline = np.linspace(smin,smax,N)
    data = np.matrix(np.empty((dim,N)))

    for k in range(N):
        s = sline[k]
        data[:,k] = traj.eval(s)

    # Plot markers
    for point in traj.way_points:
        plt.plot(point[0],point[1],marker="+",color="r",markersize=14)

    plt.plot(data[0,:].T,data[1,:].T,color="k")








