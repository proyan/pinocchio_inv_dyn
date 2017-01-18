import pinocchio as se3 
from pinocchio import SE3, rnea
from pinocchio.utils import *

import numpy as np
import matplotlib.pyplot as plt

class Struct():
  None

def checkTorqueLimoits(robot,q_t,v_t,a_t):
  model = robot.model
  data = robot.data

  num_points = q_t.shape[1]
  res = [None]*num_points

  tau_lb = -model.effortLimit[6:]
  tau_ub = model.effortLimit[6:]

  for k in range(num_points):
    q = q_t[:,k]
    v = v_t[:,k]
    a = a_t[:,k]
    tau = rnea(model,data,q,v,a)

    res[k] = (tau[6:] >= tau_lb).all() and (tau[6:] <= tau_ub).all()

  return res


def checkBounds(robot, q_t):
  q_lower = robot.model.lowerPositionLimit
  q_upper = robot.model.upperPositionLimit

  _,N = q_t.shape
  lb_activation = np.array(np.zeros(q_t.shape), dtype=bool)
  ub_activation = np.array(np.zeros(q_t.shape), dtype=bool)

  for n in range(7,int(q_t.shape[0])):
    for i in range(q_t.shape[1]):
      lb_constraint = -q_t[n,i] + q_lower[n,0]
      ub_constraint = q_t[n,i] - q_upper[n,0]
      if lb_constraint >= 0:
        lb_activation[n,i] = True
      if ub_constraint >= 0:
        ub_activation[n,i] = True

  if lb_activation.sum() != 0:
    print "One of the lower bounds is active"
  else:
    print "No violated lower bounds"

  if ub_activation.sum() != 0:
    print "One of the upper bounds is active"
  else:
    print "No violated upper bounds"

  for n in range(7,int(q_t.shape[0])):
    if lb_activation[n,:].sum() != 0:
      print "Joint",n," is reaching its lower bound limit :", q_lower[n]
      print "Min value reached :", q_t[n,:].min()

    if ub_activation[n,:].sum() != 0:
      print "Joint",n," is reaching its upper bound limit :", q_upper[n]
      print "Max value reached :", q_t[n,:].max()

  return lb_activation, ub_activation

def plotActivation(activation_l, fig_id = -1):
  N = len(activation_l)

  if fig_id == -1:
    fig = plt.figure()
  else:
    fig = plt.figure(fig_id)

  for k in range(N):
    plt.subplot('{}1{}'.format(N,k+1))
    plt.imshow(activation_l[k], cmap='Greys')

  return fig
    
def computeResultTrajectory(robot, t_t, q_t, v_t, a_t):

  model = robot.model
  data = robot.data
  
  N = q_t.shape[1]

  ZMP_t = np.matrix(np.zeros([3,N]))
  waist_t = np.matrix(np.zeros([3,N]))
  pcom_t = np.matrix(np.empty([3,N]))
  vcom_t = np.matrix(np.empty([3,N]))
  acom_t = np.matrix(np.empty([3,N]))
  tau_t = np.matrix(np.empty([robot.nv,N]))

  wrench_t = np.matrix(np.empty([6,N]))

  waist_orientation_t = np.matrix(np.empty([3,N]))

  # Sample end effector traj
  ee_t = dict()
  RF_t = []
  ee_t["rf"] = (RF_t)
  LF_t = []
  ee_t["lf"] = (LF_t)
  RH_t = []
  ee_t["rh"] = (RH_t)
  LH_t = []
  ee_t["lh"] = (LH_t)

  Mee = dict()
  Mee["rf"] = robot.Mrf
  Mee["lf"] = robot.Mlf
  Mee["rh"] = robot.Mrh
  Mee["lh"] = robot.Mlh

  ee_names = list(ee_t.viewkeys())

  for k in range(N):
    q = q_t[:,k]
    v = v_t[:,k]
    a = a_t[:,k]

    #M = robot.mass(q)
    #b = robot.biais(q,v)

    #robot.dynamics(q,v,0*v)
    se3.rnea(model,data,q,v,a)
    #se3.forwardKinematics(model,data,q)
    #robot.computeJacobians(q)
    
    pcom, vcom, acom = robot.com(q,v,a)

    # Update EE placements
    for ee in ee_names:
      ee_t[ee].append(Mee[ee](q,update_kinematics=False).copy())


    # Update CoM data
    pcom_t[:,k] = pcom
    vcom_t[:,k] = vcom
    acom_t[:,k] = acom

    #oXi_s = robot.data.oMi[1].inverse().np.T
    #phi0 = oXi_s * (M[:6,:] * a + b[:6])
    tau_t[:,k] = data.tau
    phi0 = data.oMi[1].act(se3.Force(data.tau[:6]))

    wrench_t[:,k] = phi0.vector

    forces = wrench_t[:3,k]
    torques = wrench_t[3:,k]

    ZMP_t[0,k] = -torques[1]/forces[2]
    ZMP_t[1,k] = torques[0]/forces[2]

    waist_t[:,k] = robot.data.oMi[1].translation
    waist_orientation_t[:,k] = matrixToRpy(robot.data.oMi[1].rotation)

  result = Struct()

  result.t_t = t_t
  result.ZMP_t = ZMP_t
  result.waist_t = waist_t
  result.waist_orientation_t = waist_orientation_t

  result.pcom_t = pcom_t
  result.vcom_t = vcom_t
  result.acom_t = acom_t

  result.wrench_t = wrench_t
  result.tau_t = tau_t

  result.q_t = q_t
  result.v_t = v_t
  result.a_t = a_t

  result.ee_t = ee_t


  return result

def writeOpenHRPConfig(robot,q,t=0.):
  def write_vector(vector, delim):
    line = ""
    for k in range(vector.shape[0]):
      line += delim + str(vector[k,0])

    return line

  line = str(t)
  delim = " "
  q_openhrp = []

  # RL
  line += write_vector(q[robot.r_leg], delim)
  q_openhrp.append(q[robot.r_leg].tolist())

  # LL
  line += write_vector(q[robot.l_leg], delim)
  q_openhrp.append(q[robot.l_leg].tolist())

  # Chest
  line += write_vector(q[robot.chest], delim)
  q_openhrp.append(q[robot.chest].tolist())

  # Head
  line += write_vector(q[robot.head], delim)
  q_openhrp.append(q[robot.head].tolist())

  # RA
  line += write_vector(q[robot.r_arm], delim)
  q_openhrp.append(q[robot.r_arm].tolist())

  # LA
  line += write_vector(q[robot.l_arm], delim)
  q_openhrp.append(q[robot.l_arm].tolist())

  # Fingers
  line += write_vector(np.matrix(np.zeros([10, 1])), delim)
  q_openhrp.append(np.matrix(np.zeros([10, 1])).tolist())

  q_openhrp = np.matrix(np.concatenate(q_openhrp))

  return q_openhrp,line

def generateOpenHRPMotion(robot, data, path, project_name):
  def write_vector(vector, delim):
    line = ""
    for k in range(vector.shape[0]):
      line += delim + str(vector[k,0])

    return line
  
  timeline = data.t_t - data.t_t[0]
  N = len(timeline)
  dt = 0.005
  timeline = np.linspace(0., (N-1)*dt, N) 
  delim = " "
  eol = "\n"

  filename_prefix = path + '/' + project_name
  
  ## ZMP trajectory ##
  filename_zmp = filename_prefix + '.zmp'
  file_zmp = open(filename_zmp, "w")

  ZMP_waist = data.ZMP_t - data.waist_t

  for k in range(N):
    line = str(timeline[k])
    for i in range(3):
      line += delim + str(ZMP_waist[i,k])

    line += eol
    file_zmp.write(line)

  file_zmp.close()

  ## Posture trajectory ##
  filename_pos = filename_prefix + '.pos'
  file_pos = open(filename_pos, "w")

  qout_l = []
  for k in range(N):
    line = str(timeline[k])
    q_openhrp = []

    # RL
    line += write_vector(data.q_t[robot.r_leg,k], delim)
    q_openhrp.append(data.q_t[robot.r_leg,k].tolist())

    # LL
    line += write_vector(data.q_t[robot.l_leg,k], delim)
    q_openhrp.append(data.q_t[robot.l_leg,k].tolist())

    # Chest
    line += write_vector(data.q_t[robot.chest,k], delim)
    q_openhrp.append(data.q_t[robot.chest,k].tolist())

    # Head
    line += write_vector(data.q_t[robot.head,k], delim)
    q_openhrp.append(data.q_t[robot.head,k].tolist())

    # RA
    line += write_vector(data.q_t[robot.r_arm,k], delim)
    q_openhrp.append(data.q_t[robot.r_arm,k].tolist())

    # LA
    line += write_vector(data.q_t[robot.l_arm,k], delim)
    q_openhrp.append(data.q_t[robot.l_arm,k].tolist())

    # Fingers
    line += write_vector(np.matrix(np.zeros([10,1])), delim)
    q_openhrp.append(np.matrix(np.zeros([10,1])).tolist())

    line += eol
    file_pos.write(line)

    qout_l.append(np.matrix(np.concatenate(q_openhrp)))

  file_pos.close()

  ## Waist orientation
  filename_hip = filename_prefix + '.hip'
  file_hip = open(filename_hip, "w")

  for k in range(N):
    line = str(timeline[k])
    
    line += write_vector(data.waist_orientation_t[:,k], delim)

    line += eol
    file_hip.write(line)
  file_hip.close()

  return qout_l


def writeKinematicsData(robot, data, path, project_name):
  def write_vector(vector, delim):
    line = ""
    for k in range(vector.shape[0]):
      line += delim + str(vector[k,0])

    return line
  
  timeline = data.t_t - data.t_t[0]
  N = len(timeline)
  dt = 0.005
  timeline = np.linspace(0., (N-1)*dt, N) 
  delim = " "
  eol = "\n"

  filename_prefix = path + '/' + project_name
  
  ## Config trajectory ##
  filename_config = filename_prefix + '_config.csv'
  file_config = open(filename_config, "w")

  for k in range(N):
    line = str(timeline[k])
    line += write_vector(data.q_t[:,k], delim)
    line += eol
    file_config.write(line)

  file_config.close()

  ## Vel trajectory ##
  filename_vel = filename_prefix + '_vel.csv'
  file_vel = open(filename_vel, "w")

  for k in range(N):
    line = str(timeline[k])
    line += write_vector(data.v_t[:,k], delim)
    line += eol
    file_vel.write(line)

  file_vel.close()

  ## Acc trajectory ##
  filename_acc = filename_prefix + '_acc.csv'
  file_acc = open(filename_acc, "w")

  for k in range(N):
    line = str(timeline[k])
    line += write_vector(data.a_t[:,k], delim)
    line += eol
    file_acc.write(line)

  file_acc.close()

