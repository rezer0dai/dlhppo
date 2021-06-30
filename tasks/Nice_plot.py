# COPY PASTE ( + little updates ) from : 
# lost github reference, if you know it please let me know to update it here!

import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.linalg import expm

GRID_DIM = 20

def plot_proxy(s, g, v, f, a=None):
    results = {}
    s = np.array(s)

    results['x'] = s[:,0]
    results['y'] = s[:,1]
    results['z'] = s[:,2]
    results['phi'] = s[:,3]
    results['theta'] = s[:,4]
    results['psi'] = s[:,5]
    results['x_velocity'] = s[:,6]
    results['y_velocity'] = s[:,7]
    results['z_velocity'] = s[:,8]
    results['phi_velocity'] = s[:,9]
    results['theta_velocity'] = s[:,10]
    results['psi_velocity'] = s[:,11]
    results['r1'] = s[:,12]
    results['r2'] = s[:,17]
    results['dist'] = s[:,19]
    results['time'] = list(range(len(s)))
    results['rotor_speed1'] = s[1:,13]
    results['rotor_speed2'] = s[1:,14]
    results['rotor_speed3'] = s[1:,15]
    results['rotor_speed4'] = s[1:,16]
    plot(results, g, v, f, a)

# Patch to 3d axis to remove margins around x, y and z limits.
# Taken from here: https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###


# Rotation using euler angles from here:
# https://gist.github.com/machinaut/29d0e21b544b4a36082c761c439144d6
def rotateByEuler(points, xyz):
    ''' Rotate vector v (or array of vectors) by the euler angles xyz '''
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    for theta, axis in zip(xyz, np.eye(3)):
        points = np.dot(np.array(points), expm(np.cross(np.eye(3), axis*-theta)).T)
    return points




def plot(results, g, values, future, a=None, fancy=True):

    # Set up axes grid. ###############################################################
    fig = plt.figure(figsize=(20,15))
    ax1 = plt.subplot2grid((20, 40), (0, 0), colspan=24, rowspan=20, projection='3d')
    ax2 = plt.subplot2grid((20, 40), (1, 28), colspan=12, rowspan=4)
    ax3 = plt.subplot2grid((20, 40), (6, 28), colspan=12, rowspan=4)
    ax4 = plt.subplot2grid((20, 40), (11, 28), colspan=12, rowspan=4)
    ax5 = plt.subplot2grid((20, 40), (16, 28), colspan=12, rowspan=4)


    # Plot 3d trajectory and copter. ##################################################
    c = 0.0
    plt.rcParams['grid.color'] = [c, c, c, 0.075]
    mpl.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.xmargin'] = 0


    plotLimitXY = GRID_DIM // 2
    plotLimitZ = GRID_DIM

    quadSize = 0.5
    nPointsRotor = 15
    pointsQuadInitial = [[-quadSize, -quadSize, 0], [-quadSize, quadSize, 0], [quadSize, quadSize, 0], [quadSize, -quadSize, 0]]
    pointsRotorInitial = np.vstack(( np.sin(np.linspace(0., 2.*np.pi, nPointsRotor)),
                                     np.cos(np.linspace(0., 2.*np.pi, nPointsRotor)),
                                     np.repeat(0.0, nPointsRotor))).T * quadSize * 0.8

    # Create 3d plot.

    #ax = fig.gca(projection='3d')
    ax1.view_init(12, -55)
    ax1.dist = 7.6

    # Plot trajectories projected.
    xLimited = [x for x in results['x'] if np.abs(x) <= plotLimitXY]
    yLimited = [y for y in results['y'] if np.abs(y) <= plotLimitXY]
    zLimited = [z for z in results['z'] if z <= plotLimitZ]
    l = min(len(xLimited), len(yLimited))
    ax1.plot(xLimited[0:l], yLimited[0:l], np.repeat(0.0, l), c='darkgray', linewidth=0.9)
    l = min(len(xLimited), len(zLimited))
    ax1.plot(xLimited[0:l], np.repeat(plotLimitXY, l), zLimited[0:l], c='darkgray', linewidth=0.9)
    l = min(len(yLimited), len(zLimited))
    ax1.plot(np.repeat(-plotLimitXY, l), yLimited[0:l], zLimited[0:l], c='darkgray', linewidth=0.9)

    # Plot trajectory 3d.
    ax1.plot(results['x'], results['y'], results['z'], c='gray', linewidth=0.5)

    # Plot copter.
    nTimesteps = len(results['x'])
    # Colors from here: https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    colors = np.array([ [230, 25, 75, 255],
                        [60, 180, 75, 255],
                        [255, 225, 25, 255],
                        [0, 130, 200, 255]]) / 255.

    for t in range(nTimesteps):
        # Plot copter position as dot on trajectory for each full second. ******
        if results['time'][t]%1.0 <= 0.025 or results['time'][t]%1.0 >= 0.975:
            ax1.scatter([results['x'][t]], [results['y'][t]], [results['z'][t]], s=5, c=[[0., 0., 0., 0.3]])
        alpha1 = 0.96*np.power(t/nTimesteps, 20)+0.04
        alpha2 = 0.5 * alpha1
        # Plot frame. **********************************************************
        if fancy or t == nTimesteps -1:
            # Rotate frame it.
            pointsQuad = rotateByEuler(pointsQuadInitial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
            # Move it.
            pointsQuad += np.array([results['x'][t], results['y'][t], results['z'][t]])
        # Plot frame projections for last time step.
        if t == nTimesteps -1:
            # Z plane.
            if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['y'][t]) <= plotLimitXY:
                ax1.plot(pointsQuad[[0,2], 0], pointsQuad[[0,2], 1], [0., 0.], c=[0., 0., 0., 0.1])
                ax1.plot(pointsQuad[[1,3], 0], pointsQuad[[1,3], 1], [0., 0.], c=[0., 0., 0., 0.1])
            # Y plane.
            if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                ax1.plot(pointsQuad[[0,2], 0], [plotLimitXY, plotLimitXY], pointsQuad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax1.plot(pointsQuad[[1,3], 0], [plotLimitXY, plotLimitXY], pointsQuad[[1,3], 2], c=[0., 0., 0., 0.1])
            # X plane.
            if np.abs(results['y'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                ax1.plot([-plotLimitXY, -plotLimitXY], pointsQuad[[0,2], 1], pointsQuad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax1.plot([-plotLimitXY, -plotLimitXY], pointsQuad[[1,3], 1], pointsQuad[[1,3], 2], c=[0., 0., 0., 0.1])
        # Plot frame for all other time steps.
        if fancy:
            ax1.plot(pointsQuad[[0,2], 0], pointsQuad[[0,2], 1], pointsQuad[[0,2], 2], c=[0., 0., 0., alpha2])
            ax1.plot(pointsQuad[[1,3], 0], pointsQuad[[1,3], 1], pointsQuad[[1,3], 2], c=[0., 0., 0., alpha2])

        # Plot rotors. *********************************************************
        # Rotate rotor.
        if fancy or t == nTimesteps -1:
            pointsRotor = rotateByEuler(pointsRotorInitial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
        # Move rotor for each frame point.
        for i, color in zip(range(4), colors):
            if fancy or t == nTimesteps -1:
                pointsRotorMoved = pointsRotor + pointsQuad[i]
            # Plot rotor projections.
            if t == nTimesteps -1:
                # Z plane.
                if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['y'][t]) <= plotLimitXY:
                    ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], pointsRotorMoved[:,1], np.repeat(0, nPointsRotor)))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # Y plane.
                if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                    ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], np.repeat(plotLimitXY, nPointsRotor), pointsRotorMoved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # X plane.
                if np.abs(results['y'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                    ax1.add_collection3d(Poly3DCollection([list(zip(np.repeat(-plotLimitXY, nPointsRotor), pointsRotorMoved[:,1], pointsRotorMoved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
            # Outline.
            if t == nTimesteps-1:
                ax1.plot(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2], c=color[0:3].tolist()+[alpha1], label='Rotor {:g}'.format(i+1))
            elif fancy:
                ax1.plot(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2], c=color[0:3].tolist()+[alpha1])
            # Fill.
            if fancy or t == nTimesteps -1:
                ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2]))], facecolor=color[0:3].tolist()+[alpha2]))

#    ax1.scatter([gx], [gy], [gz], s=500, marker="*")
    ax1.scatter(g[:, 0], g[:, 1], g[:, 2], s=500, marker="*")
    if a is not None: ax1.scatter(a[:, 0], a[:, 1], a[:, 2], s=500, marker="x")

    ax1.legend(bbox_to_anchor=(0.0 ,0.0 , 0.95, 0.85), loc='upper right')
    c = 'r'
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_xlim(-plotLimitXY, plotLimitXY)
    ax1.set_ylim(-plotLimitXY, plotLimitXY)
    ax1.set_zlim(0, plotLimitZ)
    ax1.set_xticks(np.arange(-plotLimitXY, plotLimitXY+2, 2))
    ax1.set_yticks(np.arange(-plotLimitXY, plotLimitXY+2, 2))
    ax1.set_zticks(np.arange(0, plotLimitZ+2, 2))

    # Plot copter angles.
    ax2.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['phi']], label='$\\alpha_x$')
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['theta']], label='$\\alpha_y$')
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['psi']], label='$\\alpha_z$')
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_ylabel('$\\alpha$ [rad]')
    ax2.legend()

    # Plot copter velocities.
    ax3.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax3.plot(results['time'], results['x_velocity'], label='$V_x$')
    ax3.plot(results['time'], results['y_velocity'], label='$V_y$')
    ax3.plot(results['time'], results['z_velocity'], label='$V_z$')
    ax3.set_ylim(-20, 20)
    ax3.set_ylabel('V [$m\,s^{1}$]')
    ax3.legend()


    # Plot copter turn rates.
    ax4.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax4.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    ax4.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    ax4.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    ax4.set_ylim(-3, 3)
    ax4.set_ylabel('$\omega$ [$rad\,s^{1}$]')
    ax4.legend()

    # Plot dimensions
    ax5.plot([0,results['time'][-1]], [0,0], c=[0,0,0], linewidth=0.5)
    ax5.plot(results['time'], results['x'], label='x')
    ax5.plot(results['time'], results['y'], label='y')
    ax5.plot(results['time'], results['z'], label='z')
    ax5.set_ylabel('dim[3D]')
    ax5.legend()

    # Done :)
    plt.show()

    plt.figure(figsize=(20, 5))
    axa = plt.subplot2grid((10,10),(0,0), colspan = 4,rowspan = 10)
    axr = plt.subplot2grid((10,10),(0,5), colspan = 4,rowspan = 10)

    # Plot rotor speeds.
    axa.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    axa.plot(results['time'][1:], results['rotor_speed1'], label='A1/s')
    axa.plot(results['time'][1:], results['rotor_speed2'], label='A2/s')
    axa.plot(results['time'][1:], results['rotor_speed3'], label='A3/s')
    axa.plot(results['time'][1:], results['rotor_speed4'], label='A4/s')
    axa.set_xlabel('policy actions')
    axa.legend()

    # Plot euclidian distance.
    axr.plot(results['time'], -results['dist'])
    axr.set_xlabel('euclidian distance')

    plt.show()

    # Plot reward.
    plt.figure(figsize=(20, 5))
    axa = plt.subplot2grid((10,10),(0,0), colspan = 4,rowspan = 10)
    axr = plt.subplot2grid((10,10),(0,5), colspan = 4,rowspan = 10)

    axa.plot(results['time'], results['r1'], label='reward')
    axa.plot(results['time'], results['r2'], label='custom')
    axa.set_xlabel("REWARDS")
    axa.legend()

    axr.plot(list(range(len(values))), values, label='explorer')
    axr.plot(list(range(len(values))), future, label='target')
    axr.set_xlabel("Q-VALUES")
    axr.legend()

    plt.show()
