import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


cam_x = np.arange(0.0, 0.01, 0.1)
cam_y = np.arange(1.9, 1.91, 0.1)
# cam_x = np.arange(0, 0.11, 0.1)
# cam_y = np.arange(0, 0.11, 0.1)

x_lim = [0.0,9.0]
# x_lim = [7.0, 7.0]
y_lim = [0.0,4.5]
# y_lim = [4.5, 4.5]
z_lim = [0.2,0.2]
# x_lim = [-3.,4.1]
# y_lim = [-2.2,2.3]

x_span = np.arange(x_lim[0], x_lim[1]+0.001, 0.25)
y_span = np.arange(y_lim[0], y_lim[1]+0.001, 0.25)
z_span = np.arange(z_lim[0], z_lim[1]+0.001, 0.25)
# yaw_span = np.arange(0, 2*np.pi, np.pi/3)

x_points, y_points = np.meshgrid(x_span, y_span)

sig_u = [3]
# resolution = 768.
# m = 160.
# u0 = resolution/2.
# v0 = resolution/2.
coeffs1 = [-0.003125, 0, 0.001029, 0, 0.007671, 0, 0.013237, 0, 1.492357]
coeffs2 = [-0.003934, 0, 0.002062, 0, 0.010611, 0, 0.017052, 0, 1.495884]
mu1 = 157.1979;     mv1 = 157.2336;     u01 = 385.21;       v01 = 385.24;
m1 = (mu1+mv1)/2.
mu2 = 156.4238;     mv2 = 156.4208;     u02 = 385.14;       v02 = 385.32;
m2 = (mu2+mv2)/2.


def calculate_error(cam_pos = (0.5, 0.5), yaw=None, sigma_u=1):
    if not yaw:
        yaw = np.arctan2((cam_pos[1]),(cam_pos[0]))       # rotate to align cameras

    # var_pixels = np.diag([sigma_u ** 2, sigma_u ** 2])
    # var_xcam = np.diag([sigma_u ** 2 / m ** 2, sigma_u ** 2 / m ** 2])

    directory = '/home/otalab/Python-dir/Fisheye_error/Errors/sigma' + str(sigma_u) + 'px/VeryDetailed/'

    T_cam1 = np.array([[4.5-cam_pos[0]], [2.25-cam_pos[1]], [0]])
    R1 = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    H_cam1 = np.array([[np.cos(yaw), -np.sin(yaw), 0, 4.5-cam_pos[0]], [np.sin(yaw), np.cos(yaw), 0, 2.25-cam_pos[1]], [0, 0, 1, 0], [0, 0, 0, 1]])

    T_cam2 = np.array([[4.5+cam_pos[0]], [cam_pos[1]+2.25], [0]])
    R2 = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    H_cam2 = np.array([[np.cos(yaw), -np.sin(yaw), 0, 4.5+cam_pos[0]], [np.sin(yaw), np.cos(yaw), 0, cam_pos[1]+2.25], [0, 0, 1, 0], [0, 0, 0, 1]])

    H = np.dot(np.linalg.inv(H_cam1),H_cam2)

    baseline = np.sqrt(H[0,3]**2 + H[1,3]**2 +H[2,3]**2)
    d_unit = np.array([1,0,0])
    D = H[0:3,3]/baseline
    if (np.abs(D-d_unit) < 1e-3).all():
        R_align = np.eye(3)
    else:
        V = np.cross(D, d_unit)
        sine = np.sqrt(V[0]**2+V[1]**2+V[2]**2)
        cosine = np.dot(D,d_unit)
        Vx = np.array([[0,-V[2], V[1]], [V[2], 0, -V[0]], [-V[1], V[0], 0]])
        R_align = np.eye(3) + Vx + np.dot(Vx,Vx)*(1-cosine)/sine**2
    R_align2 = np.dot(R_align,np.dot(R1.transpose(),R2))

    # print H_cam1
    # print H_cam2
    # print R_align
    # print R_align2
    # print baseline

    # X_rand = np.zeros((len(x_span), len(y_span)))
    # Y_rand = np.zeros((len(x_span), len(y_span)))
    # Z_rand = np.zeros((len(x_span), len(y_span)))

    SIGMA = np.zeros((len(x_span), len(y_span), len(z_span)))

    # fig_fisheye = plt.figure()
    # ax_fisheye = fig_fisheye.add_subplot('111')

    # fig = plt.figure()
    # ax = Axes3D(fig)

    # fig2 = plt.figure()
    # ax2 = Axes3D(fig2)

    fig3 = plt.figure()
    ax3 = Axes3D(fig3)

    # fig4 = plt.figure()
    # ax4 = Axes3D(fig4)

    # fig_beta = plt.figure()
    # ax_beta = fig_beta.add_subplot('111')

    for i in range(len(x_span)):
        for j in range(len(y_span)):
            for k in range(len(z_span)):
                X = np.array([[x_span[i]], [y_span[j]], [z_span[k]], [1.]])
                # Cam 1
                X1 = np.dot(np.linalg.inv(H_cam1), X)

                theta = np.arctan2(np.sqrt(X1[0,0]**2+X1[1,0]**2), X1[2,0])
                r = np.dot(coeffs1, [theta**9, 0, theta**7, 0 ,theta**5, 0, theta**3, 0, theta]) + 1e-12

                phi = np.arctan2(X1[1,0], X1[0,0])
                x_c = r*np.cos(phi)
                y_c = r*np.sin(phi)

                u = mu1*x_c + u01
                v = mv1*y_c + v01

                # ax_fisheye.scatter(u,v,c='r',marker='.')

                # Random for noise
                # u_rand = u + sigma_u*np.random.randn(50,)
                # v_rand = v + sigma_u*np.random.randn(50,)
                #
                # x_c_rand = (u_rand-u01)/mu1
                # y_c_rand = (v_rand-v01)/mv1
                # phi_rand = np.arctan2(y_c_rand, x_c_rand)
                # r_rand = np.sqrt(x_c_rand*x_c_rand + y_c_rand*y_c_rand)
                #
                # theta_rand = []
                # for r_sample in r_rand:
                #     p = coeffs1[:]
                #     p.append(-r_sample)
                #     # p.append(-r_rand)
                #     thetas = np.roots(p)
                #     for thet in thetas:
                #         if np.imag(thet) == 0:
                #             if 0 < np.real(thet) < np.pi / 2 + 0.1:
                #                 theta_rand.append(np.double(np.real(thet)))
                #                 # theta_rand=np.double(np.real(thet))
                #                 break
                #     else:
                #         print "Unable to find theta", thetas
                #         print r_sample
                #
                # theta_rand = np.array(theta_rand)
                #
                # u_cam = np.array([[np.sin(theta_rand) * np.cos(phi_rand)], [np.sin(theta_rand) * np.sin(phi_rand)], [np.cos(theta_rand)]])
                # rect_r = np.zeros(u_cam.shape)
                # for ind in range(u_cam.shape[2]):
                #     vec = u_cam[:,:,ind]
                #     rect_r[:,:,ind] = np.dot(R_align,vec)
                #
                # #
                # # rect_r = np.dot(R_align,u_cam)
                # # psi_rand = np.arcsin(rect_r[0,0,:])
                # # beta_rand = np.arctan2(rect_r[1,0,:], rect_r[2,0,:])
                # psi_rand = np.arcsin(rect_r[0, 0, :])
                # beta_rand = np.arctan2(rect_r[1, 0, :], rect_r[2, 0, :])
                ######################

                rect_X = np.dot(R_align,X1[0:3,:])
                norm = np.sqrt(rect_X[0,0]**2 + rect_X[1,0]**2 + rect_X[2,0]**2)
                beta = np.arctan2(rect_X[1,0]/norm, rect_X[2,0]/norm)
                psi = np.arcsin(rect_X[0,0]/norm)

                J_theta = coeffs1[8] + 3*coeffs1[6]*theta**2 + 5*coeffs1[4]*theta**4 + 7*coeffs1[2]*theta**6 + 9*coeffs1[0]*theta**8
                var_theta_phi = np.diag([sigma_u**2/(m1*J_theta)**2, sigma_u**2/(m1*r)**2])

                c_theta = np.cos(theta)
                s_theta = np.sin(theta)
                c_phi = np.cos(phi)
                s_phi = np.sin(phi)
                c_psi = np.cos(psi)
                s_psi = np.sin(psi)
                c_beta = np.cos(beta)
                s_beta = np.sin(beta)

                J_ucam = np.array([[c_theta*c_phi, -s_theta*s_phi], [c_theta*s_phi, s_theta*c_phi], [-s_theta, 0]])

                var_ucam = np.dot(np.dot(J_ucam, var_theta_phi), J_ucam.transpose())

                var_R = np.dot(np.dot(R_align, var_ucam), R_align.transpose())

                # J_beta_psi = np.array([[c_psi, -s_psi*s_beta, -s_psi*c_beta], [0, c_beta/c_psi, -s_beta/c_psi]])
                J_beta_psi = np.array([[c_psi, 0], [-s_psi*s_beta, c_psi*c_beta], [-s_psi*c_beta, -c_psi*s_beta]])
                J_beta_psi = np.linalg.pinv(J_beta_psi)

                var_beta_psi = np.dot(np.dot(J_beta_psi, var_R), J_beta_psi.transpose())

                # Cam 2 (right camera)
                X2 = np.dot(np.linalg.inv(H_cam2), X)

                theta2 = np.arctan2(np.sqrt(X2[0,0]**2+X2[1,0]**2), X2[2,0])
                r2 = np.dot(coeffs2, [theta2**9, 0, theta2**7, 0 ,theta2**5, 0, theta2**3, 0, theta2]) + 1e-12

                phi2 = np.arctan2(X2[1,0], X2[0,0])
                x_c2 = r2*np.cos(phi2)
                y_c2 = r2*np.sin(phi2)

                u2 = mu2*x_c2 + u02
                v2 = mv2*y_c2 + v02

                # Random for noise
                # u_rand2 = u2 + sigma_u * np.random.randn(50, )
                # v_rand2 = v2 + sigma_u * np.random.randn(50, )
                #
                # x_c_rand2 = (u_rand2 - u02) / mu2
                # y_c_rand2 = (v_rand2 - v02) / mv2
                # phi_rand2 = np.arctan2(y_c_rand2, x_c_rand2)
                # r_rand2 = np.sqrt(x_c_rand2 * x_c_rand2 + y_c_rand2 * y_c_rand2)
                # theta_rand2 = []
                # for r_sample in r_rand2:
                #     p = coeffs[:]
                #     p.append(-r_sample)
                #     # p.append(-r_rand2)
                #     thetas = np.roots(p)
                #     for thet in thetas:
                #         if np.imag(thet) == 0:
                #             if 0 < np.real(thet) < np.pi / 2 + 0.1:
                #                 theta_rand2.append(np.double(np.real(thet)))
                #                 # theta_rand2 = np.double(np.real(thet))
                #                 break
                #     else:
                #         print "Unable to find theta", thetas
                #         print r_sample
                #
                # theta_rand2 = np.array(theta_rand2)
                # u_cam2 = np.array(
                #     [[np.sin(theta_rand2) * np.cos(phi_rand2)], [np.sin(theta_rand2) * np.sin(phi_rand2)], [np.cos(theta_rand2)]])
                # rect_r2 = np.zeros(u_cam2.shape)
                # for ind in range(u_cam2.shape[2]):
                #     vec = u_cam2[:, :, ind]
                #     rect_r2[:, :, ind] = np.dot(R_align2, vec)
                # # rect_r2 = np.dot(R_align2,u_cam2)
                # # psi_rand2 = np.arcsin(rect_r2[0, 0, :])
                # # beta_rand2 = np.arctan2(rect_r2[1, 0, :], rect_r2[2, 0, :])
                # psi_rand2 = np.arcsin(rect_r2[0, 0, :])
                # beta_rand2 = np.arctan2(rect_r2[1, 0, :], rect_r2[2, 0, :])
                ######################

                rect_X2 = np.dot(R_align2, X2[0:3,:])
                norm2 = np.sqrt(rect_X2[0,0]**2 + rect_X2[1,0]**2 + rect_X2[2,0]**2)
                beta2 = np.arctan2(rect_X2[1,0]/norm2, rect_X2[2,0]/norm2)
                psi2 = np.arcsin(rect_X2[0,0]/norm2)

                J_theta2 = coeffs2[8] + 3*coeffs2[6]*theta2**2 + 5*coeffs2[4]*theta2**4 + 7*coeffs2[2]*theta2**6 + 9*coeffs2[0]*theta2**8
                var_theta_phi2 = np.diag([sigma_u**2/(m2*J_theta2)**2, sigma_u**2/(m2*r2)**2])

                c_theta2 = np.cos(theta2)
                s_theta2 = np.sin(theta2)
                c_phi2 = np.cos(phi2)
                s_phi2 = np.sin(phi2)
                c_psi2 = np.cos(psi2)
                s_psi2 = np.sin(psi2)
                c_beta2 = np.cos(beta2)
                s_beta2 = np.sin(beta2)

                J_ucam2 = np.array([[c_theta2*c_phi2, -s_theta2*s_phi2], [c_theta2*s_phi2, s_theta2*c_phi2], [-s_theta2, 0]])

                var_ucam2 = np.dot(np.dot(J_ucam2, var_theta_phi2), J_ucam2.transpose())

                var_R2 = np.dot(np.dot(R_align2, var_ucam2), R_align2.transpose())
                # J_beta_psi2 = np.array([[c_psi2, -s_psi2*s_beta2, -s_psi2*c_beta2], [0, c_beta2/c_psi2, -s_beta2/c_psi2]])
                J_beta_psi2 = np.array([[c_psi2, 0], [-s_psi2 * s_beta2, c_psi2 * c_beta2], [-s_psi2 * c_beta2, -c_psi2 * s_beta2]])
                J_beta_psi2 = np.linalg.pinv(J_beta_psi2)

                var_beta_psi2 = np.dot(np.dot(J_beta_psi2, var_R2), J_beta_psi2.transpose())

                disparity = psi - psi2

                # From camera 1
                # var_beta_psi_disp = np.zeros((3,3))
                # var_beta_psi_disp[2,2] = var_disparity
                # var_beta_psi_disp[0:2, 0:2] = var_beta_psi
                #
                # J_p = np.zeros((3,3))
                # temp = -baseline * s_psi2 / np.sin(disparity) * np.array([[c_psi], [-s_psi * s_beta], [-s_psi * c_beta]])
                # J_p[0,0] = temp[0,0]
                # J_p[1,0] = temp[1,0]
                # J_p[2,0] = temp[2,0]
                # temp = baseline * c_psi2 / np.sin(disparity) * np.array([[0], [c_psi * c_beta], [-c_psi * s_beta]])
                # J_p[0,1] = temp[0,0]
                # J_p[1,1] = temp[1,0]
                # J_p[2,1] = temp[2,0]
                # temp = baseline * (s_psi2/np.sin(disparity) - c_psi2*np.cos(disparity)/(np.sin(disparity))**2) * np.array([[s_psi], [c_psi * s_beta], [c_psi * c_beta]])
                # J_p[0,2] = temp[0,0]
                # J_p[1,2] = temp[1,0]
                # J_p[2,2] = temp[2,0]
                #
                # var_p = np.dot(np.dot(J_p, var_beta_psi_disp), J_p.transpose())

                # U, s, rotation = np.linalg.svd(var_p)
                # radii = np.sqrt(s)*3
                #
                # u = np.linspace(0.0, 2.0 * np.pi, 18)
                # v = np.linspace(0.0, np.pi, 18)
                #
                # x = radii[0] * np.outer(np.cos(u), np.sin(v))
                # y = radii[1] * np.outer(np.sin(u), np.sin(v))
                # z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
                #
                # center = [x_span[i], y_span[j], 0.5]
                #
                # for a in range(len(x)):
                #     for b in range(len(x)):
                #         [x[a, b], y[a, b], z[a, b]] = np.dot([x[a, b], y[a, b], z[a, b]], rotation) + center
                #
                # ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='k', alpha=0.2)

                # From camera 2
                # var_beta_psi_disp2 = np.zeros((3, 3))
                # var_beta_psi_disp2[2, 2] = var_disparity
                # var_beta_psi_disp2[0:2, 0:2] = var_beta_psi2
                #
                # J_p2 = np.zeros((3, 3))
                # temp = -baseline * s_psi / np.sin(disparity) * np.array([[c_psi2], [-s_psi2 * s_beta2], [-s_psi2 * c_beta2]])
                # J_p2[0, 0] = temp[0, 0]
                # J_p2[1, 0] = temp[1, 0]
                # J_p2[2, 0] = temp[2, 0]
                # temp = baseline * c_psi / np.sin(disparity) * np.array([[0], [c_psi2 * c_beta2], [-c_psi2 * s_beta2]])
                # J_p2[0, 1] = temp[0, 0]
                # J_p2[1, 1] = temp[1, 0]
                # J_p2[2, 1] = temp[2, 0]
                # temp = -baseline * (s_psi / np.sin(disparity) + c_psi2 * np.cos(disparity) / (np.sin(disparity)) ** 2) * np.array(
                #     [[s_psi2], [c_psi2 * s_beta2], [c_psi2 * c_beta2]])
                # J_p2[0, 2] = temp[0, 0]
                # J_p2[1, 2] = temp[1, 0]
                # J_p2[2, 2] = temp[2, 0]

                # var_p2 = np.dot(np.dot(J_p2, var_beta_psi_disp2), J_p2.transpose())
                #
                # U, s, rotation = np.linalg.svd(var_p2)
                # radii = np.sqrt(s) * 3
                #
                # u = np.linspace(0.0, 2.0 * np.pi, 18)
                # v = np.linspace(0.0, np.pi, 18)
                #
                # x = radii[0] * np.outer(np.cos(u), np.sin(v))
                # y = radii[1] * np.outer(np.sin(u), np.sin(v))
                # z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
                #
                # center = [x_span[i], y_span[j], 0.5]
                #
                # for a in range(len(x)):
                #     for b in range(len(x)):
                #         [x[a, b], y[a, b], z[a, b]] = np.dot([x[a, b], y[a, b], z[a, b]], rotation) + center
                #
                # ax2.plot_wireframe(x, y, z, rstride=4, cstride=4, color='g', alpha=0.2)

                # From camera 1, based on psi1, beta1, psi2, beta2
                var_combi = np.zeros((4,4))
                var_combi[0:2,0:2] = var_beta_psi
                var_combi[2:,2:] = var_beta_psi2

                J_p3 = np.zeros((3,4))
                temp = -baseline*(np.array([[c_psi2*(-s_psi*np.cos(disparity)/(np.sin(disparity)**2) + c_psi/np.sin(disparity))],
                                            [-c_psi2*s_beta*(c_psi*np.cos(disparity)/(np.sin(disparity)**2) + s_psi/np.sin(disparity))],
                                            [-c_psi2*c_beta*(c_psi*np.cos(disparity)/(np.sin(disparity)**2) + s_psi/np.sin(disparity))]]))
                J_p3[0, 0] = temp[0, 0]
                J_p3[1, 0] = temp[1, 0]
                J_p3[2, 0] = temp[2, 0]
                temp = baseline*c_psi2/np.sin(disparity) * np.array([[0], [c_psi*c_beta], [-c_psi*s_beta]])
                J_p3[0, 1] = temp[0, 0]
                J_p3[1, 1] = temp[1, 0]
                J_p3[2, 1] = temp[2, 0]
                temp = baseline*(c_psi2*np.cos(disparity)/(np.sin(disparity))**2 - s_psi2/np.sin(disparity)) * np.array([[s_psi], [c_psi*s_beta], [c_psi*c_beta]])
                J_p3[0, 2] = temp[0, 0]
                J_p3[1, 2] = temp[1, 0]
                J_p3[2, 2] = temp[2, 0]

                var_p3 = np.dot(np.dot(J_p3, var_combi), J_p3.transpose())
                var_p3 = np.dot(np.dot(np.dot(R1,R_align.transpose()), var_p3), np.dot(R_align, R1.transpose()))

                U, s, rotation = np.linalg.svd(var_p3)
                radii = np.sqrt(s) #* 3.
                # SIGMA[i,j,k] = np.sqrt((radii[0]/3.)**2 + (radii[1]/3.)**2 + (radii[2]/3.)**2)
                SIGMA[i, j, k] = np.sqrt((radii[0])**2 + (radii[1])**2 + (radii[2])**2)

                # if abs(z_span[k]-0.3) < 1e-6:
                u = np.linspace(0.0, 2.0 * np.pi, 18)
                v = np.linspace(0.0, np.pi, 18)

                x = radii[0] * np.outer(np.cos(u), np.sin(v))
                y = radii[1] * np.outer(np.sin(u), np.sin(v))
                z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

                center = [x_span[i], y_span[j], z_span[k]]

                for a in range(len(x)):
                    for b in range(len(x)):
                        [x[a, b], y[a, b], z[a, b]] = np.dot([x[a, b], y[a, b], z[a, b]], rotation) + center

                ax3.plot_wireframe(x, y, z, rstride=4, cstride=4, color='b', alpha=0.2)

                # U, s, rotation = np.linalg.svd(var_p4)
                # radii = np.sqrt(s) * 3
                #
                # u = np.linspace(0.0, 2.0 * np.pi, 18)
                # v = np.linspace(0.0, np.pi, 18)
                #
                # x = radii[0] * np.outer(np.cos(u), np.sin(v))
                # y = radii[1] * np.outer(np.sin(u), np.sin(v))
                # z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
                #
                # center = [x_span[i], y_span[j], 0.5]
                #
                # for a in range(len(x)):
                #     for b in range(len(x)):
                #         [x[a, b], y[a, b], z[a, b]] = np.dot([x[a, b], y[a, b], z[a, b]], rotation) + center
                #
                # ax4.plot_wireframe(x, y, z, rstride=4, cstride=4, color='c', alpha=0.2)

                # Random stuff
                # rho = baseline*np.cos(psi_rand2)/np.sin(psi_rand-psi_rand2)
                # x_out = rho*np.sin(psi_rand)
                # y_out = rho*np.cos(psi_rand)*np.sin(beta_rand)
                # z_out = rho*np.cos(psi_rand)*np.cos(beta_rand)
                #
                # X_out = np.array([x_out, y_out, z_out, np.ones(x_out.shape)])
                # H_align = np.eye(4)
                # H_align[0:3, 0:3] = R_align.transpose()
                # H_align = np.dot(H_cam1, H_align)
                # X_out = np.dot(H_align,X_out)

                # ax.scatter(x_out, y_out, z_out, c='r', s=5)
                # ax2.scatter(x_out, y_out, z_out, c='r', s=5)
                # ax3.scatter(X_out[0], X_out[1], X_out[2], c='r', s=5)
                # ax4.scatter(x_out, y_out, z_out, c='r', s=5)

                # X_rand[i,j] = np.mean(X_out[0])
                # Y_rand[i,j] = np.mean(X_out[1])
                # Z_rand[i,j] = np.mean(X_out[2])

                # print np.abs(beta_rand-beta_rand2).max()

                # U, s, rotation = np.linalg.svd(var_beta_psi)
                # radii = np.sqrt(s) * 3
                #
                # u = np.linspace(0.0, 2.0 * np.pi, 18)
                # # v = np.linspace(0.0, np.pi, 18)
                #
                # x = radii[0] * np.cos(u)
                # y = radii[1] * np.sin(u)
                # # z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
                #
                # center = [x_span[i], y_span[j]]
                #
                # for a in range(len(x)):
                #     [x[a], y[a]] = np.dot([x[a], y[a]], rotation) + center

                # ax_beta.plot(x, y, color='b', alpha=0.2)
                # ax_beta.scatter(x_span[i]+(psi_rand-psi), y_span[j]+(beta_rand-beta), c='r', s=5)

    # ax.plot_wireframe(x_points, y_points, SIGMA.transpose())
    # ax.set_xlim([-0, 9])
    # ax.set_ylim([-2, 7])
    # # ax.set_zlim([-5, 5])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('$\sigma$')

    # ax2.set_xlim([-5, 5])
    # ax2.set_ylim([-5, 5])
    # ax2.set_zlim([-5, 5])
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')

    ax3.scatter(4.5-cam_pos[0], 2.25-cam_pos[1], 0, c='r', marker='*', s=100)
    ax3.scatter(4.5+cam_pos[0], 2.25+cam_pos[1], 0, c='r', marker='*', s=100)

    ax3.set_xlim([0, 9])
    ax3.set_ylim([-2, 7])
    ax3.set_zlim([0, 1])
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.view_init(elev=-135, azim=-75)

    # ax_fisheye.set_xlim([0,768])
    # ax_fisheye.set_ylim([0,768])

    # ax4.set_xlim([-5, 5])
    # ax4.set_ylim([-5, 5])
    # ax4.set_zlim([-5, 5])
    # ax4.set_xlabel('x')
    # ax4.set_ylabel('y')
    # ax4.set_zlabel('z')

    # fig_x = plt.figure()
    # ax_x = fig_x.add_subplot('111',projection='3d')
    # ax_x.set_xlabel('x')
    # ax_x.set_ylabel('y')
    # ax_x.set_zlabel('$\sigma_x$')
    #
    # ax_x.plot_wireframe(x_points, y_points, SIGMA_X.transpose(),color='r')
    # ax_x.plot_wireframe(x_points, y_points, SIGMA_X_rand.transpose(),color='b')
    # # ax_x.plot_wireframe(x_points, y_points, SIGMA_X2.transpose(),color='g')
    # ax_x.plot_wireframe(x_points, y_points, SIGMA_X3.transpose(),color='k')
    # # ax_x.plot_wireframe(x_points, y_points, np.fmin(SIGMA_X,SIGMA_X2).transpose(),color='k')
    # # ax_x.plot_wireframe(x_points, y_points, 0.5*(SIGMA_X+SIGMA_X2).transpose(),color='g')
    #
    # fig_y = plt.figure()
    # ax_y = fig_y.add_subplot('111',projection='3d')
    # ax_y.set_xlabel('x')
    # ax_y.set_ylabel('y')
    # ax_y.set_zlabel('$\sigma_y$')
    #
    # ax_y.plot_wireframe(x_points, y_points, SIGMA_Y.transpose(),color='r')
    # ax_y.plot_wireframe(x_points, y_points, SIGMA_Y_rand.transpose(),color='b')
    # # ax_y.plot_wireframe(x_points, y_points, SIGMA_Y2.transpose(),color='g')
    # ax_y.plot_wireframe(x_points, y_points, SIGMA_Y3.transpose(),color='k')
    # # ax_y.plot_wireframe(x_points, y_points, np.fmin(SIGMA_Y,SIGMA_Y2).transpose(),color='k')
    # # ax_y.plot_wireframe(x_points, y_points, 0.5*(SIGMA_Y+SIGMA_Y2).transpose(),color='g')
    #
    # fig_z = plt.figure()
    # ax_z = fig_z.add_subplot('111',projection='3d')
    # ax_z.set_xlabel('x')
    # ax_z.set_ylabel('y')
    # ax_z.set_zlabel('$\sigma_z$')
    #
    # ax_z.plot_wireframe(x_points, y_points, SIGMA_Z.transpose(),color='r')
    # ax_z.plot_wireframe(x_points, y_points, SIGMA_Z_rand.transpose(),color='b')
    # # ax_z.plot_wireframe(x_points, y_points, SIGMA_Z2.transpose(),color='g')
    # ax_z.plot_wireframe(x_points, y_points, SIGMA_Z3.transpose(),color='k')
    # # ax_z.plot_wireframe(x_points, y_points, np.fmin(SIGMA_Z,SIGMA_Z2).transpose(),color='k')
    # # ax_z.plot_wireframe(x_points, y_points, 0.5*(SIGMA_Z+SIGMA_Z2).transpose(),color='g')

    # fig_projection = plt.figure()
    # ax = fig_projection.add_subplot('111', projection='3d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #
    # ax.scatter(x_points, y_points, 1.5, color='r')
    # ax.scatter(X_rand, Y_rand, Z_rand, color='b')

    # print "Difference 1: ", (SIGMA_X-SIGMA_X_rand).mean(), ", ", (SIGMA_X-SIGMA_X_rand).std(), ", ", \
    #     (SIGMA_Y-SIGMA_Y_rand).mean(), ", ", (SIGMA_Y-SIGMA_Y_rand).std(), ", ", \
    #     (SIGMA_Z-SIGMA_Z_rand).mean(), ", ", (SIGMA_Z-SIGMA_Z_rand).std()
    #
    # print "Difference 2: ", (SIGMA_X3-SIGMA_X_rand).mean(), ", ", (SIGMA_X3-SIGMA_X_rand).std(), ", ", \
    #     (SIGMA_Y3-SIGMA_Y_rand).mean(), ", ", (SIGMA_Y3-SIGMA_Y_rand).std(), ", ", \
    #     (SIGMA_Z3-SIGMA_Z_rand).mean(), ", ", (SIGMA_Z3-SIGMA_Z_rand).std()

    # fig_beta = plt.figure()
    # ax_beta = fig_beta.add_subplot('111',projection='3d')
    # ax_beta.set_xlabel('x')
    # ax_beta.set_ylabel('y')
    # ax_beta.set_zlabel('$\sigma_{beta}$')
    #
    # ax_beta.plot_wireframe(x_points, y_points, SIGMA_beta.transpose(),color='r')
    # ax_beta.plot_wireframe(x_points, y_points, SIGMA_beta2.transpose(),color='b')
    print SIGMA.mean(), SIGMA.min(), SIGMA.max()
    # print SIGMA[:,len(y_span)/2]
    #
    # filename = 'graph_var%02d-%02d' % (cam_pos[0]*10, cam_pos[1]*10)
    # filename = directory + filename + '.png'
    plt.show()
    # fig3.savefig('Optimal3D.png', dpi=600, facecolor='w', edgecolor='w',
    #         format='png', transparent=True, bbox_inches='tight', pad_inches=0.1,
    #         frameon=None)
    # plt.close(fig3)
    return SIGMA

for sig in sig_u:
    MAX = np.zeros((len(cam_x), len(cam_y)))
    MEAN = np.zeros((len(cam_x), len(cam_y)))
    directory = '/home/otalab/Python-dir/Fisheye_error/Errors/sigma'+str(sig)+'px/VeryDetailed/'

    for a in range(len(cam_x)):
        for b in range(len(cam_y)):
            if cam_x[a] == 0. and cam_y[b] == 0.:
                MAX[a,b] = 1e16
                MEAN[a,b] = 1e16
                continue
            covariance = calculate_error((cam_x[a], cam_y[b]),sigma_u=sig)
            MAX[a,b] = covariance.max()
            MEAN[a,b] = covariance.mean()

            # filename = 'sigma%02d-%02d' % (cam_x[a]*10, cam_y[b]*10)
            # filename = directory + filename + '.npy'
            # np.save(filename, covariance)

    min_loc = np.argmin(MAX)
    min_index = np.unravel_index(min_loc,MAX.shape)

    print "The position at which the maximum error is minimum is at (" + str(cam_x[min_index[0]]) + ", " + str(cam_y[min_index[1]]) + "), with max error of " + str(MAX.min()) + "."

    min_loc = np.argmin(MEAN)
    min_index = np.unravel_index(min_loc,MEAN.shape)

    print "The position at which the mean error is minimum is at (" + str(cam_x[min_index[0]]) + ", " + str(cam_y[min_index[1]]) + "), with mean error of " + str(MEAN.min()) + "."

    # filename = directory + 'max.npy'
    # np.save(filename, MAX)
    # filename = directory + 'mean.npy'
    # np.save(filename, MEAN)

