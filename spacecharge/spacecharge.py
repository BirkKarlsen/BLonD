'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


import copy, h5py, sys
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi

from beams.slices import *
from beams.matching import match_transverse, match_longitudinal, unmatched_inbucket
from solvers.poissonfft import *


re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p


class SpaceCharge():

    def __init__(self, beam=None, beam_type=None, extension_x=1, extension_y=1, nx=128, ny=128):

        self.poisson = PoissonFFT(extension_x, extension_y, nx, ny)

        self.ex1 = np.zeros((ny, nx))
        self.ey1 = np.zeros((ny, nx))
        self.phi1 = np.zeros((ny, nx))
        self.rho1 = np.zeros((ny, nx))

        if beam:
            self.beam = beam
            self.beam_type = beam_type
            self.beam.kx = np.zeros(beam.n_macroparticles)
            self.beam.ky = np.zeros(beam.n_macroparticles)

            self.ex2 = np.zeros((ny, nx))
            self.ey2 = np.zeros((ny, nx))
            self.phi2 = np.zeros((ny, nx))
            self.rho2 = np.zeros((ny, nx))

    # def track(): pass

    # def add_poisson(self, extent_x, extent_y, nx, ny, other=None):

    #     self.poisson_self = PoissonFFT(extent_x, extent_y, nx, ny)
    #     self.kx = np.zeros(self.n_macroparticles)
    #     self.ky = np.zeros(self.n_macroparticles)

    #     if other:
    #         other.poisson_other = copy.deepcopy(self.poisson_self)
    #         other.kx = np.zeros(other.n_macroparticles)
    #         other.ky = np.zeros(other.n_macroparticles):

    def push_beam(self, beam, cf1, dt=0, ix=None):

        # Push beam
        beam.xp[ix] += cf1 * beam.kx[ix]
        beam.yp[ix] += cf1 * beam.ky[ix]

    def push_cloud(self, cloud, cf1, dt=0, ix=None):

        # Push cloud
        cloud.xp += cf1 * cloud.kx
        cloud.yp += cf1 * cloud.ky
        cloud.x += dt * cloud.xp
        cloud.y += dt * cloud.yp

    # @profile
    def track(self, beam):

        if not hasattr(beam, 'kx'):
            beam.kx = np.zeros(beam.n_macroparticles)
            beam.ky = np.zeros(beam.n_macroparticles)

        beam.compute_statistics() # need not be done here
        self.beam.reinitialize()

        # phi1 = plt.zeros((beam.poisson_other.ny, beam.poisson_other.nx))
        # phi2 = plt.zeros((beam.poisson_other.ny, beam.poisson_other.nx))

        index_after_bin_edges = np.cumsum(beam.slices.n_macroparticles)[:-3]
        index_after_bin_edges[0] = 0

        for i in xrange(beam.slices.n_slices):
            ix = np.s_[index_after_bin_edges[i]:index_after_bin_edges[i + 1]]

            self.poisson.gather_from(beam.x[ix], beam.y[ix], self.rho1)
            self.poisson.compute_potential(self.rho1, self.phi1)
            self.poisson.compute_fields(self.phi1, self.ex1, self.ey1)
            # beam.poisson_other.compute_potential_fgreenm2m(beam.poisson_other.x, beam.poisson_other.y,
            #                                                 phi1, beam.poisson_other.rho)
            # beam.poisson_other.compute_potential_fgreenp2m(beam.x, beam.y,
            #                                                 beam.poisson_other.x, beam.poisson_other.y,
            #                                                 phi2, beam.poisson_other.rho)
            if self.beam_type == 'cloud':
                self.poisson.scatter_to(self.ex1, self.ey1, self.beam.x, self.beam.y, self.beam.kx, self.beam.ky)
                dt = (beam.slices.dz_centers[5] - beam.slices.dz_centers[4]) / (beam.beta * c)
                cf1 = -2 * c * re * 1 / beam.beta * beam.n_particles / beam.n_macroparticles
                # print dt, cf1
                #   = -2 * c ** 2 * re  * ex / dz * dt * 1 / gamma
                self.push_cloud(self.beam, cf1, dt)

                # Cloud track
                self.poisson.gather_from(self.beam.x, self.beam.y, self.rho2)
                self.poisson.compute_potential(self.rho2, self.phi2)
                self.poisson.compute_fields(self.phi2, self.ex2, self.ey2)
                self.poisson.scatter_to(self.ex2, self.ey2, beam.x[ix], beam.y[ix], beam.kx[ix], beam.ky[ix])
                cf1 = -2 * 1 * rp * 1 / (beam.gamma * beam.beta ** 2) * self.beam.n_particles / self.beam.n_macroparticles
                # print cf1, self.beam.n_particles / self.beam.n_macroparticles, beam.n_particles / beam.n_macroparticles
                #   = -2 * c ** 2 * rp  * ex / dL * dL * 1 / gamma / (beta * c) ** 2 # WATCH OUT HERE!!! dL / dz???
                self.push_beam(beam, cf1, ix=ix)
                # print "***********************************"

                # if i == 0:
                #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
                # [ax.cla() for ax in (ax1, ax2, ax3, ax4)]
                # # [ax.set_aspect('equal') for ax in (ax1, ax2, ax3, ax4)]
                # ax1.contour(self.poisson.fgreen.T, 100)
                # ax2.plot(self.phi1[self.poisson.ny / 2,:], '-g')
                # # ax2.plot(phi1[bunch.poisson_other.ny / 2,:], '-r')
                # # ax2.plot(phi2[bunch.poisson_other.ny / 2,:], '-', c='orange')
                # # ax3.contourf(self.poisson_self.x, self.poisson_self.y, 10 * plt.log10(self.poisson_self.rho), 100)
                # # print self.phi1 # 10 * plt.log10(self.rho2)
                # ax3.imshow(10 * plt.log10(self.rho2), origin='lower', aspect='auto', vmin=50, vmax=1e2,
                #            extent=(self.poisson.x[0,0], self.poisson.x[0,-1], self.poisson.y[0,0], self.poisson.y[-1,0]))
                # # ax3.scatter(self.x[::20], self.y[::20], c='b', marker='.')
                # # ax3.quiver(self.x[::50], self.y[::50], self.kx[::50], self.ky[::50], color='g')
                # # ax3.contour(p.x, p.y, p.phi, 100, lw=2)
                # # ax3.scatter(bunch.x[ix], bunch.y[ix], c='y', marker='.', alpha=0.8)
                # ax4.imshow(plt.sqrt(self.ex1 ** 2 + self.ey1 ** 2), origin='lower', aspect='auto',
                #            extent=(self.poisson.x[0,0], self.poisson.x[0,-1], self.poisson.y[0,0], self.poisson.y[-1,0]))

                # plt.draw()
