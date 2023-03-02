# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.cavity_feedback

:Authors: **Birk Emil Karlsen-Baeck**, **Helga Timko**
"""

import unittest
import numpy as np
import os
from scipy.constants import c

from blond.llrf.cavity_feedback import SPSOneTurnFeedback, SPSCavityFeedback, CavityFeedbackCommissioning
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker
from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageTime
from blond.impedances.impedance_sources import TravelingWaveCavity

this_directory = os.path.dirname(os.path.realpath(__file__))

class TestSPSCavityFeedback(unittest.TestCase):

    def setUp(self):
        C = 2*np.pi*1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1/gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]
        h = 4620                    # 200 MHz system harmonic
        phi = 0.                    # 200 MHz RF phase

        # With this setting, amplitude in the two four-section, five-section
        # cavities must converge, respectively, to
        # 2.0 MV = 4.5 MV * 4/18 * 2
        # 2.5 MV = 4.5 MV * 5/18 * 2
        V = 4.5e6                   # 200 MHz RF voltage

        N_t = 1                     # Number of turns to track

        self.ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
        self.rf = RFStation(self.ring, h, V, phi)

        N_m = 1e6                   # Number of macro-particles for tracking
        N_b = 288 * 2.3e11               # Bunch intensity [ppb]

        # Gaussian beam profile
        self.beam = Beam(self.ring, N_m, N_b)
        sigma = 1.0e-9
        bigaussian(self.ring, self.rf, self.beam, sigma, seed=1234,
                   reinsertion=False)

        n_shift = 1550  # how many rf-buckets to shift beam
        self.beam.dt += n_shift * self.rf.t_rf[0, 0]

        self.profile = Profile(
            self.beam, CutOptions=CutOptions(
                cut_left=(n_shift-1.5)*self.rf.t_rf[0, 0],
                cut_right=(n_shift+2.5)*self.rf.t_rf[0, 0],
                n_slices=4*64))
        self.profile.track()

        # Cavities
        l_cav = 32*0.374
        v_g = 0.0946
        tau = l_cav/(v_g*c)*(1 + v_g)
        f_cav = 200.222e6
        n_cav = 4   # factor 2 because of two four/five-sections cavities
        short_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                            f_cav, 2*np.pi*tau)
        shortInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                 [short_cavity])
        l_cav = 43*0.374
        tau = l_cav/(v_g*c)*(1 + v_g)
        n_cav = 2
        long_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                           f_cav, 2*np.pi*tau)
        longInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                [long_cavity])
        self.induced_voltage = TotalInducedVoltage(
            self.beam, self.profile, [shortInducedVoltage, longInducedVoltage])
        self.induced_voltage.induced_voltage_sum()

        self.cavity_tracker = RingAndRFTracker(
            self.rf, self.beam, Profile=self.profile, interpolation=True,
            TotalInducedVoltage=self.induced_voltage)

        self.OTFB = SPSCavityFeedback(
            self.rf, self.beam, self.profile, G_llrf=20, G_tx=[1.0355739238973907, 1.078403005653143],
            a_comb=63/64, turns=1000, post_LS2=True, df=[0.18433333e6, 0.2275e6],
            Commissioning=CavityFeedbackCommissioning(open_FF=True))


        self.OTFB_tracker = RingAndRFTracker(self.rf, self.beam,
                                             Profile=self.profile,
                                             TotalInducedVoltage=None,
                                             CavityFeedback=self.OTFB,
                                             interpolation=True)

    def test_FB_pre_tracking(self):

        digit_round = 3

        Vind3_mean = np.mean(np.absolute(self.OTFB.OTFB_1.V_ANT[-self.OTFB.OTFB_1.n_coarse:]))/1e6
        Vind3_std = np.std(np.absolute(self.OTFB.OTFB_1.V_ANT[-self.OTFB.OTFB_1.n_coarse:]))/1e6
        Vind3_mean_exp = 2.7047955940118764
        Vind3_std_exp = 2.4121534046270847e-12

        Vind4_mean = np.mean(np.absolute(self.OTFB.OTFB_2.V_ANT[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_std = np.std(np.absolute(self.OTFB.OTFB_2.V_ANT[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_mean_exp = 1.8057100857806163
        Vind4_std_exp = 1.89451253314611e-12

        self.assertAlmostEqual(Vind3_mean, Vind3_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of four-section cavity differs')
        self.assertAlmostEqual(Vind3_std, Vind3_std_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: standard ' +
                               'deviation of four-section cavity differs')

        self.assertAlmostEqual(Vind4_mean, Vind4_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of five-section cavity differs')
        self.assertAlmostEqual(Vind4_std, Vind4_std_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: standard '
                               + 'deviation of five-section cavity differs')

    def test_FB_pre_tracking_IQ_v1(self):
        rtol = 1e-2         # relative tolerance
        atol = 0            # absolute tolerance
        # interpolate from coarse mesh to fine mesh
        V_fine_tot_3 = np.interp(
            self.profile.bin_centers, self.OTFB.OTFB_1.rf_centers,
            self.OTFB.OTFB_1.V_IND_COARSE_GEN[-self.OTFB.OTFB_1.n_coarse:])
        V_fine_tot_4 = np.interp(
            self.profile.bin_centers, self.OTFB.OTFB_2.rf_centers,
            self.OTFB.OTFB_2.V_IND_COARSE_GEN[-self.OTFB.OTFB_2.n_coarse:])

        V_tot_3 = V_fine_tot_3/1e6
        V_tot_4 = V_fine_tot_4/1e6

        V_sum = self.OTFB.V_sum/1e6

        # expected generator voltage is only in Q
        V_tot_3_exp = 2.7j*np.ones(256)
        V_tot_4_exp = 1.8j*np.ones(256)
        V_sum_exp = 4.5j*np.ones(256)

        np.testing.assert_allclose(V_tot_3, V_tot_3_exp,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_FB_pretracking_IQ: total voltage ' +
                                   'in four-section cavity differs')

        np.testing.assert_allclose(V_tot_4, V_tot_4_exp,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_FB_pretracking_IQ: total voltage ' +
                                   'in five-section cavity differs')

        np.testing.assert_allclose(V_sum, V_sum_exp,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_FB_pretracking_IQ: voltage sum ' +
                                   ' differs')

    def test_rf_voltage(self):

        digit_round = 7

        # compute voltage
        self.cavity_tracker.rf_voltage_calculation()

        # compute voltage after OTFB pre-tracking
        self.OTFB_tracker.rf_voltage_calculation()

        # Since there is a systematic offset between the voltages,
        # compare the maxium of the ratio
        max_ratio = np.max(self.cavity_tracker.rf_voltage
                           / self.OTFB_tracker.rf_voltage)

        max_ratio_exp = 1.0690779399272086#1.0001336336515099
        self.assertAlmostEqual(max_ratio, max_ratio_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_rf_voltage: '
                               + 'RF-voltages differ')

    def test_beam_loading(self):
        digit_round = 7
        # Compute voltage with beam loading
        self.cavity_tracker.rf_voltage_calculation()
        cavity_tracker_total_voltage = self.cavity_tracker.rf_voltage \
            + self.cavity_tracker.totalInducedVoltage.induced_voltage

        self.OTFB.track()
        self.OTFB_tracker.rf_voltage_calculation()
        OTFB_tracker_total_voltage = self.OTFB_tracker.rf_voltage

        max_ratio = np.max(cavity_tracker_total_voltage /
                           OTFB_tracker_total_voltage)


        max_ratio_exp = 1.0690779399245092 #1.0055233047525063

        self.assertAlmostEqual(max_ratio, max_ratio_exp, places=digit_round,
                               msg='In TestCavityFeedback test_beam_loading: '
                               + 'total voltages differ')

    @unittest.skip("FIXME")
    def test_Vsum_IQ(self):
        rtol = 1e-7         # relative tolerance
        atol = 0              # absolute tolerance

        self.OTFB.track()

        V_sum = self.OTFB.V_sum/1e6


        V_sum_exp = np.array([0.01479927484371188+4.510480895814644j, 0.01479927484372684+4.510480895814655j,
                            0.014799274843736804+4.510480895814641j, 0.014799274843731789+4.510480895814639j,
                            0.014799274843716785+4.5104808958146405j, 0.014799274843716764+4.510480895814651j,
                            0.014799274843736717+4.510480895814647j, 0.014799274843726706+4.510480895814645j,
                            0.01479927484373667+4.510480895814647j, 0.01479927484373665+4.510480895814663j,
                            0.014799274843736627+4.5104808958146485j, 0.014799274843736606+4.510480895814639j,
                            0.014799274843726597+4.510480895814653j, 0.014799274843726576+4.510480895814661j,
                            0.01479927484373654+4.510480895814646j, 0.014799274843751499+4.510480895814642j,
                            0.01479927484372651+4.51048089581466j, 0.014799274843726489+4.510480895814661j,
                            0.014799274843726467+4.510480895814655j, 0.014799274843711463+4.510480895814648j,
                            0.014799274843716436+4.510480895814655j, 0.01479927484371142+4.510480895814667j,
                            0.014799274843746352+4.510480895814651j, 0.01479927484373135+4.510480895814649j,
                            0.014799274843726335+4.510480895814663j, 0.014799274843731305+4.510480895814663j,
                            0.014799274843736278+4.510480895814653j, 0.01479927484372627+4.51048089581465j,
                            0.014799274843716258+4.510480895814655j, 0.014799274843731218+4.510480895814662j,
                            0.014799274843736191+4.510480895814649j, 0.014799274843731175+4.510480895814646j,
                            0.01479927484372616+4.510480895814657j, 0.014799274843746112+4.5104808958146645j,
                            0.014799274843726116+4.510480895814661j, 0.014799274843726095+4.510480895814658j,
                            0.014799274843726071+4.510480895814659j, 0.014799274843736039+4.510480895814664j,
                            0.014799274843726028+4.510480895814655j, 0.01479927484373599+4.510480895814654j,
                            0.014799274843730977+4.5104808958146645j, 0.014799274843755924+4.510480895814666j,
                            0.014799274843745915+4.510480895814659j, 0.014799274843740898+4.510480895814658j,
                            0.01479927484373089+4.510480895814663j, 0.01479927484373586+4.510480895814668j,
                            0.014799274843735838+4.510480895814651j, 0.014799274843750798+4.510480895814651j,
                            0.014799274843715822+4.510480895814661j, 0.014799274843725787+4.510480895814657j,
                            0.014799274843715779+4.510480895814663j, 0.014799274843715756+4.510480895814653j,
                            0.01479927484372572+4.510480895814659j, 0.014799274843705724+4.510480895814673j,
                            0.01479927484371569+4.510480895814663j, 0.014799274843720662+4.510480895814655j,
                            0.014799274843730626+4.510480895814668j, 0.014799274843730602+4.510480895814653j,
                            0.01479927484372559+4.510480895814666j, 0.01479927484373056+4.510480895814665j,
                            0.014799274843720551+4.51048089581467j, 0.014799274843730515+4.510480895814669j,
                            0.014799274843735489+4.5104808958146645j, 0.014799274843730472+4.510480895814661j,
                            0.014799274843735452+4.510480895814669j, 0.014651787218559609+4.510502773537189j,
                            0.01450434518949208+4.510524650656808j, 0.014504390785435603+4.510524650052046j,
                            0.014504436381234316+4.510524649445435j, 0.014376593420634722+4.510601302287685j,
                            0.014248792354473941+4.510677937122168j, 0.014027926109320472+4.5108781792939885j,
                            0.013606874225271105+4.511299323530271j, 0.01305156734395231+4.511999185728238j,
                            0.012083420902610345+4.513501071962331j, 0.010769540782963489+4.5160062825983385j,
                            0.009818007598385695+4.518189891847422j, 0.008951931244301223+4.521349413275657j,
                            0.007987252223720279+4.526166088818855j, 0.00743381372344398+4.532693112774662j,
                            0.007464765109469153+4.541624577582733j, 0.008488287841842038+4.551694331833949j,
                            0.010944194493775337+4.5636500660536266j, 0.016178653377279084+4.580261535862432j,
                            0.027905799890013565+4.607469338529264j, 0.045953089294647635+4.64120182760906j,
                            0.07031577097446909+4.677197072757502j, 0.10991929575035633+4.724417879733518j,
                            0.16210557059542258+4.776640422352565j, 0.2295108265722965+4.831056940244109j,
                            0.3216331045169427+4.891975438018363j, 0.4512883974833155+4.959797165198393j,
                            0.6227638842912119+5.030063809448461j, 0.8393887139725591+5.0943001265343595j,
                            1.1132201670675523+5.147399068328872j, 1.455735392436072+5.179077535380332j,
                            1.85893093335233+5.177968282821074j, 2.334990883549414+5.128159944686226j,
                            2.911035273795893+5.010675994428396j, 3.590051879984493+4.800868913981582j,
                            4.374295584757511+4.471919905257593j, 5.257387892112483+3.994833144753212j,
                            6.236468089913465+3.3344597210115077j, 7.29920067439701+2.45491591518776j,
                            8.441348957381154+1.3020698965483457j, 9.61977180184159+-0.1431617203781351j,
                            10.78586231559474+-1.9029101069293082j, 11.904466776320334+-4.0121476092371475j,
                            12.932664230748388+-6.523029957847709j, 13.816628361564073+-9.472264172808334j,
                            14.482376314565732+-12.886826861868157j, 14.851417955920088+-16.741001459473146j,
                            14.839911346237194+-20.956733587824658j, 14.367021771253754+-25.592441537811048j,
                            13.358372819245508+-30.59884208719641j, 11.728623392149776+-35.91156438324952j,
                            9.405834422564743+-41.480153034253426j, 6.3627495687688285+-47.14248965535178j,
                            2.5517832441178085+-52.81568303243523j, -2.0603706632496044+-58.40968576323217j,
                            -7.4431831317937+-63.77128670332761j, -13.568499610965471+-68.77701570247041j,
                            -20.418186070696297+-73.33355419980394j, -27.971716362196425+-77.34974429380777j,
                            -36.17707650678707+-80.72873213239765j, -44.88135821688268+-83.35275955822611j,
                            -53.86300413421221+-85.12635069744884j, -63.04268438244274+-86.0125243114934j,
                            -72.35082641638594+-85.99686624223762j, -81.55034485343772+-85.07815274071774j,
                            -90.4364043136237+-83.2986132115449j, -98.96601885901798+-80.69402495347747j,
                            -107.06690894940083+-77.32379760088676j, -114.64012190721947+-73.25708376917012j,
                            -121.57324117896822+-68.61042753037916j, -127.73778090374356+-63.53508476347076j,
                            -133.1082799654868+-58.1473043822226j, -137.6921182481451+-52.53929502647198j,
                            -141.4931632248349+-46.8286907641909j, -144.50856644043992+-41.16147290390865j,
                            -146.79545502354935+-35.60360469640764j, -148.40158289541642+-30.27210097411176j,
                            -149.3796331906408+-25.298228012217738j, -149.8198489580434+-20.69998669183162j,
                            -149.80594270205077+-16.457009045911203j, -149.41419414231794+-12.601201201297991j,
                            -148.72543174342215+-9.194283961444743j, -147.81409730798782+-6.229413920342085j,
                            -146.7402153772137+-3.6664192527829957j, -145.577122435231+-1.509142052874557j,
                            -144.3840181746944+0.25988148098715624j, -143.19039500416426+1.7007889520531307j,
                            -142.0233842924279+2.856859254195526j, -140.9337758958107+3.744901922240847j,
                            -139.93854889056354+4.403433025472635j, -139.0159642671132+4.891902426783195j,
                            -138.18118243509093+5.235629835274246j, -137.44393419045062+5.458698070103716j,
                            -136.8085821190379+5.586757875225235j, -136.26598732416855+5.642790420498434j,
                            -135.82439721097575+5.648578364730166j, -135.477374730157+5.620536114917869j,
                            -135.1877526639592+5.570905937570374j, -134.95020817054748+5.508943267126949j,
                            -134.76111204295938+5.44292136837977j, -134.61381331125955+5.3788157079065915j,
                            -134.49955657880014+5.319653193951355j, -134.40879040477805+5.265208920687224j,
                            -134.3376684551519+5.218053200658914j, -134.28008052855347+5.175664497085748j,
                            -134.23365638054733+5.1405240846579545j, -134.19967387030263+5.118691233752794j,
                            -134.17074352346853+5.100873148543474j, -134.14442082088053+5.084961646139331j,
                            -134.1211856046365+5.07439709540568j, -134.09971729822564+5.0683715651123045j,
                            -134.07925605051025+5.064376545325417j, -134.05923288870983+5.061591798098676j,
                            -134.03932390853618+5.061223866003148j, -134.01948203809482+5.061790099585471j,
                            -133.99973271155332+5.062704156092411j, -133.98001489950047+5.063960892405476j,
                            -133.96035854647388+5.06542692071836j, -133.94031876150626+5.067531754363622j,
                            -133.9199453007942+5.070134417447717j, -133.89957175617408+5.072736278299749j,
                            -133.87908768019506+5.075437432449332j, -133.85873833583972+5.078074000426825j,
                            -133.8385292843571+5.080659525377927j, -133.81818535187242+5.083307983260842j,
                            -133.79770097943702+5.0860058347943715j, -133.7772165526365+5.0887028492209065j,
                            -133.7567320714965+5.091399026539334j, -133.7362475360426+5.094094366748566j,
                            -133.7157629463006+5.096788869847533j, -133.69527830229603+5.099482535835124j,
                            -133.67479360405468+5.102175364710233j, -133.65430885160208+5.104867356471789j,
                            -133.633824044964+5.107558511118685j, -133.61333918416605+5.110248828649843j,
                            -133.59285426923392+5.112938309064186j, -133.57236930019323+5.1156269523605715j,
                            -133.5518842770697+5.118314758537983j, -133.53139919988888+5.121001727595294j,
                            -133.51091406867656+5.123687859531423j, -133.49042888345832+5.12637315434526j,
                            -133.46994364425987+5.1290576120357585j, -133.44945835110684+5.131741232601823j,
                            -133.42897300402493+5.134424016042361j, -133.40848760303976+5.13710596235628j,
                            -133.38800214817698+5.139787071542538j, -133.3675166394623+5.142467343600006j,
                            -133.3470310769214+5.145146778527629j, -133.32654546057987+5.147825376324312j,
                            -133.30605979046342+5.150503136988991j, -133.2855740665977+5.153180060520573j,
                            -133.26508828900836+5.1558561469179836j, -133.24460245772107+5.158531396180136j,
                            -133.22411657276155+5.161205808305996j, -133.20363063415533+5.163879383294442j,
                            -133.18314464192818+5.166552121144419j, -133.16265859610576+5.169224021854834j,
                            -133.14217249671367+5.171895085424621j, -133.1216863437776+5.174565311852731j,
                            -133.10120013732325+5.177234701138056j, -133.0807138773762+5.179903253279535j,
                            -133.06022756396223+5.182570968276128j, -133.03974119710693+5.185237846126707j,
                            -133.01925477683594+5.187903886830233j, -132.99876830317493+5.190569090385639j,
                            -132.9782817761496+5.193233456791852j, -132.9577951957856+5.195896986047806j,
                            -132.9373085621086+5.198559678152437j, -132.91682187514428+5.201221533104665j,
                            -132.89633513491822+5.203882550903446j, -132.87584834145613+5.20654273154769j,
                            -132.8553614947837+5.209202075036341j, -132.83487459492653+5.2118605813683425j,
                            -132.81438764191037+5.2145182505426355j, -132.79390063576082+5.217175082558148j,
                            -132.77341357650357+5.219831077413814j, -132.7529264641643+5.222486235108567j,
                            -132.73243929876855+5.225140555641392j, -132.71195208034214+5.227794039011183j,
                            -132.69146480891067+5.23044668521689j, -132.67097748449982+5.233098494257457j,
                            -132.65049010713517+5.23574946613184j, -132.6300026768425+5.238399600838972j,
                            -132.6095151936474+5.241048898377796j, -132.58902765757557+5.243697358747252j,
                            -132.56854006865265+5.246344981946297j, -132.5480524269043+5.248991767973885j,
                            -132.52756473235618+5.251637716828931j, -132.507076985034+5.254282828510395j,
                            -132.48658918496338+5.2569271030172455j, -132.46610133216993+5.259570540348426j])

        np.testing.assert_allclose(V_sum_exp, V_sum,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_Vsum_IQ: total voltage ' +
                                   'is different from expected values!')

class TestSPSOneTurnFeedback(unittest.TestCase):

    def setUp(self):
        # Parameters ----------------------------------------------------------
        C = 2 * np.pi * 1100.009                # Ring circumference [m]
        gamma_t = 18.0                          # Transition Gamma [-]
        alpha = 1 / (gamma_t ** 2)              # Momentum compaction factor [-]
        p_s = 450e9                             # Synchronous momentum [eV]
        h = 4620                                # 200 MHz harmonic number [-]
        V = 10e6                                # 200 MHz RF voltage [V]
        phi = 0                                 # 200 MHz phase [-]

        # Parameters for the Simulation
        N_m = 1e5                               # Number of macro-particles for tracking
        N_b = 1.0e11                            # Bunch intensity [ppb]
        N_t = 1                                 # Number of turns to track


        # Objects -------------------------------------------------------------

        # Ring
        self.ring = Ring(C, alpha, p_s, Proton(), N_t)

        # RFStation
        self.rfstation = RFStation(self.ring, [h], [V], [phi], n_rf=1)

        # Beam
        self.beam = Beam(self.ring, N_m, N_b)
        self.profile = Profile(self.beam, CutOptions=CutOptions(cut_left=0.e-9,
                                                      cut_right=self.rfstation.t_rev[0], n_slices=4620))
        self.profile.track()

        # Cavity
        self.Commissioning = CavityFeedbackCommissioning()


        self.OTFB = SPSOneTurnFeedback(self.rfstation, self.beam, self.profile, 3, a_comb=63 / 64,
                                          Commissioning=self.Commissioning)

        self.OTFB.update_variables()

        self.turn_array = np.linspace(0, 2 * self.rfstation.t_rev[0], 2 * self.OTFB.n_coarse)

    def test_set_point(self):
        self.OTFB.set_point()
        t_sig = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        t_sig[-self.OTFB.n_coarse:] = (4/9) * 10e6 * np.exp(1j * (np.pi/2 - self.rfstation.phi_rf[0,0]))

        np.testing.assert_allclose(self.OTFB.V_SET, t_sig)


    def test_error_and_gain(self):
        self.OTFB.error_and_gain()

        np.testing.assert_allclose(self.OTFB.DV_GEN, self.OTFB.V_SET * self.OTFB.G_llrf)


    def test_comb(self):
        sig = np.zeros(self.OTFB.n_coarse)
        self.OTFB.DV_COMB_OUT = np.sin(2 * np.pi * self.turn_array / self.rfstation.t_rev[0])
        self.OTFB.DV_GEN = -np.sin(2 * np.pi * self.turn_array / self.rfstation.t_rev[0])
        self.OTFB.a_comb = 0.5

        self.OTFB.comb()

        np.testing.assert_allclose(self.OTFB.DV_COMB_OUT[-self.OTFB.n_coarse:], sig)


    def test_one_turn_delay(self):
        self.OTFB.DV_COMB_OUT = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_COMB_OUT[self.OTFB.n_coarse] = 1

        self.OTFB.one_turn_delay()

        self.assertEqual(np.argmax(self.OTFB.DV_DELAYED), 2 * self.OTFB.n_coarse - self.OTFB.n_mov_av)


    def test_mod_to_fr(self):
        self.OTFB.DV_DELAYED = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_DELAYED[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB.dphi_mod)
        self.OTFB.mod_to_fr()
        ref_DV_MOD_FR = np.load(os.path.join(this_directory, "ref_DV_MOD_FR.npy"))

        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse:], ref_DV_MOD_FR)

        self.OTFB.DV_DELAYED = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_DELAYED[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_fr()

        time_array = self.OTFB.rf_centers - 0.5*self.OTFB.T_s
        ref_sig = np.cos((self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse]) - \
                  1j * np.sin((self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse])

        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse:], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi


    def test_mov_avg(self):
        sig = np.zeros(self.OTFB.n_coarse-1)
        sig[:self.OTFB.n_mov_av] = 1
        self.OTFB.DV_MOD_FR = np.zeros(2 * self.OTFB.n_coarse)
        self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse + 1:] = sig

        self.OTFB.mov_avg()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[:self.OTFB.n_mov_av] = (1/self.OTFB.n_mov_av) * np.array(range(self.OTFB.n_mov_av))
        sig[self.OTFB.n_mov_av: 2 * self.OTFB.n_mov_av] = (1/self.OTFB.n_mov_av) * (self.OTFB.n_mov_av
                                                                                    - np.array(range(self.OTFB.n_mov_av)))

        np.testing.assert_allclose(np.abs(self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:]), sig)


    def test_mod_to_frf(self):
        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB.dphi_mod)
        self.OTFB.mod_to_frf()
        ref_DV_MOD_FRF = np.load(os.path.join(this_directory, "ref_DV_MOD_FRF.npy"))

        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:], ref_DV_MOD_FRF)

        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_frf()

        time_array = self.OTFB.rf_centers - 0.5*self.OTFB.T_s
        ref_sig = np.cos(-(self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse]) - \
                  1j * np.sin(-(self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse])

        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi

    def test_sum_and_gain(self):
        self.OTFB.V_SET[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)

        self.OTFB.sum_and_gain()

        sig = 2 * np.ones(self.OTFB.n_coarse) * self.OTFB.G_tx / self.OTFB.TWC.R_gen

        np.testing.assert_allclose(self.OTFB.I_GEN[-self.OTFB.n_coarse:], sig)


    @unittest.skip("FIXME")
    def test_gen_response(self):
        # Tests generator response at resonant frequency.
        self.OTFB.I_GEN = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN[self.OTFB.n_coarse] = 1

        self.OTFB.TWC.impulse_response_gen(self.OTFB.TWC.omega_r, self.OTFB.rf_centers)
        self.OTFB.gen_response()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[1:1 + self.OTFB.n_mov_av] = 4 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[0] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[self.OTFB.n_mov_av + 1] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig *= self.OTFB.T_s

        np.testing.assert_allclose(np.abs(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse:]), sig,
                                   atol=5e-5)

        # Tests generator response at carrier frequency.
        self.OTFB.TWC.impulse_response_gen(self.OTFB.omega_c, self.OTFB.rf_centers)

        self.OTFB.I_GEN = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN[self.OTFB.n_coarse] = 1

        self.OTFB.gen_response()

        ref_V_IND_COARSE_GEN = np.load(os.path.join(this_directory, "ref_V_IND_COARSE_GEN.npy"))
        np.testing.assert_allclose(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse:], ref_V_IND_COARSE_GEN)



if __name__ == '__main__':
    unittest.main()
