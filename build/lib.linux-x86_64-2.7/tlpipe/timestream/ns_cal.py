"""Relative phase calibration using the noise source signal.

Inheritance diagram
-------------------

.. inheritance-diagram:: NsCal
   :parts: 2

"""

import warnings
import os
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import h5py
from caput import mpiutil
from caput import mpiarray
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import time
import ephem
import aipy


# def bl2pol_feed(bls, badchn = None):

def bl2pol_feed_inds(bls, feeds, pol = np.array(['xx','yy'])):
    '''
        Convert bls in raw_timestream object to pol and feeds in timestream object.

        Parameter:
        ----------
        bls: (N,2) array, baselines
        feeds: (M,) array, feeds
        pol: (2,) array, pols, should only be ['xx', 'yy'] or ['yy', 'xx'], default ['xx','yy']

        Return:
        -------
        pf: (K, 4) list, K is for bls index, 4 is for pol index and feed index range in form of [pol_i, feed_i, pol_j, feed_j]
        bl_select: is used to get rid of the baselines that have used feeds that are not in feeds array. use bls[bl_select] to get good bls

    '''
    bls = np.array(bls)
    feeds = np.array(feeds)
    pol = np.array(pol)
    
    lbls = len(bls)
    lfeeds = len(feeds)

    incongruity = False

    xpol_ind = np.where(pol == 'xx')[0][0]

    pf = []
    bl_select = []
    for bli, blj in bls:
        fi = int(np.ceil(bli/2.))
        fj = int(np.ceil(blj/2.))
        if not (fi in feeds and fj in feeds):
            if not incongruity:
                print('Detected incongruity between feeds and baselines, should not use all of the present baselines!')
                incongruity = True
            bl_select += [False]
            continue
        else:
            bl_select += [True]
        fi_ind = np.where(feeds == fi)[0][0]
        fj_ind = np.where(feeds == fj)[0][0]
        ip = bli%2-1-xpol_ind # bli is odd -> xpol -> ip = xpol_in ~ -xpol_ind ~ 1-1-xpol_ind
        jp = blj%2-1-xpol_ind # blj is even -> ypol -> jp = xpol_in+1 ~ -xpol_ind-1
        pf += [[ip,fi_ind,jp,fj_ind]]
    bl_select = np.array(bl_select)
    return pf, bl_select

def uni_gain(ps_data, ns_data, phs_only = False):
#    ps_data = h5.File(psfile,'r')
    ps_gain = ps_data['gain'][:] # freq, pol, feed
#    ns_data = h5.File(nsfile,'r')
    if phs_only:
        ns_gain = np.exp(1.J * ns_data['ns_cal_phase'][:])# * ns_data['ns_cal_amp'][:] #time, freq, bl
    else:
        ns_gain = np.exp(1.J * ns_data['ns_cal_phase'][:]) * ns_data['ns_cal_amp'][:] #time, freq, bl
    blorder = ns_data['bl_order'][:]
    polfeed, bl_select = bl2pol_feed_inds(blorder, ps_data['gain'].attrs['feed'], ps_data['gain'].attrs['pol'])
    ns_gain = ns_gain[:,:,bl_select]

    new_ns_gain = np.zeros_like(ns_gain)
    for ii,(ip,fi,jp,fj) in enumerate(polfeed):
        new_ns_gain[:,:,ii] = ns_gain[:,:,ii]*ps_gain[None,:,ip,fi]*ps_gain[None,:,jp,fj].conj()
    return new_ns_gain

class NotEnoughPointToInterpolateWarning(Warning):
    pass
class TransitMasked(Warning):
    pass

class NoPsGainFile(Exception):
    pass
class ImproperAbsGain(Exception):
    pass
class TransitGainNotRecorded(Exception):
    pass
class BeforeStableTime(Exception):
    pass
class AllMasked(Exception):
    pass

class NsCal(timestream_task.TimestreamTask):
    """Relative phase calibration using the noise source signal.

    The noise source can be viewed as a near-field source, its visibility
    can be expressed as

    .. math:: V_{ij}^{\\text{ns}} = C \\cdot e^{i k (r_{i} - r_{j})}

    where :math:`C` is a real constant.

    .. math::

        V_{ij}^{\\text{on}} &= G_{ij} (V_{ij}^{\\text{sky}} + V_{ij}^{\\text{ns}} + n_{ij}) \\\\
        V_{ij}^{\\text{off}} &= G_{ij} (V_{ij}^{\\text{sky}} + n_{ij})

    where :math:`G_{ij}` is the gain of baseline :math:`i,j`.

    .. math::

        V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}} &= G_{ij} V_{ij}^{\\text{ns}} \\\\
                                       &=|G_{ij}| e^{i k \\Delta L} C \\cdot e^{i k (r_{i} - r_{j})} \\\\
                                       & = C |G_{ij}| e^{i k (\\Delta L + (r_{i} - r_{j}))}

    where :math:`\\Delta L` is the equivalent cable length.

    .. math:: \\text{Arg}(V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}}) = k (\\Delta L + (r_{i} - r_{j})) = k \\Delta L + const.

    To compensate for the relative phase change (due to :math:`\\Delta L`) of the
    visibility, we can do

    .. math:: V_{ij}^{\\text{rel-cal}} = e^{-i \\; \\text{Arg}(V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}})} \\, V_{ij}

    .. note::
        Note there is still an unknown (constant) phase factor to be determined in
        :math:`V_{ij}^{\\text{rel-cal}}`, which may be done by absolute calibration.

    """

    params_init = {
                    'num_mean': 5, # use the mean of num_mean signals
                    'unmasked_only': False, # cal for unmasked time points only
                    'phs_only': True, # phase cal only
                    'save_gain': False,
                    'gain_file': 'ns_cal/gain.hdf5',
                    'plot_gain': False, # plot the gain change
                    'phs_unit': 'radian', # or degree
                    'fig_name': 'ns_cal/gain_change',
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'freq_incl': 'all', # or a list of include freq idx
                    'freq_excl': [],
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                    'feed_no': False, # True to use feed number (true baseline) else use channel no
                    'order_bl': True, # True to make small feed no first
                    'absolute_gain_filename': None, # do the absolute calibration using the data from file
                    'use_center_data': False, # if True get rid of the beginning and end points of a noise signal to avoid imcomplete integration period
                    'normal_gain_file': 'ns_cal/normal_gain.hdf5', # the file to store the phase and amplitude as a reference, should be the gain of the transit point. however for dish case, it should be the gain when the dish is rotated to the target region
                    'ns_stable': False, # if True, use the ns point after the dish got stable as normalize signal
                    'last_transit2stable': 40*60. # time in second. the last source transit after this time, the noise will be stable enough to do the ns cal.
                  }

    prefix = 'nc_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        if not 'ns_on' in rt.iterkeys():
            raise RuntimeError('No noise source info, can not do noise source calibration')

        local_bl_size, local_bl_start, local_bl_end = mpiutil.split_local(len(rt['blorder']))

        rt.redistribute('baseline')

        num_mean = self.params['num_mean']
        phs_only = self.params['phs_only']
        save_gain = self.params['save_gain']
        tag_output_iter = self.params['tag_output_iter']
        gain_file = self.params['gain_file']
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        freq_incl = self.params['freq_incl']
        freq_excl = self.params['freq_excl']
        ns_stable = self.params['ns_stable']
        last_transit2stable = self.params['last_transit2stable']

#===================================
        normal_gain_file = self.params['normal_gain_file']
        rt.gain_file = self.params['gain_file']
        absolute_gain_filename = self.params['absolute_gain_filename']
        use_center_data = self.params['use_center_data']
        if use_center_data and rt['ns_on'].attrs['on_time'] <= 2:
            warnings.warn('The period %d <= 2, cannot get rid of the beginning and ending points. Use the whole average automatically!')
            use_center_data = False
#===================================
        # apply abs gain to data if ps_first
        if rt.ps_first:
            if absolute_gain_filename is None:
                raise NoPsGainFile('Absent parameter absolute_gain_file. In ps_first mode, absolute_gain_filename is required to process!')
            if not os.path.exists(output_path(absolute_gain_filename)):
                raise NoPsGainFile('No absolute gain file %s, do the ps calibration first!'%output_path(absolute_gain_filename))
            with h5py.File(output_path(absolute_gain_filename, 'r')) as agfilein:
                if not ns_stable:
#                    try: # if there is transit in this batch of data, build up transit amp and phs
#                        rt.normal_index = np.where(np.abs(rt['jul_date'] - agfilein.attrs['transit_time']) < 2.e-6)[0][0] # to avoid numeric error. 1.e-6 julian date is about 0.1s
                    if rt['sec1970'][0] <= agfilein.attrs['transit_time'] and rt['sec1970'][-1] >= agfilein.attrs['transit_time']:
                        rt.normal_index = np.int64(np.around((agfilein.attrs['transit_time'] - rt['sec1970'][0])/rt.attrs['inttime']))
                        if mpiutil.rank0:
                            print('Detected transit, time index %d, build up transit normalization gain file!'%rt.normal_index)
                        build_normal_file = True
                        rt.normal_phs = np.zeros((rt['vis'].shape[1], local_bl_size))
                        rt.normal_phs.fill(np.nan)
                        if not phs_only:
                            rt.normal_amp = np.zeros((rt['vis'].shape[1], local_bl_size))
                            rt.normal_amp.fill(np.nan)
#                    except IndexError: # no transit in this batch of data, load transit amp and phs from file
                    else: 
                        rt.normal_index = None # no transit flag
                        build_normal_file = False
                        if not os.path.exists(output_path(normal_gain_file)):
                            time_info = (aipy.phs.juldate2ephem(agfilein.attrs['transit_jul'] + 8./24.), aipy.phs.juldate2ephem(rt['jul_date'][0] + 8./24.), aipy.phs.juldate2ephem(rt['jul_date'][-1] + 8./24.))
                            raise TransitGainNotRecorded('The transit %s is not in time range %s to %s and no transit normalization gain was recorded!'%time_info)
                        else:
                            if mpiutil.rank0:
                                print('No transit, use existing transit normalization gain file!')
                            with h5py.File(output_path(normal_gain_file), 'r') as filein:
                                rt.normal_phs = filein['phs'][:, local_bl_start:local_bl_end]
                                if not phs_only:
                                    rt.normal_amp = filein['amp'][:, local_bl_start:local_bl_end]
                else:
                    rt.normal_index = None # no transit flag
                    rt.normal_phs = np.zeros((rt['vis'].shape[1], local_bl_size))
                    rt.normal_phs.fill(np.nan)
                    if not phs_only:
                        rt.normal_amp = np.zeros((rt['vis'].shape[1], local_bl_size))
                        rt.normal_amp.fill(np.nan)
                    try:
                        stable_time = rt['pointingtime'][-1,-1]
                    except KeyError:
                        stable_time = rt['transitsource'][-1, 0]
                    if stable_time > 0: # the time when the antenna will be pointing at the target region was recorded, use it
                        stable_time += 300 # plus 5 min to ensure stable
                    else:
                        stable_time = rt['transitsource'][-2, 0] + last_transit2stable
                    stable_time_jul = datetime.utcfromtimestamp(stable_time)
                    stable_time_jul = ephem.julian_date(stable_time_jul) # convert to julian date, convenient to display
                    if not os.path.exists(output_path(normal_gain_file)):
                        if mpiutil.rank0:
                            print('Normalization gain file has not been built yet. Try to build it!')
                            print('Recorded transit Time: %s'%aipy.phs.juldate2ephem(agfilein.attrs['transit_jul'] + 8./24.))
                            print('Last transit Time: %s'%aipy.phs.juldate2ephem(ephem.julian_date(datetime.utcfromtimestamp(rt['transitsource'][-2,0])) + 8./24.))
                            print('First time point of this data: %s'%aipy.phs.juldate2ephem(rt['jul_date'][0] + 8./24.))
                        build_normal_file = True
#                        if rt['jul_date'][0] < stable_time:
                        if rt.attrs['sec1970'][0] < stable_time:
                            raise BeforeStableTime('The beginning time point is %s, but only after %s, will the noise source be stable. Abort the noise calibration!'%(aipy.phs.juldate2ephem(rt['jul_date'][0] + 8./24.), aipy.phs.juldate2ephem(stable_time_jul + 8./24.)))
                    else:
                        if mpiutil.rank0:
                            print('Use existing normalization gain file!')
                        build_normal_file = False
                        with h5py.File(output_path(normal_gain_file), 'r') as filein:
                            rt.normal_phs = filein['phs'][:, local_bl_start:local_bl_end]
                            if not phs_only:
                                rt.normal_amp = filein['amp'][:, local_bl_start:local_bl_end]
                # apply absolute gain
                absgain = agfilein['gain'][:]
                polfeed, _ = bl2pol_feed_inds(rt.local_bl, agfilein['gain'].attrs['feed'][:], agfilein['gain'].attrs['pol'][:])
                for ii, (ipol, ifeed, jpol, jfeed) in enumerate(polfeed):
#                    rt.local_vis[:,:,ii] /= (absgain[None,:,ipol,ifeed-1] * absgain[None,:,jpol,jfeed-1].conj())
                    rt.local_vis[:,:,ii] /= (absgain[None,:,ipol,ifeed] * absgain[None,:,jpol,jfeed].conj())
#===================================
        nt = rt.local_vis.shape[0]
        if num_mean <= 0:
            raise RuntimeError('Invalid num_mean = %s' % num_mean)
        ns_on = rt['ns_on'][:]
        ns_on = np.where(ns_on, 1, 0)
        diff_ns = np.diff(ns_on)
        inds = np.where(diff_ns==1)[0] # NOTE: these are inds just 1 before the first ON
        if not rt.FRB_cal: # for FRB there might be just one noise point, avoid waste
            if inds[0]-1 < 0: # no off data in the beginning to use
                inds = inds[1:]
            if inds[-1]+2 > nt-1: # no on data in the end to use
                inds = inds[:-1]

        if save_gain:
            num_inds = len(inds)
            shp = (num_inds,)+rt.local_vis.shape[1:]
            dtype = rt.local_vis.real.dtype
            # create dataset to record ns_cal_time_inds
            rt.create_time_ordered_dataset('ns_cal_time_inds', inds)
            # create dataset to record ns_cal_phase
            ns_cal_phase = np.empty(shp, dtype=dtype)
            ns_cal_phase[:] = np.nan
            ns_cal_phase = mpiarray.MPIArray.wrap(ns_cal_phase, axis=2, comm=rt.comm)
            rt.create_freq_and_bl_ordered_dataset('ns_cal_phase', ns_cal_phase, axis_order=(None, 1, 2))
            rt['ns_cal_phase'].attrs['unit'] = 'radians'
            if not phs_only:
                # create dataset to record ns_cal_amp
                ns_cal_amp = np.empty(shp, dtype=dtype)
                ns_cal_amp[:] = np.nan
                ns_cal_amp = mpiarray.MPIArray.wrap(ns_cal_amp, axis=2, comm=rt.comm)
                rt.create_freq_and_bl_ordered_dataset('ns_cal_amp', ns_cal_amp, axis_order=(None, 1, 2))

        if bl_incl == 'all':
            bls_plt = [ tuple(bl) for bl in rt.bl ]
        else:
            bls_plt = [ bl for bl in bl_incl if not bl in bl_excl ]

        if freq_incl == 'all':
            freq_plt = range(rt.freq.shape[0])
        else:
            freq_plt = [ fi for fi in freq_incl if not fi in freq_excl ]

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        if rt.ps_first:
            rt.freq_and_bl_data_operate(self.cal, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False, num_mean=num_mean, inds=inds, bls_plt=bls_plt, freq_plt=freq_plt, build_normal_file = build_normal_file)
        else:
            rt.freq_and_bl_data_operate(self.cal, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False, num_mean=num_mean, inds=inds, bls_plt=bls_plt, freq_plt=freq_plt)

        if save_gain:
            interp_mask_ratio = mpiutil.allreduce(np.sum(rt.interp_mask_count))/1./mpiutil.allreduce(np.size(rt.interp_mask_count)) * 100.
            if interp_mask_ratio > 50. and rt.normal_index is not None:
                warnings.warn('%.1f%% of the data was masked due to shortage of noise points for interpolation(need at least 4 to perform cubic spline)! The pointsource calibration may not be done due to too many masked points!'%interp_mask_ratio, NotEnoughPointToInterpolateWarning)
            if interp_mask_ratio > 80.:
                rt.interp_all_masked = True
            # gather bl_order to rank0
            bl_order = mpiutil.gather_array(rt['blorder'].local_data, axis=0, root=0, comm=rt.comm)
            # gather ns_cal_phase / ns_cal_amp to rank 0
            ns_cal_phase = mpiutil.gather_array(rt['ns_cal_phase'].local_data, axis=2, root=0, comm=rt.comm)
            phs_unit = rt['ns_cal_phase'].attrs['unit']
            rt.delete_a_dataset('ns_cal_phase', reserve_hint=False)
            if not phs_only:
                ns_cal_amp = mpiutil.gather_array(rt['ns_cal_amp'].local_data, axis=2, root=0, comm=rt.comm)
                rt.delete_a_dataset('ns_cal_amp', reserve_hint=False)

            if tag_output_iter:
                gain_file = output_path(gain_file, iteration=self.iteration)
            else:
                gain_file = output_path(gain_file)

            if rt.ps_first:
                phase_finite_count = mpiutil.allreduce(np.isfinite(rt.normal_phs).sum())
                if not phs_only:
                    amp_finite_count = mpiutil.allreduce(np.isfinite(rt.normal_amp).sum())
            if mpiutil.rank0:
                if rt.ps_first and build_normal_file:
                    if phase_finite_count == 0:
                        raise AllMasked('All values are masked when calculating phase!')
                    if not phs_only:
                        if amp_finite_count == 0:
                            raise AllMasked('All values are masked when calculating amplitude!')
                    with h5py.File(output_path(normal_gain_file), 'w') as tapfilein:
                        tapfilein.create_dataset('amp',(rt['vis'].shape[1], rt['vis'].shape[2]))
                        tapfilein.create_dataset('phs',(rt['vis'].shape[1], rt['vis'].shape[2]))
                with h5py.File(gain_file, 'w') as f:
                    # save time
                    f.create_dataset('time', data=rt['jul_date'][:])
                    f['time'].attrs['unit'] = 'Julian date'
                    # save freq
                    f.create_dataset('freq', data=rt['freq'][:])
                    f['freq'].attrs['unit'] = rt['freq'].attrs['unit']
                    # save bl
                    f.create_dataset('bl_order', data=bl_order)
                    # save ns_cal_time_inds
                    f.create_dataset('ns_cal_time_inds', data=rt['ns_cal_time_inds'][:])
                    # save ns_cal_phase
                    f.create_dataset('ns_cal_phase', data=ns_cal_phase)
                    f['ns_cal_phase'].attrs['unit'] = phs_unit
                    f['ns_cal_phase'].attrs['dim'] = '(time, freq, bl)'
                    if not phs_only:
                        # save ns_cal_amp
                        f.create_dataset('ns_cal_amp', data=ns_cal_amp)

                    # save channo
                    f.create_dataset('channo', data=rt['channo'][:])
                    f['channo'].attrs['dim'] = rt['channo'].attrs['dimname']
                    if rt.exclude_bad:
                        f['channo'].attrs['badchn'] = rt['channo'].attrs['badchn']

                    if not (absolute_gain_filename is None):
                        if not os.path.exists(output_path(absolute_gain_filename)):
                            raise NoPsGainFile('No absolute gain file %s, do the ps calibration first!'%output_path(absolute_gain_filename))
                        with h5py.File(output_path(absolute_gain_filename,'r')) as abs_gain:
                            new_gain = uni_gain(abs_gain, f, phs_only = phs_only)
                            f.create_dataset('uni_gain', data = new_gain)
                            f['uni_gain'].attrs['dim'] = '(time, freq, bl)'

            if rt.ps_first and build_normal_file:
                if mpiutil.rank0:
                    print('Start write normalization gain into %s'%output_path(normal_gain_file))
                for i in range(10):
                    try:
                        for ii in range(mpiutil.size):
                            if ii == mpiutil.rank:
                                with h5py.File(output_path(normal_gain_file), 'r+') as fileout:
                                    fileout['phs'][:,local_bl_start:local_bl_end] = rt.normal_phs[:,:]
                                    if not phs_only:
                                        fileout['amp'][:,local_bl_start:local_bl_end] = rt.normal_amp[:,:]
                            mpiutil.barrier()
                        break
                    except IOError:
                        time.sleep(0.5)
                        continue
            rt.delete_a_dataset('ns_cal_time_inds', reserve_hint=False)

        return super(NsCal, self).process(rt)

    def cal(self, vis, vis_mask, li, gi, fbl, rt, **kwargs):
        """Function that does the actual cal."""

        unmasked_only = self.params['unmasked_only']
        phs_only = self.params['phs_only']
        save_gain = self.params['save_gain']
        plot_gain = self.params['plot_gain']
        phs_unit = self.params['phs_unit']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']
        feed_no = self.params['feed_no']
        order_bl = self.params['order_bl']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration
        num_mean = kwargs['num_mean']
        inds = kwargs['inds'] # inds is the point before noise on
        bls_plt = kwargs['bls_plt']
        freq_plt = kwargs['freq_plt']
        if rt.ps_first:
            build_normal_file = kwargs['build_normal_file']
        use_center_data = self.params['use_center_data']
        ns_stable = self.params['ns_stable']
        

        if np.prod(vis.shape) == 0 :
            return

        lfi, lbi = li # local freq and bl index
        fi = gi[0] # freq idx for this cal
        bl = tuple(fbl[1]) # bl for this cal

        nt = vis.shape[0]
        on_time = rt['ns_on'].attrs['on_time']
        # off_time = rt['ns_on'].attrs['off_time']
        period = rt['ns_on'].attrs['period']

        # the calculated phase and amp will be at the ind just 1 before ns ON (i.e., at the ind of the last ns OFF)
        valid_inds = []
        phase = []
        if not phs_only:
            amp = []
        ii_range = []
        for ii, ind in enumerate(inds):
            # drop the first and the last ind, as it may lead to exceptional vals
            if (ind == inds[0] or ind == inds[-1]) and not rt.FRB_cal:
                continue

            lower = ind - num_mean
            off_sec = np.ma.array(vis[lower:ind], mask=(~np.isfinite(vis[lower:ind]))&vis_mask[lower:ind])
            if off_sec.count() == 0: # all are invalid values
                continue
            if unmasked_only and off_sec.count() < max(2, num_mean/2): # more valid sample to make stable
                continue

            valid = True
            upper = ind + 1 + on_time
            off_mean = np.ma.mean(off_sec)
            if use_center_data:
                if ind + 2 < upper - 1:
                    this_on = np.ma.masked_invalid(vis[ind+2:upper-1]) # all on signal
                else:
                    continue
            else:
                this_on = np.ma.masked_invalid(vis[ind+1:upper]) # all on signal
            # just to avoid the case of all invalid on values
            if this_on.count() > 0:
                on_mean = np.ma.mean(this_on) # mean for all valid on signals
            else:
                continue
            diff = on_mean - off_mean
            phs = np.angle(diff) # in radians
            if not np.isfinite(phs):
                valid = False
            if not phs_only:
                amp_ = np.abs(diff)
                if not (np.isfinite(amp_) and amp_ > 1.0e-8): # amp_ should > 0
                    valid = False
            if not valid:
                continue
            valid_inds.append(ind)
            if save_gain:
                rt['ns_cal_phase'].local_data[ii, lfi, lbi] = phs
            phase.append( phs ) # in radians
            if not phs_only:
                if save_gain:
                    ii_range += [ii]
                    rt['ns_cal_amp'].local_data[ii, lfi, lbi] = amp_
                amp.append( amp_ )

        # not enough valid data to do the ns_cal
        num_valid = len(valid_inds)
        if (num_valid <= 3 and not rt.FRB_cal) or num_valid < 1:
            print 'Only have %d valid points, mask all for fi = %d, bl = (%d, %d)...' % (num_valid, fbl[0], fbl[1][0], fbl[1][1])
            vis_mask[:] = True # mask the vis as no ns_cal has done
            return

        phase = np.unwrap(phase) # unwrap 2pi discontinuity
        if not phs_only:
            if not rt.ps_first:
                rt['ns_cal_amp'].local_data[ii_range, lfi, lbi] = rt['ns_cal_amp'].local_data[ii_range, lfi, lbi] / np.median(amp)
                amp = np.array(amp) / np.median(amp) # normalize

        # split valid_inds into consecutive chunks
        intervals = [0] + (np.where(np.diff(valid_inds) > 5 * period)[0] + 1).tolist() + [num_valid]
        itp_inds = []
        itp_phase = []
        if not phs_only:
            itp_amp = []
        for i in xrange(len(intervals) -1):
            this_chunk = valid_inds[intervals[i]:intervals[i+1]]
            if len(this_chunk) > 3:
                itp_inds.append(this_chunk)
                itp_phase.append(phase[intervals[i]:intervals[i+1]])
                if not phs_only:
                    itp_amp.append(amp[intervals[i]:intervals[i+1]])

        # if no such chunk, mask all the data
        num_itp = len(itp_inds)
        if num_itp == 0:
            rt.interp_mask_count.append(1)
            vis_mask[:] = True
        else:
            rt.interp_mask_count.append(0)

        # get itp pairs
        itp_pairs = []
        for it in itp_inds:
            # itp_pairs.append((max(0, it[0]-off_time), min(nt, it[-1]+period)))
            itp_pairs.append((max(0, it[0]-5), min(nt, it[-1]+5))) # not to out interpolate two much, which may lead to very inaccurate values

        # get mask pairs
        mask_pairs = []
        for i in xrange(num_itp):
            if i == 0:
                mask_pairs.append((0, itp_pairs[i][0]))
            if i == num_itp - 1:
                mask_pairs.append((itp_pairs[i][-1], nt))
            else:
                mask_pairs.append((itp_pairs[i][-1], itp_pairs[i+1][0]))

        # set mask for inds in mask_pairs
        for mp1, mp2 in mask_pairs:
            vis_mask[mp1:mp2] = True

        # interpolate for inds in itp_inds
        all_phase = np.array([np.nan]*nt)
        if rt.ps_first:
            if rt.normal_index is not None:
                normal_phs = None
                normal_amp = None
                interp_inds = np.array([])
                new_amp = np.array([])
            elif not ns_stable:
                normal_phs = rt.normal_phs[lfi,lbi]
                normal_amp = rt.normal_amp[lfi,lbi]
                interp_inds = np.array([])
                new_amp = np.array([])
            else:
                if build_normal_file:
                    rt.normal_phs[lfi, lbi] = rt['ns_cal_phase'].local_data[0, lfi, lbi]
                    rt.normal_amp[lfi, lbi] = rt['ns_cal_amp'].local_data[0, lfi, lbi]
                normal_phs = rt.normal_phs[lfi,lbi]
                normal_amp = rt.normal_amp[lfi,lbi]
                interp_inds = np.array([])
                new_amp = np.array([])
        for this_inds, this_phase, (i1, i2) in zip(itp_inds, itp_phase, itp_pairs):
            # no need to interpolate for auto-correlation
            if bl[0] == bl[1]:
                all_phase[i1:i2] = 0
                if rt.ps_first:
                    interp_inds = np.concatenate([interp_inds, np.arange(i1,i2)])
                    if rt.normal_index>=i1 and rt.normal_index<i2 and rt.normal_index is not None:
                        normal_phs = 0
                        rt.normal_phs[lfi, lbi] = normal_phs
            else:
                f = InterpolatedUnivariateSpline(this_inds, this_phase)
                this_itp_phs = f(np.arange(i1, i2))
                # # make the interpolated values in the appropriate range
                # this_itp_phs = np.where(this_itp_phs>np.pi, np.pi, this_itp_phs)
                # this_itp_phs = np.where(this_itp_phs<-np.pi, np.pi, this_itp_phs)
                all_phase[i1:i2] = this_itp_phs
                # do phase cal for this range of inds
                if rt.ps_first:
                    interp_inds = np.concatenate([interp_inds, np.arange(i1,i2)])
                    if rt.normal_index>=i1 and rt.normal_index<i2 and rt.normal_index is not None:
                        normal_phs = this_itp_phs[rt.normal_index - i1]
                        rt.normal_phs[lfi, lbi] = normal_phs
                vis[i1:i2] = vis[i1:i2] * np.exp(-1.0J * this_itp_phs)
        if rt.ps_first:
            interp_inds = np.int64(interp_inds)
            if normal_phs is None:
                normal_phs = np.nan
                warnings.warn('The transit point has been masked for fi = %d, bl = (%d, %d)... when calculating phase! Maybe the noise points are too sparse!'%(fbl[0], fbl[1][0], fbl[1][1]),TransitMasked)
            vis[interp_inds] = vis[interp_inds]*np.exp(1.0J * normal_phs)
            rt['ns_cal_phase'].local_data[ii_range, lfi, lbi] = rt['ns_cal_phase'].local_data[ii_range, lfi, lbi] - normal_phs

        if not phs_only:
            all_amp = np.array([np.nan]*nt)
            for this_inds, this_amp, (i1, i2) in zip(itp_inds, itp_amp, itp_pairs):
                f = InterpolatedUnivariateSpline(this_inds, this_amp)
                this_itp_amp = f(np.arange(i1, i2))
                all_amp[i1:i2] = this_itp_amp
                # do amp cal for this range of inds
                if rt.ps_first:
                    new_amp = np.concatenate([new_amp, this_itp_amp])
                    if rt.normal_index>=i1 and rt.normal_index<i2 and rt.normal_index is not None:
                        normal_amp = this_itp_amp[rt.normal_index - i1]
                        rt.normal_amp[lfi, lbi] = normal_amp
                else:
                    vis[i1:i2] = vis[i1:i2] / this_itp_amp

            if rt.ps_first:
                if normal_amp is None:
                    normal_amp = np.nan
                    warnings.warn('The transit point has been masked for frequency %d baseline %d when calculating amplitude! Maybe the noise points are too sparse!'%(lfi,lbi),TransitMasked)
                new_amp = new_amp / normal_amp
                vis[interp_inds] = vis[interp_inds]/new_amp
                rt['ns_cal_amp'].local_data[ii_range, lfi, lbi] = rt['ns_cal_amp'].local_data[ii_range, lfi, lbi] / normal_amp

        if plot_gain and (bl in bls_plt and fi in freq_plt):
            plt.figure()
            if phs_only:
                fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots(2, sharex=True)
            ax_val = np.array([ (datetime.utcfromtimestamp(sec) + timedelta(hours=8)) for sec in rt['sec1970'][:] ])
            xlabel = '%s' % ax_val[0].date()
            ax_val = mdates.date2num(ax_val)
            if order_bl and (bl[0] > bl[1]):
                # negate phase as for the conj of vis
                all_phase = np.where(np.isfinite(all_phase), -all_phase, np.nan)
                phase = np.where(np.isfinite(phase), -phase, np.nan)
            if phs_unit == 'degree': # default to radians
                all_phase = np.degrees(all_phase)
                phase = np.degrees(phase)
                ylabel = r'$\Delta \phi$ / degree'
            else:
                ylabel = r'$\Delta \phi$ / radian'
            if phs_only:
                ax.plot(ax_val, all_phase)
                ax.plot(ax_val[valid_inds], phase, 'ro')
                ax1 = ax
            else:
                ax[0].plot(ax_val, all_amp)
                ax[0].plot(ax_val[valid_inds], amp, 'ro')
                ax[0].set_ylabel(r'$\Delta |g|$')
                ax[1].plot(ax_val, all_phase)
                ax[1].plot(ax_val[valid_inds], phase, 'ro')
                ax1 = ax[1]
            duration = (ax_val[-1] - ax_val[0])
            dt = duration / nt
            ext = max(0.05*duration, 5*dt)
            # if phs_unit == 'degree': # default to radians
            #     ax1.set_ylim([-180, 180])
            #     ax1.set_yticks([-180, -120, -60, 0, 60, 120, 180])
            # else:
            #     ax1.set_ylim([-np.pi, np.pi])
            ax1.set_xlim([ax_val[0]-ext, ax_val[-1]+ext])
            ax1.xaxis_date()
            date_format = mdates.DateFormatter('%H:%M')
            ax1.xaxis.set_major_formatter(date_format)
            if rotate_xdate:
                # set the x-axis tick labels to diagonal so it fits better
                fig.autofmt_xdate()
            else:
                # reduce the number of tick locators
                locator = MaxNLocator(nbins=6)
                ax1.xaxis.set_major_locator(locator)
                ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)

            if feed_no:
                pol = rt['bl_pol'].local_data[li[1]]
                bl = tuple(rt['true_blorder'].local_data[li[1]])
                if order_bl and (bl[0] > bl[1]):
                    bl = (bl[1], bl[0])
                fig_name = '%s_%f_%d_%d_%s.png' % (fig_prefix, fbl[0], bl[0], bl[1], rt.pol_dict[pol])
            else:
                fig_name = '%s_%f_%d_%d.png' % (fig_prefix, fbl[0], fbl[1][0], fbl[1][1])
            if tag_output_iter:
                fig_name = output_path(fig_name, iteration=iteration)
            else:
                fig_name = output_path(fig_name)
            plt.savefig(fig_name)
            plt.close()
