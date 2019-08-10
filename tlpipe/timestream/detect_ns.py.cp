"""Detect noise source signal.

Inheritance diagram
-------------------

.. inheritance-diagram:: Detect
   :parts: 2

"""

import warnings
from collections import Counter
import numpy as np
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from caput import mpiutil
from caput import mpiarray
from tlpipe.utils.path_util import output_path
import os

def cmp_cd(cpinds, cninds, pinds, ninds, period, threshold = 2./3):
    '''
    compare calculated points and detected points to detect abnormal and lost points
    cpinds: on points from calculation
    cninds: off points from calculation
    pinds: on points from detection
    ninds: off points from detection
    return: signal, n
        signal = 'lost': elements in cninds and cpinds with index >= n should minus one
        signal = 'add': elements in cninds and cpinds with index >= n should add one
        signal = 'ok': no lost points, no added points and no abnormal
        signal = 'abnormal': abnormal
    '''
    pinds = set(pinds)
    ninds = set(ninds)
    cpinds = np.array(cpinds)
    cninds = np.array(cninds)

    pcommon = sorted(pinds.intersection(cpinds))
    ncommon = sorted(ninds.intersection(cninds))
    lpcommon = sorted(pinds.intersection(cpinds - 1))
    lncommon = sorted(ninds.intersection(cninds - 1))
    apcommon = sorted(pinds.intersection(cpinds + 1))
    ancommon = sorted(ninds.intersection(cninds + 1))
    maxcommon = max(len(pcommon), len(ncommon), len(lpcommon), len(lncommon), len(apcommon), len(ancommon))

    if len(lpcommon) + len(pcommon) > maxcommon or len(lncommon) + len(ncommon) > maxcommon:
        if len(lpcommon) + len(pcommon) >= threshold*len(cpinds) or len(lncommon) + len(ncommon) >= threshold*len(cninds): # may lost one in the data
            if pcommon[-1] + period == lpcommon[0] + 1:
                return 'lost', list(cpinds - 1).index(lpcommon[0]) # elements in cpinds and cninds with index >= this should minus one
            elif ncommon[-1] + period == lncommon[0] + 1:
                return 'lost', list(cninds - 1).index(lncommon[0]) # elements in cpinds and cninds with index >= this should minus one
    if len(apcommon) + len(pcommon) > maxcommon or len(ancommon) + len(ncommon) > maxcommon:
        if len(apcommon) + len(pcommon) >= threshold*len(cpinds) or len(ancommon) + len(ncommon) >= threshold*len(cninds): # may added one in the data
            if pcommon[-1] + period == apcommon[0] - 1:
                return 'add', list(cpinds + 1).index(apcommon[0]) # elements in cpinds and cninds with index >= this should plus one
            elif ncommon[-1] + period == ancommon[0] - 1:
                return 'add', list(cninds + 1).index(ancommon[0]) # elements in cpinds and cninds with index >= this should plus one
    if len(pcommon) >= threshold*len(cpinds) or len(ncommon) >= threshold*len(cninds): # no abnormal and no lost
        return 'ok', -1
    elif len(lpcommon) >= threshold*len(cpinds) or len(lncommon) >= threshold*len(cninds): # lost one at beginning
        return 'lost', 0
    elif len(apcommon) >= threshold*len(cpinds) or len(ancommon) >= threshold*len(cninds): # lost one at beginning
        return 'add', 0
    else: # unknown error, abnormal data(may be badchn)
        return 'abnormal', -2
class IncontinuousData(Exception):
    pass
class OverlapData(Exception):
    pass
class NoiseNotEnough(Exception):
    pass
class AbnormalPoints(Exception):
    pass
class NoNoisePoint(Exception):
    pass
class DetectNoiseFailure(Exception):
    pass
class LostPointBegin(Warning):
    pass
class LostPointMiddle(Warning):
    pass
class AdditionalPointBegin(Warning):
    pass
class AdditionalPointMiddle(Warning):
    pass
class DetectedLostAddedAbnormal(Warning):
    pass
class ChangeReferenceTime(Warning):
    pass
class Detect(timestream_task.TimestreamTask):
    """Detect noise source signal.

    This task automatically finds out the time points that the noise source
    is **on**, and creates a new bool dataset "ns_on" with elements *True*
    corresponding to time points when the noise source is **on**.

    """

    params_init = {
                    'channel': None, # use auto-correlation of this channel
                    'sigma': 3.0,
                    'mask_near': 1, # how many extra near ns_on int_time to be masked
                    'FRB_cal': False, # do real-time calibration for FRB, wait untill enough noise point is recorded
                    'ns_arr_file': 'ns_cal/ns_arr_file.npz', # contains the noise points and total length, once the period and start point of noise is gotten, it will not be change
                    'ns_prop_file': 'ns_cal/ns_prop_file.npz', # contains the period, on_time, off_time, reference_time of the noise signal
                    'num_noise': 10, # in FRB case, number of noise points that are used to do calibration, 2*num_noise/3 noise points are required to do the calibration
                    'change_reference_tol': 2
                  }

    prefix = 'dt_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        channel = self.params['channel']
        sigma = self.params['sigma']
        mask_near = max(0, int(self.params['mask_near']))
#================================================
        rt.FRB_cal = self.params['FRB_cal']
        ns_arr_file = self.params['ns_arr_file']
        ns_prop_file = self.params['ns_prop_file']
        num_noise = self.params['num_noise']
        change_reference_tol = self.params['change_reference_tol']
        abnormal_ns_save_file = 'ns_cal/abnormal_ns.npz'
#================================================

        rt.redistribute(0) # make time the dist axis

        time_span = rt.local_vis.shape[0]
        total_time_span = mpiutil.allreduce(time_span)

        auto_inds = np.where(rt.bl[:, 0]==rt.bl[:, 1])[0].tolist() # inds for auto-correlations
        channels = [ rt.bl[ai, 0] for ai in auto_inds ] # all chosen channels
        if channel is not None:
            if channel in channels:
                bl_ind = auto_inds[channels.index(channel)]
            else:
                bl_ind = auto_inds[0]
                if mpiutil.rank0:
                    print 'Warning: Required channel %d doen not in the data, use channel %d instead' % (channel, rt.bl[bl_ind, 0])
        else:
            bl_ind = auto_inds[0]
        # move the chosen channel to the first
        auto_inds.remove(bl_ind)
        auto_inds = [bl_ind] + auto_inds
        if rt.FRB_cal:
            ns_arr_end_time = mpiutil.bcast(rt['jul_date'][total_time_span - 1], root = mpiutil.size - 1)
            if os.path.exists(output_path(ns_arr_file)) and not os.path.exists(output_path(ns_prop_file)):
                filein = np.load(output_path(ns_arr_file))
                # add 0 to avoid a single float or int become an array
                ns_arr_pinds = filein['on_inds'] + 0
                ns_arr_ninds = filein['off_inds'] + 0
                ns_arr_pinds = list(ns_arr_pinds)
                ns_arr_ninds = list(ns_arr_ninds)
                ns_arr_len = filein['time_len'] + 0
                ns_arr_num = filein['ns_num'] + 0
                ns_arr_bl = filein['auto_chn'] + 0
                ns_arr_start_time = filein['start_time']
                overlap_index_span = np.around(np.float128(filein['end_time'] - mpiutil.bcast(rt['jul_date'][0], root = 0))*24.*3600./rt.attrs['inttime']) + 1
                if overlap_index_span > 0:
                    raise OverlapData('Overlap of data occured when trying to build up noise property file! In julian date, the end time of previous data is %.10f, while the start time of this data is %.10f. The overlap span in index is %d.'%(filein['end_time'], mpiutil.bcast(rt['jul_date'][0], root = 0), overlap_index_span))
            elif not os.path.exists(output_path(ns_prop_file)):
                ns_arr_pinds = []
                ns_arr_ninds = []
                ns_arr_len = 0
                ns_arr_num = -1 # to distinguish the first process

                ns_arr_start_time = mpiutil.bcast(rt['jul_date'][0], root = 0)

        if rt.FRB_cal and os.path.exists(output_path(ns_prop_file)):
            print('Use existing ns property file %s to do calibration.'%output_path(ns_prop_file))
            ns_prop_data = np.load(output_path(ns_prop_file))
            period = ns_prop_data['period'] + 0
            on_time = ns_prop_data['on_time'] + 0
            off_time = ns_prop_data['off_time'] + 0
            reference_time = ns_prop_data['reference_time'] + 0 # in julian date, is a start point of noise
            if 'lost_count' in ns_prop_data.files:
                lost_count_before = ns_prop_data['lost_count']
            else:
                lost_count_before = 0
            if 'added_count' in ns_prop_data.files:
                added_count_before = ns_prop_data['added_count']
            else:
                added_count_before = 0

            this_time_start = mpiutil.bcast(rt['jul_date'][0], root=0)
            skip_inds = int(np.around((this_time_start - reference_time)*86400.0/rt.attrs['inttime'])) # the number of index between the reference time and the start point
            on_start = period - skip_inds%period
#            if total_time_span < period:
#                raise Exception('Time span of data %d is shorter than period %d!'%(total_time_span, period))
            if total_time_span < on_start + on_time:
                raise NoNoisePoint('Calculated from previous data, this data contains no noise point or does not contain complete noise signal!')
            # check whether there are lost points
            # only consider that the case that there are one lost point and only consider the on points
#================================================
            abnormal_count = -1
            lost_one_point = 0
            added_one_point = 0
            abnormal_list = []
            lost_one_list = []
            added_one_list = []
            lost_one_pos = []
            added_one_pos = []
            complete_period_num = (total_time_span - on_start - on_time - 1)//period # the number of periods in the data
            on_points = [on_start + i*period for i in range(complete_period_num + 1)]
            off_points = [on_start + on_time + i*period for i in range(complete_period_num + 1)]
            for bl_ind in auto_inds:
                this_chan = rt.bl[bl_ind, 0] # channel of this bl_ind
                vis = np.ma.array(rt.local_vis[:, :, bl_ind].real, mask=rt.local_vis_mask[:, :, bl_ind])
                cnt = vis.count() # number of not masked vals
                total_cnt = mpiutil.allreduce(cnt)
                vis_shp = rt.vis.shape
                ratio = float(total_cnt) / np.prod((vis_shp[0], vis_shp[1])) # ratio of un-masked vals
                if ratio < 0.5: # too many masked vals
                    continue

                if abnormal_count < 0:
                    abnormal_count = 0
                tt_mean = mpiutil.gather_array(np.ma.mean(vis, axis=-1).filled(0), root=None) # mean for all freq, for a specific auto bl 
                df =  np.diff(tt_mean, axis=-1)
                pdf = np.where(df>0, df, 0)
                pinds = np.where(pdf>pdf.mean() + sigma*pdf.std())[0]
                pinds = pinds + 1
#====================================
                if len(pinds) == 0: # no raise, might be badchn, continue
                    continue
#====================================
                pinds1 = [pinds[0]]
                for pi in pinds[1:]:
                    if pi - pinds1[-1] > 1:
                        pinds1.append(pi)
                pinds = np.array(pinds1)

                ndf = np.where(df<0, df, 0)
                ninds = np.where(ndf<ndf.mean() - sigma*ndf.std())[0]
                ninds = ninds + 1
                ninds = ninds[::-1]
#====================================
                if len(ninds) == 0: # no raise, might be badchn, continue
                    continue
#====================================
                ninds1 = [ninds[0]]
                for ni in ninds[1:]:
                    if ni - ninds1[-1] < -1:
                        ninds1.append(ni)
                ninds = np.array(ninds1[::-1])
                cmp_signal, cmp_res = cmp_cd(on_points, off_points, pinds, ninds, period, 2./3)
                if cmp_signal == 'ok': # normal
                    continue
                elif cmp_signal == 'abnormal': # abnormal
                    abnormal_count += 1
                    abnormal_list += [this_chan]
                elif cmp_signal == 'lost': # lost point
                    lost_one_point += 1
                    lost_one_list += [this_chan]
                    lost_one_pos += [cmp_res]
                    continue
                elif cmp_signal == 'add': # added point
                    added_one_point += 1
                    added_one_list += [this_chan]
                    added_one_pos += [cmp_res]
                    continue
                else:
                    raise Exception('Unknown compare signal!')
                    
#                if on_start in pinds or on_start + on_time in ninds:
#                    # to avoid the effect of interference
#                    continue
#                elif on_start - 1 in pinds or on_start + on_time - 1 in ninds:
#                    lost_one_point += 1
#                    lost_one_list += [this_chan]
#                    continue
#                elif on_start - 1 < 0 and on_start + period - 1 in pinds:
#                    lost_one_point += 1
#                    lost_one_list += [this_chan]
#                    continue
#                else:
#                    abnormal_count += 1
#                    abnormal_list += [this_chan]


            if abnormal_count < 0:
                raise NoNoisePoint('No noise points are detected from this data or the data contains too many masked points!')
            elif abnormal_count > len(auto_inds)/3:
                if mpiutil.rank0:
                    np.savez(output_path(abnormal_ns_save_file), on_inds = pinds, off_inds = ninds, on_start = on_start, period = period, on_time = on_time)
                mpiutil.barrier()
                raise AbnormalPoints('Something rather than one lost point happened. The expected start point is %d, period %d, on_time %d, but the pinds and ninds are: '%(on_start, period, on_time), pinds, ninds)
            elif lost_one_point > 2*len(auto_inds)/3:
                uniques, counts = np.unique(lost_one_pos, return_counts = True)
                maxcount = np.argmax(counts)
                if counts[maxcount] > 2*len(auto_inds)/3:
                    lost_position = uniques[maxcount]
                else:
                    raise AbnormalPoints('More than 2/3 baselines have detected lost points but do not have a universal lost position, some unexpected error happened!\nChannels that probably lost one point and the position: %s'%str(zip(lost_one_list, lost_one_pos)))
                if mpiutil.rank0:
                    if lost_position == 0:
                        warnings.warn('One lost point before the data is detected!', LostPointBegin)
                    else:
                        warnings.warn('One lost point before index %d is detected!'%on_points[lost_position], LostPointMiddle)
#                on_start = on_start - 1
                if lost_position == 0:
                    lost_count_before += 1
                elif lost_count_before == 0:
                    lost_count_before += 1
                on_points = np.array(on_points)
                on_points[lost_position:] = on_points[lost_position:] - 1
                off_points = np.array(off_points)
                off_points[lost_position:] = off_points[lost_position:] - 1
                if on_points[0] < 0:
                    on_points = on_points[1:]
                    off_points = off_points[1:]
                if mpiutil.rank0:
                    if lost_count_before >= change_reference_tol:
                        reference_time -= 1/86400.0*rt.attrs['inttime']
                        warnings.warn('Move the reference time one index earlier to compensate!',ChangeReferenceTime)
                        np.savez(output_path(ns_prop_file), period = period, on_time = on_time, off_time = off_time, reference_time = reference_time, lost_count = 0, added_count = 0)
                    else:
                        print('The number of recorded lost points was %d while tolerance is %d. Do not change the reference time.'%(lost_count_before, change_reference_tol))
                        np.savez(output_path(ns_prop_file), period = period, on_time = on_time, off_time = off_time, reference_time = reference_time, lost_count = lost_count_before, added_count = 0)
#                mpiutil.barrier()
            elif added_one_point > 2*len(auto_inds)/3:
                uniques, counts = np.unique(added_one_pos, return_counts = True)
                maxcount = np.argmax(counts)
                if counts[maxcount] > 2*len(auto_inds)/3:
                    added_position = uniques[maxcount]
                else:
                    raise AbnormalPoints('More than 2/3 baselines have detected additional points but do not have a universal adding position, some unexpected error happened!\nChannels that probably added one point and the position: %s'%str(zip(added_one_list, added_one_pos)))
                if mpiutil.rank0:
                    if added_position == 0:
                        warnings.warn('One additional point before the data is detected!', AdditionalPointBegin)
                    else:
                        warnings.warn('One additional point before index %d is detected!'%on_points[added_position], AdditionalPointMiddle)
                if added_position == 0:
                    added_count_before += 1
                elif added_count_before == 0:
                    added_count_before += 1
#                on_start = on_start - 1
                on_points = np.array(on_points)
                on_points[added_position:] = on_points[added_position:] + 1
                off_points = np.array(off_points)
                off_points[added_position:] = off_points[added_position:] + 1
                if off_points[-1] >= total_time_span:
                    on_points = on_points[:-1]
                    off_points = off_points[:-1]
                if mpiutil.rank0:
                    if added_count_before >= change_reference_tol:
                        warnings.warn('Move the reference time one index later to compensate!',ChangeReferenceTime)
                        reference_time += 1/86400.0*rt.attrs['inttime']
                        np.savez(output_path(ns_prop_file), period = period, on_time = on_time, off_time = off_time, reference_time = reference_time, lost_count = 0, added_count = 0)
                    else:
                        print('The number of recorded added points was %d while tolerance is %d. Do not change the reference time.'%(added_count_before, change_reference_tol))
                        np.savez(output_path(ns_prop_file), period = period, on_time = on_time, off_time = off_time, reference_time = reference_time, lost_count = 0, added_count = added_count_before)
#                mpiutil.barrier()
            elif lost_one_point > 0 or abnormal_count > 0 or added_one_point > 0:
                if mpiutil.rank0:
                    np.savez(output_path(ns_prop_file), period = period, on_time = on_time, off_time = off_time, reference_time = reference_time, lost_count = 0, added_count = 0)
                    warnings.warn('Abnormal points are detected for some channel, number of abnormal bl is %d, number of channel that probably lost one point is %d, number of channel that probably added one point is %d'%(abnormal_count, lost_one_point, added_one_point), DetectedLostAddedAbnormal)
                    warnings.warn('Abnomal channels: %s\nChannels that probably lost one point: %s\nChannels that probably added one point: %s'%(str(abnormal_list), str(lost_one_list), str(added_one_list)), DetectedLostAddedAbnormal)
            else:
                if mpiutil.rank0:
                    np.savez(output_path(ns_prop_file), period = period, on_time = on_time, off_time = off_time, reference_time = reference_time, lost_count = 0, added_count = 0)
                
#================================================
            if mpiutil.rank0:
                print 'Noise source: period = %d, on_time = %d, off_time = %d' % (period, on_time, off_time)
#            num_period = np.int(np.ceil(total_time_span / np.float(period)))
#            ns_on = np.array([False] * on_start + ([True] * on_time + [False] * off_time) * num_period)[:total_time_span]
            ns_on = np.array([False]*total_time_span)
            for i,j in zip(on_points, off_points):
                ns_on[i:j] = True


#            if mpiutil.rank0:
#                np.save('ns_on', ns_on)
        elif not rt.FRB_cal or ns_arr_num < num_noise:

#            min_inds = 0 # min_inds = min(len(pinds), len(ninds), min_inds) if min_inds != 0 else min(len(pinds), len(ninds))
            ns_num_add = []
            for ns_arr_index, bl_ind in enumerate(auto_inds):
                this_chan = rt.bl[bl_ind, 0] # channel of this bl_ind
                vis = np.ma.array(rt.local_vis[:, :, bl_ind].real, mask=rt.local_vis_mask[:, :, bl_ind])
                cnt = vis.count() # number of not masked vals
                total_cnt = mpiutil.allreduce(cnt)
                vis_shp = rt.vis.shape
                ratio = float(total_cnt) / np.prod((vis_shp[0], vis_shp[1])) # ratio of un-masked vals
                if ratio < 0.5: # too many masked vals
                    if rt.FRB_cal and ns_arr_num == -1:
                        ns_arr_pinds += [np.array([])]
                        ns_arr_ninds += [np.array([])]
                    if mpiutil.rank0:
                        warnings.warn('Too many masked values for auto-correlation of Channel: %d, does not use it' % this_chan)
                    continue
                tt_mean = mpiutil.gather_array(np.ma.mean(vis, axis=-1).filled(0), root=None) # mean for all freq, for a specific auto bl 
                df =  np.diff(tt_mean, axis=-1)
                pdf = np.where(df>0, df, 0)
                pinds = np.where(pdf>pdf.mean() + sigma*pdf.std())[0]
#====================================
                if len(pinds) == 0: # no raise, might be badchn, continue
                    if rt.FRB_cal and ns_arr_num == -1:
                        ns_arr_pinds += [np.array([])]
                        ns_arr_ninds += [np.array([])]
                    if mpiutil.rank0:
                        warnings.warn('No noise on signal is detected for Channel %d, it may be bad channel.' % this_chan)
                    continue
#====================================
                pinds = pinds + 1
                pinds1 = [pinds[0]]
                for pi in pinds[1:]:
                    if pi - pinds1[-1] > 1:
                        pinds1.append(pi)
                pinds = np.array(pinds1)
                pT = Counter(np.diff(pinds)).most_common(1)[0][0] # period of pinds
                
#            print('pT:',pT)
#            np.save('pinds',pinds)
#            np.save('tt_mean',tt_mean)
#            np.save('df',df)

                ndf = np.where(df<0, df, 0)
                ninds = np.where(ndf<ndf.mean() - sigma*ndf.std())[0]
                ninds = ninds + 1
                ninds = ninds[::-1]
#====================================
                if len(ninds) == 0: # no raise, might be badchn, continue
                    if rt.FRB_cal and ns_arr_num == -1:
                        ns_arr_pinds += [np.array([])]
                        ns_arr_ninds += [np.array([])]
                    if mpiutil.rank0:
                        warnings.warn('No noise off signal is detected for Channel %d, it may be bad channel.' % this_chan)
                    continue
#====================================
                ninds1 = [ninds[0]]
                for ni in ninds[1:]:
                    if ni - ninds1[-1] < -1:
                        ninds1.append(ni)
                ninds = np.array(ninds1[::-1])
                nT = Counter(np.diff(ninds)).most_common(1)[0][0] # period of ninds

#                if min_inds == 0:
#                    min_inds = min(len(pinds), len(ninds))
#                else:
#                    min_inds = min(len(pinds), len(ninds), min_inds)
                ns_num_add += [min(len(pinds), len(ninds))]
                if rt.FRB_cal:
                    if ns_arr_num == -1:
#                        if mpiutil.rank0:
#                            print(ninds)
                        ns_arr_pinds += [pinds]
                        ns_arr_ninds += [ninds]
                    else:
#                        if mpiutil.rank0:
#                            print(ninds)
#                            print(ns_arr_ninds[ns_arr_index])
                        ns_arr_pinds[ns_arr_index] = np.concatenate([ns_arr_pinds[ns_arr_index], pinds + ns_arr_len])
                        ns_arr_ninds[ns_arr_index] = np.concatenate([ns_arr_ninds[ns_arr_index], ninds + ns_arr_len])
#==============================================
                # continue for non-FRB case
                if pT != nT: # failed to detect correct period
                    if mpiutil.rank0:
                        warnings.warn('Failed to detect correct period for auto-correlation of Channel: %d, positive T %d != negative T %d, does not use it' % (this_chan, pT, nT))
                    continue
                else:
                    period = pT

                ninds = ninds.reshape(-1, 1)
                dinds = (ninds - pinds).flatten()
                on_time = Counter(dinds[dinds>0] % period).most_common(1)[0][0]
                off_time = Counter(-dinds[dinds<0] % period).most_common(1)[0][0]

                if period != on_time + off_time: # incorrect detect
                    if mpiutil.rank0:
                        warnings.warn('Incorrect detect for auto-correlation of Channel: %d, period %d != on_time %d + off_time %d, does not use it' % (this_chan, period, on_time, off_time))
                    continue
                else:
                    if 'noisesource' in rt.iterkeys():
                        if rt['noisesource'].shape[0] == 1: # only 1 noise source
                            start, stop, cycle = rt['noisesource'][0, :]
                            int_time = rt.attrs['inttime']
                            true_on_time = np.round((stop - start)/int_time)
                            true_period = np.round(cycle / int_time)
                            if on_time != true_on_time and period != true_period: # inconsistant with the record in the data
                                if mpiutil.rank0:
                                    warnings.warn('Detected noise source info is inconsistant with the record in the data for auto-correlation of Channel: %d: on_time %d != record_on_time %d, period != record_period %d, does not use it' % (this_chan, on_time, true_on_time, period, true_period))
                                continue
                        elif rt['noisesource'].shape[0] >= 2: # more than 1 noise source
                            if mpiutil.rank0:
                                warnings.warn('More than 1 noise source, do not know how to deal with this currently')

                    # break if succeed
                    if not rt.FRB_cal: # for FRB case, record all baseline
                        break

            else:
                if not rt.FRB_cal:
                    raise DetectNoiseFailure('Failed to detect noise source signal')

            if mpiutil.rank0:
                print 'Detected noise source: period = %d, on_time = %d, off_time = %d' % (period, on_time, off_time)
            on_start = Counter(pinds % period).most_common(1)[0][0]
            num_period = np.int(np.ceil(len(tt_mean) / np.float(period)))
            ns_on = np.array([False] * on_start + ([True] * on_time + [False] * off_time) * num_period)[:len(tt_mean)]
#==============================================
            if rt.FRB_cal:
                ns_arr_len += total_time_span
                ns_arr_from_time = np.float128(ns_arr_end_time - ns_arr_start_time)*24.*3600./rt.attrs['inttime'] + 1
                if np.abs(ns_arr_len - ns_arr_from_time) < 1: # avoid numeric errors, which do appeare
                    pass
                else:
                    raise IncontinuousData('Incontinuous data. Index span calculated from time is %.2f, while the sum of array length is %d! Can not deal with incontinuous data at present!'%(ns_arr_from_time, ns_arr_len))
#                    print('Detected incontinuous data, use index span %d calculated from time instead of the sum of array length %d!'%(ns_arr_from_time, ns_arr_len))
#                    ns_arr_len = ns_arr_from_time
                if ns_arr_num < 0:
                    ns_arr_num = 0
#                ns_arr_num += min_inds
                ns_arr_num += np.around(np.average(ns_num_add))
                ns_arr_bl = rt.bl[auto_inds, 0]
            if mpiutil.rank0 and rt.FRB_cal:
                np.savez(output_path(ns_arr_file), on_inds = ns_arr_pinds, off_inds = ns_arr_ninds, time_len = ns_arr_len, ns_num = ns_arr_num, auto_chn = ns_arr_bl, start_time = ns_arr_start_time, end_time = ns_arr_end_time)
                if ns_arr_num < num_noise:
                    raise NoiseNotEnough('Number of noise points %d is not enough for calibration(need %d), wait for next file!'%(ns_arr_num, num_noise))
#            mpiutil.barrier()
#=================================================================
        if rt.FRB_cal and (not os.path.exists(output_path(ns_prop_file))) and ns_arr_num >= num_noise:
            print('Got %d noise points (need %d) to build up noise property file!'%(ns_arr_num, num_noise))
            for ns_arr_index, (pinds, ninds) in enumerate(zip(ns_arr_pinds, ns_arr_ninds)):
                if len(pinds) < num_noise*2./3. or len(ninds) < num_noise*2./3.:
                    print('Channel %d does not have enough noise points(%d, need %d) for calibration. Do not use it.'%(ns_arr_bl[ns_arr_index], len(pinds), int(2*num_noise/3.)))
                    continue
                pT = Counter(np.diff(pinds)).most_common(1)[0][0] # period of pinds
                nT = Counter(np.diff(ninds)).most_common(1)[0][0] # period of ninds

#=================================================================
                if pT != nT: # failed to detect correct period
                    if mpiutil.rank0:
                        warnings.warn('Failed to detect correct period for auto-correlation of Channel: %d, positive T %d != negative T %d, does not use it' % (ns_arr_bl[ns_arr_index], pT, nT))
                    continue
                else:
                    period = pT

                ninds = ninds.reshape(-1, 1)
                dinds = (ninds - pinds).flatten()
                on_time = Counter(dinds[dinds>0] % period).most_common(1)[0][0]
                off_time = Counter(-dinds[dinds<0] % period).most_common(1)[0][0]
#            print('on_time:',on_time)
#            print('off_time:',off_time)

                if period != on_time + off_time: # incorrect detect
                    if mpiutil.rank0:
                        warnings.warn('Incorrect detect for auto-correlation of Channel: %d, period %d != on_time %d + off_time %d, does not use it' % (ns_arr_bl[ns_arr_index], period, on_time, off_time))
                    continue
                else:
                    if 'noisesource' in rt.iterkeys():
                        if rt['noisesource'].shape[0] == 1: # only 1 noise source
                            start, stop, cycle = rt['noisesource'][0, :]
                            int_time = rt.attrs['inttime']
                            true_on_time = np.round((stop - start)/int_time)
                            true_period = np.round(cycle / int_time)
                            if on_time != true_on_time and period != true_period: # inconsistant with the record in the data
                                if mpiutil.rank0:
                                    warnings.warn('Detected noise source info is inconsistant with the record in the data for auto-correlation of Channel: %d: on_time %d != record_on_time %d, period != record_period %d, does not use it' % (this_chan, on_time, true_on_time, period, true_period))
                                continue
                        elif rt['noisesource'].shape[0] >= 2: # more than 1 noise source
                            if mpiutil.rank0:
                                warnings.warn('More than 1 noise source, do not know how to deal with this currently')

                    # break if succeed

                    break

            else:
                raise DetectNoiseFailure('Failed to detect noise source signal')

            if mpiutil.rank0:
                print 'Detected noise source: period = %d, on_time = %d, off_time = %d' % (period, on_time, off_time)
            on_start = Counter(pinds % period).most_common(1)[0][0]
            this_time_len = total_time_span
            first_ind = ns_arr_len - this_time_len
            skip_inds = first_ind - on_start
            on_start = period - skip_inds%period
            num_period = np.int(np.ceil(total_time_span / np.float(period)))
            ns_on = np.array([False] * on_start + ([True] * on_time + [False] * off_time) * num_period)[:total_time_span]

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(np.where(ns_on, np.nan, tt_mean))
        # # plt.plot(pinds, tt_mean[pinds], 'RI')
        # # plt.plot(ninds, tt_mean[ninds], 'go')
        # plt.savefig('df.png')
        # err

        ns_on1 = mpiarray.MPIArray.from_numpy_array(ns_on)

        rt.create_main_time_ordered_dataset('ns_on', ns_on1)
        rt['ns_on'].attrs['period'] = period
        rt['ns_on'].attrs['on_time'] = on_time
        rt['ns_on'].attrs['off_time'] = off_time

        if (not rt['jul_date'][on_start] is None) and not os.path.exists(output_path(ns_prop_file)):
            np.savez(output_path(ns_prop_file), period = period, on_time = on_time, off_time = off_time, reference_time = np.float128(rt['jul_date'][on_start]))
#            mpiutil.barrier()
            print('Save noise property file to %s'%output_path(ns_prop_file))

        # set vis_mask corresponding to ns_on
        on_inds = np.where(rt['ns_on'].local_data[:])[0]
        rt.local_vis_mask[on_inds] = True

        if mask_near > 0:
            on_inds = np.where(ns_on)[0]
            new_on_inds = on_inds.tolist()
            for i in xrange(1, mask_near+1):
                new_on_inds = new_on_inds + (on_inds-i).tolist() + (on_inds+i).tolist()
            new_on_inds = np.unique(new_on_inds)

            if rt['vis_mask'].distributed:
                start = rt.vis_mask.local_offset[0]
                end = start + rt.vis_mask.local_shape[0]
            else:
                start = 0
                end = rt.vis_mask.shape[0]
            global_inds = np.arange(start, end).tolist()
            new_on_inds = np.intersect1d(new_on_inds, global_inds)
            local_on_inds = [ global_inds.index(i) for i in new_on_inds ]
            rt.local_vis_mask[local_on_inds] = True # set mask using global slicing

        return super(Detect, self).process(rt)
