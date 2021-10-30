import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats
import seaborn as sns
import mne
import matplotlib.pyplot as plt
from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import XdawnCovariances, ERPCovariances
import time

import warnings
warnings.simplefilter(action='ignore')

''' Data Investigation'''
def investigate(data_paths,picks= ["Fz","Fc1","Fc2","C3","Cz","C4","Pz"],plot_amplitude =True,plot_mne =False,plot_woi=False, plot_r2=True,save_averages=False,sess_name='Unknown',mean_amplitude_tactilos=False):
    #mne.set_log_level('WARNING')
    col_list = ["#d8b365", "#5ab4ac", "#ef8a62", "#67a9cf"] # colors were choosen to be readable for the colorblind according to the website colorbrewer2 (https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=3)
    raw_list = []
    if not isinstance(data_paths,list):
        data_paths = [data_paths]

    if mean_amplitude_tactilos:
        mean_amplitudes = pd.DataFrame(columns=["Session","Electrode","Tactilo","Mean Amplitude","Condition"])

        for data_path in data_paths:
            if 'Calib' in data_path:
                raw_list =[]
            with open(data_path, "rb") as file:
                data_tuple = pickle.load(file)
                raw_list += data_tuple[0]
            if 'Calib' in data_path:
                continue
            sess = data_path[-10:-7]
            for i, raw in enumerate(raw_list):
                raw_list[i] = raw.pick(picks).filter(l_freq=0.1, h_freq=40) # band-pass filter + notch filter
            raw = mne.concatenate_raws(raw_list)
            events, event_id = mne.events_from_annotations(raw)
            event_id.pop('10')
            epochs = mne.Epochs(raw, events,
                                event_id=event_id,
                                tmin=-0.1, tmax=0.8, #epochs of 900ms starting 100ms before stimulus onset
                                preload=True,
                                event_repeated='merge',
                                baseline=(None, 0), #baseline correction with the first 100ms
                                picks=picks,
                                reject=dict(eeg=100e-6)) # reject trials with peak-to-peak amplitudes of 100microVolt
            rejected = 0
            for entry in epochs.drop_log:
                if len(entry) < 1:
                    continue
                if entry[0] in picks:
                    rejected += 1
            for tactilo in range(1,7):
                target_tac = '1/10'+str(tactilo)
                df =pd.DataFrame((epochs[target_tac].average()._data*1e6).T)
                df.index = epochs.times
                for electrode,amp in enumerate(df[0.35: 0.6].mean().to_list()):
                    mean_amplitudes.loc[len(mean_amplitudes)] =[sess,picks[electrode],str(tactilo),amp,"Target"]
            for tactilo in range(1,7):
                non_target_tac = '0/10'+str(tactilo)
                df =pd.DataFrame((epochs[non_target_tac].average()._data*1e6).T)
                df.index = epochs.times
                for electrode,amp in enumerate(df[0.35: 0.6].mean().to_list()):
                    mean_amplitudes.loc[len(mean_amplitudes)] =[sess,picks[electrode],str(tactilo),amp,"nonTarget"]

        #mean_amplitudes.to_csv(r"D:\Master\Masterarbeit\Data\Amplitudes\average_amplitudes_350_600.csv")
    if sess_name == 'all':
        sess= 'all'
    for data_path in data_paths:
        with open(data_path, "rb") as file:
            data_tuple = pickle.load(file)
        raw_list += data_tuple[0]
    for i, raw in enumerate(raw_list):
        raw_list[i] = raw.pick(picks).filter(l_freq=0.1, h_freq=50)

    raw = mne.concatenate_raws(raw_list)
    events, event_id = mne.events_from_annotations(raw)
    event_id.pop('10')
    epochs= mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=0.8, preload=True,event_repeated='merge',baseline=(None,0),picks=picks,reject=dict(eeg=100e-6))  # timewindow -0.85 so its' 0.8 after resampling baseline=(-0.100,0)
    #epochs = epochs_org.resample(50)
    nt = epochs['0'].average()._data*1e6 #mne data in Volt! create evoked
    t = epochs['1'].average()._data*1e6
    times = epochs.times
    #epochs.event_id.pop('10')
    new_event_id = {} # empty dictionary
    for key, value in epochs.event_id.items(): # set all targets to 1, non_targets to 0
        new_event_id[value] = 0 if key[0] == "0" else 1


    event_labels = pd.DataFrame(epochs.events[:, 2])
    event_labels.replace(new_event_id, inplace=True)
    y = np.array(event_labels.loc[((event_labels[0] == 0) | (event_labels[0] == 1))][0])


    if save_averages:
        t_df =  pd.DataFrame(t.T,index= times)
        nt_df = pd.DataFrame(nt.T,index =times)
        t_df.to_csv(os.path.join(r"D:\Master\Masterarbeit\Data\filtered_avg_targets_" + str(sess) + ".csv"))
        nt_df.to_csv(os.path.join(r"D:\Master\Masterarbeit\Data\filtered_avg_nontargets_" + str(sess) + ".csv"))


    if plot_r2:
        getData = epochs.get_data()

        (n_epochs,n_channels,n_samples) = getData.shape

        r_value_mat = np.empty((n_channels,n_samples))
        r_value_mat[:] = np.nan # empty doesn't create an empty array!!
        y_mat = np.tile(y.reshape(1,1,-1).T, (1,n_channels, n_samples))


        for timepoint in range(n_samples):
            for electrode in range(n_channels):
                slope, intercept, r_value, p_value, std_err = stats.linregress(getData[:,electrode, timepoint],y_mat[:,electrode, timepoint])
                r_value_mat[electrode, timepoint] = r_value

        r2_value_mat = np.square(r_value_mat)

        fig, ax = plt.subplots(figsize=(10, 7))
        xticks = xticks = np.linspace(0, n_samples - 1, round(n_samples / 51.2) + 1, dtype=np.int)
        xticklabels = np.around(np.arange(round(epochs.tmin,3),round(epochs.tmax,3), 0.1),3).tolist()
        sns.heatmap(r2_value_mat, cmap="viridis", fmt="", ax=ax, yticklabels=picks, xticklabels=xticklabels)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45)
        ax.set_xlabel("seconds")
        ax.set_ylabel("Electrodes")
        ax.set_title("R² of session "+str(sess))
        plt.savefig(os.path.join(r"D:\Master\Masterarbeit\Data\Plots\R2_filtered" + "\R2_filtered_"+str(sess)+".png"))

    if plot_mne:
        for electrode in ["Fz","Pz","Cz"]:
            evokeds = [epochs[name].average(picks=[electrode]) for name in ('1', '0')]
            evokeds[0].comment = 'Targets'
            evokeds[1].comment = 'Non-Targets'
            fig = mne.viz.plot_compare_evokeds(evokeds,show =False, title ="Average over "+electrode+" in sess" +str(sess))
            plt.show()
            plt.savefig(os.path.join(r"D:\Master\Masterarbeit\Data\Plots\P300_MNE_filtered" + "\P300_MNE_filtered_"+str(sess)+"_"+electrode+".png"))

    if plot_woi:
        for electrode in ["Cz"]:

            e = picks.index(electrode)
            fig,(ax,ax2) = plt.subplots(1,2,figsize=(10, 4), sharey=True)
            ax.plot(times,t[e], label="Target",c=col_list[1])
            ax.plot(times,nt[e], label="Non-Target", c=col_list[0])
            ax.axvline(x=0, c='k', lw=0.5)
            ax.axhline(y=0, c='k', lw=0.5)
            ax.axvspan(0.3,0.5, facecolor='r', alpha=0.25)
            ax.set_xlabel("seconds")
            ax.set_ylabel("µV")
            ax.margins(x=0)
            ax2.plot(times,t[e], label="Target",c=col_list[1])
            ax2.plot(times,nt[e],label="Non-Target", c=col_list[0])
            ax2.axvline(x=0, c='k', lw=0.5)
            ax2.axhline(y=0, c='k', lw=0.5)
            ax2.axvspan(0.35,0.60, facecolor='r', alpha=0.25)
            ax2.set_xlabel("seconds")
            ax2.margins(x=0)
            #fig.suptitle("Average over "+ electrode+ " all sessions")
            #fig.tight_layout()

            plt.subplots_adjust(wspace=.08, top=0.859)
            ax2.legend(title = "Average over all sessions at Cz",loc='center',bbox_to_anchor=(-0.04, 1.1),ncol=2)

            plt.show()

    if plot_amplitude:

        fig, (Fz,Cz,Pz) = plt.subplots(1,3,figsize=(14, 5),sharey=True)
        for e in ["Fz","Pz","Cz"]:
            electrode = picks.index(e)
            eval(e).plot(times, t[electrode], label="Target",c=col_list[1])
            eval(e).plot(times, nt[electrode], label="Non-Target", c=col_list[0])
            eval(e).axvline(x=0, c='k', lw=0.5)
            eval(e).axhline(y=0, c='k', lw=0.5)
            eval(e).axvspan(0.35, 0.6, facecolor='r', alpha=0.25)
            eval(e).set_xlabel("seconds")

            eval(e).margins(x=0)
            eval(e).set_title(e)
        Fz.set_ylabel("µV")
        Cz.legend(title = "Average over all sessions",loc='center',bbox_to_anchor=(0.5, 1.16),ncol=2)
        #fig.suptitle("Average over all sessions")
        fig.tight_layout()
        plt.show()




'''LOAD DATA'''
def load_epochs(data_paths, cov_estimator=None,picks = 'all', resampled_riemann = True, calib_runs=3, xDawn=False,spatial_filter=None): #calibration session variation not implemented yet
    mne.set_log_level('WARNING')
    raw_list = []
    if not isinstance(data_paths,list):
        data_paths = [data_paths]
    for data_path in data_paths:
        with open(data_path, "rb") as file:
            data_tuple = pickle.load(file)
        if calib_runs ==1:
            raw = data_tuple[0][0]
            break
        else:
            raw_list += data_tuple[0]
    raw = mne.concatenate_raws(raw_list) if calib_runs != 1 else raw
    events, event_id = mne.events_from_annotations(raw)
    event_id.pop('10') 
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=0.8, preload=True,
                        event_repeated='merge',baseline=None,picks=picks)  # timewindow -0.85 so its' 0.8 after resampling baselien=(-0.100,0)


    # prepare data for covariance estimation
    X = epochs.copy().resample(20).get_data() if resampled_riemann else epochs.get_data()

    new_event_id = {} # empty dictionary
    for key, value in epochs.event_id.items(): # set all targets to 1, non_targets to 0
        new_event_id[value] = 0 if key[0] == "0" else 1

    event_labels = pd.DataFrame(epochs.events[:, 2])
    event_labels.replace(new_event_id, inplace=True)
    y = np.array(event_labels.loc[((event_labels[0] == 0) | (event_labels[0] == 1))][0])


    if not cov_estimator:
        cov_estimator = ERPCovariances(estimator='lwf') if not xDawn else XdawnCovariances(estimator='lwf',nfilter=2)
        cov_matrices = cov_estimator.fit_transform(X,y) #will be of size n_channels*(n_classes+1) - t/nt -> 2 classes -> cov matrices size = 36*36 ( xDawn 6*6)

    else:
        cov_matrices = cov_estimator.transform(X)

    # Initiate xDawn spatial filter:

    if xDawn:
        t =time.time()
        if not spatial_filter:
            spatial_filter = Xdawn(nfilter=2)
            filtered_epochs = spatial_filter.fit_transform(epochs.get_data(),y=y)
        else:
            filtered_epochs = spatial_filter.transform(epochs.get_data())
        print("spatial filtering for resampled epochs took: {}".format(time.time()-t))

        edf_td = epochs.to_data_frame(time_format='timedelta')
        edf_filtered = edf_td[["time", "condition", "epoch"]]
        feature_names = ["feature_" + str(i) for i in range(filtered_epochs.shape[1])]
        edf_filtered[feature_names] = filtered_epochs.reshape(edf_td.shape[0],filtered_epochs.shape[1])
        edf=edf_filtered
    else:
        epochs_resampled = epochs.copy().resample(20)
        edf = epochs_resampled.to_data_frame(time_format='timedelta')

    edf = edf.set_index("time", drop=False)

    first_round = True
    epoch_grouper = edf.groupby('epoch')

    for name, epoch_complete in epoch_grouper:
        if epoch_complete['condition'].unique() =='10': #kic
            continue
        epoch_info = np.array([epoch_complete['condition'].iloc[0][0], epoch_complete['condition'].iloc[0][-1], epoch_complete['epoch'].iloc[0]],dtype=int)
        if xDawn:
            epoch_complete = epoch_complete.resample("50ms").mean()
            epoch_complete = epoch_complete[:-1] # drop 800ms time point to be consistent with the resampling done by mne
            epoch_signal = epoch_complete.drop(columns=['epoch']).to_numpy()
        else:
            epoch_signal = epoch_complete.drop(columns=['time', 'condition', 'epoch']).to_numpy()
        epoch_signal = epoch_signal.flatten()
        if first_round:
            epoch_df =  pd.DataFrame(np.append(epoch_info,epoch_signal)).T
            first_round=False
        else:
            epoch_df = epoch_df.append(pd.DataFrame(np.append(epoch_info,epoch_signal)).T)

    epoch_df = epoch_df.rename(columns={0:"condition",1:"tactilo",2:"epoch"}).reset_index(drop=True)
    epoch_info_df = epoch_df[["condition","tactilo","epoch"]]
    x = epoch_df.drop(columns=["condition","tactilo","epoch"])
    return x,y, epoch_info_df,cov_estimator, cov_matrices,spatial_filter


''' LOAD INFO'''
def load_session_dates():
    with open(r"D:\Master\Masterarbeit\Data\date_dict.pickle", "rb") as file:
        SESS_DATES = pickle.load(file)
    return SESS_DATES

def load_bci2k_ac():
    bci2k = pd.read_fwf(r"D:\Master\Masterarbeit\Data\Classifier_Results\bci2000.txt", header=None)[[0, 1]].to_numpy()
    temp_df = pd.DataFrame(columns=["Accuracy", "Classifier", "Session", "Ep2Avg"])
    temp_df["Accuracy"] = bci2k[:, 1] * 0.01
    temp_df["Classifier"] = "BCI2000"
    temp_df["Session"] = [i for i in range(1, int(len(bci2k) / 8) + 1) for _ in range(8)]
    temp_df["Ep2Avg"] = bci2k[:, 0]


''' ACCURACY CALCULATION'''



def evaluate_independent_epochs(probability_df, info_df):
    ''' Berechnet die Accuracy aus der gemittelten Wahrscheinlichkeit von 1-8 einzeln evaluierten Epochen (ep2avg), der Tacitlo mit dem maximalsten Wert "gewinnt"'''
    evaluation = []
    for ep2avg in range(1,9):
        eval_array = np.empty((18, 6, 2))
        for tactilo in range(6):
            trial_idx = np.split(info_df["epoch"].loc[(info_df["tactilo"] == tactilo + 1)].unique(),18)  # 18 weil 3 Runs a 6 "Runden" pro Taktilo (1*Target, 5*Non-Target)
            # trial idx gibt an welche epoch zu einem Trial gehören (von dem aktuellen Taktilo)
            for trial in range(len(trial_idx)):
                ground_truth = info_df["condition"].loc[info_df["epoch"].isin(trial_idx[trial])].mean()  # bildet den mittelwert aus ep2avg epochen (single value)
                assert ground_truth == 1 or ground_truth == 0  # make sure the epochs didn't mix
                prediction = pd.DataFrame(probability_df)[1].loc[info_df["epoch"].isin(trial_idx[trial])][:ep2avg].mean()  # mittelwert aus ep2avg Epochen (single value)
                eval_array[trial, tactilo, 0] = ground_truth
                eval_array[trial, tactilo, 1] = prediction

        eval_tactilo = eval_array.argmax(axis=1)
        evaluation.append((eval_tactilo[:, 0] == eval_tactilo[:, 1]).mean())

    return evaluation


def results_template(accuracy,classifer,session, ep2avg=range(1,9)):
    #template for saving the results in a unified manner
    temp_df = pd.DataFrame(columns=["Accuracy", "Classifier", "Session", "Ep2Avg"])
    temp_df["Accuracy"] = accuracy
    temp_df["Classifier"] = classifer
    temp_df["Session"] = session
    temp_df["Ep2Avg"] = ep2avg
    return temp_df