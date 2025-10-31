import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def load_from_json(path="../data/iwai2025/log_centered_large.json"):
    # load experiment data from json file
    logdata = pd.read_json(path)
    # remove unnecessary columns
    logdata.drop(['mouse_init_x', 'mouse_init_y','dt'], axis=1, inplace=True)
    # adjust time to start from 0
    logdata['t'] = logdata['t'] - logdata['t'].iloc[0]

    plotdata = logdata.where(logdata['type'] == 'mousemove').loc[:, ['x', 'y','t']].dropna()

    # Add savgol filter and velocities
    plotdata['x_savgol'] = savgol_filter(plotdata['x'], 5, 3)
    plotdata['x_vel'] = savgol_filter(plotdata['x'], 5, 3, deriv=1)
    plotdata['x_acc'] = savgol_filter(plotdata['x'], 5, 3, deriv=2)

    targetdata = logdata.where(logdata['type'] == 'target/hit').loc[:, ['x', 'y','r','t']].dropna()
    # drop targets where x is 900
    targetdata = targetdata[targetdata['x'] != 900.0]

    startdata  = logdata.where(logdata['type'] == 'target/init').loc[:, ['x', 'y','r','t']].dropna()

    ## Create trial data
    trialdata = targetdata.copy()

    trialdata["start_time"] = startdata.loc[:,'t'].values
    trialdata["end_time"] = targetdata['t'].values
    trialdata.drop(["t"], axis=1, inplace=True)

    trialdata["movement_time"] = trialdata["end_time"] - trialdata['start_time']

    # Calculate Fitts law index of difficulty
    trialdata["width"] = 2*trialdata.loc[:,'r'].values
    trialdata["distance"] =  np.abs(trialdata.loc[:,'x'].values - 900.0)
    trialdata["ID"] = np.log2(trialdata["distance"] /trialdata["width"] + 1)

    # Add Trial ID, direction and target_id to targetdata and startdata
    trialdata["trial_id"] = range(len(trialdata))

    # get target data of start_data_trials
    trialdata["direction"] = "left"
    trialdata["right_bool"] = trialdata['x'] > 900.0
    trialdata.loc[trialdata["right_bool"], "direction"] = "right"
    trialdata.drop("right_bool", axis=1, inplace=True)

    # Get unique targets from target_info
    unique_targets = trialdata.drop_duplicates(subset=['x', 'y', 'r'])
    # Sort targets by radius and x position
    unique_targets = unique_targets.sort_values(by=['r','x'], ascending=False)

    unique_targets.drop(["trial_id","end_time","start_time","movement_time"], axis=1, inplace=True)
    unique_targets.reset_index(drop=True, inplace=True)
    unique_targets["target_id"] = range(len(unique_targets))

    # Add unique target id to target_info
    trialdata = trialdata.merge(unique_targets[['x', 'y', 'r','target_id']], on=['x', 'y', 'r'], how='left')

    # Get mean and std movement time per unique ID
    unique_IDs = unique_targets['ID'].unique()
    mean_movement_times = {}
    std_movement_times = {}
    for id in unique_IDs:
        mean_mt = trialdata.where(trialdata['ID'] == id).dropna()["movement_time"].mean()
        std_mt = trialdata.where(trialdata['ID'] == id).dropna()["movement_time"].std()
        mean_movement_times[id] = mean_mt
        std_movement_times[id] = std_mt

    # Add column for mean and std to target_info
    trialdata['mean_movement_time'] = trialdata['ID'].map(mean_movement_times)
    trialdata['std_movement_time'] = trialdata['ID'].map(std_movement_times)

    # Add column for outlier to target_info
    trialdata['outlier'] = trialdata['movement_time'] > (trialdata['mean_movement_time'] + 3 * trialdata['std_movement_time'])

    # Remove outliers from trialdata
    trialdata = trialdata.loc[trialdata['outlier'] == False]

    # Data about clicks
    click_data = logdata.where(logdata['type'] == 'click').drop("r", axis=1).dropna()
    click_data['target_hit'] = (logdata.iloc[click_data.index +1]['type'] == 'target/hit').values
    
    # Split data into single click movements
    single_trial_data = []
    total_clicks = []
    for i in range(len(trialdata)):
        t_start = trialdata['start_time'].iloc[i]
        t_end = trialdata['end_time'].iloc[i]

        single_trial_data.append(plotdata.loc[(plotdata['t'] >= t_start) & (plotdata['t'] <= t_end)])
        trial_clicks = click_data.loc[(click_data['t'] >= t_start) & (click_data['t'] <= t_end)]
        trial_clicks = trial_clicks.drop(trial_clicks.index[0])
        trial_clicks.loc[:,'t'] = trial_clicks['t'] - t_end

        total_clicks.append(trial_clicks)

    # Set t for each trial starting at 0
    for i in range(len(single_trial_data)):
        single_trial_data[i].loc[:,"t"] = single_trial_data[i].loc[:,'t'] - single_trial_data[i]['t'].values[-1]

    return logdata, trialdata, unique_targets, plotdata, click_data, single_trial_data, total_clicks

def get_trials_for_target(trialdata, click_data, plotdata, target_id = 0, dt=0.02):
    # Get all trials for a specific target
    target_trials = trialdata.loc[trialdata['target_id'] == target_id]

    # Split data into single click movements
    trial_data_id = []
    total_clicks = []
    for i in range(len(target_trials)):
        t_start = target_trials.loc[:,'start_time'].iloc[i]
        t_end = target_trials.loc[:,'end_time'].iloc[i]

        trial_data_id.append(plotdata.loc[(plotdata['t'] >= t_start) & (plotdata['t'] <= t_end)])
        trial_clicks = click_data.loc[(click_data['t'] >= t_start) & (click_data['t'] <= t_end)]
        trial_clicks = trial_clicks.drop(trial_clicks.index[0])
        trial_clicks.loc[:,'t'] = trial_clicks['t'] - t_end

        total_clicks.append(trial_clicks)

    # Set t for each trial starting at 0
    trial_data_interp = []
    for i in range(len(trial_data_id)):
        trial_data_id[i].loc[:,"t"] = trial_data_id[i]['t'] - trial_data_id[i]['t'].iloc[0]

        # Interpolate to match dt
        interpolate_x_savgol = interp1d(trial_data_id[i].loc[:,"t"], trial_data_id[i].loc[:,"x_savgol"], kind='linear', fill_value="extrapolate")
        interpolate_x = interp1d(trial_data_id[i].loc[:,"t"], trial_data_id[i].loc[:,"x"], kind='linear', fill_value="extrapolate")

        t_new = np.arange(trial_data_id[i]['t'].iloc[0], trial_data_id[i]['t'].iloc[-1], dt)
        x_savgol_new = interpolate_x_savgol(t_new)   # use interpolation function returned by `interp1d`
        x_new = interpolate_x(t_new)   # use interpolation function returned by `interp1d`
        # get velocity of new x
        v_savgol_new = np.gradient(x_savgol_new, dt)
        v_new = np.gradient(x_new, dt)
        trial_data_interp.append(pd.DataFrame({'t': t_new, 'x': x_new, 'v': v_new, 'x_savgol': x_savgol_new, 'v_savgol': v_savgol_new}))

    return trial_data_id, total_clicks, trial_data_interp

def get_trials_for_target_id_from_json(path, target_id, dt=0.02):
    logdata, trialdata, unique_targets, plotdata, click_data, single_trial_data, total_clicks = load_from_json(path)
    trial_data_id, total_clicks, trial_data_interp = get_trials_for_target(trialdata, click_data, plotdata, target_id, dt)
    return trial_data_id, total_clicks, trial_data_interp