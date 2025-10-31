import matplotlib.pyplot as plt
import numpy as np

def move_legend_out(ax, fontsize=12, order=None):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    if order is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles_new = [handles[idx] for idx in order]
        labels_new = [labels[idx] for idx in order]
    else:
        handles_new, labels_new = ax.get_legend_handles_labels()
    
    ax.legend(handles=handles_new, labels=labels_new, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    

def plot_results(dt, xx, oo, bb, aa, aa_applied, lll, llr, NEFE_PLAN=[], PRAGMATIC_PLAN=[], INFO_GAIN_PLAN=[], NEFES=[], PRAGMATICS=[], INFO_GAINS=[], 
                 buttons=[(0.1,0.02)], bb_sys=[], bb_noise=[], bb_after_rt=[], reaction_time_steps=None, belief_button=False, click_state=False, additional_obs=False, exp_normal_sys_params=False,
                 obs_missclick=False, button_switch_state=False, switch_on_click=False, cur_button=0, diff_control=False, 
                 plot_axes=['pos', 'vel', 'finger', 'button', 'click_state', 'acc_applied', 'button_nr', 'target_position', 'target_radius', 'target_position_change', 'target_radius_change', 'missclick', 'acc', 'click', 'loss', 'nefe', 'nefe_comp', 'bb_sys', 'bb_noise'],
                 distance_unit="m", ic_timesteps=None, ic_pred_error=None, fig=None, ax=None, figsize_x=None):
    rows = []
    
    if 'pos' in plot_axes:
        rows.append(['pos', 'pos'])
    if 'vel' in plot_axes:
        rows.append(['vel', 'vel'])
    if 'finger' in plot_axes:
        rows.append(['finger', 'finger'])
    if 'button' in plot_axes:
        rows.append(['button', 'button'])
    
    if click_state:
        if 'click_state' in plot_axes:
            rows.append(['click_state', 'click_state'])
        # fig, ax = plt.subplot_mosaic([['pos', 'pos'],
        #                         ['finger', 'finger'],
        #                         ['button', 'button'],
        #                         ['click_state', 'click_state'],
        #                         ['vel', 'vel'],
        #                         ['acc', 'click'],
        #                         ['loss', 'lr'],
        #                         ['nefe', 'nefe_comp']],
        #                         figsize=(10, 30))

    if diff_control:
        if 'acc_applied' in plot_axes:
            rows.append(['acc_applied', 'acc_applied'])
            #move third entry to the end (acc_applied)
            for i in range(len(xx)):
                x = xx[i]
                x = np.hstack((x[:2],x[3:],x[2]))
                xx[i] = x

                b = bb[i]
                columns = [j for (j,c) in enumerate(b[1][0,:])]
                new_order = columns[:2]+columns[3:]+[2]
                b_cov = np.array(b[1])
                b_cov[:] = [b_cov[i] for i in new_order]
                for row in b_cov:
                    row[:] = [row[i] for i in new_order]
                b = [np.hstack((b[0][:2],b[0][3:],b[0][2])),b_cov]
                bb[i] = b
    elif button_switch_state:
        if 'button_nr' in plot_axes:
            rows.append(['button_nr', 'button_nr'])
    elif len(xx[0]) > 4:
        if 'target_position' in plot_axes:
            rows.append(['target_position', 'target_position'])
        if 'target_radius' in plot_axes:
            rows.append(['target_radius', 'target_radius'])
    if len(xx[0]) > 6:
        if 'target_position_change' in plot_axes:
            rows.append(['target_position_change', 'target_position_change'])
        if 'target_radius_change' in plot_axes:
            rows.append(['target_radius_change', 'target_radius_change'])

    if obs_missclick:
        if 'missclick' in plot_axes:
            rows.append(['missclick', 'missclick'])
    if 'acc' in plot_axes and 'click' in plot_axes:
        rows.append(['acc', 'click'])
    elif 'acc' in plot_axes:
        rows.append(['acc', 'acc'])
    elif 'click' in plot_axes:
        rows.append(['click', 'click'])
    if 'loss' in plot_axes and 'lr' in plot_axes:
        rows.append(['loss', 'lr'])
    elif 'loss' in plot_axes and len(lll) > 0:
        rows.append(['loss', 'loss'])
    elif 'lr' in plot_axes and len(llr) > 0:
        rows.append(['lr', 'lr'])
    if 'nefe' in plot_axes and 'nefe_comp' in plot_axes and len(NEFE_PLAN) > 0:
        rows.append(['nefe', 'nefe_comp'])
    elif 'nefe' in plot_axes and len(NEFE_PLAN) > 0:
        rows.append(['nefe', 'nefe'])
    elif 'nefe_comp' in plot_axes and len(PRAGMATICS) > 0:
        rows.append(['nefe_comp', 'nefe_comp'])

    if 'bb_sys' in plot_axes:
        if len(bb_sys) > 0:
            for i in range(0, len(bb_sys[0][0]), 2):
                rows.append([f'sys_{i}', f'sys_{i+1}'])
    if 'bb_noise' in plot_axes:
        if len(bb_noise) > 0:
            for i in range(0, len(bb_noise[0][0]), 2):
                rows.append([f'noise_{i}', f'noise_{i+1}'])

    print(f"Rows: {rows}")
    if fig is None:
        if figsize_x is None:
            figsize_x = 7
        fig, ax = plt.subplot_mosaic(rows, figsize=(figsize_x, 3*len(rows)))
    

    numsteps = len(aa)

    std_mult = 3
    t = np.arange(0, numsteps * dt + 1e-6, dt)

    # Mouse clicks
    num_buttons = len(buttons)
    
    clicks = []
    button_switch = []
    missclicks = []
    if not additional_obs:
        for i in range(numsteps):
            if xx[i][2] < 0.05 and xx[i+1][2] > 0.05:
                clicks.append(t[i+1])
                if xx[i+1][3] > 0.5:
                    button_switch.append(i+1)
    else:
        for i in range(numsteps):
            if oo[i][3] > 0.5:
                clicks.append(t[i+1])
                if i < numsteps - 1:
                    if oo[i][2] > 0.5 or switch_on_click:
                        button_switch.append(i+1)
                if obs_missclick:
                    if oo[i][5] < -0.5:
                        missclicks.append(t[i+1])
            # if xx[i][2] < 0.05 and xx[i][3] > 0.05:
            #     clicks.append(t[i])
            #     if i < numsteps - 1:
            #         if oo[i][2] > 0.5:
            #             button_switch.append(i)

    ## Position
    if 'pos' in plot_axes:
        if len(button_switch) > 0:
            ax['pos'].fill_between(t[:button_switch[0]+1], [buttons[cur_button][0] - buttons[cur_button][1] for _ in t[:button_switch[0]+1]], [buttons[cur_button][0] + buttons[cur_button][1] for _ in t[:button_switch[0]+1]], alpha=0.2, label='target', color='red',  hatch='//')
            for i, _ in enumerate(button_switch):
                if i == len(button_switch) - 1:
                    t_interval = t[button_switch[i]:]
                else:
                    t_interval = t[button_switch[i]:button_switch[i+1]+1]
                cur_button = (cur_button + 1) % num_buttons
                ax['pos'].fill_between(t_interval, [buttons[cur_button][0] - buttons[cur_button][1] for _ in t_interval], [buttons[cur_button][0] + buttons[cur_button][1] for _ in t_interval], alpha=0.2, color='red',  hatch='//')
        else:
            ax['pos'].fill_between(t, [buttons[cur_button][0] - buttons[cur_button][1] for _ in t], [buttons[cur_button][0] + buttons[cur_button][1] for _ in t], alpha=0.2, label='target', color='red',  hatch='//')
        
        # Real position
        ax['pos'].plot(t, [x[0] for x in xx], label=['$s_1(t)$'], color="blue")
        ax['pos'].plot(t[1:], [o[0] for o in oo], label=['$o_1(t)$'], color = "orange")

        # ax['pos'].vlines(t[-10], -0.2, 0.3, color='red', linestyle='--')

        line_max = np.array([x[0] for x in xx]).max()
        line_min = np.array([x[0] for x in xx]).min()
        miss_click_labeled = False
        button_click_labeled = False
        for i,c in enumerate(clicks):
            color = 'blue'
            if obs_missclick:
                if c in missclicks:
                    color = 'red'
            if not miss_click_labeled and obs_missclick and c in missclicks:
                ax['pos'].vlines(c, line_min, line_max, linestyle='--', label=f'missed', color=color)
                miss_click_labeled = True
            if not button_click_labeled and c not in missclicks:
                ax['pos'].vlines(c, line_min, line_max, linestyle='--', label=f'hit', color=color)
                button_click_labeled = True
            else:
                ax['pos'].vlines(c, line_min, line_max, linestyle='--', color=color)

        # Buttons position and width belief
        if belief_button:
            for i,b in enumerate(bb_sys):
                ax['pos'].fill_between(t[i], [b[0][0] - b[0][1] for _ in t], [b[0][0] + b[0][1] for _ in t], alpha=0.5, color='orange')

        ax['pos'].set_xlabel('Time [s]')
        ax['pos'].set_ylabel(f'Cursor Position [{distance_unit}]')



    ## Finger State
    if 'finger' in plot_axes:
        ax['finger'].plot(t, [x[3] for x in xx], label=['$s_3(t)$'], color="blue")
        mean = [b[0][3] for b in bb]
        var = [b[1][3, 3] for b in bb]
        ax['finger'].plot(t[1:], [o[1] for o in oo], label=['$o_2(t)$'], color='orange')
        ax['finger'].plot(t, mean, label=['$Q^s_3(t-\\tau)$'], color='purple')
        ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
        ax['finger'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
        ax['finger'].hlines(0.05, 0, t[-1], color='green', linestyle='--', label='threshold')

        line_max = np.array([x[3] for x in xx]).max()
        line_min = np.array([x[3] for x in xx]).min()
        miss_click_labeled = False
        button_click_labeled = False
        for i,c in enumerate(clicks):
            color = 'blue'
            if obs_missclick:
                if c in missclicks:
                    color = 'red'
            if not miss_click_labeled and obs_missclick and c in missclicks:
                ax['finger'].vlines(c, line_min, line_max, linestyle='--', label=f'missed', color=color)
                miss_click_labeled = True
            if not button_click_labeled and c not in missclicks:
                ax['finger'].vlines(c, line_min, line_max, linestyle='--', label=f'hit', color=color)
                button_click_labeled = True
            else:
                ax['finger'].vlines(c, line_min, line_max, linestyle='--', color=color)

        ax['finger'].set_ylim([-0.1, 0.1])
        ax['finger'].set_xlabel('Time [s]')
        ax['finger'].set_ylabel('Mouse Button Displacement')
        ax['finger'].legend()


    ## Button State
    if 'button' in plot_axes:
        if not additional_obs:
            ax['button'].scatter(t, [x[3] for x in xx], label=['button state'], color="blue", marker='x')
            mean = [b[0][3] for b in bb]
            var = [b[1][3, 3] for b in bb]
            ax['button'].plot(t, mean, label=['button belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['button'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
        
        ax['button'].scatter(t[1:], [o[2] for o in oo], label=['button observation'], color='orange')

        ax['button'].set_xlabel('Time [s]')
        ax['button'].set_ylabel('Button State')
        ax['button'].legend()

    ## Click State
    if click_state:
        if 'click_state' in plot_axes:
            if not additional_obs:
                mean = [b[0][4] for b in bb]
                var = [b[1][4, 4] for b in bb]
                ax['click_state'].plot(t, mean, label=['mouse click belief'], color='purple')
                ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
                ax['click_state'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            
            ax['click_state'].scatter(t[1:], [o[3] for o in oo], label=['click observation'], color='orange')

            ax['click_state'].set_xlabel('Time [s]')
            ax['click_state'].set_ylabel('Click')
            ax['click_state'].legend()

    if obs_missclick:
        if 'missclick' in plot_axes:
            ax['missclick'].scatter(t[1:], [o[4] for o in oo], label=['missclick observation'], color='orange')
            ax['missclick'].set_xlabel('Time [s]')
            ax['missclick'].set_ylabel('Missclick')
            ax['missclick'].legend()

    ## Velocity
    if 'vel' in plot_axes:
        ax['vel'].plot(t, [x[1] for x in xx], label=['velocity'])

        mean = [b[0][1] for b in bb]
        var = [b[1][1, 1] for b in bb]
        ax['vel'].plot(t, mean, label=['belief velocity'], color='purple')
        ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
        ax['vel'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor="purple", alpha=1 / 4)

        ax['vel'].set_xlabel('Time [s]')
        ax['vel'].set_ylabel(f'Velocity [{distance_unit}/s]')
        ax['vel'].legend()

    if diff_control:
        if 'acc_applied' in plot_axes:
            ax['acc_applied'].plot(t, [x[-1] for x in xx], label='applied mouse acceleration')
            mean = [b[0][4] for b in bb]
            var = [b[1][4, 4] for b in bb]
            ax['acc_applied'].plot(t, mean, label='applied mouse acceleration belief', color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['acc_applied'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            ax['acc_applied'].set_xlabel('Time [s]')
            ax['acc_applied'].set_ylabel(f'Control [{distance_unit}/s^2]')
            ax['acc_applied'].legend()
    elif button_switch_state:
        if 'button_nr' in plot_axes:
            ax['button_nr'].plot(t, [x[4] for x in xx], label=['button state'], color="blue", marker='x')
            mean = [b[0][4] for b in bb]
            var = [b[1][4, 4] for b in bb]
            ax['button_nr'].plot(t, mean, label=['button belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['button_nr'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
    elif len(xx[0]) > 4:
        if 'target_position' in plot_axes:
            ax['target_position'].plot(t, [x[4] for x in xx], label=['target_position'])
            mean = [b[0][4] for b in bb]
            var = [b[1][4, 4] for b in bb]
            ax['target_position'].plot(t, mean, label=['target_position belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['target_position'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            ax['target_position'].plot(t[1:], [o[5] for o in oo], label=['target_position observation'], color='orange', linestyle='--')        
            ax['target_position'].set_xlabel('Time [s]')
            ax['target_position'].set_ylabel('Target Position')
            ax['target_position'].legend()
        if 'target_radius' in plot_axes:
            mean = [b[0][5] for b in bb]
            var = [b[1][5, 5] for b in bb]
            ax['target_radius'].plot(t, [x[5] for x in xx], label=['target_radius'])
            ax['target_radius'].plot(t, mean, label=['target_radius belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['target_radius'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            ax['target_radius'].plot(t[1:], [o[6] for o in oo], label=['target_radius observation'], color='orange', linestyle='--')
            ax['target_radius'].set_xlabel('Time [s]')
            ax['target_radius'].set_ylabel('Target Radius')
            ax['target_radius'].legend()

    if len(xx[0]) > 6:
        if 'target_position_change' in plot_axes:
            ax['target_position_change'].plot(t, [x[6] for x in xx], label=['target_position_change'])
            mean = [b[0][6] for b in bb]
            var = [b[1][6, 6] for b in bb]
            ax['target_position_change'].plot(t, mean, label=['target_position_change belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['target_position_change'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            ax['target_position_change'].set_xlabel('Time [s]')
            ax['target_position_change'].set_ylabel('Target Position Change')
            ax['target_position_change'].legend()

        if 'target_radius_change' in plot_axes:
            ax['target_radius_change'].plot(t, [x[7] for x in xx], label=['target_radius_change'])
            mean = [b[0][7] for b in bb]
            var = [b[1][7, 7] for b in bb]
            ax['target_radius_change'].plot(t, mean, label=['target_radius_change belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['target_radius_change'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            ax['target_radius_change'].set_xlabel('Time [s]')
            ax['target_radius_change'].set_ylabel('Target Radius Change')
            ax['target_radius_change'].legend()



    ## Control
    # Mouse acceleration
    if 'acc' in plot_axes:
        ax['acc'].plot(t[:-1], [a[0] for a in aa], label='chosen mouse acceleration')
        ax['acc'].plot(t[:-1], [a[0] for a in aa_applied], label='applied mouse acceleration')
        ax['acc'].set_xlabel('Time [s]')
        ax['acc'].set_ylabel(f'Control [{distance_unit}/s^2]')
        ax['acc'].legend()

    if 'click' in plot_axes:
        ax['click'].plot(t[:-1], [a[1] for a in aa], label='chosen finger force', color='purple', alpha=0.5)
        ax['click'].plot(t[:-1], [a[1] for a in aa_applied], label='applied finger force')
        ax['click'].set_xlabel('Time [s]')
        ax['click'].set_ylabel('Finger Force')
        ax['click'].legend()


    ## Loss in belief update
    if 'loss' in plot_axes:
        for i, l in enumerate(lll):
            ax['loss'].plot(l, color='purple', alpha=0.1)

    ## Learning rate
    if 'lr' in plot_axes:
        ax['lr'].plot(t[1:], llr, label='learning rate')
        ax['lr'].set_xlabel('Time [s]')
        ax['lr'].set_ylabel('Learning Rate')
        ax['lr'].legend()

    if 'nefe' in plot_axes:
        if len(NEFE_PLAN) > 0:
            ax['nefe'].plot(t[1:], NEFE_PLAN, label='NEFE')
            if len(PRAGMATIC_PLAN) > 0:
                ax['nefe'].plot(t[1:], PRAGMATIC_PLAN, label='Pragmatic')
            if len(INFO_GAIN_PLAN) > 0:
                ax['nefe'].plot(t[1:], INFO_GAIN_PLAN, label='Info Gain')
            ax['nefe'].set_xlabel('Time [s]')
            ax['nefe'].set_ylabel('NEFE')
            ax['nefe'].legend()
        if len(NEFES) > 0:
            ax['nefe'].violinplot(NEFES, positions=t[1:], showmeans=False, showmedians=False, widths=0.1, showextrema=False)

            # calculate normalised components of Pragmatic and Info Gain based on NEFES
            if 'nefe_comp' in plot_axes:
                pragmatic_normalised = []
                info_gain_normalised = []
                for i in range(len(NEFES)):
                    min_pragmatic = np.min(PRAGMATICS[i])
                    min_info_gain = np.min(INFO_GAINS[i])
                    pragmatic_normalised.append(PRAGMATIC_PLAN[i] / min_pragmatic)
                    info_gain_normalised.append(INFO_GAIN_PLAN[i] / min_info_gain)
                ax['nefe_comp'].plot(t[1:], pragmatic_normalised, label='Pragmatic Normalised')
                ax['nefe_comp'].plot(t[1:], info_gain_normalised, label='Info Gain Normalised')
                ax['nefe_comp'].set_xlabel('Time [s]')
                ax['nefe_comp'].set_ylabel('NEFE Components')
                ax['nefe_comp'].legend()

    t_noise_sys = t

    if 'bb_sys' in plot_axes:
        if len(bb_sys) > 0:
            for i in range(len(bb_sys[0][0])):
                if exp_normal_sys_params:
                    mean = [np.exp(b[0][i]) for b in bb_sys]
                else:
                    mean = [b[0][i] for b in bb_sys]
                var = [b[1][i, i] for b in bb_sys]
                ax[f'sys_{i}'].plot(t_noise_sys, mean, label=[f'sys_{i} belief'], color='purple')
                ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
                ax[f'sys_{i}'].fill_between(t_noise_sys, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
                ax[f'sys_{i}'].set_xlabel('Time [s]')
                ax[f'sys_{i}'].set_ylabel(f'Sys {i} State')
                ax[f'sys_{i}'].legend()
                
    if 'bb_noise' in plot_axes:
        if len(bb_noise) > 0:
            for i in range(len(bb_noise[0][0])):
                mean = [b[0][i] for b in bb_noise]
                var = [b[1][i, i] for b in bb_noise]
                ax[f'noise_{i}'].plot(t_noise_sys, mean, label=[f'noise_{i} belief'], color='purple')
                ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
                ax[f'noise_{i}'].fill_between(t_noise_sys, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
                ax[f'noise_{i}'].set_xlabel('Time [s]')
                ax[f'noise_{i}'].set_ylabel(f'Noise {i} State')
                ax[f'noise_{i}'].legend()

    if reaction_time_steps is not None and len(bb_after_rt) > 0:
        ax['pos'].plot(t[reaction_time_steps+1:], [o[0] for o in oo[:-reaction_time_steps]], label=['$o_1(t-\\tau)$'], color = "pink")
        mean = [b[0][0] for b in bb_after_rt]
        var = [b[1][0, 0] for b in bb_after_rt]
        ax['pos'].plot(t[:-1], mean, label=['$\\tilde{Q}^s_1(t)$'], color='green')
        ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
        ax['pos'].fill_between(t[:-1], ribbons[:, 0], ribbons[:, 1], facecolor="green", alpha=1 / 6)
        # ax['pos'].legend()

        ax['pos'].set_xlabel(None)
        # ax['pos'].set_xticklabels([])

        ax['finger'].plot(t[reaction_time_steps+1:], [o[1] for o in oo[:-reaction_time_steps]], label=['$o_2(t-\\tau)$'], color = "pink")
        mean = [b[0][3] for b in bb_after_rt]
        var = [b[1][3, 3] for b in bb_after_rt]
        ax['finger'].plot(t[:-1], mean, label=['$\\tilde{Q}^s_3(t)$'], color='green')
        ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
        ax['finger'].fill_between(t[:-1], ribbons[:, 0], ribbons[:, 1], facecolor="green", alpha=1 / 6)
        ax['finger'].legend()

    if ic_timesteps is not None:
        line_max = np.array([x[0] for x in xx]).max()
        line_min = np.array([x[0] for x in xx]).min()
        # IC timesteps
        ax['pos'].vlines(np.array(ic_timesteps)*dt, ymin=line_min, ymax=line_max, color='grey', linestyle='dotted', label='IC Timesteps', alpha=0.5)
        if ic_pred_error is not None:
            for ic_timestep, pred_belief_state, prediction_error in ic_pred_error:
                # Plot box displaying the predicted belief at that time
                ax['pos'].errorbar(ic_timestep*dt, pred_belief_state[0][0], yerr=3*np.sqrt(pred_belief_state[1][0, 0]), color='grey', fmt="x")
                ax['pos'].text(ic_timestep*dt, line_min-0.1, f"{prediction_error:.2f}", fontsize=8, color='grey', ha='center', va='bottom')

    fig.tight_layout()

    return fig, ax, t, clicks, button_switch
    