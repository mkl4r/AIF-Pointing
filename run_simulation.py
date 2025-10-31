import os
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental.ode import odeint
from jax.nn import sigmoid
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime

from difai.aif import AIF_Env, AIF_Agent, AIF_Simulation


# Configuration
jax.config.update("jax_enable_x64", True)

# Simulation constants
MAX_STEPS = 100
MAX_COUNTS_PER_TARGET = 10
REACTION_TIME = 0.1
DT = 0.02

# System parameters
DEFAULT_K = 0.0
DEFAULT_D = 24.0
DEFAULT_K_FINGER = 10.0

# Target configuration
TARGET_BOUNDS = np.array([[50, 1750], [5, 50]])
NUM_X_TARGETS = 6
NUM_R_TARGETS = 3


class Mouse_Cursor_Click_Finger_Obs_Missclick(AIF_Env):
    """
    Mouse cursor environment with finger dynamics and click detection.
    
    Models a spring-damper system for mouse movement with finger state tracking
    and click/missclick observation capabilities.
    """

    def __init__(self, x0=jnp.array([-0.1, 0.0, 0.0, 0.0]), k=40.0, d=0.7, 
                 k_finger=7.5, target_position=0.1, target_radius=0.02, dt=0.01):
        """
        Initialize mouse cursor environment.
        
        Args:
            x0: Initial state [position, velocity, finger_old, finger_new]
            k: Spring stiffness parameter
            d: Damping parameter  
            k_finger: Finger stiffness parameter
            target_position: Target position for cursor
            target_radius: Target radius for successful clicks
            dt: Time step duration
        """
        self.x0 = x0
        self.dt = dt
        self.sys_params = {
            'k': k, 'd': d, 'k_finger': k_finger, 
            'target_position': target_position, 'target_radius': target_radius
        }
        self.non_negative_sys_params = [0, 1, 2, 3, 4]
        self.dim_action = 2
        self.dim_observation = 5
        self.jitable = True


    @staticmethod
    def _forward_complete(x, u, dt, random_realisation, key, k, d, k_finger, target_position, target_radius):
        """
        Computes one forward step with the spring-damper system (unit mass).
        
        Args:
            x: Initial state (position, velocity, finger state old, finger state new)
            u: Control value (acceleration, click)
            dt: Time step duration in seconds
            random_realisation: Not used here
            key: Not used here
            k: Stiffness parameter
            d: Damping parameter
            k_finger: Finger stiffness parameter
            target_position: Target position for the mouse cursor
            target_radius: Target radius for the mouse cursor
            
        Returns:
            x: Resulting state
        """
        # 1. Define system matrix A
        A = jnp.array([[0,          1,          0,          0,      0],    
                       [-k ,        -d,         0,          1,      0], 
                       [0,          0,       -k_finger,     0,      1],
                       [0,          0,          0,          0,      0],
                       [0,          0,          0,          0,      0]])

        step_fn = lambda y, t: A @ y
        
        # 2. Compute next state using ODE solver
        y = jnp.hstack([x[0], x[1], x[3], u]) # initial state: position, velocity, finger state, control
        solution = odeint(step_fn, y, jnp.array([0,dt]), rtol=1.4e-8, atol=1.4e-8)
        y = solution[1]

        x = x.at[:2].set(y[:2]) # update cursor position and velocity
        x = x.at[2].set(x[3]) # update previous and new finger state
        x = x.at[3].set(y[2]) 
       
        return x 

    @staticmethod
    def _get_observation_complete(x, k, d, k_finger, target_position, target_radius):
        """
        Generate observations from current state.
        
        Returns position, finger state, button click, mouse click, and missclick indicators.
        """
        sigmoid_steepness = 1e6
        click_trigger_threshold = 0.05
        mouse_clicked = (sigmoid(sigmoid_steepness*(x[3]-click_trigger_threshold)) * 
                        sigmoid(sigmoid_steepness*(click_trigger_threshold-x[2])))
        button_clicked = (mouse_clicked * 
                         sigmoid(-sigmoid_steepness*(jnp.abs(x[0] - target_position) - target_radius)))
        return jnp.hstack([x[0], x[3], button_clicked, mouse_clicked, button_clicked-mouse_clicked])


def create_mouse_env(targets, start_target, target_id):
    """
    Create mouse cursor environment for specified target.
    
    Args:
        targets: Array of target positions and radii
        start_target: Starting position
        target_id: Index of target to use
        
    Returns:
        Tuple of (mouse_cursor, sys_params, x0, noise_params, buttons)
    """
    buttons = [start_target, targets[target_id]]
    x0 = jnp.array([buttons[0][0], 0.0, 0.0, 0.0])

    k = np.float64(DEFAULT_K)
    d = np.float64(DEFAULT_D)
    k_finger = np.float64(DEFAULT_K_FINGER)
    sys_params = jnp.array([k, d, k_finger, buttons[1][0], buttons[1][1]])

    mouse_cursor = Mouse_Cursor_Click_Finger_Obs_Missclick(x0, *sys_params, dt=DT)
    mouse_cursor.reset()

    noise_params = {
        'observation_std': {'id': np.array([0,1]), 'value': jnp.array([0.001, 0.001])}
    }

    return mouse_cursor, sys_params, x0, noise_params, buttons


def create_agent(sys_params, x0, dt):
    """
    Create AIF agent with mouse cursor model.
    
    Args:
        sys_params: System parameters for the mouse cursor
        x0: Initial state
        dt: Time step duration
        
    Returns:
        Tuple of (agent, intermittent_action_selection)
    """
    intermittent_action_selection = 0

    # Create Generative Model (Mouse Cursor Model)
    mouse_cursor_model = Mouse_Cursor_Click_Finger_Obs_Missclick(x0, *sys_params, dt=dt)
    mouse_cursor_model.reset()

    noise_params = {'observation_std': {'id': np.array([0,1,2,3,4])}}
    agent = AIF_Agent(mouse_cursor_model, noise_params)

    initial_belief_sys_cov = jnp.diag(jnp.array([0.000001, 0.2, 0.2, 0.000001, 0.000001])**2)
    initial_belief_state_cov = jnp.diag(jnp.array([0.001, 0.0001, 0.00005, 0.00005])**2)

    agent.set_params_with_defaults(
        n_samples_a=30,
        n_samples_a_noise_sys=10,
        n_samples_a_combine=200,
        lr_o=3e-4,
        use_complete_ukf=True, 
        n_steps_o=30,
        n_samples_o=300,
        horizon=12,
        n_plans=3000,
        action_prior=[jnp.array([0.0, 0.0]), jnp.array([[50**2,0], [0, 1**2]])],
        use_info_gain=False, 
        n_samples_obs_pref_o=3,
        n_samples_obs_pref_s=50,
        select_max_pi=True
    )
    
    agent.set_initial_beliefs(
        initial_belief_state=[x0, initial_belief_state_cov], 
        initial_belief_noise=[jnp.log(jnp.array([0.001, 0.001, 0.000001, 0.000001, 0.000001])), 
                             0.0000001*jnp.eye(agent.params['dim_noise'])],
        initial_belief_sys=[sys_params, initial_belief_sys_cov]
    )

# Preference distribution for observing button click and not observing missclick
    agent.set_preference_distribution(
        C=[jnp.array([1.0, 1.0, 0.0]), jnp.diag(jnp.array([0.01**2, 0.01**2, 0.001**2]))],
        C_index=[0,2,4],
        sys_dependent_C=np.array([[0],[3]]), # first entry: index in C mean that depends on system parameter, second entry: index in system parameter that is used for the dependency (target position)
        use_observation_preference=True
    )
    agent.initialize()    

    return agent, intermittent_action_selection


def setup_targets():
    """
    Create target configuration for the experiment.
    
    Returns:
        Tuple of (targets, start_target, target_selection)
    """
    num_targets = NUM_X_TARGETS * NUM_R_TARGETS
    targetr = TARGET_BOUNDS[1][0] + np.linspace(5, TARGET_BOUNDS[1][1] - TARGET_BOUNDS[1][0], NUM_R_TARGETS)
    
    assert NUM_X_TARGETS % 2 == 0
    center = (TARGET_BOUNDS[0][0] + TARGET_BOUNDS[0][1]) // 2
    distance_from_center = np.linspace(
        (TARGET_BOUNDS[0][0] + TARGET_BOUNDS[0][1]) // 8, 
        (TARGET_BOUNDS[0][0] + TARGET_BOUNDS[0][1]) // 2 - TARGET_BOUNDS[1,1], 
        NUM_X_TARGETS // 2
    )
    
    start_target = np.array([center, 3.0])
    targets_right = center + distance_from_center
    targets_left = center - distance_from_center
    targetx = np.hstack([targets_left, targets_right])
        
    targetsxr = np.meshgrid(targetx, targetr)
    targetsxr = np.stack((targetsxr[0], targetsxr[1]), axis=-1).reshape(num_targets, 2).tolist()
    targets = np.array([np.array((xr[0], xr[1])) for xr in targetsxr])

    # Scale for AIF
    start_target = start_target / 1000
    targets = targets / 1000

    # Sort targets by width and position descending
    targets = targets[np.lexsort((-targets[:, 0], -targets[:, 1]))]

    # Center targets around 0 for AIF
    targets[:, 0] = targets[:, 0] - start_target[0]
    start_target[0] = 0.0

    target_selection = range(num_targets)
    return targets, start_target, target_selection


# Setup experiment configuration
targets, start_target, target_selection = setup_targets()


# Obtain System Parameters
_, sys_params, x0, _, _ = create_mouse_env(targets, start_target, 0)
agent, intermittent_action_selection = create_agent(sys_params=sys_params, x0=x0, dt=DT)
reaction_time_steps = int(REACTION_TIME // DT)
dim_observation = agent.params['dim_observation']
dim_state = agent.params['dim_state']
dim_noise = agent.params['dim_noise']
dim_action = agent.params["dim_action"]
initial_belief_sys_cov = agent.params['initial_belief_sys'][1]


def main():
    """Main simulation execution."""
    key = random.key(0)

    for target_id in target_selection:
        mouse_cursor, sys_params, x0, noise_params, buttons = create_mouse_env(targets, start_target, target_id)
        sim = AIF_Simulation(agent, mouse_cursor, noise_params)

        for repeat in range(MAX_COUNTS_PER_TARGET):
            print(f"Running target {target_id} (repeat {repeat+1}/{MAX_COUNTS_PER_TARGET}).")
            save_path = f"./data/sim_data/{datetime.today().strftime('%Y%m%d')}/target_{target_id}_rep_{repeat}.pickle"
            if os.path.exists(save_path):
                print(f"File {save_path} already exists. Skipping...")
                continue

            agent.belief_sys = [jnp.hstack([sys_params[:3], start_target[0], 0.03]), 
                               jnp.diag(jnp.hstack([jnp.diag(initial_belief_sys_cov)[:3], 
                                                   jnp.array([0.9, 0.02])]))]

            belief_state = agent.belief_state
            action_buffer = jnp.zeros((reaction_time_steps, agent.params['dim_action']))
            observation_buffer = jnp.zeros((reaction_time_steps, agent.params['dim_observation']))
            mouse_cursor.reset()
            
            bb_a = [belief_state]
            bb_o = [belief_state]
            bb_sys = [agent.belief_sys]
            bb = [belief_state]
            bb_after_rt = []
            lll = []
            lll_sys = []
            LR = []
            xx = [mouse_cursor.x]
            oo = []
            aa = []
            aa_applied = []
            BBS = []
            NEFE_PLAN = []
            PRAGMATIC_PLAN = []
            INFO_GAIN_PLAN = []
            NEFES = []
            PRAGMATICS = []
            INFO_GAINS = []
            
            for i in tqdm(range(MAX_STEPS)):
                if REACTION_TIME > 0 and i >= reaction_time_steps:
                    agent.belief_sys = [sys_params, initial_belief_sys_cov]
                    
                belief_state_after_rt = belief_state
                for j in range(reaction_time_steps):
                    a = action_buffer[j]
                    key, use_key = random.split(key)
                    belief_state_after_rt, BS = agent.update_belief_state(belief_state_after_rt, agent.belief_noise, agent.belief_sys, a, key=use_key)

                bb_after_rt.append(belief_state_after_rt)

                key, use_key = random.split(key)
                sel_plan, nefe_plan, pragmatic_plan, info_gain_plan, plans, nefes, pragmatics, info_gains = agent.select_action(belief_state_after_rt, agent.belief_noise, agent.belief_sys, key=use_key)
                a_new = sel_plan[0]

                key, use_key = random.split(key)
                o, x, a_applied = sim.step(a_new, debug=True, key=use_key)

                observation_buffer = jnp.roll(observation_buffer, -1, axis=0)
                observation_buffer = observation_buffer.at[-1].set(o)

                NEFE_PLAN.append(nefe_plan)
                PRAGMATIC_PLAN.append(pragmatic_plan)
                INFO_GAIN_PLAN.append(info_gain_plan)
                NEFES.append(nefes)
                PRAGMATICS.append(pragmatics)
                INFO_GAINS.append(info_gains)
                xx.append(x)
                aa.append(action_buffer[0])
                aa_applied.append(a_applied)
                oo.append(o)

                a = action_buffer[0]
                key, use_key = random.split(key)
                belief_state, BS = agent.update_belief_state(belief_state, agent.belief_noise, agent.belief_sys, a, key=use_key)

                action_buffer = jnp.roll(action_buffer, -1, axis=0)
                action_buffer = action_buffer.at[-1].set(a_new)

                if i >= (reaction_time_steps-1):
                    o = observation_buffer[0]
                    key, use_key = random.split(key)
                    belief_state, ll, lr = agent.update_belief_state_obs(belief_state, agent.belief_noise, agent.belief_sys, o, key=use_key)
                    lll.append(ll)
                    LR.append(lr)
      
                bb.append(belief_state)
                bb_sys.append(agent.belief_sys)

                if jnp.isnan(belief_state[0]).any() or np.isnan(belief_state[1]).any():
                    print("NAN in belief. Breaking...")
                    break

                if observation_buffer[-1][2] > 0.9:
                    print("Button clicked. Breaking...")
                    break
            else:
                print("Max steps reached! Switching condition.")

            create_dir = os.path.dirname(save_path)
            if not os.path.exists(create_dir):
                os.makedirs(create_dir)
                print(f"Directory {create_dir} created.")
                
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'xx': xx, 'oo': oo, 'bb': bb, 'bb_after_rt': bb_after_rt,
                    'aa': aa, 'aa_applied': aa_applied, 'lll': lll, 'llr': LR,
                    'nefess': NEFES, 'pragmatics': PRAGMATICS, 'info_gains': INFO_GAINS,
                    'nefe_plan': NEFE_PLAN, 'pragmatic_plan': PRAGMATIC_PLAN,
                    'info_gain_plan': INFO_GAIN_PLAN, 'bb_sys': bb_sys,
                    'belief_noise': agent.belief_noise, 'C_index': agent.params['C_index'],
                    'C': agent.C, 'params': agent.params, 'sys_params_real': sys_params,
                    'noise_params_real': noise_params, 'noise_params_model': noise_params,
                    'buttons': buttons, 'dt': DT
                }, f)


if __name__ == "__main__":
    main()

for target_id in target_selection:

    # Create Generative Process (real system)
    mouse_cursor, sys_params, x0, noise_params, _, buttons = create_mouse_env(targets, start_target, target_id)

    # Create Simulation
    sim = AIF_Simulation(agent, mouse_cursor, noise_params)

    for repeat in range(MAX_COUNTS_PER_TARGET):
        print(f"Running target {target_id} (repeat {repeat+1}/{MAX_COUNTS_PER_TARGET}).")
        # check if file already exists
        save_path = f"./data/sim_data/{datetime.today().strftime('%Y%m%d')}/target_{target_id}_rep_{repeat}.pickle"
        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping...")
            continue

        # Set agents belief of the target to be very uncertain and centred around the start
        agent.belief_sys = [jnp.hstack([sys_params[:3], start_target[0], 0.03]), jnp.diag(jnp.hstack([jnp.diag(initial_belief_sys_cov)[:3], jnp.array([0.9, 0.02])]))]

        belief_state = agent.belief_state

        action_buffer = jnp.zeros((reaction_time_steps, agent.params['dim_action']))
        observation_buffer = jnp.zeros((reaction_time_steps, agent.params['dim_observation']))

        mouse_cursor.reset()
        
        # Logging
        bb_a = [belief_state]
        bb_o = [belief_state]
        bb_sys = [agent.belief_sys]
        bb = [belief_state]
        bb_after_rt = []
        lll = []
        lll_sys = []
        LR = []
        xx = [mouse_cursor.x]
        oo = []
        aa = []
        aa_applied = []
        BBS = []
        NEFE_PLAN = []
        PRAGMATIC_PLAN = []
        INFO_GAIN_PLAN = []
        NEFES = []
        PRAGMATICS = []
        INFO_GAINS = []
        for i in tqdm(range(MAX_STEPS)):
            
            if reaction_time > 0 and i >= reaction_time_steps:
                # If we are past the reaction time, let the agent know about the target
                agent.belief_sys = [sys_params, initial_belief_sys_cov]
                
            # Predict the state after reaction time
            belief_state_after_rt = belief_state
            for j in range(reaction_time_steps):
                a = action_buffer[j]
                key, use_key = random.split(key)
                belief_state_after_rt, BS = agent.update_belief_state(belief_state_after_rt, agent.belief_noise, agent.belief_sys, a, key=use_key)

            bb_after_rt.append(belief_state_after_rt)

            # Select action as usual
            key, use_key = random.split(key)
            sel_plan, nefe_plan, pragmatic_plan, info_gain_plan, plans, nefes, pragmatics, info_gains = agent.select_action(belief_state_after_rt, agent.belief_noise, agent.belief_sys, key=use_key)
            a_new = sel_plan[0]

            # Make system step
            key, use_key = random.split(key)
            o, x, a_applied = sim.step(a_new, debug=True, key=use_key)

            # Fill observation buffer with the observation
            observation_buffer = jnp.roll(observation_buffer, -1, axis=0)
            observation_buffer = observation_buffer.at[-1].set(o)

            NEFE_PLAN.append(nefe_plan)
            PRAGMATIC_PLAN.append(pragmatic_plan)
            INFO_GAIN_PLAN.append(info_gain_plan)
            NEFES.append(nefes)
            PRAGMATICS.append(pragmatics)
            INFO_GAINS.append(info_gains)
            xx.append(x)
            aa.append(a)
            aa_applied.append(a_applied)
            oo.append(o)

            a = action_buffer[0]
            key, use_key = random.split(key)
            belief_state, BS = agent.update_belief_state(belief_state, agent.belief_noise, agent.belief_sys, a, key=use_key)

            action_buffer = jnp.roll(action_buffer, -1, axis=0)
            action_buffer = action_buffer.at[-1].set(a_new)

            if i >= (reaction_time_steps-1):
                o = observation_buffer[0]
                key, use_key = random.split(key)
                belief_state, ll, lr = agent.update_belief_state_obs(belief_state, agent.belief_noise, agent.belief_sys, o, key=use_key)
                lll.append(ll)
                LR.append(lr)
  
            bb.append(belief_state)
            bb_sys.append(agent.belief_sys)
            

            if jnp.isnan(belief_state[0]).any() or np.isnan(belief_state[1]).any():
                print("NAN in belief. Breaking...")
                break

            if observation_buffer[-1][2] > 0.9:
                print("Button clicked. Breaking...")
                break

        else:
            print("Max steps reached! Switching condition.")

        create_dir = os.path.dirname(save_path)
        if not os.path.exists(create_dir):
            os.makedirs(create_dir)
            print(f"Directory {create_dir} created.")
            
        with open(save_path, 'wb') as f:
            pickle.dump({
                'xx': xx,
                'oo': oo,
                'bb': bb,
                'bb_after_rt': bb_after_rt,
                'aa': aa,
                'aa_applied': aa_applied,
                'lll': lll,
                'llr': LR,
                'nefess': NEFES,
                'pragmatics': PRAGMATICS,
                'info_gains': INFO_GAINS,
                'nefe_plan': NEFE_PLAN,
                'pragmatic_plan': PRAGMATIC_PLAN,
                'info_gain_plan': INFO_GAIN_PLAN,
                'bb_sys': bb_sys,
                'belief_noise': agent.belief_noise,
                'C_index': agent.params['C_index'],
                'C': agent.C,
                'params': agent.params,
                'sys_params_real': sys_params,
                'noise_params_real': noise_params,
                'noise_params_model': noise_params,
                'buttons': buttons,
                'dt': dt
            }, f)

