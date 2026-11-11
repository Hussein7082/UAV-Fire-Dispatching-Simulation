"""Verify the manuscript's DQN observation and transition description.
Run python verify_drl_environment.py. Does not train or alter saved outputs.
"""
import copy
import numpy as np
import paper1_uav_fire_dispatching_training as m


def main():
    env=m.FireDispatchEnv(np.array([0.,0.]),np.array([0.27,0.01]),np.array([1.,0.05]),0.,0.)
    s=env.reset(wind=0.1)
    np.testing.assert_allclose(s,[1.,2/3,1.,0.,0.5,2/15,1.])
    assert len(s)==m.FireDispatchEnv.N_STATES==7
    # Full simulator state cannot be recovered from the observation: same
    # observation/action, different termination because the step count is hidden.
    late=copy.deepcopy(env); late.steps=m.DRL_MAX_STEPS-1
    np.testing.assert_array_equal(env._state(),late._state())
    assert not env.step(2)[2] and late.step(2)[2]
    assert env.wind==late.wind==0.1
    # A high-risk incident beyond its RARD deadline is admitted by the DQN env.
    env.reset(wind=0.1)
    arrival_min, go, hover, back = env._mission_energy(0)
    assignment = m.compute_matrices(
        env.f_lat, env.f_lon, env.f_risk,
        np.array([env.b_lat]), np.array([env.b_lon]), export=False,
    )
    deadline = assignment["deadline"][0]
    np.testing.assert_allclose(arrival_min, assignment["delay"][0, 0])
    assert arrival_min > deadline and env._feasible(0)
    assert not assignment["feasible"][0, 0]
    _,_,done=env.step(0)
    assert env.served==1 and not done
    np.testing.assert_allclose(env.battery,m.BATTERY_CAPACITY_WH-go-hover)
    np.testing.assert_allclose([env.lat,env.lon],[0.,0.27])
    assert env.pending==[1] and env.cursor==0
    # Explicit terminal return charges travel cost but does not move/debit state.
    before=(env.lat,env.lon,env.battery)
    _,reward,done=env.step(3)
    assert done and (env.lat,env.lon,env.battery)==before
    np.testing.assert_allclose(reward,env._return_cost())
    # Exact reserve boundary; below it the attempted target is discarded.
    e=m.FireDispatchEnv(np.array([0.]),np.array([0.]),np.array([1.]),0.,0.)
    e.reset(wind=0.)
    boundary=m.HOVER_POWER_W*m.HOVER_TIME_MIN/60+m.BATTERY_CAPACITY_WH*m.SAFETY_RESERVE_RATIO
    e.battery=boundary; assert e._feasible(0)
    e.battery=np.nextafter(boundary,-np.inf); assert not e._feasible(0)
    before=e.battery; state,reward,done=e.step(0)
    assert done and e.served==0 and not e.pending and e.battery==before
    np.testing.assert_allclose(reward,-m.A_DELAY_RL*m.RETARGET_TIME_MIN)
    np.testing.assert_array_equal(state[[1,2,6]],[0.,0.,0.])
    # Recharge uses the same base and fixed delay, restoring full capacity.
    env.reset(wind=0.1); env.step(0)
    t,e=env._leg(env.lat,env.lon,env.b_lat,env.b_lon)
    _,reward,done=env.step(1)
    assert not done and env.battery==m.BATTERY_CAPACITY_WH
    assert (env.lat,env.lon)==(env.b_lat,env.b_lon)
    np.testing.assert_allclose(reward,-m.A_DELAY_RL*(t+m.CHARGE_TIME_MIN)-m.B_ENERGY_RL*e)
    print('PASS: seven feature values; hidden-step observation alias; fixed wind; admission beyond RARD deadline; service battery debit and cursor; exact reserve boundary; failed-target removal; terminal return accounting; recharge location, delay and battery reset.')
    print(f'Counterexample: arrival {arrival_min:.6f} min (RARD deadline {deadline:.1f} min); DQN admission ignores that deadline.')

if __name__=='__main__': main()
