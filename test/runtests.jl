using POMDPReinforce
import POMDPModels: GridWorld, LightDark1D
using Reinforce
using Base.Test

type RandomPolicy <: AbstractPolicy end
Reinforce.action(policy::RandomPolicy, r, s, A) = rand(A)

policy = RandomPolicy()

# Test MDP env
env = MDPEnv(GridWorld(), MersenneTwister(1))
last_state = Void()
last_reward = Void()
ep = Episode(env, policy)
for (s, a, r, sp) in ep
    info("State $s, action $a, reward $r")
    last_state = s
    last_reward = r
end
@test last_state == [8.0, 8.0, 0.0]
@test last_reward == 3.0

# Test POMDP env
env = POMDPEnv(LightDark1D(), MersenneTwister(1))
last_obs = Void()
last_reward = Void()
ep = Episode(env, policy)
for (o, a, r, op) in ep
    info("Observation $o, action $a, reward $r")
    last_obs = o
    last_reward = r
end
@test_approx_eq_eps last_obs [7.53396] 1e-3
@test last_reward == 10.0


