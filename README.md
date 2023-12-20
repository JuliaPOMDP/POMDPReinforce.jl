# THIS PACKAGE IS DEPRECATED

[Reinforce.jl](https://github.com/JuliaML/Reinforce.jl) is discontinued. Please reference [CommonRLInterface](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl).

# POMDPReinforce

[![Build Status](https://travis-ci.org/etotheipluspi/POMDPReinforce.jl.svg?branch=master)](https://travis-ci.org/etotheipluspi/POMDPReinforce.jl)

[![Coverage Status](https://coveralls.io/repos/etotheipluspi/POMDPReinforce.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/etotheipluspi/POMDPReinforce.jl?branch=master)

[![codecov.io](http://codecov.io/github/etotheipluspi/POMDPReinforce.jl/coverage.svg?branch=master)](http://codecov.io/github/etotheipluspi/POMDPReinforce.jl?branch=master)

This package wraps [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) problems into reinforcement learning environments with [Reinforce.jl](https://github.com/JuliaML/Reinforce.jl) interface. 
This allows POMDP problems to be used with a variety of deep reinforcement learning algorithms. 

## Quick Start

We will use the `Episode` type from `Reinforce` to run a quick simulation. 

```julia
using POMDPReinforce
using Reinforce
import POMDPModels: GridWorld, LightDark1D

# Create a random Reinforce.jl policy 
type RandomPolicy <: AbstractPolicy end
Reinforce.action(policy::RandomPolicy, r, s, A) = rand(A)
policy = RandomPolicy()


# Simulating an grid world MDP environment
env = MDPEnv(GridWorld())
ep = Episode(env, policy)
for (s, a, r, sp) in ep
    # do some custom pre-processing if needed
    info("State $s, action $a, reward $r")
end

# Simulating a light dark POMDP environment
env = POMDPEnv(LightDark1D(), MersenneTwister(1))
ep = Episode(env, policy)
for (o, a, r, op) in ep
    # do some custom pre-processing if needed
    info("Observation $o, action $a, reward $r")
end
```

## OpenAI Gym Like Interface

You can also simulate wrapped `POMDPs` problems in a similar way to [OpenAI Gym](https://github.com/openai/gym).

```julia
using POMDPReinforce
import POMDPModels: GridWorld

env = MDPEnv(GridWorld())

s = reset!(env)
while !finished(env, s)
    a = rand(actions(env, s))
    r, sp = step!(env, s, a)
    info("State $s, action $a, reward $r")
    s = sp
end
```

