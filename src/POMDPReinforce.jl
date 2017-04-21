__precompile__(true)

module POMDPReinforce

using POMDPs
importall Reinforce

export
    POMDPEnv,
    MDPEnv,

    reset!,
    actions,
    step!,
    finished,

    state,
    reward,
    ismdp,

    in
    


type MDPEnv <: AbstractEnvironment
    problem::MDP
    s::Array{Float64}
    r::Float64
    t::Type
    rng::AbstractRNG
end
function MDPEnv(p::MDP, rng::AbstractRNG=MersenneTwister()) 
    s = initial_state(p,rng)
    MDPEnv(p, convert(Array{Float64},initial_state(p,rng),p), 0.0, typeof(s), rng)
end

type POMDPEnv <: AbstractEnvironment
    problem::POMDP
    s
    o::Array{Float64}
    r::Float64
    t::Type
    rng::AbstractRNG
end
function POMDPEnv(p::POMDP, rng::AbstractRNG=MersenneTwister()) 
    s = initial_state(p,rng)
    o = convert(Array{Float64}, generate_o(p, s, rng), p)
    return POMDPEnv(p, s, o, 0.0, typeof(o), rng)
end

type POMDPActionSpace
    action_space
    rng::AbstractRNG
end

# dummies for now, POMDPs doesn't really check spaces like this
Base.in(as::POMDPActionSpace, a) = true
Base.in(a, as::POMDPActionSpace) = true

Reinforce.ismdp(env::MDPEnv) = true
Reinforce.ismdp(env::POMDPEnv) = false

finished(env::MDPEnv, s) = isterminal(env.problem, convert(env.t, s, env.problem))
finished(env::POMDPEnv, o) = isterminal(env.problem, env.s)

Reinforce.actions(env::MDPEnv, s) = POMDPActionSpace(POMDPs.actions(env.problem, s), env.rng)
Reinforce.actions(env::POMDPEnv, s) = POMDPActionSpace(POMDPs.actions(env.problem, s), env.rng)

Base.rand(as::POMDPActionSpace) = rand(as.rng, as.action_space)

Reinforce.state(env::MDPEnv) = env.s
Reinforce.state(env::POMDPEnv) = env.o

Reinforce.reward(env::MDPEnv) = env.r
Reinforce.reward(env::POMDPEnv) = env.r

function Reinforce.reset!(env::MDPEnv)
    s = initial_state(env.problem, env.rng)
    env.s = convert(Array{Float64}, s, env.problem)
    return env.s
end

function Reinforce.reset!(env::POMDPEnv)
    s = initial_state(env.problem, env.rng)
    o = generate_o(env.problem, s, env.rng)
    env.s, env.o = s, convert(Array{Float64}, o, env.problem)
    return env.o
end

function Reinforce.step!(env::MDPEnv, s, a)
    sp, r = generate_sr(env.problem, convert(env.t, s, env.problem), a, env.rng)
    env.r, env.s = r, convert(Array{Float64}, sp, env.problem)
    return r, env.s
end

function Reinforce.step!(env::POMDPEnv, o, a)
    sp, op, r = generate_sor(env.problem, env.s, a, env.rng)
    env.s, env.o, env.r = sp, convert(Array{Float64}, op, env.problem), r
    return r, env.o
end

end # module
