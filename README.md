## How to run the code

```
pip install -e .
python -m assistance_games.run
```

You should see the environment below:

![RedBlueAssistanceProblem](docs/redblue1.gif)


## Overview of files:

### core.py
contain the core classes, such as:
  * POMDP: Defines a POMDP and inherits gym.Env
  * AssistanceGame
  * AssistanceProblem
  * POMDPPolicy: Policy returned by solvers
  * Human policy functions: Functions that map an assistance game and a reward to a human policy
### solver.py
POMDP solvers
  * exact\_vi : Exact solver, can only solve very small environments
  * pbvi : Approximate anytime solver, relatively fast for medium-sized environments
  * deep\_rl\_solve : 5-line wrapper around stable\_baselines.PPO2
### envs.py
Contain a few instantiations of the core classes, such as:
  * RedBlueAssistanceGame
  * RedBlueAssistanceProblem
  * FourThreeMaze
  * TwoStatePOMDP
### rendering.py
Rendering utils for envs.
### parser.py
Parser for .pomdp files (mostly for testing/benchmarking solvers).
### run.py
Simple script to run an environment.
### utils.py
Some utils.
### tests.py
Very basic smoke tests.

## POMDP class:

The current design uses a flexible, compositional architecture in which a POMDP has the following components:
- A state space, which also contains the initial state distribution
- An action space
- A transition model, which generates next states
- A sensor model, which generates "senses"\*
- An observation model, which generates the observations to be returned from env.reset and env.step
- A reward model, which generates rewards

This supposedly makes for a very flexible implementation, in which the spaces can be continuous or infinite, transitions and rewards can be non-Markovian and time-dependent, and the environment specification can be different from the POMDP specification (e.g. different versions of observation models: observations can be (i) beliefs, (ii) human actions, (iii) human actions concatenated with state features).

To get a particular behavior, it might be necessary to add implementations of new models, but that should be only a few lines of code.

\* The motivation for having both 'senses' and 'observations' is that there is that there are "observations" in the POMDP definition, and "observations" returned from the gym.Env, and they can be different (e.g. in a belief-space MDP the gym.Env observations would be beliefs, instead of "POMDP observations"). To differentiate them, we use 'senses' for the POMDP definition, and 'observations' for the gym.Env implementation.

## AssistanceProblem.\_\_init\_\_

A similar decomposition is made on the construction of an AssistanceProblem: the constructor takes arbitrary functions that create the POMDP models from the assistance game and the human\_policy\_fn.

## Solvers:
For understanding the solvers, I recommend:
1.  skimming these slides: https://www.cs.cmu.edu/~ggordon/780-fall07/lectures/POMDP_lecture.pdf
2.  Reading sections 2.2 to 4.1 of this survey: https://www.cs.mcgill.ca/~jpineau/files/jpineau-jaamas12-finalcopy.pdf

If running PBVI on your task gives different results each time, you might want to increase the number of iterations to make sure it finds the optimal solution - for the tasks I tested, PBVI found the optimal solution consistently and I did not have to worry about this.

## Human Policies:

For computing a human policy, the human has to use the transition matrix from the AssistanceGame, which gives P(s' | a\_h, a\_r, s). Thus, the human needs to have some robot model to be able to compute transitions. This might be something like 'assume robot acts randomly', or 'assume robot has full information', or something more complex - still, it cannot have a perfect model of the robot, since the full robot behavior depends on the results of the POMDP solvers.

## Back sensors:

Normally, in POMDPs, we have a sensor model that gives probabilities O(o |s', a) of observations, given the action taken and the resulting **next state**. However, we are dealing with a very particular type of POMDP, in which our original state space is fully observable and deterministic, with the only uncertainty being the reward; and since all the information about the reward is contained in the human's actions, we can instead treat the human's actions as observations (this can be done as long as the state is fully determined by the actions (i.e. fixed initial positions and deterministic transitions)).

To do that however, we need a sensor model that relates the **previous state** with the observation, instead of the next state; so we include an optional 'back\_sensor', which gives probabilities O'(o' | s, a). This change greatly reduces the complexity of solvers (e.g. in RedBlueAssistanceProblem, it reduces an exponent in the complexity from 24 to 2, changing it from intractable to solvable 1-3 seconds).
