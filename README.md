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
  * POMDP: Defines a POMDP and inherits gym.Env as a belief-space MDP
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

## Solvers:
For understanding the solvers, I recommend:
1.  skimming these slides: https://www.cs.cmu.edu/~ggordon/780-fall07/lectures/POMDP_lecture.pdf
2.  Reading sections 2.2 to 4.1 of this survey: https://www.cs.mcgill.ca/~jpineau/files/jpineau-jaamas12-finalcopy.pdf


## Back sensors:

Normally, in POMDPs, we have a sensor model that gives probabilities O(o |s', a) of observations, given the action taken and the resulting **next state**. However, we are dealing with a very particular type of POMDP, in which our original state space is fully observable and deterministic, with the only uncertainty being the reward; and since all the information about the reward is contained in the human's actions, we can instead treat the human's actions as observations (this can be done as long as the state is fully determined by the actions (i.e. fixed initial positions and deterministic transitions)).

To do that however, we need a sensor model that relates the **previous state** with the observation, instead of the next state; so we include an optional 'back_sensor', which gives probabilities O'(o' | s, a). This change greatly reduces the complexity of solvers (e.g. in RedBlueAssistanceProblem, it reduces an exponent in the complexity from 24 to 2, changing it from intractable to solvable 1-3 seconds).


## Current limitations:

The main limitation is that all the code assumes that transitions, rewards and sensors are in the form of full matrices. This highly limits how much the code can scale, since a full transition matrix needs |S|**2|A_H||A_R| entries, which for a 4x4 grid with 2 agents and nothing else, this is 1048576 entries (though only 4096 non-zero entries). So, it is necessary to adapt the code to functions (instead of matrices), and maybe also sparse matrices.

