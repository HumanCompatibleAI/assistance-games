"""Parser for .pomdp files.

Specification can be found at:
https://cs.brown.edu/research/ai/pomdp/examples/pomdp-file-spec.html

File repository found at:
https://cs.brown.edu/research/ai/pomdp/examples/index.html
"""

from collections import defaultdict
import numpy as np

from lark import Lark, Transformer
from gym.spaces import Discrete

from functools import partial


from assistance_games.core import DiscreteDistribution

pomdp_parser = Lark(r"""
    pomdp_file: preamble param_list

    ?preamble: param_type+

    ?param_type: discount
               | values
               | states
               | actions
               | observations
               | start

    discount: "discount:"i number

    values: "values:"i values_type
    !values_type: "reward"i
                | "cost"i

    states: "states:"i (integer | ident_list)
    actions: "actions:"i (integer | ident_list)
    observations: "observations:"i (integer | ident_list)
    
    start: "start:"i u_matrix

    ?param_list: (transitions | sensors | rewards)+

    ?transitions: "T:" action (  ":" state ":" state prob
                               | ":" state u_matrix
                               | ui_matrix)

    ?sensors: "O:" action (  ":" state ":" obs prob
                           | ":" state u_matrix
                           | u_matrix)

    ?rewards: "R:" action ":" state (  ":" state ":" obs number
                                     | ":" state num_matrix
                                     | num_matrix)

    ?state: value
    ?action: value
    ?obs: value

    !value: integer | name | "*"

    !ui_matrix: uniform | identity | num_matrix
    ?u_matrix: uniform | num_matrix

    uniform: "uniform"i
    identity: "identity"i
    ?num_matrix: number+

    ident_list: name+ 

    integer: INT
    prob: NUMBER
    number: SIGNED_NUMBER

    name: NAME
    NAME: /[a-zA-Z\-_]+/

    COMMENT: /#.*\n/

    %import common.ESCAPED_STRING
    %import common.LETTER
    %import common.SIGNED_NUMBER
    %import common.NUMBER
    %import common.INT
    %import common.WS
    %ignore WS
    %ignore COMMENT

    """, start='pomdp_file')

     

class TreeToPOMDP(Transformer):
    def pomdp_file(self, items):
        def process_matrix(matrix, shape):
            l = shape[0]
            if matrix == 'uniform':
                return np.ones(shape) / l
            elif matrix == 'identity':
                return np.eye(l)
            else:
                return np.array(matrix).reshape(shape)

        preamble, specs = items

        discount = preamble['discount']
        values_type = preamble['values']

        get_len = lambda x : x if type(x) == int else len(x)

        states = preamble['states']
        actions = preamble['actions']
        obs = preamble['observations']

        num_states = get_len(states)
        num_actions = get_len(actions)
        num_obs = get_len(obs)

        initial_state_distribution = preamble.get('start', 'uniform')
        initial_state_distribution = process_matrix(initial_state_distribution, (num_states,))
        initial_state_distribution = DiscreteDistribution(initial_state_distribution)

        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        O = np.zeros((num_actions, num_states, num_obs))

        def get_raw_fn(space):
            def fn(item):
                if item == '*':
                    return slice(None)
                elif type(item) == str:
                    return space.index(item)
                else:
                    return item
            return fn

        get_raw_state = get_raw_fn(states)
        get_raw_act = get_raw_fn(actions)
        get_raw_ob = get_raw_fn(obs)

        for rule in specs['transitions']:
            act, *rest = rule
            if len(rest) == 3:
                state, next_state, prob = rest
                val = prob
            elif len(rest) == 2:
                state, matrix = rest
                next_state = slice(None)
                val = process_matrix(matrix, (num_states,))
            else:
                matrix, *_ = rest
                state = slice(None)
                next_state = slice(None)
                val = process_matrix(matrix, (num_states, num_states))
            state = get_raw_state(state)
            act = get_raw_act(act)
            next_state = get_raw_state(next_state)
            T[state, act, next_state] = val


        for rule in specs['sensors']:
            act, *rest = rule
            if len(rest) == 3:
                next_state, ob, prob = rest
                ob = get_raw_state(ob)
            elif len(rest) == 2:
                next_state, matrix = rest
                ob = slice(None)
                val = process_matrix(matrix, (num_obs,))
            else:
                matrix, *_ = rest
                ob = slice(None)
                next_state = slice(None)
                val = process_matrix(matrix, (num_states, num_obs))
            act = get_raw_act(act)
            next_state = get_raw_state(next_state)
            ob = get_raw_ob(ob)
            O[act, next_state, ob] = val

        # ignoring obs in reward
        for rule in specs['rewards']:
            act, state, *rest = rule
            if len(rest) == 3:
                next_state, ob, val = rest
            elif len(rest) == 2:
                next_state, matrix = rest
                val = process_matrix(matrix, (num_obs,)).mean()
            else:
                matrix, *_ = rest
                next_state = slice(None)
                val = process_matrix(matrix, (num_states, num_obs)).mean(axis=1)
            state = get_raw_state(state)
            act = get_raw_act(act)
            next_state = get_raw_state(next_state)
            R[state, act, next_state] = val


        # We assume observations depend only on state
        assert all((O[0] == O[i]).all() for i in range(num_actions)), "Observations must depend only on state"
        return T, R, O[0], discount, initial_state_distribution

    def preamble(self, items):
        headers = {}
        for item in items:
            headers.update(item)
        return headers

    def param_list(self, items):
        d = defaultdict(list)
        for item in items:
            for k, v in item.items():
                d[k].append(v)
        return d

    integer = lambda _, items : int(items[0])
    number = lambda _, items : float(items[0])
    discount = lambda _, items : {'discount' : float(items[0])}

    states = lambda _, items : {'states' : items[0]}
    values = lambda _, items : {'values' : items[0]}
    actions = lambda _, items : {'actions' : items[0]}
    observations = lambda _, items : {'observations' : items[0]}
    start = lambda _, items : {'start' : items[0]}

    transitions = lambda _, items : {'transitions' : items}
    rewards = lambda _, items : {'rewards' : items}
    sensors = lambda _, items : {'sensors' : items}

    ui_matrix = lambda _, items : items[0]
    u_matrix = lambda _, items : items[0]
    num_matrix = lambda _, items : items

    name = lambda _, items : items[0][:]
    value = lambda _, items : items[0]
    values_type = lambda _, items : items[0]

    uniform = lambda *_: 'uniform'
    identity = lambda *_: 'identity'

    ident_list = list


def read_pomdp(filename):
    with open(filename) as f:
        text = f.read()
    tree = pomdp_parser.parse(text)
    pomdp = TreeToPOMDP().transform(tree)
    return pomdp
