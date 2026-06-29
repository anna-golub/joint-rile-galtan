import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import RILE, GALTAN, JOINT

prompt_lookup = {
    RILE: """Question: What political position is expressed in this statement?
Statement: {sent}
Option A: Right-wing
Option B: Left-wing
Option C: Neutral
Keep your response short (up to 10 words) by choosing exactly one option!
Correct option:""",

    GALTAN: """Question: What political position is expressed in this statement?
Statement: {sent}
Option A: Green-Alternative-Liberal
Option B: Traditional-Authoritarian-Nationalist
Option C: Neutral
Keep your response short (up to 10 words) by choosing exactly one option!
Correct option:""",

    JOINT: """Question: What political position is expressed in this statement?
Statement: {sent}
Choose exactly one option from each of the two lists below.
List 1 (economic policy): 
Option A: Right-wing
Option B: Left-wing
Option C: Neutral
List 2 (socio-cultural policy): 
Option D: Green-Alternative-Liberal
Option E: Traditional-Authoritarian-Nationalist
Option F: Neutral
Keep your response short (up to 10 words)!
Correct options:"""
}

rile_labels_order_prompt = ['right', 'left', 'neutral']
galtan_labels_order_prompt = ['libertarian', 'authoritarian', 'neutral']

str2label = {RILE: {
    'Option A': rile_labels_order_prompt[0],
    'Option B': rile_labels_order_prompt[1],
    'Option C': rile_labels_order_prompt[2],

    'Option A: Right-wing': 'right',
    'Option B: Left-wing': 'left',
    'Option C: Neutral': 'neutral',

    'A: Right-wing': 'right',
    'B: Left-wing': 'left',
    'C: Neutral': 'neutral',

    'A': rile_labels_order_prompt[0],
    'B': rile_labels_order_prompt[1],
    'C': rile_labels_order_prompt[2],

    'Right-wing': 'right',
    'Left-wing': 'left',
    'Neutral': 'neutral',
},
    GALTAN: {
        'Option A': galtan_labels_order_prompt[0],
        'Option B': galtan_labels_order_prompt[1],
        'Option C': galtan_labels_order_prompt[2],

        'Option A: Green-Alternative-Liberal': 'libertarian',
        'Option B: Traditional-Authoritarian-Nationalist': 'authoritarian',
        'Option C: Neutral': 'neutral',

        'A: Green-Alternative-Liberal': 'libertarian',
        'B: Traditional-Authoritarian-Nationalist': 'authoritarian',
        'C: Neutral': 'neutral',
        'C Neutral': 'neutral',

        'Green-Alternative-Liberal': 'libertarian',
        'Traditional-Authoritarian-Nationalist': 'authoritarian',
        'Neutral': 'neutral',

        'A': galtan_labels_order_prompt[0],
        'B': galtan_labels_order_prompt[1],
        'C': galtan_labels_order_prompt[2],

        'Green': 'libertarian',
        'Alternative': 'libertarian',
        'Liberal': 'libertarian',
        'Traditional': 'authoritarian',
        'Authoritarian': 'authoritarian',
        'Nationalist': 'authoritarian',

        'Green-Liberal': 'libertarian',
        'Authoritarian-Nationalist': 'authoritarian',
        'National-Authoritarian-Nationalist': 'authoritarian',
        'Green-Alternative-Democratic': 'libertarian'
    },
    JOINT: {
        'A': 'right',
        'B': 'left',
        'C': 'neutral',
        'D': 'libertarian',
        'E': 'authoritarian',
        'F': 'neutral',

        'right-wing': 'right',
        'right': 'right',
        'left-wing': 'left',
        'left': 'left',
        'green-alternative-liberal': 'libertarian',
        'green': 'libertarian',
        'alternative': 'libertarian',
        'liberal': 'libertarian',
        'traditional-authoritarian-nationalist': 'authoritarian',
        'traditional': 'authoritarian',
        'authoritarian': 'authoritarian',
        'nationalist': 'authoritarian',
        'neutral': 'neutral',
    }
}
