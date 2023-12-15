import numpy as np

from wsd.data.tokens import TokenInput
from wsd.vectorise.bert import BertVectoriser

text = [
    "As he carefully peeked out of the trench , he saw three tanks approaching him .",
    "He had a keen interest in army stuff such as tanks"
    "Wow , that move really did tank our score !",
    "His excitement tanked when he saw who was going to perform tonight ."
    "They were happy that the tank out back was still filled to the brim .",
    "I think i have a tank of gasoline in the shed"
    "After the student 's motivation dwindled , their grades went into the tank ."
    "Her hitting sta",
]
