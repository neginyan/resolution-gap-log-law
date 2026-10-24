"""Stage AA: the 377 states of stage Z (361 stored + 16 adversarial) as Gaussian terms in (v, w).  Python 3.11."""
import json
import numpy as np


def stored_terms():
    from z_chain import jobs_stored
    return jobs_stored()
