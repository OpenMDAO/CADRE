"""
Run the CADRE model and make sure the value compare to the saved pickle.
"""
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem

from CADRE.CADRE_group import CADRE
from CADRE.test.util import load_validation_data


# set verbose to True for debugging
verbose = False


class TestCADRE(unittest.TestCase):

    def test_CADRE(self):
        n, m, h, setd = load_validation_data(idx='0')

        prob = Problem(CADRE(n=n, m=m))
        prob.setup(check=verbose)

        input_data = [
            'CP_P_comm',
            'LD',
            'cellInstd',
            'CP_gamma',
            'finAngle',
            'lon',
            'CP_Isetpt',
            'antAngle',
            't',
            'r_e2b_I0',
            'lat',
            'alt',
            'iSOC'
        ]
        for inp in input_data:
            prob[inp] = setd[inp]

        prob.run_driver()

        if verbose:
            inputs = prob.model.list_inputs()
            outputs = prob.model.list_outputs()
        else:
            inputs = prob.model.list_inputs(out_stream=None)
            outputs = prob.model.list_outputs(out_stream=None)

        good = []
        fail = []

        for varpath, meta in inputs + outputs:
            var = varpath.split('.')[-1]
            if var in setd:
                actual = setd[var]
                computed = prob[var]

                if isinstance(computed, np.ndarray):
                    rel = np.linalg.norm(actual - computed) / np.linalg.norm(actual)
                else:
                    rel = np.abs(actual - computed) / np.abs(actual)

                if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                    if rel > 1e-3:
                        fail.append(varpath)
                        if verbose:
                            print(varpath, 'failed:\n', computed, '\n---\n', actual, '\n')
                else:
                    good.append(varpath)
                    if verbose:
                        print(varpath, 'is good.')
            elif verbose:
                print(var, 'not in data.')

        fails = len(fail)
        total = fails + len(good)

        self.assertEqual(fails, 0,
                         '%d (of %d) mismatches with validation data:\n%s'
                         % (fails, total, '\n'.join(sorted(fail))))


if __name__ == "__main__":
    unittest.main()
