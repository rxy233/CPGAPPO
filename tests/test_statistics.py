import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze import paired_statistics,holm


class StatisticsTests(unittest.TestCase):
    def test_exact_five_pair_resolution(self):
        self.assertEqual(paired_statistics([1,2,3,4,5])['exact_signflip_p'],.0625)

    def test_zero_difference(self):
        result=paired_statistics([0]*5)
        self.assertEqual(result['paired_t_p'],1)
        self.assertEqual(result['exact_signflip_p'],1)
        self.assertEqual(result['ci_low'],0)

    def test_paired_interval(self):
        result=paired_statistics([1,2,3,4,5])
        self.assertAlmostEqual(result['difference_pp'],3)
        self.assertAlmostEqual(result['ci_low'],1.0367568385)
        self.assertAlmostEqual(result['ci_high'],4.9632431615)

    def test_holm_order_and_monotonicity(self):
        np.testing.assert_allclose(holm([.04,.001,.01]),[.04,.003,.02])


if __name__=='__main__':unittest.main()
