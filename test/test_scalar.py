import sys
import pytest

# For higher order import
sys.path.append("..")

from AutoDiff.scalarCalculation.values import variable, constant
from AutoDiff.scalarCalculation.operators import Arithmetic


def test_add():
    test_var = variable.Variables(100)
    test_constant = constant.Constant(35)

    test_add = Arithmetic.Add(test_var, test_constant)
    assert test_add.forward() == 100 + 35

def test_error_wrong_inp():
    # Must raise typesError
    with pytest.raises(Exception) as excinfo:
        test_var = variable.Variables("One")
        test_var2 = variable.Variables(False)
    assert str(excinfo.value) == "Variable has to be int or float!!"
