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

def test_multi():
    test_var = variable.Variables(100)
    test_constant = constant.Constant(35)

    test_add = Arithmetic.Multiply(test_var, test_constant)
    assert test_add.forward() == 100 * 35

def test_error_wrong_inp():
    # Must raise typesError
    with pytest.raises(Exception) as excinfo:
        test_var = variable.Variables("One")
        test_var2 = variable.Variables(False)
    assert str(excinfo.value) == "Variable has to be int or float!!"

def test_error_wrong_inp2():
    # Must raise typesError
    with pytest.raises(Exception) as excinfo:
        test_var = constant.Constant("One")
        test_var2 = constant.Constant(False)
    assert str(excinfo.value) == "Constant has to be int or float!!"

def test_accessing_grad_constant():
    test_var = constant.Constant(100)
    constant_grad = test_var.get_grad()
    assert constant_grad == 0

def test_accessing_grad_simple_backward():
    test_var_1 = variable.Variables(100)
    test_var_2 = variable.Variables(10)

    add_both_var = Arithmetic.Add(test_var_1, test_var_2)

    # Initalize with grad = 1 to start the backpropagation....
    add_both_var.backward(1)

    var_grad_1 = test_var_1.get_grad()
    var_grad_2 = test_var_2.get_grad()
    assert var_grad_1 == 1 and var_grad_2 == 1


def test_accessing_grad_simple_backward2():
    test_var_1 = variable.Variables(100)
    test_var_2 = constant.Constant(10)

    add_both_var = Arithmetic.Multiply(test_var_1, test_var_2)

    # Initalize with grad = 1 to start the backpropagation....
    add_both_var.backward(1)

    var_grad_1 = test_var_1.get_grad()
    assert var_grad_1 == 10
