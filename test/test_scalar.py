import sys
import pytest

# For higher order import
sys.path.append("..")

from AutoDiff.scalarCalculation.values import variable, constant
from AutoDiff.scalarCalculation.operators import Arithmetic, sigmoid, square

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

def test_neuron_forward():
    a = variable.Variables(1.0)
    b = variable.Variables(2.0)
    c = variable.Variables(-3.0)
    x = variable.Variables(-1.0)
    y = variable.Variables(3.0)

    ax = Arithmetic.Multiply(a, x)
    by = Arithmetic.Multiply(b, y)
    axpby = Arithmetic.Add(ax, by)
    axpbypc = Arithmetic.Add(axpby, c)
    s = sigmoid.Sigmoid(axpbypc)

    assert round(s.forward(), 4) == 0.8808

def test_neuron_backward():
    a = variable.Variables(1.0)
    b = variable.Variables(2.0)
    c = variable.Variables(-3.0)
    x = variable.Variables(-1.0)
    y = variable.Variables(3.0)

    ax = Arithmetic.Multiply(a, x)
    by = Arithmetic.Multiply(b, y)
    axpby = Arithmetic.Add(ax, by)
    axpbypc = Arithmetic.Add(axpby, c)
    s = sigmoid.Sigmoid(axpbypc)

    s.backward(1)

    assert round(a.get_grad(), 3) == -0.105
    assert round(b.get_grad(), 3) == 0.315
    assert round(c.get_grad(), 3) == 0.105
    assert round(x.get_grad(), 3) == 0.105
    assert round(y.get_grad(), 3) == 0.210

def test_minus():
    a = variable.Variables(100)
    b = variable.Variables(10)

    c = Arithmetic.Minus(a, b)
    assert c.forward() == 100 - 10

def test_minus_grad():
    a = variable.Variables(100)
    b = variable.Variables(10)

    c = Arithmetic.Minus(a, b)
    c.backward(1)

    assert a.get_grad() == 1
    assert b.get_grad() == -1

def test_square():
    a = variable.Variables(100)
    b = square.Square(a)

    assert b.forward() == 100**2

def test_square_backward():
    a = variable.Variables(100)
    b = square.Square(a)

    b.backward(1)

    assert a.get_grad() == 2
