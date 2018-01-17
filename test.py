from AutoDiff.scalarCalculation.visualize import visualize

from AutoDiff.scalarCalculation.values import variable, constant
from AutoDiff.scalarCalculation.operators import Arithmetic, sigmoid, square

a = variable.Variables(1.0, name="Variable1")
b = variable.Variables(2.0, name="Variable2")
c = variable.Variables(-3.0, name="Variable3")
x = variable.Variables(-1.0, name="Variable4")
y = variable.Variables(3.0, name="Variable5")

ax = Arithmetic.Multiply(a, x, name="Multiply1")
by = Arithmetic.Multiply(b, y, name="Multiply2")
axpby = Arithmetic.Add(ax, by, name="add1")
axpbypc = Arithmetic.Add(axpby, c, name="add2")
s = sigmoid.Sigmoid(axpbypc)

# test_var_1 = variable.Variables(100, name="test_var_1")
# test_var_2 = variable.Variables(10, name="test_var_2")
#
# add_both_var = Arithmetic.Add(test_var_1, test_var_2)

visualize.plot_graph(s)
