import sys

# For higher order import
sys.path.append("..")

from src.scalarCalculation.values import variable, constant
from src.scalarCalculation.operators import add

test_var = variable.Variables(100)
test_constant = constant.Constant(35)

test_add = add.Add(test_var, test_constant)

print(test_var.forward())
