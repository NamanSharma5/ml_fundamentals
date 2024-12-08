from enum import Enum
from typing import Optional
"""
Purpose of this engine is to provide primitives which support automatic differentiation via backpropagation (chain rule).

Some example expressions:
a = Value(2)
b = Value(3)
c = a + b
d = a * b + c = a * b + a + b 

What interface do we want to provide???
- well the value method should support the basic operations of addition, subtraction, multiplication, and division + the power operator(i.e. repeated multiplication)
- for any value object, we should be able to call the backward method with respect to a given value object
-- i.e. for c.backward(a), we should be able to compute the derivative of c with respect to a
"""

"""
Order of implmentation:
1) Value class 
- attributes to support forward pass (data, operation, children)

"""

class Operation(Enum):
    # ID? 
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    POW = 5

class Value:
    def __init__(self, data):
        self.data = data
        # this gradient is the contribution of this self node to the gradient of the node where we called backward (note if this is not t)
        self.grad = 0 
        self.operation:Optional[Enum] = None
        self._children:list[Value] = []
        self._backward_fn = None

    def __repr__(self):
        return f"Value({self.data}) with grad {self.grad}"
    
    # NOTE: equality on data attribute does not hold since lineage is not considered
    def __equal__(self, other):
        return self.data == other.data and self.operation == other.operation and self._children == other.children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        new_value = Value(self.data + other.data)
        new_value.operation = Operation.ADD
        new_value._children = [self, other]
        return new_value
    
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        new_value = Value(self.data * other.data)
        new_value.operation = Operation.MUL
        new_value._children = [self, other]
        return new_value

    def backward(self, with_respect_to:Optional['Value'] = None, parent_grad:Optional[float] = 1):
        # compute gradient from node w.r.t. value 
        # NOTE: if backward called a value object, with the with_respect_to_argument as None, then for each child node we consider gradient w.r.t. to that child node
        # search for the node w.r.t. which we are computing the gradient
        # whilst looking for the node take a running product of gradients encountered
        if isinstance(parent_grad, Value):
            print(f"Assuming you made a typo and w.r.p to be {parent_grad}")
            #TODO
        
        # only nodes with no children contribute to the gradient
        if len(self._children) == 0:
            if with_respect_to is None or with_respect_to == self:
                self.grad = parent_grad
            else:
                self.grad = 0
            return self.grad

        match self.operation:
            case Operation.ADD:
                for child in self._children:
                    child.backward(with_respect_to=with_respect_to, parent_grad=parent_grad)


            case Operation.MUL:
                for index, child in enumerate(self._children):
                    # assumes two children
                    other_child = self._children[1 - index]
                    child_grad = parent_grad * other_child.data
                    child.backward(with_respect_to=with_respect_to, parent_grad=child_grad)        

            case _: 
                raise NotImplementedError(f"Operation {self.operation} not implemented")
