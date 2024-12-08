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
        self.operation:Optional[Enum] = None
        self.children:list[Value] = []

    def __repr__(self):
        return f"Value({self.data})"
    
    # NOTE: equality on data attribute does not hold since lineage is not considered
    def __equal__(self, other):
        return self.data == other.data and self.operation == other.operation and self.children == other.children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        new_value = Value(self.data + other.data)
        new_value.operation = Operation.ADD
        new_value.children = [self, other]
        return new_value
    
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        new_value = Value(self.data * other.data)
        new_value.operation = Operation.MUL
        new_value.children = [self, other]
        return new_value

    def backward(self, parent_grad = 1, with_respect_to = None):
        # compute gradient from node w.r.t. value 
        # search for the node w.r.t. which we are computing the gradient
        # whilst looking for the node take a running product of gradients encountered
        if isinstance(parent_grad, Value):
            print(f"Assuming you made a typo and w.r.p to be {parent_grad}")
            with_respect_to = parent_grad
            parent_grad = 1

        if with_respect_to is None: #TODO
            #TODO: what behavior do we want to have here?
            pass 
        
        if self == with_respect_to:
            return parent_grad

        if self.operation is None:
            return 0
        
        ## this assumes that we have exactly 2 children
        elif self.operation == Operation.ADD:
            return self.children[0].backward(parent_grad, with_respect_to) + self.children[1].backward(parent_grad, with_respect_to)
        
        elif self.operation == Operation.MUL:
            # product rule f0 = f1 * f2 => f0' = f1' * f2 + f1 * f2' 
            return self.children[0].backward(parent_grad * self.children[1].data, with_respect_to) + self.children[1].backward(parent_grad * self.children[0].data, with_respect_to)


