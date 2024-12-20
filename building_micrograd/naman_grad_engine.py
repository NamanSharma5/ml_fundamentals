from enum import Enum
from typing import Optional
from functools import reduce
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

ML Learnings:
 - its a graph, not a tree!! (can have multiple parents)
 i.e. V(3) = V(2) + V(1)  ; V(4) = V(3) + V(1) ; V(4).backward()

1) Leaf Value nodes can feed into the computation graph in multiple operations, so need to be able to accumulate gradients from multiple sources
- otherwise need to change when graph is being constructed with operations to create a new value object if we are likely to reuse a node

SWE Learnings:
1) Keep backward on operation on operation method rather than backward function

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
        # this gradient is the contribution of this self node to the gradient of the node where we called backward
        # to implement w.r.p simplest thing is just to search for the w.r.p node in graph
        self.grad = 0
        self.operation:Optional[Enum] = None
        self._children:set[Value] = set()
        self._backward_fn = lambda: None # base case for leaf nodes in no graph

    def __repr__(self):
        return f"Value({self.data}) with grad {self.grad}"

    # NOTE: equality on data attribute does not hold since lineage is not considered
    def __equal__(self, other):
        return self.data == other.data and self.operation == other.operation and self._children == other.children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        new_value = Value(self.data + other.data)
        new_value.operation = Operation.ADD
        new_value._children = {self, other}

        def _backward():
            assert new_value.grad is not None, "Gradient is None"
            for child in new_value._children:
                # in addition we just pass through the gradient to the children,
                #   irrespective of the other values in the operation
                child.grad += new_value.grad

        new_value._backward_fn = _backward
        return new_value

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        new_value = Value(self.data * other.data)
        new_value.operation = Operation.MUL
        new_value._children = {self, other}

        def _backward():
            # backward is harder to reason about when you have multiple children
            total_product = reduce(lambda x, y: x * y, [child.data for child in new_value._children])
            for child in new_value._children:
                child: Value
                remaining_product = total_product / child.data
                child.grad += new_value.grad * remaining_product

        new_value._backward_fn = _backward
        return new_value

    def backward(self, with_respect_to:Optional['Value'] = None, parent_grad:Optional[float] = 1):
        # compute gradient from node w.r.t. value
        # NOTE: if backward called a value object, with the with_respect_to_argument as None, then for each child node we consider gradient w.r.t. to that child node
        # search for the node w.r.t. which we are computing the gradient
        # whilst looking for the node take a running product of gradients encountered
        if isinstance(parent_grad, Value):
            print(f"Assuming you made a typo and w.r.p to be {parent_grad}")
            #TODO

        visited = set()
        differentiation_order = []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                differentiation_order.append(v) # if here, its preorder
                for child in v._children:
                    build_topo(child)
                # differentiation_order.append(v) if here, its postorder, hence need to reverse (i.e. differentiate parent before children)
            return differentiation_order

        differentiation_order = build_topo(self)
        self.grad = parent_grad
        # print(f"Differentiation order: {differentiation_order}")
        # print(f"Parent grad: {parent_grad}")
        for node in differentiation_order:
            node: Value
            node._backward_fn()
            if with_respect_to is not None:
                if node == with_respect_to:
                    break
                else:
                    node.grad = 0
