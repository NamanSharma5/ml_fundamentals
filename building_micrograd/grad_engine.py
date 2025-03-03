from enum import Enum
from typing import Optional
from functools import reduce
from math import tanh, exp, log
"""
Purpose of this engine is to provide primitives which support automatic differentiation via backpropagation (chain rule)

Some example expressions:
a = Value(2)
b = Value(3)
c = a + b
d = a * b + c = a * b + a + b

What interface do we want to provide???
- well the value method should support the basic operations of addition, subtraction, multiplication, and division
    + the power operator(i.e. repeated multiplication)
- for any value object, we should be able to call the backward method with respect to a given value object
-- i.e. for c.backward(a), we should be able to compute the derivative of c with respect to a
"""

"""
Order of implmentation:
1) Value class
- attributes to support forward pass (data, operation, children)

2) figure out how to calculate forward pass FLOPs given a computation graph

3) Create a MLP

ML Learnings:
 - its a graph, not a tree!! (can have multiple parents)
 i.e. V(3) = V(2) + V(1)  ; V(4) = V(3) + V(1) ; V(4).backward()

1) Leaf Value nodes can feed into the computation graph in multiple operations,
    so need to be able to accumulate gradients from multiple sources (i.e. += new gradients)

    - otherwise need to change when graph is being constructed with operations to create a new value object
        if we are likely to reuse a node

SWE Learnings:
1) Keep backward operation on operation method rather than backward function

"""


class Operation(Enum):
    # ID?
    ADD = 1
    SUB = 2
    MUL = 3
    POW = 4
    TANH = 5
    EXP = 6


class Value:
    def __init__(self, data):
        self.data = data
        # this gradient is the contribution of this self node to the gradient of the node where we called backward
        # to implement w.r.p simplest thing is just to search for the w.r.p node in graph
        self.grad = 0
        self.operation: Optional[Enum] = None
        self._children: set[Value] = set()
        self._backward_fn = lambda: None  # base case for leaf nodes in no graph

    def __repr__(self):
        return f"Value({self.data}) with grad {self.grad}"

    # NOTE: equality on data attribute does not hold since lineage is not considered
    def __equal__(self, other):
        return self.data == other.data and self.operation == other.operation and self._children == other.children

    def get_children(self) -> set['Value']:
        return self._children

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

    def __radd__(self, other):
        return self.__add__(other)

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
                # NOTE: this does not support self.data * self.data
                remaining_product = total_product / child.data
                child.grad += new_value.grad * remaining_product

        new_value._backward_fn = _backward
        return new_value

    def __rmul__(self, other):
        # NOTE: other is the left operand
        return self.__mul__(other)

    def __neg__(self):
        """
        implement as negative multiplication
        """
        if isinstance(self, Value) and isinstance(self.data, (int, float)):
            return -1 * self
        elif isinstance(self, (int, float)):
            return -1 * Value(self)
        else:
            raise ValueError("Only supporting negation for int/float values")

    def __sub__(self, other):
        """
        implement as negative addition
        """
        # check if both self or other are Value objects
        # if not can we convert them to Value objects
        # if not raise an error
        # if not isinstance(other, Value):
        #     if isinstance(other, (int, float)):
        #         other = Value(other)
        #     else:
        #         raise ValueError("Only supporting subtraction of Value objects or int/float values")

        # if not isinstance(self, Value):
        #     if isinstance(self, (int, float)):
        #         self = Value(self)
        #     else:
        #         raise ValueError("Only supporting subtraction of Value objects or int/float values")

        return self + (-other)

    def tanh(self):
        new_data = tanh(self.data)
        new_value = Value(new_data)
        new_value.operation = Operation.TANH
        new_value._children = {self}

        def _backward():
            if self._children:
                child = self._children.pop()
                # do grad work -> update flowing (global) gradient with respect to the local gradient
                child.grad += new_value.grad * (1 - self.data ** 2)
                self._children.add(child)

        new_value._backward_fn = _backward

        return new_value

    def exp(self):
        """
        e ^ self.data
        """

        new_data = exp(self.data)
        new_value = Value(new_data)
        new_value.operation = Operation.EXP
        new_value._children = {self}

        def _backward():
            if self._children:
                child = self._children.pop()
                child.grad += new_value.grad * self.data
                self._children.add(child)

        new_value._backward_fn = _backward

        return new_value

    def __pow__(self, other: 'Value'):
        """
        self ^ other
        """
        if not isinstance(other, Value):
            if isinstance(other, (int, float)):
                other = Value(other)
            else:
                raise ValueError("Only supporting int/float powers for now")

        new_data = self.data ** other.data
        new_value = Value(new_data)
        new_value.operation = Operation.POW
        new_value._children = [self, other]  # NOTE: for powers, operand order matters so not a set

        def _backward(verbose=False):
            if new_value._children:
                # contribution of children:{self / other} to the gradient of the new POW value node
                # f(self, other) / dself = other * self ^ (other - 1)
                # f(self, other) / dother = self ^ other * log(self) , where log = natural log

                self_child, other_child = new_value._children[0], new_value._children[1]
                self_child.grad += new_value.grad * (other_child.data * self.data ** (other_child.data - 1))
                if self.data > 0:
                    other_child.grad += new_value.grad * ((self.data ** other_child.data) * log(self.data))
                elif verbose:
                    print(f"Warning: Logarithm of negative number {self.data} when calculating gradient of {new_value}")

        new_value._backward_fn = _backward

        return new_value

    def backward(self, with_respect_to: Optional['Value'] = None, parent_grad: Optional[float] = 1):
        # compute gradient from node w.r.t. value
        # NOTE: if backward called a value object, with the with_respect_to_argument as None, then for each child node
        #   we consider gradient w.r.t. to that child node
        # search for the node w.r.t. which we are computing the gradient
        # whilst looking for the node take a running product of gradients encountered
        if isinstance(parent_grad, Value):
            # TODO
            print(f"Assuming you made a typo and w.r.p to be {parent_grad}")

        visited = set()
        differentiation_order = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                differentiation_order.append(v)  # if here, its preorder
                for child in v._children:
                    build_topo(child)
                # differentiation_order.append(v) if here, its postorder, hence need to reverse
                # (i.e. differentiate parent before children)
            return differentiation_order

        differentiation_order = build_topo(self)
        self.grad = parent_grad
        # print(f"Differentiation order: {differentiation_order}")
        # print(f"Parent grad: {parent_grad}")
        for node in differentiation_order:
            node: Value
            node._backward_fn()

        # for with respect to, we need to set the gradient to 0 for all nodes that are not the node we are looking for
        if with_respect_to is not None:
            for node in differentiation_order:
                if node != with_respect_to:
                    node.grad = 0

    def reset_grad(self, all_children=True):
        self.grad = 0

        if all_children:
            for child in self._children:
                child.reset_grad()

    def calculate_inference_flops(self):
        """
        Calculate the number of FLOPs required to compute the forward pass, assuming:
            - all operations are of constant cost

        just do a graph traversal and call switch on operation type to get the cost
        """

        visited = set()  # to avoid duplicates

        def dfs(v_node):
            if v_node not in visited:
                v_node: Value
                visited.add(v_node)
                cost = 0
                match v_node.operation:
                    case Operation.ADD:
                        cost = 1
                    case Operation.MUL:
                        cost = 1
                    case _:
                        pass

                return cost + sum([dfs(child) for child in v_node.get_children()])

            return 0

        return dfs(self)


if __name__ == "__main__":
    # generate some code to test if my flops calculation is correct
    a = Value(2)
    b = Value(3)
    c = a + b
    d = a * b + c
    d.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)
    print("Done")
    print(d.calculate_inference_flops())
