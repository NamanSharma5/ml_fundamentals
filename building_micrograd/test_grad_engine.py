import unittest
from grad_engine import Value
# from reference_engine import Value


class TestGradients(unittest.TestCase):

    def test_addition(self):
        a = Value(2)
        b = Value(3)
        c = a + b
        c.backward()
        self.assertEqual(a.grad, 1, f"Expected gradient of a to be 1, got {a.grad}")
        self.assertEqual(b.grad, 1, f"Expected gradient of b to be 1, got {b.grad}")

    def test_addition_with_respect_to_one_variable(self):
        a = Value(2)
        b = Value(3)
        c = a + b
        c.backward(with_respect_to=a)
        self.assertEqual(a.grad, 1, f"Expected gradient of a to be 1, got {a.grad}")
        self.assertEqual(b.grad, 0, f"Expected gradient of b to be 1, got {b.grad}")

    def test_multiplication(self):
        a = Value(2)
        b = Value(3)
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 3, f"Expected gradient of a to be 3, got {a.grad}")
        self.assertEqual(b.grad, 2, f"Expected gradient of b to be 2, got {b.grad}")

    def test_negation(self):
        a = Value(2)
        b = -a
        assert (b.data == -2)
        b.backward()
        # NOTE: why is the gradient of b not -1?
        print(b.grad)

    def test_subtraction(self):
        a = Value(2)
        b = Value(3)
        # NOTE: b memory address is different object in c computation graph
        c: Value = a - b
        assert (c.data == -1)
        c.backward()
        """
        reference:
        [Value(data=-1, grad=1), Value(data=2, grad=0), Value(data=-3, grad=0), Value(data=-1, grad=0), Value(data=3, grad=0)]

        my solution:
        [Value(-1) with grad 0, Value(-3) with grad 0, Value(-1) with grad 0, Value(3) with grad 0, Value(2) with grad 0]

        """
        self.assertEqual(a.grad, 1, f"Expected gradient of a to be 1, got {a.grad}")
        # NOTE: for some reason b here is different to b that is updated when c.backward() is called
        self.assertEqual(b.grad, -1, f"Expected gradient of b to be -1, got {b.grad}")

    def test_multiplication_and_addition(self):
        a = Value(2)
        b = Value(3)
        c = a + b
        d = a * c

        d.backward()
        self.assertEqual(b.grad, 2, f"Expected gradient of b to be 2, got {b.grad}")
        self.assertEqual(a.grad, 7, f"Expected gradient of a to be 5, got {a.grad}")

    def test_complex_expression(self):
        # So this is a bad example because the nodes in the tree are not unique
        # meaning that they will have different gradients but the same data
        # my solution does not accumulate the gradients in the same way as the reference solution
        a = Value(2)
        b = Value(3)
        c = a + b  # 5
        self.assertEqual(c.data, 5, f"Expected d to be 6, got {c.data}")
        d = a * b  # 6
        self.assertEqual(d.data, 6, f"Expected d to be 6, got {d.data}")
        e = c + d  # 11
        self.assertEqual(e.data, 11, f"Expected e to be 121, got {e.data}")
        f: Value = e ** 2  # 121 # (c + d)^2 = c^2 + 2cd + d^2 =
        self.assertEqual(f.data, 121, f"Expected f to be 121, got {f.data}")

        # test gradients starting lower down in the computation graph
        c.backward()
        self.assertEqual(a.grad, 1, f"Expected gradient of a to be 1, got {a.grad}")
        self.assertEqual(b.grad, 1, f"Expected gradient of b to be 1, got {b.grad}")

        c.reset_grad()  # need to reset the gradients to 0 after each backward pass
        self.assertEqual(a.grad, 0, f"Expected gradient of a to be 0, got {a.grad}")
        self.assertEqual(b.grad, 0, f"Expected gradient of b to be 0, got {b.grad}")

        d.backward()
        self.assertEqual(a.grad, b.data, f"Expected gradient of a to be {b.data}, got {a.grad}")
        d.reset_grad()

        e.backward(with_respect_to=a)
        self.assertEqual(a.grad, 1 + b.data, f"Expected gradient of a to be {1 + b.data}, got {a.grad}")
        e.reset_grad()

        f.backward()
        self.assertEqual(e.grad, 2 * e.data, f"Expected gradient of e to be {2 * e.data}, got {e.grad}")
        f.reset_grad()


if __name__ == "__main__":
    unittest.main()
