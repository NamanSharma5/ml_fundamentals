import unittest
from naman_grad_engine import Value
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
        c = a + b # 5
        d = a * b # 6
        e = c + d # 11
        f = e * e # 121 # (c + d)^2 = c^2 + 2cd + d^2 = 
        self.assertEqual(f.data, 121, f"Expected f to be 121, got {f.data}")
        # c.backward(with_respect_to=a)
        c.backward()
        self.assertEqual(a.grad, 1, f"Expected gradient of a to be 1, got {a.grad}")
        # e.backward(with_respect_to=a)
        e.backward()
        self.assertEqual(a.grad, b.data + 1, f"Expected gradient of a to be {b.data + 1}, got {a.grad}")
        # f.backward(with_respect_to=e)
        # self.assertEqual(e.grad, 2 * e.data, f"Expected gradient of e to be {2 * e.data}, got {e.grad}")
        # f.backward(with_respect_to=a)
        # self.assertEqual(a.grad, 2 * a.data + 2 * b.data, f"Expected gradient of a to be {2 * a.data + 2 * b.data}, got {a.grad}")

    def test_chatgpt_suggested(self):
        a = Value(2)
        b = Value(3)
        c = a + b    # 5
        d = a * b    # 6
        e = c + d    # 11
        f = e * e    # 121

        # Forward pass check
        self.assertEqual(f.data, 121, f"Expected f to be 121, got {f.data}")

        # Backward pass for f
        f.backward()

        # Correct gradient for a
        self.assertEqual(a.grad, 88, f"Expected gradient of a to be 88, got {a.grad}")

if __name__ == "__main__":
    unittest.main()
