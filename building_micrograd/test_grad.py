import unittest
from naman_grad_engine import Value

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

    def test_complex_expression(self):
        a = Value(2)
        b = Value(3)
        c = a + b
        d = a * b + c
        d.backward()
        self.assertEqual(a.grad, 4, f"Expected gradient of a to be 4, got {a.grad}")
        self.assertEqual(b.grad, 3, f"Expected gradient of b to be 3, got {b.grad}")

if __name__ == "__main__":
    unittest.main()
