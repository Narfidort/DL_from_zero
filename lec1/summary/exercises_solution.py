"""
実装の演習問題 — 解答例
"""

import numpy as np


def perceptron(x, w, b):
    x = np.array(x, dtype=float)
    w = np.array(w, dtype=float)
    return 1 if np.dot(w, x) + b > 0 else 0


def AND(x1, x2):
    return perceptron((x1, x2), w=(0.5, 0.5), b=-0.7)

def NAND(x1, x2):
    return perceptron((x1, x2), w=(-0.5, -0.5), b=0.7)

def OR(x1, x2):
    return perceptron((x1, x2), w=(0.5, 0.5), b=-0.2)


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def half_adder(a, b):
    s = XOR(a, b)
    c = AND(a, b)
    return c, s

def full_adder(a, b, carry_in):
    c1, s1 = half_adder(a, b)
    c2, s2 = half_adder(s1, carry_in)
    carry_out = OR(c1, c2)
    return carry_out, s2


def n_bit_adder(a, b, n):
    carry = 0
    result_bits = []
    for i in range(n):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        carry, s = full_adder(a_bit, b_bit, carry)
        result_bits.append(s)
    result_bits.append(carry)
    return sum(bit << i for i, bit in enumerate(result_bits))


def self_check():
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    assert [AND(x1, x2) for x1, x2 in inputs] == [0, 0, 0, 1]
    assert [NAND(x1, x2) for x1, x2 in inputs] == [1, 1, 1, 0]
    assert [OR(x1, x2) for x1, x2 in inputs] == [0, 1, 1, 1]
    assert [XOR(x1, x2) for x1, x2 in inputs] == [0, 1, 1, 0]
    print("演習1〜3 OK ✓")

    assert half_adder(0, 0) == (0, 0)
    assert half_adder(1, 0) == (0, 1)
    assert half_adder(0, 1) == (0, 1)
    assert half_adder(1, 1) == (1, 0)
    assert full_adder(1, 1, 1) == (1, 1)
    print("演習4 OK ✓")

    for n in (2, 4):
        ok = all(n_bit_adder(a, b, n) == a + b
                  for a in range(2**n) for b in range(2**n))
        assert ok
    print("演習5(発展) OK ✓")

    print("\n全ての assert が通りました ✓")


if __name__ == "__main__":
    self_check()
