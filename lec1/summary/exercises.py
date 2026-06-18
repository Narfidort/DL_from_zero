"""
実装の演習問題

TODO を実装して、最後の self_check() の assert が全て通ることを確認する。
答え合わせは perceptron_implementations.py を見ればよい。

  $ python exercises.py
"""

import numpy as np


# =====================================================================
# 演習1: n次元パーセプトロン(汎用)を実装する
#   y = 1 if w・x + b > 0 else 0
# =====================================================================

def perceptron(x, w, b):
    # TODO: np.dot(w, x) + b > 0 なら 1, そうでなければ 0 を返す
    raise NotImplementedError


# =====================================================================
# 演習2: 2入力ゲート AND/NAND/OR を実装する
#   演習1の perceptron を、決まった w, b で呼ぶだけ
# =====================================================================

def AND(x1, x2):
    # TODO: perceptron((x1,x2), w=(0.5,0.5), b=-0.7) を返す
    raise NotImplementedError

def NAND(x1, x2):
    # TODO
    raise NotImplementedError

def OR(x1, x2):
    # TODO
    raise NotImplementedError


# =====================================================================
# 演習3（本セッションのメイン）: XOR を既存ゲートの合成で実装する
#   s1 = NAND(x1,x2), s2 = OR(x1,x2), y = AND(s1,s2)
# =====================================================================

def XOR(x1, x2):
    # TODO
    raise NotImplementedError


# =====================================================================
# 演習4: 半加算器・全加算器を実装する
# =====================================================================

def half_adder(a, b):
    """戻り値 (carry, sum)"""
    # TODO: sum = XOR(a,b), carry = AND(a,b)
    raise NotImplementedError

def full_adder(a, b, carry_in):
    """戻り値 (carry_out, sum)。半加算器を2回使う。"""
    # TODO
    raise NotImplementedError


# =====================================================================
# 演習5（発展）: nビット加算器を実装する
#   全加算器を下位ビットから繰り返し適用し、桁上げを次のビットへ伝える
# =====================================================================

def n_bit_adder(a, b, n):
    """a, b: 0以上 2^n 未満の整数。a+b を返す（最後の桁上げも含めてよい）。"""
    # TODO
    raise NotImplementedError


# =====================================================================
# 自己チェック
# =====================================================================

def self_check():
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    assert [AND(x1, x2) for x1, x2 in inputs] == [0, 0, 0, 1], "AND が正しくない"
    assert [NAND(x1, x2) for x1, x2 in inputs] == [1, 1, 1, 0], "NAND が正しくない"
    assert [OR(x1, x2) for x1, x2 in inputs] == [0, 1, 1, 1], "OR が正しくない"
    assert [XOR(x1, x2) for x1, x2 in inputs] == [0, 1, 1, 0], "XOR が正しくない"
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
        assert ok, f"{n}ビット加算器が正しくない"
    print("演習5(発展) OK ✓")

    print("\n全ての assert が通りました ✓")


if __name__ == "__main__":
    self_check()
