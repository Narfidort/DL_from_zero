"""
パーセプトロン OOP版
応用数学研究部 機械学習班 自主ゼミ 2部 対応

このファイルは「classを使う意義」を正直に示すように構成している。

  1. AND/NAND/OR は w,b が違うだけ（=データの違い）なので、
     Perceptron を継承せず、ただインスタンス化するだけにする。

  2. XOR/HalfAdder/FullAdder は複数の部品（ゲート）を組み立てる「合成」なので
     class で表現する。

(学習させる発展編は bonus_gakushu_perceptron.py に分けてあります。
 当日の説明はここまでで完結します。)

    $ python perceptron_oop.py
"""

import numpy as np


# =====================================================================
# 基底クラス: 重み w とバイアス b を持つ、素のパーセプトロン
# =====================================================================

class Perceptron:
    """w, b を属性として保持し、呼び出し可能(callable)にしたパーセプトロン。

    数学的定義: y = 1 if w・x + b > 0 else 0   (x, w は同じ次元のベクトル)
    """

    def __init__(self, w, b):
        self.w = np.array(w, dtype=float)
        self.b = float(b)

    def __call__(self, *xs):
        """p(x1, x2, ...) のように、インスタンスを関数として呼べる。"""
        x = np.array(xs, dtype=float)
        if x.shape != self.w.shape:
            raise ValueError(f"入力の次元 {x.shape} が重みの次元 {self.w.shape} と一致しません")
        return 1 if np.dot(self.w, x) + self.b > 0 else 0

    def __repr__(self):
        w_str = tuple(float(v) for v in self.w)
        return f"{self.__class__.__name__}(w={w_str}, b={self.b})"


# =====================================================================
# AND/NAND/OR は w,b が違うだけ -> サブクラス化せず、ただインスタンス化する。
#   「データの違いにサブクラスを使わない」というのも、設計判断として明示しておく。
# =====================================================================

AND = Perceptron(w=(0.5, 0.5), b=-0.7)
NAND = Perceptron(w=(-0.5, -0.5), b=0.7)
OR = Perceptron(w=(0.5, 0.5), b=-0.2)


# =====================================================================
# 多層パーセプトロン: 複数のパーセプトロン(オブジェクト)を「合成」する
#   __init__ で部品(ゲート)を受け取って組み立て、__call__ で配線どおりに信号を流す
#   デフォルト引数で「差し替え可能」にしておくと、別実装のANDを注入することもできる
# =====================================================================

class XOR:
    """s1 = NAND(x1,x2), s2 = OR(x1,x2), y = AND(s1,s2) を1つのオブジェクトにまとめる。"""

    def __init__(self, nand=NAND, or_gate=OR, and_gate=AND):
        self.nand = nand
        self.or_ = or_gate
        self.and_ = and_gate

    def __call__(self, x1, x2):
        s1 = self.nand(x1, x2)
        s2 = self.or_(x1, x2)
        return self.and_(s1, s2)

    def __repr__(self):
        return "XOR(NAND -> AND <- OR)"


XOR_GATE = XOR()  # 既定の組み立て済みXORを1つ用意しておく


# =====================================================================
# 2.6: NAND/AND/OR/XOR だけを部品にして加算器を組む
# =====================================================================

class HalfAdder:
    """1ビット加算。carry = AND(a,b), sum = XOR(a,b)。"""

    def __init__(self, xor_gate=XOR_GATE, and_gate=AND):
        self.xor = xor_gate
        self.and_ = and_gate

    def __call__(self, a, b):
        return self.and_(a, b), self.xor(a, b)  # (carry, sum)


class FullAdder:
    """繰り上がり入力つき1ビット加算。半加算器2個 + OR。"""

    def __init__(self, or_gate=OR):
        self.ha1 = HalfAdder()
        self.ha2 = HalfAdder()
        self.or_ = or_gate

    def __call__(self, a, b, carry_in):
        c1, s1 = self.ha1(a, b)
        c2, s2 = self.ha2(s1, carry_in)
        carry_out = self.or_(c1, c2)
        return carry_out, s2


class TwoBitAdder:
    """2ビット整数(0..3)同士を全加算器2段で足す。"""

    def __init__(self):
        self.fa0 = FullAdder()
        self.fa1 = FullAdder()

    def __call__(self, a, b):
        a0, a1 = a & 1, (a >> 1) & 1
        b0, b1 = b & 1, (b >> 1) & 1
        c0, s0 = self.fa0(a0, b0, 0)
        c1, s1 = self.fa1(a1, b1, c0)
        return (c1 << 2) | (s1 << 1) | s0


# =====================================================================
# 動作確認
# =====================================================================

def main():
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print("=== ゲートの repr (オブジェクトが w, b を抱えていることが分かる) ===")
    for g in (AND, NAND, OR):
        print(" ", g)
    print(" ", XOR_GATE)

    print("\n=== 真理値表 ===")
    print(" x1 x2 | AND NAND  OR | XOR")
    print("-------+--------------+-----")
    for x1, x2 in inputs:
        print(f"  {x1}  {x2} |  {AND(x1,x2)}   {NAND(x1,x2)}    {OR(x1,x2)} |  {XOR_GATE(x1,x2)}")

    print("\n=== 半加算器 ===")
    ha = HalfAdder()
    for a, b in inputs:
        c, s = ha(a, b)
        print(f"  {a} + {b} = carry {c}, sum {s}")

    print("\n=== 2ビット加算器の検算 (全16通り) ===")
    adder = TwoBitAdder()
    ok = all(adder(a, b) == a + b for a in range(4) for b in range(4))
    print("  全て一致:", "✓" if ok else "✗ 不一致あり")

    print("\n=== 不正な入力次元は ValueError になることの確認 ===")
    try:
        AND(1, 2, 3)  # 3次元を渡してしまう例
    except ValueError as e:
        print("  期待どおり ValueError:", e)


if __name__ == "__main__":
    main()
