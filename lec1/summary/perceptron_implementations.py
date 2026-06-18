"""
パーセプトロン 要点まとめ - 実装コード
応用数学研究部 機械学習班 自主ゼミ

含むもの:
  1. n次元パーセプトロン(汎用)
  2. 2入力ゲート(AND/NAND/OR) -- 1.の特殊ケース
  3. XOR -- ゲートの合成
  4. 半加算器・全加算器
  5. nビット加算器(全加算器の繰り返し適用、桁上げ伝搬)

  $ python perceptron_implementations.py
"""

import numpy as np


# =====================================================================
# 1. n次元パーセプトロン(汎用) -- すべての土台
#    y = 1 if w・x + b > 0 else 0      (x, w は同じ長さのベクトル)
# =====================================================================

def perceptron(x, w, b):
    x = np.array(x, dtype=float)
    w = np.array(w, dtype=float)
    return 1 if np.dot(w, x) + b > 0 else 0


# =====================================================================
# 2. 2入力ゲート -- 1.の w, b を固定しただけの特殊ケース
# =====================================================================

def AND(x1, x2):
    return perceptron((x1, x2), w=(0.5, 0.5), b=-0.7)

def NAND(x1, x2):
    return perceptron((x1, x2), w=(-0.5, -0.5), b=0.7)

def OR(x1, x2):
    return perceptron((x1, x2), w=(0.5, 0.5), b=-0.2)


# =====================================================================
# 3. XOR -- 既存ゲートの合成(多層化)
#    s1 = NAND(x1,x2), s2 = OR(x1,x2), y = AND(s1,s2)
# =====================================================================

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


# =====================================================================
# 4. 半加算器・全加算器 -- XOR/ANDをさらに部品として使う
# =====================================================================

def half_adder(a, b):
    """1ビット同士の加算。戻り値 (carry, sum)。"""
    s = XOR(a, b)
    c = AND(a, b)
    return c, s

def full_adder(a, b, carry_in):
    """繰り上がり入力つき1ビット加算。半加算器2個 + OR。戻り値 (carry_out, sum)。"""
    c1, s1 = half_adder(a, b)
    c2, s2 = half_adder(s1, carry_in)
    carry_out = OR(c1, c2)
    return carry_out, s2


# =====================================================================
# 5. nビット加算器 -- 全加算器を繰り返し適用(リプルキャリー方式)
#    a, b: 0以上 2^n 未満の整数。戻り値は最大 2^(n+1)-2 (桁上げ含む)
# =====================================================================

def n_bit_adder(a, b, n):
    carry = 0
    result_bits = []
    for i in range(n):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        carry, s = full_adder(a_bit, b_bit, carry)
        result_bits.append(s)
    result_bits.append(carry)   # 最後の桁上げも結果に含める
    return sum(bit << i for i, bit in enumerate(result_bits))


# =====================================================================
# 動作確認
# =====================================================================

def main():
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print("=== 真理値表 ===")
    print(" x1 x2 | AND NAND  OR | XOR")
    print("-------+--------------+-----")
    for x1, x2 in inputs:
        print(f"  {x1}  {x2} |  {AND(x1,x2)}   {NAND(x1,x2)}    {OR(x1,x2)} |  {XOR(x1,x2)}")

    print("\n=== 半加算器 ===")
    for a, b in inputs:
        c, s = half_adder(a, b)
        print(f"  {a} + {b} = carry {c}, sum {s}")

    print("\n=== 全加算器(繰り上がり入力つき) ===")
    for a, b in inputs:
        for cin in (0, 1):
            c, s = full_adder(a, b, cin)
            print(f"  {a} + {b} + cin={cin} -> carry {c}, sum {s}")

    print("\n=== nビット加算器の検算 ===")
    for n in (2, 4, 8):
        ok = all(n_bit_adder(a, b, n) == a + b
                  for a in range(2**n) for b in range(2**n))
        print(f"  n={n}bit: 全{4**n}通り検算 -> {'✓ 全て一致' if ok else '✗ 不一致あり'}")


if __name__ == "__main__":
    main()
