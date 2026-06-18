"""
発展（完全に任意・当日説明しません）: パーセプトロンに「学習」をさせてみる

このファイルは当日の演習の範囲外。気になる人だけ、家でコメントを読みながら
眺めてもらえれば、というだけの完成済みコード（TODOなし、すでに動く）。
説明できなくても問題ない——コードと出力を読めば流れが分かるように書いてある。

  $ python bonus_gakushu_perceptron.py

--------------------------------------------------------------------
背景（読みたい人だけ）:
これまでのAND/OR/NANDは、w(重み)とb(バイアス)を人間が「これなら真理値表に
合う」と考えて決めていた（2.2節参照）。では、人間が決めなくても、データを
見せるだけで w, b が自動で正しい値に近づいていく、ということはできるか？
できる。それが「パーセプトロン学習規則」(Rosenblatt, 1958)。

規則はこれだけ:
    w <- w + eta * (t - y) * x
    b <- b + eta * (t - y)

  t: 正解（教師データ。例えばANDなら (0,0)->0, (1,1)->1 など）
  y: 今の w, b で実際に出てくる出力
  eta: 学習率（1回の更新でどれだけ動くか。小さい正の数）

直感: 出力y が正解tと合っていれば (t-y=0) 何も更新しない。ずれていれば、
ずれの方向に w, b を少しだけ動かす。これをデータに対して何度も繰り返すと、
やがて全部正解するw, bに落ち着く（線形分離可能なデータなら、有限回の
更新で必ず収束することが理論的に保証されている: パーセプトロン収束定理）。
--------------------------------------------------------------------
"""

import numpy as np


class Perceptron:
    """これまでと同じ、評価だけのパーセプトロン。"""
    def __init__(self, w, b):
        self.w = np.array(w, dtype=float)
        self.b = float(b)

    def __call__(self, *xs):
        x = np.array(xs, dtype=float)
        return 1 if np.dot(self.w, x) + self.b > 0 else 0


class TrainablePerceptron(Perceptron):
    """Perceptron に「学習」という新しい振る舞いを追加しただけ。
    評価(__call__)は親クラスのものをそのまま使う。"""

    def __init__(self, n_inputs, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)               # 再現性のため乱数の種を固定
        w0 = rng.normal(scale=0.1, size=n_inputs)        # 最初はランダムな小さい値から始める
        super().__init__(w=w0, b=0.0)
        self.lr = lr

    def nudge(self, x, t):
        """1個のデータ (x, t) を見て、w, b を少しだけ更新する。"""
        x = np.array(x, dtype=float)
        y = self(*x)                  # 今の w,b での出力
        error = t - y                 # 合っていれば0、ずれていれば+1か-1
        self.w = self.w + self.lr * error * x
        self.b = self.b + self.lr * error
        return error

    def fit(self, X, T, epochs=20):
        """データ集合全体を epochs 回繰り返し学習する。"""
        for _ in range(epochs):
            for x, t in zip(X, T):
                self.nudge(x, t)
        return self


def demo(name, T, epochs=20):
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    p = TrainablePerceptron(n_inputs=2, lr=0.1, seed=0)
    before = [p(*x) for x in inputs]
    p.fit(inputs, T, epochs=epochs)
    after = [p(*x) for x in inputs]
    ok = after == T
    print(f"--- {name} を学習 ---")
    print(f"  学習前の出力: {before}")
    print(f"  学習後の出力: {after}   (目標 {name} = {T})")
    print(f"  学習後の w, b: {p.w}, {p.b}")
    print(f"  正解と一致: {'✓' if ok else '✗'}\n")


if __name__ == "__main__":
    demo("AND", T=[0, 0, 0, 1])
    demo("OR", T=[0, 1, 1, 1])
    demo("NAND", T=[1, 1, 1, 0])

    print("--- 参考: XORを学習させようとすると ---")
    p = TrainablePerceptron(n_inputs=2, lr=0.1, seed=0)
    p.fit([(0,0),(1,0),(0,1),(1,1)], T=[0,1,1,0], epochs=200)
    out = [p(x1,x2) for x1,x2 in [(0,0),(1,0),(0,1),(1,1)]]
    print(f"  200エポック学習しても出力: {out}  (目標 XOR = [0,1,1,0])")
    print("  -> 2.4節で証明した『線形分離不可能』はこういう形で現れる。")
    print("     学習回数を増やしても、単層パーセプトロンはXORには収束しない。")
