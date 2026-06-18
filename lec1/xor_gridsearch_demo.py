"""
2.4節「サンプルコード」で一緒に読む用: XORを満たす(w1,w2,b)を力づくで探してみる。
証明(背理法)を、実際に手で探索して裏付ける小さなデモ。
"""
import numpy as np

inputs = [(0,0),(1,0),(0,1),(1,1)]
target_xor = [0,1,1,0]

def satisfies(w1, w2, b, target):
    for (x1, x2), t in zip(inputs, target):
        y = 1 if w1*x1 + w2*x2 + b > 0 else 0
        if y != t:
            return False
    return True

# w1,w2,b を -2.0 から 2.0 まで 0.1刻みで全探索
grid = np.arange(-2.0, 2.01, 0.1)
found = None
count_checked = 0
for w1 in grid:
    for w2 in grid:
        for b in grid:
            count_checked += 1
            if satisfies(w1, w2, b, target_xor):
                found = (w1, w2, b)
                break
        if found:
            break
    if found:
        break

print(f"{count_checked}通りのパラメータを調べた結果:")
if found:
    print(f"  見つかった: {found}")
else:
    print("  XORを満たす(w1,w2,b)は見つからなかった。")
    print("  -> 証明(背理法)が言っていた『そもそも存在しない』を、力づくの探索でも確認できた。")

print("\n参考: ANDなら同じ探索ですぐ見つかる")
target_and = [0,0,0,1]
for w1 in grid:
    for w2 in grid:
        for b in grid:
            if satisfies(w1, w2, b, target_and):
                print(f"  見つかった: ({w1:.1f}, {w2:.1f}, {b:.1f})")
                raise SystemExit
