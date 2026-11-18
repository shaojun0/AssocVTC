"""计参校准：448x448 / patch32 (N=196) 下 enc/dec 各 ~480M 的宽度选型。

在远程仓库根目录运行（PYTHONPATH 指向 AssocVTC）：
    /root/miniconda3/bin/python count_448_960m.py
输出 enc/dec/total 参数量，用于确定 enc_dim / dec_dim。
"""
import sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/root/AssocVTC")
from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn


def build(enc_dim, enc_heads, dec_dim, dec_heads, img=448, patch=32, K=32, T=16, ph=1024):
    steps = ",".join(str(i) for i in range(1, T + 1))
    cfg = LinAttnConfig(
        num_specials=K, enc_attn="std", enc_dim=enc_dim, enc_depth=13,
        enc_heads=enc_heads, dec_dim=dec_dim, dec_depth=7, dec_heads=dec_heads,
        dec_recurrent=True, dec_steps=steps, dec_step_plan="fixed", dec_block=2,
        img_size=img, patch=patch, pixel_hidden=ph)
    m = SRPhase1LinAttn(cfg)
    pc = m.count_params()
    del m
    return pc


def show(tag, pc):
    print("  %-38s enc %7.2fM  dec %7.2fM  total %7.2fM" % (
        tag, pc["encoder"] / 1e6, pc["decoder"] / 1e6, pc["total"] / 1e6))


print("== sanity: 240M @224/p16 (README 期望 enc 244.43 / dec 237.11) ==")
show("1248x13 + 1440x7 @224/p16", build(1248, 16, 1440, 16, img=224, patch=16))
print("== 同宽度换 448/p32（patch_embed / pixel_head 变大）==")
show("1248x13 + 1440x7 @448/p32", build(1248, 16, 1440, 16))

print("== enc 宽度扫描 (depth13 h16)，dec 固定 1440 ==")
for d in (1600, 1664, 1696, 1728, 1760, 1792, 1824, 1856):
    show("enc %dx13 h16" % d, build(d, 16, 1440, 16))

print("== dec 宽度扫描 (depth7 h16)，enc 固定 1248 ==")
for d in (1792, 1856, 1920, 1952, 1984, 2016, 2048, 2080):
    show("dec %dx7 h16" % d, build(1248, 16, d, 16))

print("== 2D 组合（目标各 ~480M）==")
for ed, dd in [(1728, 1984), (1728, 2016), (1760, 1984), (1760, 2016),
               (1760, 2048), (1792, 2016), (1792, 2048)]:
    show("enc %dx13 + dec %dx7" % (ed, dd), build(ed, 16, dd, 16))
