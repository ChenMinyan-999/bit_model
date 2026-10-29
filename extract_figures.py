"""按版面裁切论文插图 -> PNG（像素自动裁边法）。

get_drawings() 的 bbox 在 form-XObject 中被变换过，不可用；
因此改为：
  1) 用题注 "Figure N:" 定下边界；
  2) 上边界 = 上方最后一个"正文字号(>=8pt)"文字块的底部（正文与图内标注靠字号区分：
     正文 9-10pt，图内刻度/图例 4-6pt）；
  3) 渲染该条带，用像素非白行/列自动收紧到墨迹外接框；
  4) 以 288dpi 重新渲染。
"""
import re, os, json
import numpy as np
import pymupdf

PDF = r"D:\北理课程\建模仿真与导论\2505.10861v1.pdf"
OUT = r"D:\北理课程\建模仿真与导论\images"
os.makedirs(OUT, exist_ok=True)
doc = pymupdf.open(PDF)

CAP = re.compile(r"^Figure\s+(\d+)\s*[:.]")
SPECIAL = {1: pymupdf.Rect(100, 512, 512, 682)}  # 图 1(TikZ) 图内文字为 9pt，会干扰字号判定
SCAN_ZOOM = 2.0      # 扫描用
FINAL_ZOOM = 4.0     # 288 dpi 输出
WHITE = 246


def text_blocks(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        txt = "".join(s["text"] for l in b["lines"] for s in l["spans"]).strip()
        if not txt:
            continue
        sizes = [s["size"] for l in b["lines"] for s in l["spans"]]
        out.append((pymupdf.Rect(b["bbox"]), txt, max(sizes)))
    return out


def ink_box(page, band):
    """渲染条带，返回非白像素的外接框（PDF 坐标）。"""
    pix = page.get_pixmap(clip=band, matrix=pymupdf.Matrix(SCAN_ZOOM, SCAN_ZOOM), alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = a[:, :, :3].min(axis=2)
    mask = gray < WHITE
    rows = mask.sum(axis=1) >= 2
    cols = mask.sum(axis=0) >= 2
    if not rows.any() or not cols.any():
        return None
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    k = SCAN_ZOOM
    return pymupdf.Rect(band.x0 + c0 / k, band.y0 + r0 / k,
                        band.x0 + (c1 + 1) / k, band.y0 + (r1 + 1) / k)


report = []
for pno in range(doc.page_count):
    page = doc[pno]
    tbs = text_blocks(page)
    caps = [(int(CAP.match(t).group(1)), r) for r, t, sz in tbs if CAP.match(t)]
    if not caps:
        continue
    caps.sort(key=lambda x: x[1].y0)
    prose = [(r, sz) for r, t, sz in tbs if sz >= 8.0 and not CAP.match(t)]

    for idx, (num, cap) in enumerate(caps):
        upper = 34.0
        for r, sz in prose:
            if r.y1 < cap.y0 - 1 and r.y1 > upper:
                upper = r.y1
        upper += 3.0
        if idx > 0:
            upper = max(upper, caps[idx - 1][1].y1 + 2.0)

        band = SPECIAL.get(num) or pymupdf.Rect(100, upper, 512, cap.y0 - 2)
        if band.height < 30:
            print(f"!! fig{num}: band too thin ({band})")
            continue
        bb = ink_box(page, band)
        if bb is None or bb.width < 40 or bb.height < 20:
            print(f"!! fig{num} p{pno+1}: bad ink box {bb}")
            continue

        bb = pymupdf.Rect(bb.x0 - 3, bb.y0 - 3, bb.x1 + 3, bb.y1 + 4) & pymupdf.Rect(56, 30, 556, 764)
        pix = page.get_pixmap(clip=bb, matrix=pymupdf.Matrix(FINAL_ZOOM, FINAL_ZOOM), alpha=False)
        name = f"fig{num:02d}.png"
        pix.save(os.path.join(OUT, name))
        kb = round(os.path.getsize(os.path.join(OUT, name)) / 1024, 1)
        report.append(dict(fig=num, page=pno + 1, file=f"images/{name}",
                           bbox=[round(v, 1) for v in bb], px=[pix.width, pix.height],
                           dpi=round(72 * FINAL_ZOOM), kb=kb))
        print(f"fig{num:>2} p{pno+1:>2}  {pix.width}x{pix.height}  {kb:>7}KB  bbox={[round(v,1) for v in bb]}")

report.sort(key=lambda r: r["fig"])
with open(os.path.join(OUT, "_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
missing = sorted(set(range(1, 28)) - {r["fig"] for r in report})
print(f"\nfigures={len(report)} missing={missing}")
