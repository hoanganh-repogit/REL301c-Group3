"""Build a research presentation using the user's original PowerPoint theme.

Run from the repository root: py latex/scripts/generate_presentation.py
Dependencies may be installed in .tools/presentation or the active environment.
"""
from __future__ import annotations

import csv
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.tools/presentation'))

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUT = ROOT / 'latex/slides'
TEMPLATE = ROOT / 'latex/slides/template.pptx'
GREEN = '788B73'
PALE = 'DDE5D8'
BLACK = '111511'
WHITE = 'F4F6F0'
prs = Presentation(TEMPLATE)
originals = list(prs.slides)
script = []


def box(slide, x, y, w, h, fill=PALE, line=BLACK):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.color.rgb = RGBColor.from_string(line)
    shape.line.width = Pt(2)
    return shape


def text(slide, value, x, y, w, h, size=30, font='Aptos', color=BLACK, bold=False, align=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.03)
    tf.margin_top = tf.margin_bottom = Inches(.02)
    for i, line in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(12)
        if align is not None:
            p.alignment = align
    return shape


def clone(source_index, cover=False):
    source = originals[source_index]
    slide = prs.slides.add_slide(source.slide_layout)
    for el in list(slide.shapes._spTree)[2:]:
        slide.shapes._spTree.remove(el)
    if source._element.cSld.bg is not None:
        slide._element.cSld.insert(0, deepcopy(source._element.cSld.bg))
    mapping = {}
    for rel in source.part.rels.values():
        if rel.reltype.endswith(('/slideLayout', '/notesSlide')):
            continue
        mapping[rel.rId] = slide.part.relate_to(rel.target_ref if rel.is_external else rel.target_part,
                                               rel.reltype, rel.is_external)
    for index, shape in enumerate(source.shapes):
        # Body slides reuse the original frame. Covers retain pixel decorations.
        if not cover and index != 0:
            continue
        if cover and shape.has_text_frame and shape.text.strip():
            continue
        element = deepcopy(shape.element)
        for node in element.iter():
            for key, value in list(node.attrib.items()):
                if key.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'):
                    if value in mapping:
                        node.set(key, mapping[value])
        slide.shapes._spTree.insert_element_before(element, 'p:extLst')
    return slide


def start(title, note, source='Source: project implementation and latex/main.tex', cover=False, last=False):
    slide = clone(9 if last else (0 if cover else 1), cover or last)
    index = len(script) + 1
    if not cover and not last:
        text(slide, title.upper(), 1.05, 1.0, 17.8, .8, 43, 'Pixellance')
        text(slide, 'SNAKE / DUELING DOUBLE DQN', 1.1, 9.95, 13, .35, 15)
        text(slide, f'{index:02d}', 17.7, 9.9, 1, .4, 18, align=PP_ALIGN.RIGHT)
        text(slide, source, 1.1, 9.4, 17.6, .4, 15)
    slide.notes_slide.notes_text_frame.text = note
    script.append((title, note))
    return slide


def bullets(slide, lines, x=1.2, y=2.3, w=17.4, size=33, gap=1.35):
    for i, line in enumerate(lines):
        box(slide, x, y + gap*i + .16, .17, .17, BLACK)
        text(slide, line, x+.45, y+gap*i, w-.45, gap-.12, size)


def cards(slide, items, y=2.65):
    w = 5.55
    for i, (label, body) in enumerate(items):
        x = 1.15 + i*6.05
        box(slide, x, y, w, 4.7)
        text(slide, label, x+.25, y+.35, w-.5, 1.2, 34, 'Pixellance')
        text(slide, body, x+.25, y+1.85, w-.5, 2.5, 29)


def table(slide, headers, rows, widths, y=2.45, rowh=.83, size=26):
    for j, row in enumerate([headers] + rows):
        x = 1.15
        for value, width in zip(row, widths):
            box(slide, x, y+j*rowh, width, rowh, BLACK if j == 0 else PALE)
            text(slide, str(value), x+.12, y+j*rowh+.12, width-.24, rowh-.18,
                 size, color=WHITE if j == 0 else BLACK, bold=j == 0)
            x += width


def figure(slide, filename):
    path = ROOT / 'latex/generated' / filename
    # Original publication plots remain unmodified, with full labels visible.
    from PIL import Image
    with Image.open(path) as img:
        ratio = img.width / img.height
    width = min(13.7, 6.1 * ratio)
    height = width / ratio
    slide.shapes.add_picture(str(path), Inches(1.2), Inches(2.3), Inches(width), Inches(height))


def wilson(k, n):
    z = 1.95996398454005
    p = k/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return c-h, c+h


def build():
    with (ROOT / 'results/cross_level_evaluation/summary.csv').open(newline='', encoding='utf-8') as handle:
        data = {(r['policy'], r['level']): r for r in csv.DictReader(handle)}
    s = start('Cross-Level Generalization of DQN in Snake',
        'Mục tiêu của đề tài là kiểm tra một Dueling Double DQN chỉ học trên Level 4 có thể chơi các level khác đến đâu. '
        'Đây là đánh giá zero-shot: giữ nguyên checkpoint và không fine-tune. Bài tập trung DQN. '
        'Tên đầy đủ trùng với báo cáo; bổ sung tên tác giả và đơn vị trước buổi trình bày.', cover=True)
    box(s, 3.4, 2.4, 13.2, 6.1, GREEN)
    text(s, 'SNAKE', 4, 2.85, 12, 1.35, 95, 'Pixellance', align=PP_ALIGN.CENTER)
    text(s, 'CROSS-LEVEL GENERALIZATION', 3.9, 4.45, 12.2, .85, 41, 'Pixellance', align=PP_ALIGN.CENTER)
    text(s, 'Evaluating a Dueling Double DQN\nTrained in a Single Snake Environment',
         4.1, 5.65, 11.8, 1.5, 32, align=PP_ALIGN.CENTER)
    text(s, 'Research presentation | October 2026', 4, 7.45, 12, .5, 23, align=PP_ALIGN.CENTER)

    s = start('The research question', 'Mở đầu bằng sự khác nhau giữa chơi tốt ở môi trường train và chuyển được sang môi trường khác. '
        'RQ1 đánh giá Level 4; RQ2 đánh giá năm level chưa thấy; RQ3 mô tả khác biệt về score, success và kết thúc episode. '
        'Các level thay đổi nhiều yếu tố cùng lúc nên RQ3 không khẳng định nguyên nhân riêng biệt.')
    text(s, 'Can one trained policy handle different Snake levels?', 1.2, 2.25, 17.3, 1.2, 43, bold=True)
    cards(s, [('RQ1 / SOURCE', 'How well does the agent perform on Level 4?'),
              ('RQ2 / TRANSFER', 'Does the frozen policy reach targets on five unseen levels?'),
              ('RQ3 / RELIABILITY', 'How do scores and failure modes vary across layouts?')], 3.8)

    s = start('Study at a glance', 'Tóm tắt thiết kế trước khi đi vào kỹ thuật: train một seed trên một level, dùng state 27 chiều và ba action tương đối. '
        'Đánh giá ba policy trên sáu level, mỗi tổ hợp 100 episode, tổng 1.800 episode. Random là baseline tối thiểu; final là kiểm tra độ nhạy checkpoint.')
    cards(s, [('TRAIN', 'Level 4 only\nOne training seed\nDueling Double DQN'),
              ('FREEZE', '27 state features\n3 relative actions\nGreedy evaluation'),
              ('EVALUATE', '6 levels × 100 episodes\nBest + final + random\n1,800 episodes total')])
    text(s, 'Single source environment → frozen policy → unseen-level evaluation', 1.3, 8.25, 17.4, .75, 30, bold=True)

    s = start('Six environments, one source', 'Level 4 là nguồn dữ liệu duy nhất của run được báo cáo. Level 1, 2, 3, 5, 6 đều chưa xuất hiện trong replay buffer của run này. '
        'Level 4 không có target, vì vậy không có success rate cho Level 4. Không so sánh raw score giữa các level như cùng một độ khó.')
    table(s, ['Level', 'Board', 'Obstacles', 'Target', 'Role'], [
        ['L1','15 × 15','0','25','Unseen'], ['L2','20 × 20','8','35','Unseen'],
        ['L3','24 × 24','16','50','Unseen'], ['L4','30 × 30','20','None','Training + evaluation'],
        ['L5','32 × 32','34','100','Unseen'], ['L6','36 × 36','68','150','Unseen']],
        [1.6,3,2.8,2.1,8.1], rowh=.85)
    text(s, 'Board size, obstacle geometry and target change together.', 1.2, 8.7, 17.5, .5, 27)

    s = start('How the agent sees the game', 'State gồm 27 feature: danger và ray 6; hướng 4; thông tin food 6; vị trí đầu và chiều dài 3; safe-area 3; thông tin tail 5. '
        'Một số feature được chuẩn hóa theo kích thước bàn. Safe-area dùng BFS có giới hạn để ước lượng vùng có thể tới. '
        'Thiết kế cho phép một network chạy trên nhiều board; hiệu quả transfer phải được xác nhận bằng evaluation.')
    table(s, ['Feature group', 'Count', 'Purpose'], [
        ['Danger + ray clearance','6','Local collision awareness'], ['Heading','4','Current movement direction'],
        ['Food direction + displacement','6','Navigation toward food'], ['Head position + body length','3','Normalized geometry'],
        ['Reachable safe areas','3','Space after candidate actions'], ['Tail direction + distance','5','Body / tail awareness']],
        [7.7,1.8,8.1], rowh=.85)

    s = start('Relative actions and rewards', 'Ba action tương đối giữ cùng ý nghĩa ở mọi level: đi thẳng, phải, trái. '
        'Reward ăn food +10, va chạm -20, bước tiến gần food +0.015, bước khác -0.025. '
        'Các giá trị bước đã gồm step penalty. Timeout khi không ăn sau hai lần số ô bàn; bộ đếm reset mỗi lần ăn food.')
    cards(s, [('STRAIGHT','Continue in the current direction.'), ('TURN RIGHT','Rotate relative to the current heading.'), ('TURN LEFT','Rotate relative to the current heading.')], 2.3)
    text(s, 'Food +10     Collision −20     Closer +0.015     Otherwise −0.025', 1.25, 7.55, 17.5, .8, 30, bold=True)
    text(s, 'No-food timeout: 2 × board cells; counter resets after eating.', 1.25, 8.5, 17.5, .6, 28)

    s = start('Dueling Double DQN', 'Network dùng MLP hai lớp 256 ReLU, chia value và advantage. Dueling kết hợp V và A để ra Q cho ba action. '
        'Double DQN dùng online network chọn action ở next state, target network định giá action đó. '
        'Hai kỹ thuật này khác nhau: một là kiến trúc, một là cách tính target.',
        'Sources: van Hasselt et al. (AAAI 2016); Wang et al. (ICML 2016); project agent.')
    cards(s, [('ENCODE','27 features\n→ 256 ReLU\n→ 256 ReLU'),
              ('DUELING','State value V(s)\nAction advantages A(s,a)\n→ three Q-values'),
              ('DOUBLE DQN','Online network selects\nTarget network evaluates\n→ Huber loss update')])
    text(s, 'Q(s,a) = V(s) + A(s,a) − mean A(s,·)', 1.3, 8.15, 17.4, .8, 34, bold=True)

    s = start('Training configuration', 'Run dùng seed 0, 14.293.725 environment steps; có 19.094 episode hoàn chỉnh và dừng trong episode 19.095. '
        'Best checkpoint xét ở mốc lưu 250 episode sau warm-up 500 episode, dựa rolling mean 100 episode. '
        'Chọn episode 17.000 với rolling mean 52,09. Các số training này không phải test score.')
    table(s, ['Setting','Value','Setting','Value'], [
        ['Learning rate','0.0003','Discount','0.995'], ['Batch size','256','Replay capacity','300,000'],
        ['Warm-up steps','5,000','Target update','5,000 steps'], ['Update frequency','Every 2 steps','Exploration ε','1.00 → 0.01'],
        ['Environment steps','14,293,725','Training seed','0']], [5,3.3,5.3,4], rowh=.9)
    text(s, 'Selected checkpoint: episode 17,000 | training rolling mean: 52.09', 1.25, 8.25, 17.3, .7, 30, bold=True)

    s = start('Learning on Level 4', 'Đường cong cho thấy learning tăng chậm giai đoạn đầu rồi tăng rõ sau đó. '
        'Rolling mean 52,09 là tiêu chí chọn checkpoint tại episode 17.000. '
        'Biến thiên episode còn lớn. Training dùng epsilon-greedy nên không đánh đồng điểm train với điểm greedy evaluation.')
    figure(s, 'training_curve.png')
    text(s, '52.09', 15.1, 2.8, 3.5, .9, 52, 'Pixellance')
    text(s, 'Selected checkpoint\n100-episode training mean', 15.1, 4.15, 3.45, 2, 27)
    text(s, 'Training statistics are separate from evaluation results.', 1.3, 8.8, 17.4, .5, 27, bold=True)

    s = start('Frozen-policy evaluation', 'Mỗi policy–level chạy 100 episode với seed 1234–1333. Network giữ nguyên; không gradient và không fine-tune. '
        'Best là policy chính, final để xem kết luận nhạy với checkpoint không, random làm baseline. '
        'Wilson CI đo bất định qua episode của policy này, không đo biến thiên giữa training seed.')
    bullets(s, ['100 episodes per policy–level pair; seeds 1234–1333.',
                'Best DQN, final DQN and uniform random policy.',
                'Frozen weights, greedy actions and headless evaluation.',
                'Scores, target success, normalized scores and failure causes.',
                '95% Wilson intervals for target success rates.'], size=31, gap=1.25)

    s = start('Zero-shot success across levels', 'Đây là kết quả chính của best checkpoint: 100%, 97%, 93%, 74%, 54% cho L1, L2, L3, L5, L6. '
        'Các thanh thể hiện success rate và đường ngang là Wilson CI 95%. L4 không có target nên không đưa vào biểu đồ success. '
        'Đặc biệt 100/100 không bảo đảm xác suất thành công thật là 100%.', 'Source: results/cross_level_evaluation/summary.csv | best_dqn | n = 100 per level')
    for i, lev in enumerate([1,2,3,5,6]):
        r = data['best_dqn', f'level{lev}']
        rate = float(r['success_rate'])
        low, high = wilson(round(rate*100), 100)
        yy = 2.6 + i*1.13
        text(s, f'L{lev}', 1.2, yy, 1, .6, 29, bold=True)
        box(s, 2.5, yy+.08, 11.8, .55, PALE)
        box(s, 2.5, yy+.08, 11.8*rate, .55, BLACK)
        box(s, 2.5+11.8*low, yy+.24, max(.015,11.8*(high-low)), .09, 'CAD2B6', 'CAD2B6')
        text(s, f'{rate:.0%}  [{low*100:.1f}–{high*100:.1f}]', 14.6, yy, 4, .7, 28)
    text(s, 'Strong transfer on smaller levels; less reliable on L5 and L6.', 1.25, 8.6, 17.4, .6, 30, bold=True)

    s = start('Score distributions reveal instability', 'Không chỉ xem mean: Level 6 median 150 nhưng P25 chỉ 1, cho thấy nhiều episode hoặc hoàn thành target hoặc gần như không tiến triển. '
        'L4 mean 144,49 SD 27,28 và median 154. Các score không trực tiếp cùng thang độ khó vì có target và kích thước khác nhau.')
    figure(s, 'score_distributions.png')
    text(s, 'LEVEL 6', 15.1, 2.8, 3.4, .7, 36, 'Pixellance')
    text(s, 'Median: 150\nP25: 1\nP75: 150', 15.1, 4, 3.4, 2.5, 30)
    text(s, 'L4 mean ± SD: 144.49 ± 27.28 | no target cap', 1.3, 8.6, 17.4, .6, 30, bold=True)

    s = start('Checkpoint choice changes transfer', 'Final checkpoint tăng L4 mean lên 153,83 nhưng L3 success giảm từ 93 xuống 83 và L5 từ 74 xuống 65. '
        'L6 mean tăng nhẹ nhưng success giảm 54 xuống 52. Đây là so sánh mô tả với cùng seed, không tuyên bố khác biệt có ý nghĩa thống kê. '
        'Random mean chỉ 0,02–0,10 và không hoàn thành target nào.')
    table(s, ['Metric','Selected DQN','Final DQN'], [
        ['L4 mean score','144.49','153.83'], ['L2 success','97%','99%'],
        ['L3 success','93%','83%'], ['L5 success','74%','65%'], ['L6 success','54%','52%']],
        [8,4.8,4.8], rowh=.9)
    text(s, 'Better source-level score did not consistently improve transfer.', 1.25, 8.15, 17.4, .7, 30, bold=True)
    text(s, 'Random baseline: mean scores 0.02–0.10; no target reached.', 1.25, 8.85, 17.4, .5, 26)

    s = start('Where the policy fails', 'L5 timeout 25%, L6 timeout 40%, thường biểu hiện thiếu tiến triển hơn là đâm tường. '
        'L4 self-collision 79% nhưng không nên diễn giải là kém an toàn hơn một cách đơn giản: L4 không có target nên episode tiếp tục khi snake dài. '
        'Không thấy wall collision ngoài L4 trong 100 episode mỗi level của best checkpoint.')
    figure(s, 'failure_modes.png')
    text(s, 'TIMEOUT', 15.1, 2.8, 3.4, .7, 35, 'Pixellance')
    text(s, 'L5: 25%\nL6: 40%\n\nL4 self-collision: 79%', 15.1, 4, 3.4, 3.5, 29)
    text(s, 'L4 has no target cap; its termination profile is different.', 1.3, 8.6, 17.4, .6, 29, bold=True)

    s = start('What the evidence supports', 'Kết quả trả lời RQ1: policy có hiệu năng nguồn cao trong evaluation. RQ2: có transfer nhưng chưa đồng đều, đặc biệt L5 L6. '
        'RQ3: phân bố và failure thay đổi giữa các level; chưa tách riêng ảnh hưởng size, obstacle, target. '
        'State và relative action là cơ chế hợp lý giúp dùng chung network nhưng chưa có ablation để khẳng định đóng góp nhân quả.')
    cards(s, [('RQ1', 'L4: 144.49 ± 27.28\nMedian score: 154\nOccasional early failure'),
              ('RQ2', '100%, 97%, 93% on L1–L3\n74%, 54% on L5–L6\nTransfer is incomplete'),
              ('RQ3', 'Timeouts rise on L5–L6\nLevel factors co-vary\nCausal effects unresolved')])
    text(s, 'A shared representation enables execution; evaluation establishes performance.', 1.25, 8.35, 17.4, .7, 29, bold=True)

    s = start('Limitations and next experiments', 'Nói rõ phạm vi một seed. 100 episode không thay thế được nhiều training run. '
        'Best chọn theo training rolling mean và chưa dùng validation riêng. '
        'Thí nghiệm tiếp theo theo ưu tiên: nhiều seed và layout mới, ablation state/reward/action, rồi so sánh curriculum và domain randomization với cùng step budget.')
    table(s, ['Current limitation','Next experiment'], [
        ['One training seed','Repeat independent training runs'],
        ['One source level; five unseen levels','Evaluate additional held-out layouts'],
        ['Board, obstacles and target co-vary','Change one factor at a time'],
        ['No representation / reward ablation','Remove feature groups and shaping'],
        ['Training-score checkpoint selection','Use a separate validation protocol']], [8.8,8.8], rowh=1.02, size=27)
    text(s, 'Then compare curricula / domain randomization under equal step budgets.', 1.25, 8.8, 17.4, .5, 27)

    s = start('Conclusion', 'Chốt ba ý: một Dueling Double DQN học duy nhất Level 4 có thể chơi các level khác không fine-tune; '
        'kết quả tốt trên ba level nhỏ hơn nhưng thiếu ổn định ở hai level lớn; mạnh hơn trên nguồn không luôn có nghĩa transfer tốt hơn. '
        'Giữ kết luận trong phạm vi policy và suite hiện tại, không gọi là giải quyết mọi môi trường Snake.')
    bullets(s, ['One Level-4-trained network transferred without fine-tuning.',
                'Success: 100%, 97%, 93%, 74%, 54% on L1, L2, L3, L5, L6.',
                'Larger unseen levels exposed high variance and timeouts.',
                'More training did not consistently improve cross-level success.'], size=32, gap=1.45)
    text(s, 'Substantial transfer within this suite — incomplete robustness.', 1.25, 8.45, 17.4, .65, 33, bold=True)

    s = start('Selected references', 'Các tài liệu nền tảng đã có trong references.bib. Slide này chỉ chọn nguồn cho DQN, Double DQN, Dueling, '
        'generalization và đánh giá RL. Các số liệu trong bài trình bày là từ thực nghiệm dự án, không phải kết quả của những paper này.',
        'Full bibliographic entries and links: latex/references.bib')
    bullets(s, ['Mnih et al. (2015). Human-Level Control through Deep Reinforcement Learning. Nature.',
                'van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-Learning. AAAI.',
                'Wang et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. ICML.',
                'Cobbe et al. (2019). Quantifying Generalization in Reinforcement Learning. ICML.',
                'Henderson et al. (2018). Deep Reinforcement Learning That Matters. AAAI.'],
            size=27, gap=1.2)

    s = start('Thank you / Questions', 'Kết thúc phần chính. Khi được hỏi về seed, ablation hay unseen levels, có thể chuyển đến các backup slide sau.', last=True)
    box(s, 3.5, 2.7, 13, 5.4, GREEN)
    text(s, 'THANK YOU', 4.2, 3.5, 11.6, 1.5, 78, 'Pixellance', align=PP_ALIGN.CENTER)
    text(s, 'QUESTIONS?', 4.2, 5.5, 11.6, 1, 58, 'Pixellance', align=PP_ALIGN.CENTER)
    text(s, 'One source level. Five unseen tests. Measured transfer.', 4.2, 7, 11.6, .6, 24, align=PP_ALIGN.CENTER)

    s = start('Backup / detailed primary results', 'Bảng chi tiết best checkpoint để trả lời câu hỏi về mean, SD, median, success và normalized score. '
        'L4 N/A là do không có target. SD là độ lệch chuẩn mô tả qua episode, không phải khoảng tin cậy.')
    rows=[]
    for lev in range(1,7):
        r=data['best_dqn',f'level{lev}']
        rows.append([f'L{lev}', f"{float(r['mean_score']):.2f} ± {float(r['std_score']):.2f}",
                     f"{float(r['median_score']):.1f}",
                     f"{float(r['success_rate']):.0%}" if r['success_rate'] else 'N/A',
                     f"{float(r['mean_normalized_score']):.3f}" if r['mean_normalized_score'] else 'N/A'])
    table(s,['Level','Mean ± SD','Median','Success','Norm. score'],rows,[1.7,5.5,3,3.3,4.1],rowh=.86)

    s = start('Backup / anticipated questions', 'Các câu trả lời cần giữ nhất quán: unseen là so với run được báo cáo; random baseline chưa đủ chứng minh ưu việt so với heuristic; '
        'không có PPO trong phạm vi nghiên cứu chính. Cần nhiều seed để kết luận độ ổn định của thuật toán. '
        'Điểm greedy cao hơn training vì lúc train vẫn còn 1% action ngẫu nhiên. Đã kiểm chứng: '
        'chạy lại Level 4 cùng 100 seed với epsilon 0,01 thì mean giảm từ 144,49 xuống 43,58, '
        'khớp rolling mean lúc train (44–52).')
    table(s,['Question','Evidence-based answer'],[
        ['Why call these levels unseen?','Only L4 supplied training transitions in this run.'],
        ['Does 100% mean guaranteed success?','No; 100/100 gives a 95% Wilson interval of 96.3–100%.'],
        ['Is the method better than all baselines?','Only random and final-checkpoint comparisons were evaluated.'],
        ['Do features cause generalization?','Plausible, but no ablation isolates their effect.'],
        ['Why no PPO comparison?','The research scope is single-source DQN transfer.']], [7.4,10.2],rowh=1.05,size=25)

    # Remove source slides only after all clones have been created.
    for _ in originals:
        item = prs.slides._sldIdLst[0]
        prs.part.drop_rel(item.rId)
        prs.slides._sldIdLst.remove(item)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'Snake_DQN_Research_Presentation.pptx'
    prs.save(path)
    notes = ['# Presentation script', '',
             'Slides: English. Speaker Notes: Vietnamese. Main talk: slides 1–19; backup: 20–21.',
             'Suggested duration: 12–15 minutes, excluding questions.', '']
    for i, (title, note) in enumerate(script,1):
        notes.extend([f'## {i:02d}. {title}', '', note, ''])
    (OUT/'speaker_notes_vi.md').write_text('\n'.join(notes),encoding='utf-8')
    reopened = Presentation(path)
    assert len(reopened.slides) == len(script) == 21
    for slide in reopened.slides:
        assert slide.notes_slide.notes_text_frame.text.strip()
        for shape in slide.shapes:
            if shape.has_text_frame:
                assert shape.left >= 0 and shape.top >= 0
                assert shape.left + shape.width <= reopened.slide_width
                assert shape.top + shape.height <= reopened.slide_height
    print(f'Created and validated {len(script)} slides: {path}')


if __name__ == '__main__':
    build()
