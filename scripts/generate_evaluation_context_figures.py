#!/usr/bin/env python3
"""Render manuscript context figures from recorded TeaLeafAI evidence.

No retraining, browser execution, dataset changes, or repository publication.
PNG (300 dpi), SVG and PDF are saved for Word/publication use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tealeaf-context-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'training/reports/evaluation-20261122/figures'
META = OUT.parent / 'data/context-figure-sources.json'
GREEN, BLUE, GOLD = '#20664f', '#367baa', '#b47726'
INK, MUTED, GRID = '#203c36', '#586a65', '#e3e9e5'


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14,
        'axes.titlesize': 14, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 13, 'axes.titleweight': 'semibold',
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': '#bccac2', 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem, out):
    paths = {}
    for ext in ('png', 'svg', 'pdf'):
        path = out / f'{stem}.{ext}'
        fig.savefig(path, dpi=300)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def history_figure(evaluation, out):
    training = evaluation['training']
    head = training['history']
    fine = training['fine_tuning']['history']
    head_epoch = int(training['selected_epoch'])
    fine_epoch = int(np.argmin(fine['val_loss']) + 1)
    assert len(head['accuracy']) == training['epochs_completed'] == 14
    assert len(fine['accuracy']) == training['fine_tuning']['epochs_completed'] == 6
    assert head_epoch == 9 and fine_epoch == 3
    assert abs(fine['val_loss'][fine_epoch - 1] - training['fine_tuning']['selected_validation_loss']) < 1e-8
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    fig.subplots_adjust(left=.105, right=.975, top=.855, bottom=.205, wspace=.28, hspace=.46)
    fig.text(.105, .959, 'Quá trình huấn luyện MobileNetV2', fontsize=19, weight='bold', color=INK)
    fig.text(.105, .918, 'Hai giai đoạn dùng trục epoch riêng; điểm chọn theo validation loss.', fontsize=12, color=MUTED)
    for row, (history, chosen, phase) in enumerate(((head, head_epoch, 'Bộ phân loại'), (fine, fine_epoch, 'Tinh chỉnh'))):
        epochs = np.arange(1, len(history['accuracy']) + 1)
        for col, metric in enumerate(('accuracy', 'loss')):
            ax = axes[row, col]
            ax.set_axisbelow(True)
            ax.grid(axis='y', color=GRID, linewidth=.8)
            ax.plot(epochs, history[metric], '-o', c=GREEN, lw=2.1, markersize=3.4, label='Train')
            ax.plot(epochs, history['val_' + metric], '-s', c=BLUE, lw=2.1, markersize=3.4, label='Validation')
            ax.axvline(chosen, color=GOLD, lw=1.25, ls=(0, (3, 3)), zorder=1)
            ax.scatter([chosen], [history['val_' + metric][chosen - 1]], s=80,
                       marker='o', facecolor='white', edgecolor=GOLD, lw=2, zorder=4)
            ax.set_title(f'{phase} · {"Accuracy" if metric == "accuracy" else "Loss"}', loc='left', pad=11)
            ax.text(.98, .96, f'Chọn epoch {chosen}/{len(epochs)}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=12, color=GOLD,
                    bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9, 'pad':2})
            ax.set_xlabel('Epoch trong giai đoạn')
            ax.set_xlim(.6, len(epochs)+.4)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            if metric == 'accuracy':
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.set_ylabel('Accuracy')
            else:
                ax.set_ylim(0, max(max(history['loss']), max(history['val_loss']))*1.2)
                ax.set_ylabel('Loss')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower left',
               bbox_to_anchor=(.092, .075), ncol=2, frameon=False, handlelength=2.1)
    fig.text(.105, .047, 'Train: có tăng cường và trọng số lớp; validation: không tăng cường.', fontsize=12, color=MUTED)
    fig.text(.105, .018, 'Không diễn giải trực tiếp khoảng cách train/validation loss như cùng một điều kiện đo.', fontsize=12, color=MUTED)
    paths = save(fig, '3-2-1-training-history', out)
    return {'section':'3.2.1', 'paths': paths,
        'caption': 'Diễn biến Accuracy và loss ở hai giai đoạn huấn luyện. Giai đoạn bộ phân loại chạy 14 epoch và chọn epoch 9; tinh chỉnh chạy 6 epoch và chọn epoch 3 theo validation loss. Các đường cong được trích trực tiếp từ nhật ký đã lưu, không huấn luyện lại.',
        'source': 'training/evaluation.json: training.history, training.selected_epoch, training.fine_tuning.history, training.fine_tuning.selected_validation_loss',
        'notes': ['Không nối epoch giữa hai giai đoạn thành một đường liên tục.',
                  'Loss huấn luyện có trọng số lớp và ảnh tăng cường; validation không tăng cường. Không so sánh cơ học độ cao hai đường loss.',
                  'Các mốc được đánh dấu là checkpoint được chọn, không phải epoch có train accuracy cao nhất.'],
        'values': {'head_epochs':14,'head_selected_epoch':9,'finetune_epochs':6,'finetune_selected_epoch':3}}


def comparison_figure(evaluation, out):
    assert evaluation['model'] == 'MobileNetV2'
    evidence_paths = ['backend/models/tea_leaf_mobilenetv2.keras', 'training/train_expanded.py',
                      'training/finetune_expanded.py', 'training/reports/reevaluation/native.json',
                      'training/reports/reevaluation/browser-metrics.json']
    assert all((ROOT / path).is_file() for path in evidence_paths)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axis('off')
    fig.subplots_adjust(left=.035, right=.97, top=.79, bottom=.20)
    fig.text(.045, .925, 'Phạm vi bằng chứng so sánh kiến trúc', fontsize=19, weight='bold')
    fig.text(.045, .865, 'Chỉ ghi nhận những thành phần có trong bộ hồ sơ hiện tại.', fontsize=12, color=MUTED)
    columns = ['Kiến trúc', 'Checkpoint', 'Mã huấn luyện', 'Kết quả test']
    rows = [
        ['MobileNetV2', 'Có bằng chứng', 'Có bằng chứng', 'Có bằng chứng'],
        ['ResNet50', 'Chưa thực nghiệm', 'Chưa thực nghiệm', 'Chưa thực nghiệm'],
        ['VGG16', 'Chưa thực nghiệm', 'Chưa thực nghiệm', 'Chưa thực nghiệm'],
    ]
    table = ax.table(cellText=rows, colLabels=columns, colWidths=[.20,.25,.28,.27],
                     cellLoc='center', loc='center', bbox=[0,.05,1,.88])
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(.8)
        cell.set_edgecolor('#dce5de')
        cell.PAD=.09
        if r == 0:
            cell.set_facecolor(GREEN); cell.get_text().set_color('white'); cell.get_text().set_weight('bold')
        elif r == 1:
            cell.set_facecolor('#edf5ee'); cell.get_text().set_color(GREEN)
        else:
            cell.set_facecolor('#f8faf8'); cell.get_text().set_color('#6b766f')
        if c==0 and r>0: cell.get_text().set_weight('bold')
    fig.text(.045, .155, 'Keras và ONNX là các dạng triển khai của cùng MobileNetV2,', fontsize=13, color=INK)
    fig.text(.045, .115, 'không phải các kiến trúc mô hình độc lập.', fontsize=13, color=INK)
    fig.text(.045, .055, '“Chưa thực nghiệm” chưa phải kết quả đo và không cho phép xếp hạng.', fontsize=12, color=MUTED)
    paths = save(fig, '3-3-4-comparison-evidence', out)
    return {'section':'3.3.4', 'paths':paths,
        'caption':'Tình trạng bằng chứng cho so sánh kiến trúc trong hồ sơ dự án. MobileNetV2 có checkpoint, mã huấn luyện và kết quả test; chưa có thực nghiệm tương ứng để so sánh với ResNet50 hoặc VGG16. Hình không biểu diễn hoặc suy đoán độ chính xác của các mô hình chưa thực nghiệm.',
        'source': evidence_paths,
        'notes':['Không tái sử dụng các số 95% hoặc bảng ResNet50/VGG16 chưa được kiểm chứng.',
                 'Không xem Keras/ONNX/ONNX Runtime Web là các kiến trúc khác nhau.',
                 'Chưa thực nghiệm là giới hạn bằng chứng hiện tại, không phải kết luận mô hình kém hơn.'],
        'values': {'architectures': rows, 'accuracy_claims_in_figure':False}}


def distribution_figure(evaluation, summary, out):
    classes = evaluation['classes']
    splits = summary['split_counts']
    totals = np.array([summary['counts'][name] for name in classes])
    assert int(totals.sum()) == summary['total_images'] == 2197
    assert all(sum(splits[split][name] for split in ('train','val','test')) == summary['counts'][name] for name in classes)
    split_totals = {split:sum(splits[split].values()) for split in ('train','val','test')}
    assert split_totals == {'train':1541,'val':327,'test':329}
    fig, ax = plt.subplots(figsize=(9, 6.8))
    fig.subplots_adjust(left=.238, right=.945, top=.79, bottom=.28)
    fig.text(.045, .937, 'Phân bố ảnh theo lớp và bộ chia', fontsize=19, weight='bold')
    fig.text(.045, .885, '2.197 ảnh · 8 lớp · nhóm ảnh gần trùng được giữ trong cùng một tập', fontsize=12, color=MUTED)
    y = np.arange(len(classes))
    healthy = classes.index('healthy')
    ax.axhspan(healthy-.43, healthy+.43, facecolor='#e5f0e5', zorder=-1)
    left = np.zeros(len(classes))
    for split, label, color in [('train','Train',GREEN),('val','Validation','#7eaa95'),('test','Test','#c1d6c8')]:
        values = np.array([splits[split][name] for name in classes])
        ax.barh(y, values, left=left, height=.56, color=color, edgecolor='white', linewidth=.55,
                label=f'{label}: {split_totals[split]:,}'.replace(',', '.'))
        left += values
    for i, total in enumerate(totals):
        ax.text(total+9, i, str(total), ha='left', va='center', fontsize=13,
                color=GREEN if i==healthy else MUTED, weight='bold' if i==healthy else 'normal')
    ax.set_yticks(y, classes)
    ax.tick_params(axis='y', length=0, pad=10, labelsize=13)
    ax.get_yticklabels()[healthy].set_color(GREEN)
    ax.get_yticklabels()[healthy].set_weight('bold')
    ax.set_xlim(0, 740)
    ax.set_ylim(len(classes)-.5, -.5)
    ax.set_xlabel('Số ảnh', labelpad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_axisbelow(True); ax.grid(axis='x', color=GRID, lw=.8)
    ax.spines['left'].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.226,.108),
                frameon=False, ncol=3, columnspacing=1.4, handlelength=1.5, fontsize=12)
    healthy_share = summary['counts']['healthy'] / summary['total_images'] * 100
    share_text = f'{healthy_share:.2f}'.replace('.', ',')
    fig.text(.045,.063,f'Healthy có 673/2.197 ảnh ({share_text}%); nhiều hơn từng lớp bệnh riêng lẻ.', fontsize=12, color=INK)
    fig.text(.045,.024,'Mất cân bằng được mô tả từ dữ liệu; hình không chứng minh nguyên nhân gây sai dự đoán.', fontsize=12, color=MUTED)
    paths = save(fig, '3-3-5-class-distribution', out)
    return {'section':'3.3.5','paths':paths,
        'caption':'Phân bố 2.197 ảnh theo tám lớp và ba tập train/validation/test. Lớp healthy có 673 ảnh, chiếm 30,63% toàn bộ dữ liệu. Đây là đặc điểm phân bố mẫu cần cân nhắc khi đọc Accuracy và macro F1; chưa phải bằng chứng nhân quả về sai số của mô hình.',
        'source':'training/reports/expanded-dataset-summary.json: counts, split_counts, total_images; training/evaluation.json: classes',
        'notes':['Tổng các tập: train 1.541, validation 327, test 329 ảnh.',
                 'Giữ nguyên tên kỹ thuật gray_light theo dữ liệu nguồn.',
                 'Không suy diễn thêm ảnh healthy tự bảo đảm cải thiện khả năng khái quát.'],
        'values':{'total_images':2197,'counts':summary['counts'],'split_counts':splits,'split_totals':split_totals,'healthy_share_percent':healthy_share}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--metadata', type=Path, default=META)
    args = parser.parse_args()
    evaluation_path = ROOT / 'training/evaluation.json'
    summary_path = ROOT / 'training/reports/expanded-dataset-summary.json'
    evaluation = json.loads(evaluation_path.read_text())
    summary = json.loads(summary_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    style()
    figures = [history_figure(evaluation, args.output_dir),
               comparison_figure(evaluation, args.output_dir),
               distribution_figure(evaluation, summary, args.output_dir)]
    metadata = {'generator':'scripts/generate_evaluation_context_figures.py',
        'evidence_policy':'Measured records only; no retraining, fabricated comparisons, or new causal claims.',
        'format':{'figure_width_inches':9,'png_dpi':300,'font':'DejaVu Sans','minimum_native_font_pt':12,'additional_formats':['SVG','PDF']},
        'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in (evaluation_path,summary_path)},
        'figures':figures}
    args.metadata.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':[f['paths'] for f in figures],'metadata':str(args.metadata)},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
