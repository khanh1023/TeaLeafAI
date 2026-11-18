"""Additional educational CNN diagrams and verified training settings."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np

GREEN, INK, GRAY = '#216b50', '#183c32', '#56665e'


def add_cnn_figures(root, save, canvas, box, arrow, leaf):
    # A conceptual CNN, deliberately distinguished from the exact MobileNetV2 diagram.
    fig, ax = canvas(5.4)
    ax.text(.5, .94, 'TỔNG QUAN CNN TRONG PHÂN LOẠI ẢNH LÁ CHÈ', ha='center', fontsize=15, weight='bold')
    xs = [.025, .28, .535, .79]
    photo = ax.inset_axes([xs[0], .32, .185, .42])
    photo.imshow(leaf); photo.set_xticks([]); photo.set_yticks([])
    for spine in photo.spines.values():
        spine.set_color('#cbd7cd')
    for offset, color in [(2, '#d0dfd1'), (1, '#a1c3ab'), (0, '#6a9c7c')]:
        x, y = .292+offset*.018, .34+offset*.033
        ax.add_patch(Rectangle((x, y), .13, .30, facecolor=color, edgecolor=GREEN, linewidth=1.2))
        for t in range(1, 5):
            ax.plot([x, x+.13], [y+t*.06, y+t*.06], color='white', alpha=.45, lw=.8)
            ax.plot([x+t*.026, x+t*.026], [y, y+.30], color='white', alpha=.45, lw=.8)
    ax.text(.37, .78, 'Conv + ReLU', ha='center', fontsize=12.5, weight='bold')
    for i, value in enumerate([.12, .25, .18, .32, .14, .23]):
        ax.add_patch(Rectangle((.553+i*.023, .36), .014, value, facecolor=GREEN, alpha=.5+i*.07))
    ax.text(.63, .78, 'Vector đặc trưng', ha='center', fontsize=12.5, weight='bold')
    for i in range(8):
        y = .32+i*.05
        ax.add_patch(Circle((.925, y), .011, facecolor=GREEN, edgecolor=GREEN))
        for source_y in [.43, .53, .63]:
            ax.plot([.825, .914], [source_y, y], color='#a0b8a7', lw=.6, zorder=0)
    for y in [.43, .53, .63]:
        ax.add_patch(Circle((.825, y), .014, facecolor='#c0d8c4', edgecolor=GREEN))
    ax.text(.88, .78, 'Dense + Softmax', ha='center', fontsize=12.5, weight='bold')
    for start, end in [((.22, .52), (.272, .52)), ((.46, .52), (.535, .52)), ((.705, .52), (.78, .52))]:
        arrow(ax, start, end)
    subtitles = [('Ảnh đầu vào', 'RGB · 224 × 224'), ('Trích xuất\nđặc trưng', 'Cạnh, gân, màu sắc'),
                 ('Tổng hợp\nđặc trưng', 'Giảm chiều dữ liệu'), ('Phân loại', '8 điểm đầu ra')]
    for x, (title, body) in zip(xs, subtitles):
        ax.text(x+.0925, .235, title, ha='center', va='top', fontsize=13, weight='bold', linespacing=1.3)
        ax.text(x+.0925, .105, body, ha='center', fontsize=10.8, color=GRAY)
    ax.text(.5, .025, 'Sơ đồ khái niệm; các ô và kết nối không biểu diễn trọng số hoặc đặc trưng đã đo của mô hình.',
            ha='center', fontsize=9.7, color=GRAY)
    save(fig, 'bo-sung-tong-quan-cnn', 'Tổng quan CNN trong phân loại ảnh lá chè',
         '2.3.1. Tổng quan về mạng nơ-ron tích chập CNN',
         'Tác giả xây dựng; ảnh đầu vào lấy từ bộ dữ liệu gốc TeaLeafAI.',
         'Dùng để giới thiệu vai trò của trích xuất đặc trưng và bộ phân loại. Kiến trúc MobileNetV2 thực tế được trình bày bằng hình riêng.')

    # Fully computed convolution, ReLU and max-pooling example; no invented activation measurements.
    values = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1],
                       [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    output = np.array([[np.sum(values[y:y+3, x:x+3]*kernel) for x in range(3)] for y in range(3)])
    relu = np.maximum(output, 0)
    pooled = np.array([[np.max(relu[y:y+2, x:x+2]) for x in range(2)] for y in range(2)])
    assert output[0, 0] == -2
    fig, ax = canvas(7.4)
    ax.text(.5, .965, 'TÍCH CHẬP, RELU VÀ MAX POOLING: VÍ DỤ SỐ', ha='center', fontsize=15, weight='bold')

    def matrix(data, pos, title, highlight=False):
        panel = ax.inset_axes(pos)
        panel.imshow(data, cmap='Greens', vmin=min(-2, int(data.min())), vmax=max(3, int(data.max())))
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                panel.text(x, y, str(data[y, x]), ha='center', va='center', fontsize=13,
                           color='white' if data[y, x] >= 3 else INK, weight='bold')
        panel.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
        panel.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
        panel.grid(which='minor', color='white', linewidth=1.2)
        panel.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in panel.spines.values():
            spine.set_visible(False)
        if highlight:
            panel.add_patch(Rectangle((-.5, -.5), 3 if data.shape[0] == 5 else 1,
                                      3 if data.shape[0] == 5 else 1, fill=False,
                                      edgecolor='#ba7330', linewidth=2.2))
        ax.text(pos[0]+pos[2]/2, pos[1]+pos[3]+.025, title, ha='center', fontsize=12.3, weight='bold')

    matrix(values, [.035, .57, .25, .28], '1. Đầu vào 5 × 5', True)
    matrix(kernel, [.395, .595, .19, .23], '2. Bộ lọc 3 × 3')
    matrix(output, [.725, .595, .20, .23], '3. Kết quả 3 × 3', True)
    ax.text(.335, .70, '×', ha='center', fontsize=23, weight='bold')
    arrow(ax, (.62, .70), (.70, .70))
    ax.text(.49, .885, 'Stride = 1 · Padding = valid · Bias = 0', ha='center', fontsize=10.6, color=GRAY)
    ax.text(.37, .485, 'Ô đầu tiên: tổng các tích = −2.', ha='center', fontsize=12)
    matrix(relu, [.725, .135, .20, .23], '4. ReLU: max(0, z)')
    matrix(pooled, [.26, .135, .20, .23], '5. Max pooling 2 × 2')
    arrow(ax, (.825, .57), (.825, .425))
    arrow(ax, (.695, .25), (.49, .25))
    ax.text(.59, .31, 'Lấy giá trị lớn nhất\nStride = 1', ha='center', fontsize=10.7, color=GRAY)
    ax.text(.5, .045, 'Minh họa CNN tổng quát. Kiến trúc TeaLeafAI dùng GAP thay cho MaxPooling trong ví dụ này.',
            ha='center', fontsize=9.9, color=GRAY)
    save(fig, 'bo-sung-tich-chap-relu-pooling', 'Minh họa phép tích chập, hàm ReLU và Max pooling',
         '2.3.2. Cấu trúc và nguyên lý hoạt động của CNN trong phân loại ảnh',
         'Tác giả xây dựng và tính toán trực tiếp từ các ma trận minh họa.',
         'Bộ lọc được nhân từng phần tử rồi cộng trên mỗi cửa sổ. ReLU đưa giá trị âm về 0; Max pooling lấy giá trị lớn nhất. Đây là ví dụ lý thuyết, không phải bản đồ kích hoạt trích từ mô hình đã huấn luyện.')

    # Typical MobileNetV2 bottleneck, with its precise residual condition and linear projection.
    fig, ax = canvas(6.0)
    ax.text(.5, .95, 'KHỐI INVERTED RESIDUAL TRONG MOBILENETV2', ha='center', fontsize=15, weight='bold')
    box(ax, .018, .60, .115, .22, 'Đầu vào', 'H × W × C', fontsize=12)
    box(ax, .20, .60, .21, .22, 'Mở rộng 1 × 1', 'C → 6C\nBN + ReLU6', fontsize=12.7)
    box(ax, .48, .60, .21, .22, 'Depthwise 3 × 3', '6C kênh · Stride 1\nBN + ReLU6', fontsize=12.3)
    box(ax, .76, .60, .21, .22, 'Chiếu 1 × 1', '6C → C\nBN · Tuyến tính', fontsize=12.7)
    for start, end in [((.14, .71), (.19, .71)), ((.42, .71), (.47, .71)), ((.70, .71), (.75, .71))]:
        arrow(ax, start, end)
    ax.plot([.075, .075, .825], [.59, .40, .40], color=GREEN, linewidth=1.6)
    arrow(ax, (.825, .40), (.832, .40))
    ax.text(.43, .44, 'Nhánh giữ nguyên đầu vào', ha='center', fontsize=12, color=GRAY)
    ax.add_patch(Circle((.865, .40), .032, facecolor='#eef5ef', edgecolor=GREEN, linewidth=1.6))
    ax.text(.865, .40, '+', fontsize=19, ha='center', va='center', weight='bold')
    arrow(ax, (.865, .59), (.865, .444))
    arrow(ax, (.865, .36), (.865, .235))
    ax.text(.865, .17, 'Đầu ra\nH × W × C', ha='center', va='center', fontsize=12.5, weight='bold')
    ax.text(.38, .20, 'BN: Batch Normalization\nKhối minh họa có hệ số mở rộng t = 6.',
            ha='center', fontsize=11.3, color=GRAY, linespacing=1.6)
    ax.text(.5, .045, 'Chỉ cộng nhánh tắt khi stride = 1 và số kênh đầu vào bằng số kênh đầu ra.', ha='center', fontsize=11, color=GRAY)
    save(fig, 'bo-sung-khoi-mobilenetv2', 'Khối Inverted Residual và Linear Bottleneck của MobileNetV2',
         '2.3.3. Lựa chọn mô hình MobileNetV2 cho bài toán phân loại bệnh lá chè',
         'Tác giả vẽ lại theo Sandler và cộng sự (2018), MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR; https://arxiv.org/abs/1801.04381.',
         'Hình mô tả một khối điển hình có kết nối tắt. Không phải mọi khối đều có t = 6 hoặc kết nối tắt. Phép chiếu cuối là tuyến tính, không đặt ReLU6 sau phép chiếu.')

    training = json.loads((root/'training/evaluation.json').read_text())['training']
    fine = training['fine_tuning']
    fig, ax = canvas(7.0)
    ax.text(.5, .95, 'THAM SỐ HUẤN LUYỆN THỰC TẾ CỦA TEALEAFAI', ha='center', fontsize=15, weight='bold')
    for x in [.025, .53]:
        ax.add_patch(FancyBboxPatch((x, .23), .445, .64, boxstyle='round,pad=.008,rounding_size=.014',
                                  facecolor='#eef5ef', edgecolor=GREEN, linewidth=1.5))
    columns = [(.055, 'GIAI ĐOẠN 1', 'Huấn luyện bộ phân loại', [
        'Backbone: đóng băng', 'Optimizer: Adam', 'Learning rate đầu: 0,001', 'Batch size: 32',
        f'Epoch tối đa: {training["maximum_epochs"]} · Đã chạy: {training["epochs_completed"]}',
        'EarlyStopping: patience = 5', 'Dropout: 0,3 và 0,2']),
        (.56, 'GIAI ĐOẠN 2', 'Tinh chỉnh mô hình', [
        '30 lớp cuối; giữ cố định BN', 'Optimizer: Adam', 'Learning rate đầu: 0,00001', 'Batch size: 16',
        f'Epoch tối đa: {fine["maximum_epochs"]} · Đã chạy: {fine["epochs_completed"]}',
        'EarlyStopping: patience = 3', 'Chọn checkpoint theo val_loss'])]
    for x, title, subtitle, rows in columns:
        ax.text(x+.192, .82, title, ha='center', fontsize=13.5, weight='bold')
        ax.text(x+.192, .765, subtitle, ha='center', fontsize=12.5, weight='bold')
        for i, text in enumerate(rows):
            ax.text(x, .675-i*.062, text, fontsize=12, ha='left', va='center')
    ax.text(.5, .15, 'Loss: Sparse Categorical Crossentropy · Seed = 42 · Có trọng số cân bằng lớp', ha='center', fontsize=10.9)
    ax.text(.5, .095, 'Lưu checkpoint có val_loss nhỏ nhất; giảm learning rate khi val_loss không cải thiện.', ha='center', fontsize=10.8, color=GRAY)
    ax.text(.5, .04, 'Tập test chỉ dùng để đánh giá; không dùng để cập nhật trọng số hoặc chọn checkpoint.', ha='center', fontsize=10.7, color=GRAY)
    save(fig, 'bo-sung-tham-so-huan-luyen', 'Các tham số của hai giai đoạn huấn luyện mô hình',
         '2.4.1. Thiết lập các tham số huấn luyện',
         'Tác giả tổng hợp từ training/train_expanded.py, training/finetune_expanded.py và training/evaluation.json.',
         'Phân biệt số epoch tối đa và số epoch thực chạy. Giai đoạn 1 huấn luyện trên đặc trưng đã lưu của backbone đóng băng; giai đoạn 2 tinh chỉnh các lớp cuối. Cả hai chọn checkpoint bằng validation loss.')
