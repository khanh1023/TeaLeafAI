"""Generate conceptual CNN layer and training diagrams with correct example dimensions."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'training/.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT=ROOT/'docs/figures/chapter2';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#183c32',
                     'mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42,
                     'figure.facecolor':'white'})
GREEN='#216b50';GRAY='#56665e';figures=[]


def canvas(height):
    fig,ax=plt.subplots(figsize=(9,height));fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.025)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    return fig,ax


def box(ax,x,y,w,h,title,body,fontsize=12.6,body_size=11.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.014',
                               facecolor='#eef5ef',edgecolor=GREEN,linewidth=1.5))
    ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontsize=fontsize,weight='bold')
    ax.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=body_size,color=GRAY,linespacing=1.5)


def arrow(ax,start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,color=GREEN,linewidth=1.6))


def save(fig,stem,caption,note):
    for suffix in ['png','svg','pdf']:
        p=OUT/f'{stem}.{suffix}';fig.savefig(p,dpi=300)
        if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig);figures.append(dict(stem=stem,caption=caption,note=note))


fig,ax=canvas(8)
ax.text(.5,.965,'CÁC LỚP CƠ BẢN VÀ LUỒNG XỬ LÝ CỦA CNN',ha='center',fontsize=15.2,weight='bold')
xs=[.025,.37,.715];w=.26;h=.19
top=[('1. Lớp đầu vào','Ảnh RGB\n224 × 224 × 3'),
     ('2. Lớp tích chập','16 bộ lọc 3 × 3\nStride 1 · Padding same\n224 × 224 × 16'),
     ('3. Hàm ReLU',r'$f(x)=\max(0,x)$'+'\n224 × 224 × 16')]
bottom=[('6. Lớp đầu ra','Softmax → 8 điểm số\nChọn lớp có điểm cao nhất'),
        ('5. Kết nối đầy đủ','GAP → vector 16 phần tử\nDense → 8 giá trị z'),
        ('4. Lớp gộp','Max Pooling 2 × 2\nStride 2\n112 × 112 × 16')]
for x,(title,body) in zip(xs,top):box(ax,x,.68,w,h,title,body)
for x,(title,body) in zip(xs,bottom):box(ax,x,.36,w,h,title,body)
for i in range(2):
    arrow(ax,(xs[i]+w+.008,.775),(xs[i+1]-.01,.775))
    arrow(ax,(xs[i+1]-.01,.455),(xs[i]+w+.008,.455))
arrow(ax,(.845,.67),(.845,.56))
ax.text(.04,.245,'Tích chập:',fontsize=12,weight='bold',va='center')
ax.text(.61,.245,r'$Y_{i,j,o}=b_o+\sum_{u,v,c}K_{u,v,c,o}\,\widetilde{X}_{i+u,j+v,c}$',
        ha='center',va='center',fontsize=15)
ax.text(.5,.18,r'$\widetilde{X}$ là ảnh đã đệm 0 một pixel mỗi phía cho bộ lọc 3 × 3, stride 1.',
        ha='center',fontsize=10.7,color=GRAY)
ax.text(.04,.102,'Softmax:',fontsize=12,weight='bold',va='center')
ax.text(.61,.102,r'$p_k=\frac{\exp(z_k)}{\sum_{j=1}^{8}\exp(z_j)}$',ha='center',va='center',fontsize=16)
ax.text(.5,.028,'Sơ đồ CNN minh họa. TeaLeafAI thực tế sử dụng MobileNetV2 với ReLU6 và GAP.',
        ha='center',fontsize=10.3,color=GRAY)
save(fig,'cnn-cac-lop-va-luong-xu-ly','Các lớp cơ bản và luồng xử lý của một CNN minh họa.',
     'Đọc từ trái sang phải ở hàng trên, xuống lớp gộp rồi từ phải sang trái. Các kích thước là ví dụ lý thuyết: Conv 3 × 3, 16 bộ lọc, stride 1, padding same giữ 224 × 224; Max pooling 2 × 2 stride 2 giảm còn 112 × 112. GAP lấy trung bình toàn cục tạo vector 16 phần tử trước Dense 8. Đây là sơ đồ giải thích nguyên lý, không phải kiến trúc MobileNetV2 thực tế. Công thức tích chập dùng ảnh đã đệm, theo phép nhân–cộng thường dùng trong CNN.')

fig,ax=canvas(7.8)
ax.text(.5,.97,'CHU TRÌNH HUẤN LUYỆN MẠNG CNN',ha='center',fontsize=16,weight='bold')
xs=[.025,.275,.525,.775]
stages=[('Ảnh và nhãn train','Dữ liệu đã gán nhãn'),
        ('Lan truyền thuận','CNN → giá trị z\nSoftmax → điểm số p'),
        ('Hàm mất mát','So sánh điểm p\nvới nhãn chuẩn y'),
        ('Lan truyền ngược','Tính gradient\ncác trọng số học')]
for x,(title,body) in zip(xs,stages):box(ax,x,.68,.20,.15,title,body,fontsize=11.8,body_size=11.1)
for i in range(3):arrow(ax,(xs[i]+.208,.755),(xs[i+1]-.01,.755))
ax.plot([.125,.125,.625],[.838,.895,.895],color=GREEN,linewidth=1.4)
arrow(ax,(.625,.895),(.625,.84));ax.text(.4,.91,'Nhãn chuẩn',ha='center',fontsize=11,color=GRAY)
box(ax,.675,.39,.30,.15,'Thuật toán tối ưu','Ví dụ: Adam cập nhật\ncác trọng số được phép học',fontsize=12.6,body_size=11.3)
arrow(ax,(.875,.67),(.875,.55))
ax.plot([.666,.375],[.465,.465],color=GREEN,linewidth=1.5)
arrow(ax,(.375,.465),(.375,.67))
ax.text(.52,.58,'Cập nhật rồi lặp\nmini-batch / epoch',ha='center',va='center',fontsize=11.5,color=GRAY,linespacing=1.6)
ax.text(.28,.305,'Cross-entropy với nhãn one-hot',ha='center',fontsize=12.2,weight='bold')
ax.text(.28,.22,r'$\mathcal{L}=-\sum_{k=1}^{8}y_k\log p_k$',ha='center',va='center',fontsize=17)
ax.text(.76,.305,'Nhãn nguyên (TeaLeafAI)',ha='center',fontsize=12.2,weight='bold')
ax.text(.76,.22,r'$\mathcal{L}=-\log p_y$',ha='center',va='center',fontsize=17)
ax.text(.5,.12,'Công thức minh họa loss của một ảnh; quá trình huấn luyện xử lý theo mini-batch.',
        ha='center',fontsize=10.8,color=GRAY)
ax.text(.5,.045,'Validation: theo dõi và chọn checkpoint · Test: đánh giá mô hình.',ha='center',fontsize=11.2,color=GRAY)
save(fig,'cnn-chu-trinh-huan-luyen','Chu trình huấn luyện CNN bằng lan truyền thuận, tính loss, lan truyền ngược và cập nhật trọng số.',
     'Lan truyền ngược tính gradient; thuật toán tối ưu như Adam dùng gradient để cập nhật các tham số được phép huấn luyện. Nhãn chuẩn và điểm dự đoán cùng được dùng để tính loss. Công thức bên trái áp dụng cho nhãn one-hot; bên phải là dạng tương đương với nhãn nguyên, phù hợp Sparse Categorical Crossentropy trong TeaLeafAI. Điểm softmax không phải Accuracy của mô hình. Validation và test không tham gia cập nhật gradient.')

source='Nguồn: Tác giả xây dựng theo nguyên lý CNN; đối chiếu định dạng nhãn và hàm mất mát trong mã huấn luyện TeaLeafAI.'
md=['# Các lớp và quá trình huấn luyện CNN','','Vị trí chèn: mục 2.3.2. Cấu trúc và nguyên lý hoạt động của CNN trong phân loại ảnh.','']
for f in figures:
    md.extend([f'## {f["caption"]}','',f'![{f["caption"]}]({f["stem"]}.png)','',f['note'],''])
md.extend([source,'','PNG 300 dpi; có SVG/PDF. Các sơ đồ minh họa lý thuyết, không phải số liệu thực nghiệm.',''])
(OUT/'cnn-cau-truc-va-huan-luyen.md').write_text('\n'.join(md),encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin,sec.bottom_margin=Cm(2),Cm(2)
sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
for i,f in enumerate(figures):
    if i:doc.add_page_break()
    p=doc.add_paragraph('CẤU TRÚC VÀ QUÁ TRÌNH HUẤN LUYỆN CNN');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
    doc.add_paragraph('Vị trí chèn: 2.3.2. Cấu trúc và nguyên lý hoạt động của CNN.').runs[0].bold=True
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(OUT/f'{f["stem"]}.png'),width=Cm(16))
    p=doc.add_paragraph(f['caption']);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
    doc.add_paragraph(source).runs[0].italic=True
    doc.add_paragraph(f['note'])
doc.save(ROOT/'downloads/TeaLeafAI-cau-truc-va-huan-luyen-CNN.docx')
print('Created two CNN diagrams (PNG 300 dpi, SVG/PDF) and Word handout.')
