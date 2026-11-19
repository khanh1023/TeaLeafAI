"""Draw a supervised deep-image-classification diagram for report section 1.4.3."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'training/.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = ROOT/'docs/figures/chapter1'
OUT.mkdir(parents=True, exist_ok=True)
STEM = 'phan-loai-anh-su-dung-hoc-sau'
GREEN, GRAY = '#216b50', '#56665e'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':'#183c32',
                     'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
fig, ax = plt.subplots(figsize=(9,7.8))
fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.025)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')


def box(x,y,title,body,fill='#eef5ef'):
    ax.add_patch(FancyBboxPatch((x,y),.20,.16,boxstyle='round,pad=.007,rounding_size=.014',
                               facecolor=fill,edgecolor=GREEN,linewidth=1.5))
    ax.text(x+.10,y+.112,title,ha='center',va='center',fontsize=12.7,weight='bold')
    ax.text(x+.10,y+.045,body,ha='center',va='center',fontsize=12.2,color=GRAY,linespacing=1.55)


def arrow(start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,color=GREEN,linewidth=1.6))


ax.text(.5,.97,'PHÂN LOẠI ẢNH SỬ DỤNG HỌC SÂU',ha='center',fontsize=17,weight='bold')
ax.text(.025,.91,'1. HUẤN LUYỆN CÓ GIÁM SÁT',fontsize=13.3,weight='bold')
xs=[.025,.275,.525,.775]
top=[('Ảnh có nhãn','Ảnh thuộc train\n+ nhãn chuẩn'),
     ('Chuẩn bị ảnh','Đổi kích thước\nTăng cường train'),
     ('Mô hình học sâu','CNN trích đặc trưng\n+ bộ phân loại'),
     ('Tính Loss','So sánh dự đoán\nvới nhãn chuẩn')]
for x,(title,body) in zip(xs,top):box(x,.64,title,body)
for i in range(3):arrow((xs[i]+.208,.72),(xs[i+1]-.01,.72))

# Both predicted outputs and ground-truth labels feed the loss.
ax.plot([.125,.125,.875],[.808,.852,.852],color=GREEN,linewidth=1.4)
arrow((.875,.852),(.875,.81))
ax.text(.5,.865,'Nhãn chuẩn',ha='center',fontsize=11.3,color=GRAY)

# Optimisation updates only parameters that are configured as trainable.
ax.plot([.875,.875,.625],[.631,.465,.465],color=GREEN,linewidth=1.5)
arrow((.625,.465),(.625,.63))
ax.text(.75,.55,'Lan truyền ngược + tối ưu\nCập nhật các trọng số\nđược phép huấn luyện',
        ha='center',va='center',fontsize=11.1,linespacing=1.55)
ax.text(.265,.55,'Lặp qua các epoch\nđể mô hình học từ dữ liệu',
        ha='center',va='center',fontsize=12,color=GRAY,linespacing=1.6)
ax.text(.5,.395,'Validation: theo dõi và chọn checkpoint · Test: đánh giá mô hình',
        ha='center',fontsize=11.2,color=GRAY)
ax.plot([.025,.975],[.365,.365],color='#cad9ce',linewidth=1.2,linestyle='--')

ax.text(.025,.31,'2. SUY LUẬN TRÊN ẢNH MỚI',fontsize=13.3,weight='bold')
bottom=[('Ảnh mới','Ảnh cần phân loại\nChưa biết nhãn'),
        ('Tiền xử lý','Đổi kích thước\nnhư khi huấn luyện'),
        ('Mô hình đã học','Trọng số đã lưu\nĐiểm số softmax'),
        ('Kết quả','Chọn 1 trong 8 lớp\ncó điểm cao nhất')]
for x,(title,body) in zip(xs,bottom):box(x,.105,title,body,fill='#f1f5f8')
for i in range(3):arrow((xs[i]+.208,.185),(xs[i+1]-.01,.185))
ax.text(.5,.035,'Minh họa TeaLeafAI: 7 nhóm bệnh và 1 nhóm lá khỏe mạnh.',
        ha='center',fontsize=11.2,color=GRAY)

for suffix in ['png','svg','pdf']:
    p=OUT/f'{STEM}.{suffix}';fig.savefig(p,dpi=300)
    if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)

caption='Phân loại ảnh sử dụng học sâu trong bài toán nhận diện bệnh lá chè.'
source='Nguồn: Tác giả xây dựng. Sơ đồ minh họa nguyên lý học có giám sát; không biểu diễn các trọng số hoặc số liệu thực nghiệm.'
note='Chèn tại mục 1.4.3. Phân loại ảnh sử dụng học sâu. Chỉ tăng cường ảnh train; giữ quy trình tiền xử lý xác định khi suy luận. Điểm softmax được dùng để chọn lớp dự đoán.'
(OUT/'README.md').write_text(
    '# Phân loại ảnh sử dụng học sâu\n\n'
    f'![{caption}]({STEM}.png)\n\n'
    f'**Vị trí chèn:** mục 1.4.3.\n\n**Chú thích:** {caption}\n\n{source}\n\n'
    'PNG 300 dpi để chèn Word; SVG/PDF là bản vector. Hai luồng tách biệt quá trình huấn luyện và suy luận.\n',encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin,sec.bottom_margin=Cm(2),Cm(2)
sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
style=doc.styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(12)
p=doc.add_paragraph('HÌNH MINH HỌA PHÂN LOẠI ẢNH SỬ DỤNG HỌC SÂU');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph('Vị trí chèn: 1.4.3. Phân loại ảnh sử dụng học sâu.').runs[0].bold=True
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT/f'{STEM}.png'),width=Cm(16))
p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph(source).runs[0].italic=True
doc.add_paragraph(note)
doc.save(ROOT/'downloads/TeaLeafAI-phan-loai-anh-hoc-sau.docx')
print('Created PNG (300 dpi), SVG, PDF and Word handout for section 1.4.3.')
