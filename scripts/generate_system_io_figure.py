"""Draw system input/output while distinguishing inference, training and evaluation."""
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
STEM='dau-vao-dau-ra-he-thong'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#183c32',
                     'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(9,8.2))
fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.025)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
GREEN='#216b50';GRAY='#56665e'


def box(x,y,w,h,title,body,fill='#eef5ef',title_size=12.7,body_size=11.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.014',
                               facecolor=fill,edgecolor=GREEN,linewidth=1.5))
    ax.text(x+w/2,y+h*.70,title,ha='center',va='center',fontsize=title_size,weight='bold')
    ax.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=body_size,color=GRAY,linespacing=1.5)


def arrow(start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,color=GREEN,linewidth=1.6))


ax.text(.5,.972,'ĐẦU VÀO VÀ ĐẦU RA CỦA HỆ THỐNG PHÂN LOẠI LÁ CHÈ',ha='center',fontsize=14.5,weight='bold')
ax.text(.025,.905,'1. DỰ ĐOÁN MỘT ẢNH DO NGƯỜI DÙNG CUNG CẤP',fontsize=12.8,weight='bold')
xs=[.025,.275,.525,.775]
main=[('Ảnh đầu vào','JPG/JPEG · PNG\nWEBP · Tối đa 8 MB'),
      ('Tiền xử lý','RGB · 224 × 224\nfloat32 · Pixel 0–255'),
      ('Mô hình CNN','MobileNetV2\nCó lớp Rescaling'),
      ('Đầu ra dự đoán','Nhãn có điểm cao nhất\nĐiểm số của 8 nhóm')]
for x,(title,body) in zip(xs,main):box(x,.70,.20,.15,title,body,body_size=10.8)
for i in range(3):arrow((xs[i]+.208,.775),(xs[i+1]-.01,.775))
ax.text(.13,.666,'Tải ảnh hoặc chụp camera',ha='center',fontsize=10.2,color=GRAY)

ax.text(.025,.56,'2. TRONG GIAI ĐOẠN HUẤN LUYỆN',fontsize=12.8,weight='bold')
box(.025,.39,.25,.12,'Ảnh train + nhãn','Nhãn mục tiêu thuộc 8 lớp',fill='#eff4f8',title_size=12.3,body_size=11)
box(.38,.39,.25,.12,'Huấn luyện CNN','Học các trọng số cho phép',fill='#eff4f8',title_size=12.3,body_size=11)
box(.725,.39,.25,.12,'Trọng số đã học','Được lưu cho suy luận',fill='#eff4f8',title_size=12.3,body_size=11)
arrow((.285,.45),(.37,.45));arrow((.64,.45),(.715,.45))
# Training weights are loaded for inference, not supplied as labels by the website user.
ax.plot([.85,.85,.625],[.518,.61,.61],color=GREEN,linewidth=1.5)
arrow((.625,.61),(.625,.69))
ax.text(.775,.632,'Nạp trọng số',ha='center',fontsize=11.2,color=GRAY)

ax.text(.025,.32,'3. ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ CÓ NHÃN',fontsize=12.8,weight='bold')
box(.025,.10,.25,.16,'Ảnh test + nhãn','Giữ riêng để đánh giá',fill='#f8f4e9',title_size=12.3)
box(.38,.10,.25,.16,'Đánh giá mô hình','Dự đoán / so sánh\nvới nhãn test',fill='#f8f4e9',title_size=12.3)
box(.725,.10,.25,.16,'Đầu ra đánh giá','Accuracy\nPrecision · Recall · F1\nMa trận nhầm lẫn',fill='#f8f4e9',title_size=12.3,body_size=11)
arrow((.285,.18),(.37,.18));arrow((.64,.18),(.715,.18))
ax.text(.5,.035,'Accuracy, Precision, Recall và F1 được tính trên tập có nhãn, không phải cho một ảnh riêng lẻ.',
        ha='center',fontsize=10.2,color=GRAY)
for suffix in ['png','svg','pdf']:
    p=OUT/f'{STEM}.{suffix}';fig.savefig(p,dpi=300)
    if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)

caption='Sơ đồ đầu vào và đầu ra của hệ thống phân loại bệnh lá chè trong dự đoán, huấn luyện và đánh giá.'
source=('Nguồn: Tác giả xây dựng từ frontend/src/api.js, frontend/src/image-preprocessing.js '
        'và các chương trình huấn luyện, đánh giá của TeaLeafAI.')
note=('Chèn tại mục 2.1.3. Đầu vào, đầu ra và yêu cầu hệ thống. '
      'Người dùng cung cấp ảnh khi dự đoán; nhãn chuẩn được dùng trong huấn luyện và đánh giá. '
      'Rescaling nằm bên trong mô hình, nên đầu vào được giữ trong khoảng 0–255. '
      'Điểm dự đoán của từng nhóm không đồng nhất với Accuracy. Các chỉ số và ma trận nhầm lẫn là kết quả đánh giá trên một tập ảnh có nhãn. '
      'Kết quả chỉ hỗ trợ nhận diện, cần đối chiếu với tình trạng thực tế của cây.')
(OUT/f'{STEM}.md').write_text(
    '# Đầu vào và đầu ra của hệ thống\n\n'
    f'![{caption}]({STEM}.png)\n\n**Vị trí chèn:** mục 2.1.3.\n\n'
    f'**Chú thích:** {caption}\n\n{source}\n\n{note}\n\nPNG 300 dpi; có SVG/PDF.\n',encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin,sec.bottom_margin=Cm(2),Cm(2)
sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
p=doc.add_paragraph('ĐẦU VÀO VÀ ĐẦU RA CỦA HỆ THỐNG');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph('Vị trí chèn: 2.1.3. Đầu vào, đầu ra và yêu cầu hệ thống.').runs[0].bold=True
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT/f'{STEM}.png'),width=Cm(16))
p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph(source).runs[0].italic=True
doc.add_paragraph(note)
doc.save(ROOT/'downloads/TeaLeafAI-dau-vao-dau-ra.docx')
print('Created input/output diagram: PNG 300 dpi, SVG/PDF and Word.')
