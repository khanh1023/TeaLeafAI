export const CLASSES = [
  {
    key: "algal_leaf",
    label: "Bệnh đốm tảo",
    name: "Algal Leaf Spot",
    description: "Nhóm ảnh lá chè có các vùng đốm liên quan đến tác nhân tảo."
  },
  {
    key: "anthracnose",
    label: "Bệnh thán thư",
    name: "Anthracnose",
    description: "Nhóm ảnh lá chè được gán nhãn bệnh thán thư trong dữ liệu huấn luyện."
  },
  {
    key: "bird_eye_spot",
    label: "Bệnh đốm mắt chim",
    name: "Bird Eye Spot",
    description: "Nhóm ảnh lá chè được gán nhãn bệnh đốm mắt chim trong dữ liệu huấn luyện."
  },
  {
    key: "brown_blight",
    label: "Bệnh đốm nâu",
    name: "Brown Blight",
    description: "Nhóm ảnh lá chè có các tổn thương được gán nhãn bệnh đốm nâu."
  },
  {
    key: "gray_light",
    label: "Bệnh đốm xám",
    name: "Gray Light",
    description: "Nhãn gray_light được giữ nguyên theo bộ dữ liệu nguồn; cần xác minh nhãn bệnh đốm xám."
  },
  {
    key: "healthy",
    label: "Lá khỏe mạnh",
    name: "Healthy",
    description: "Nhóm ảnh lá chè khỏe mạnh theo nhãn của dữ liệu huấn luyện."
  },
  {
    key: "red_leaf_spot",
    label: "Bệnh đốm đỏ",
    name: "Red Leaf Spot",
    description: "Nhóm ảnh lá chè được gán nhãn bệnh đốm đỏ trong dữ liệu huấn luyện."
  },
  {
    key: "white_spot",
    label: "Bệnh đốm trắng",
    name: "White Spot",
    description: "Nhóm ảnh lá chè được gán nhãn bệnh đốm trắng trong dữ liệu huấn luyện."
  }
];

export const LABELS = Object.fromEntries(
  CLASSES.map(({ key, label }) => [key, label])
);
