# dengue-forecasting-gnns
Dự đoán và trực quan hóa Bệnh sốt xuất huyết ở Brazil ở mô hình Spatio-temporal Graph Attention Networks
Đồ án được xây dựng bởi Huỳnh Lê Thanh Hải

Dengue Forecasting with Graph Neural Networks (GNNs)
Giới thiệu

Đồ án này tập trung vào việc dự báo số ca mắc sốt xuất huyết tại Brazil dựa trên dữ liệu dịch tễ, khí hậu, dân số và liên kết không gian. Phương pháp chính được áp dụng là Graph Neural Networks (GNNs), đặc biệt là Graph Attention Networks (GAT), nhằm khai thác cả yếu tố không gian và thời gian trong dữ liệu.

tải dashboard_chinhthuc về để xem kết  quả cuối cùng

Dataset được sử dụng từ Mosqlimate và các nguồn mở chính thức từ Brazil khác, bao gồm:

Dữ liệu dịch tễ (sốt xuất huyết theo tuần)

Dữ liệu khí hậu (nhiệt độ, mưa, độ ẩm,…)

Dữ liệu dân số (IBGE)

Bản đồ hành chính Brazil (GeoJSON)

Dữ liệu kết nối không gian (REGIC 2018) + bản đồ Brazil geojs-100-mun

vì lí do bảo mật nên không công khai data và data visual

Pipeline dự án

Giai đoạn 1. Xử lý dữ liệu

Làm sạch và chuẩn hóa dữ liệu thô (dịch tễ, khí hậu, dân số).

Kết hợp dữ liệu không gian bằng GeoJSON và REGIC.

Tạo node features (đặc trưng theo thành phố, theo năm).

Tạo edge list bằng:

Contiguity (Queen adjacency)

KNN cho các node cô lập

Xuất dữ liệu dạng tensor (.pt) phục vụ huấn luyện GNN.

Giai đoạn 2. Xây dựng và huấn luyện mô hình

Mô hình chính: Graph Attention Network (GAT) đa-head.

Các bước:

Chuẩn hóa dữ liệu (StandardScaler, RobustScaler, log-transform).

Tạo edge_index, x (features), y (labels).

Tách dữ liệu train/val/test.

Xây dựng module theo cấu trúc:

models/ (mô hình GAT)

trainers/ (hàm huấn luyện, early stopping)

metrics/ (MAE, RMSE, R², MAPE,…)

utils/ (chạy theo năm, chạy toàn cục).

File chạy chính: run_global_gat.py.

Giai đoạn 3. Trực quan hóa và Dashboard

Trực quan kết quả bằng:

Matplotlib, Seaborn (biểu đồ kết quả huấn luyện).

GeoPandas (bản đồ tĩnh).

Plotly (dashboard động).

Dashboard offline:

Play/Pause/Restart timeline.

Điều chỉnh tốc độ phát theo số tuần/giây.

Lớp biểu diễn: Truth, Prediction, Residual, Density, Centroids, Edges, Biome.

Colorbar tự động sắp xếp để tránh chồng lấn.

Thư viện sử dụng

Data Processing

Pandas, NumPy, PyArrow

GeoPandas, Shapely

Scikit-learn

Openpyxl

Machine Learning

PyTorch

PyTorch Geometric (PyG)

Scikit-learn (baseline models)

NetworkX

Visualization

Matplotlib, Seaborn

Plotly (graph_objects, mapbox)

GeoPandas

Tools

argparse

tqdm

Microsoft Office (báo cáo, trình bày)

