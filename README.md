# CTD2026_DT049

## Thông Tin Đề Tài

**Tên đề tài:** Xây dựng mô hình chấm điểm ESG cho nhà bán hàng lĩnh vực thời trang trên sàn thương mại điện tử bằng Machine Learning

**Mã đề tài:** DT049

**Đơn vị:** Trường Công nghệ và Thiết kế, Đại học Kinh tế Thành phố Hồ Chí Minh

**Giải thưởng:** Giải thưởng Tinh hoa Học thuật CTD 2026

## Thành Viên Nhóm

| STT | Họ và tên | MSSV |
|---:|---|---|
| 1 | Nguyễn Thị Thúy Vân | 31231023797 |
| 2 | Lê Bảo Ngọc | 31231027188 |
| 3 | Phạm Thị Ngọc Diệu | 31231026760 |
| 4 | Nguyễn Võ Lan Thanh | 31231026118 |
| 5 | Bùi Trọng Nguyên | 31231027037 |

## Giới Thiệu Dự Án

Repository này chứa toàn bộ mã nguồn, notebook xử lý dữ liệu, dữ liệu đã xử lý, kết quả mô hình và dashboard minh họa cho đề tài nghiên cứu khoa học về chấm điểm ESG cho nhà bán hàng thời trang trên sàn thương mại điện tử.

Đề tài xuất phát từ khoảng trống khi các hệ thống đánh giá nhà bán hàng hiện nay chủ yếu dựa trên các chỉ số vận hành như rating, lượt bán, phản hồi khách hàng hoặc tốc độ xử lý đơn hàng. Những chỉ số này phản ánh hiệu quả thương mại, nhưng chưa đo lường trực tiếp trách nhiệm môi trường, xã hội và quản trị của nhà bán hàng. Vì vậy, nhóm nghiên cứu đề xuất một quy trình chuyển hóa dữ liệu sản phẩm và đánh giá khách hàng thành các tín hiệu ESG có thể đo lường, tổng hợp và mô hình hóa bằng Machine Learning.

Bối cảnh thực nghiệm của dự án là dữ liệu sản phẩm và review trên Amazon, tập trung vào ngành hàng thời trang. Kết quả nghiên cứu được sử dụng như một prototype phương pháp luận, có thể mở rộng sang các nền tảng thương mại điện tử khác trong tương lai.

## Mục Tiêu Nghiên Cứu

Đề tài hướng đến các mục tiêu chính sau:

1. Xây dựng bộ chỉ báo ESG phù hợp với nhà bán hàng thời trang trên sàn thương mại điện tử.
2. Xây dựng ESG Knowledge Base từ báo cáo ESG, dữ liệu vật liệu thời trang, nhãn sinh thái và đánh giá khách hàng.
3. Ánh xạ các chỉ báo ESG sang dữ liệu Amazon thông qua ESG Operationalization và Feature Engineering.
4. Tính điểm ESG tham chiếu ở cấp nhà bán hàng bằng phương pháp AHP.
5. Huấn luyện và so sánh các mô hình Machine Learning để dự báo điểm ESG.
6. Xây dựng dashboard Streamlit nhằm trực quan hóa điểm ESG, điểm thành phần E-S-G và các chỉ báo liên quan.

## Cấu Trúc Repository

```text
CTD2026_DT049/
├── Dataset/
│   └── processed/
│       ├── ESG_Features/
│       │   ├── environment_feature_dataset.csv
│       │   └── social_governance_feature_dataset.csv
│       ├── ESG_Knowledge_Base/
│       │   ├── candidate_esg_indicators.csv
│       │   ├── eco_label_dictionary.csv
│       │   ├── esg_keyword_dictionary.csv
│       │   ├── fashion_esg_complaint_ontology.csv
│       │   └── sustainable_material_dictionary.csv
│       ├── ESG_Operationalization/
│       │   ├── feature_mapping.csv
│       │   └── feature_specification.csv
│       ├── cleaned_amazon_dataset.csv
│       ├── environment_feature_dataset.csv
│       ├── seller_esg_feature_dataset.csv
│       └── social_governance_feature_dataset.csv
├── ESG_Modeling/
│   ├── ahp_consistency_results.csv
│   ├── ahp_fixed_weights.csv
│   ├── ahp_indicator_mapping.csv
│   ├── feature_importance.csv
│   ├── indicator_weight.csv
│   ├── model_metadata.json
│   ├── model_performance.csv
│   ├── ridge_best_params.csv
│   ├── ridge_tuning_results.csv
│   ├── seller_esg_model.pkl
│   └── seller_esg_reference_dataset.csv
├── notebooks/
│   ├── 00_merge_dataset.ipynb
│   ├── 01_preEDA_cleaning_preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_build_esg_knowledge_base.ipynb
│   ├── 04_esg_operationalization.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_seller_esg_modeling.ipynb
├── requirements_streamlit.txt
├── streamlit_app.py
└── README.md
```

## Nguồn Dữ Liệu

Nghiên cứu sử dụng nhiều bộ dữ liệu thứ cấp để xây dựng quy trình chấm điểm ESG:

| Bộ dữ liệu | Vai trò trong nghiên cứu |
|---|---|
| Amazon Products & Reviews Dataset | Dữ liệu chính để trích xuất đặc trưng ESG ở cấp sản phẩm, review và nhà bán hàng |
| Amazon Fashion Reviews Dataset | Corpus review phục vụ xây dựng Complaint Dictionary và nhận diện tín hiệu trải nghiệm khách hàng |
| ESG Sustainability Reports Dataset | Corpus chuyên ngành để xây dựng ESG Keyword Dictionary |
| Fast Fashion Eco Dataset | Nguồn dữ liệu phục vụ xây dựng Sustainable Material Dictionary và Eco Label Dictionary |

Sau quá trình làm sạch và tổng hợp, bộ dữ liệu seller-level gồm **261 nhà bán hàng** và **9 đặc trưng ESG**, đại diện cho ba chiều Environmental, Social và Governance.

## Quy Trình Nghiên Cứu

### 1. Gộp Dữ Liệu Và Tiền Xử Lý

Dữ liệu sản phẩm và dữ liệu review được gộp thông qua mã định danh sản phẩm. Bước này giúp liên kết thông tin nhà bán hàng, mô tả sản phẩm, rating, review text, sentiment score, verified purchase và helpful vote trong cùng một cấu trúc dữ liệu.

Notebook chính:

```text
notebooks/00_merge_dataset.ipynb
notebooks/01_preEDA_cleaning_preprocessing.ipynb
```

### 2. Phân Tích Khám Phá Dữ Liệu

Giai đoạn EDA được thực hiện nhằm mô tả cấu trúc dữ liệu, phân phối danh mục sản phẩm, số lượng review, tình trạng thiếu dữ liệu, phân phối rating và đặc điểm văn bản review. Kết quả EDA cho thấy nhóm Clothing, Shoes & Jewelry chiếm tỷ trọng chủ đạo, phù hợp với trọng tâm nghiên cứu về nhà bán hàng thời trang.

Notebook chính:

```text
notebooks/02_EDA.ipynb
```

### 3. Xây Dựng ESG Knowledge Base

ESG Knowledge Base được xây dựng nhằm chuẩn hóa nguồn tri thức phục vụ nhận diện tín hiệu ESG từ dữ liệu sản phẩm và đánh giá khách hàng. Kết quả gồm bốn nhóm từ điển:

- `esg_keyword_dictionary.csv`
- `sustainable_material_dictionary.csv`
- `eco_label_dictionary.csv`
- `fashion_esg_complaint_ontology.csv`

Phương pháp xử lý gồm Text Preprocessing, Keyword Extraction, Frequency Analysis, Dictionary/Lexicon-based Matching và Rule-based Pattern Matching.

Notebook chính:

```text
notebooks/03_build_esg_knowledge_base.ipynb
```

### 4. ESG Operationalization

Các chỉ báo ESG ứng viên được đánh giá theo bốn tiêu chí:

1. Có cơ sở từ tài liệu hoặc ESG Knowledge Base.
2. Có tín hiệu quan sát được trong dữ liệu Amazon.
3. Có khả năng chuyển thành biến định lượng.
4. Phù hợp với mục tiêu chấm điểm ESG ở cấp nhà bán hàng.

Kết quả của bước này gồm:

```text
Dataset/processed/ESG_Operationalization/feature_mapping.csv
Dataset/processed/ESG_Operationalization/feature_specification.csv
```

Notebook chính:

```text
notebooks/04_esg_operationalization.ipynb
```

### 5. Feature Engineering Và Xây Dựng Seller-Level Dataset

Thông tin sản phẩm và review được chuyển thành các biến định lượng đại diện cho tín hiệu ESG. Bộ đặc trưng cuối cùng gồm 9 biến:

| Chiều ESG | Đặc trưng |
|---|---|
| Environmental | `sustainable_material_count`, `eco_label_count`, `environmental_keyword_count` |
| Social | `product_quality_complaint_count`, `product_damage_complaint_count`, `product_safety_complaint_count` |
| Governance | `customer_service_complaint_count`, `counterfeit_complaint_count`, `governance_keyword_count` |

Các đặc trưng được tổng hợp từ cấp bản ghi sản phẩm-review lên cấp nhà bán hàng bằng giá trị trung bình theo `seller_name`. Đối với các biến khiếu nại dạng nhị phân, giá trị trung bình được diễn giải như tỷ lệ bản ghi của nhà bán hàng có xuất hiện loại khiếu nại tương ứng.

Notebook chính:

```text
notebooks/05_feature_engineering.ipynb
```

### 6. Tính Seller ESG Score Và Huấn Luyện Mô Hình

Điểm ESG tham chiếu được tính dựa trên trọng số AHP của ba chiều E-S-G và các chỉ báo thành phần. Sau đó, dữ liệu seller-level được dùng để huấn luyện các mô hình hồi quy nhằm dự báo điểm ESG.

Các mô hình được đánh giá:

- Linear Regression
- Ridge Regression
- XGBoost Regressor
- Random Forest Regressor

Notebook chính:

```text
notebooks/06_seller_esg_modeling.ipynb
```

## Kết Quả Mô Hình

Các mô hình được đánh giá bằng R2, MSE và MAE.

| Mô hình | R2 | MSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 1.000000 | 6.638e-31 | 2.000e-16 |
| Ridge Regression | 0.961805 | 0.000165 | 0.001782 |
| XGBoost Regressor | 0.878778 | 0.000524 | 0.006902 |
| Random Forest Regressor | 0.786380 | 0.000924 | 0.008927 |

Kết quả cho thấy Linear Regression đạt hiệu quả cao nhất trong cấu trúc dữ liệu hiện tại. Điều này phù hợp với cách xây dựng biến mục tiêu `ESG_reference_score`, vì điểm ESG tham chiếu được tính từ tổ hợp tuyến tính có trọng số của chính các đặc trưng đầu vào. Do đó, kết quả chủ yếu xác nhận tính nhất quán của quy trình chuẩn hóa, gán trọng số và tính điểm ESG, chưa phải bằng chứng xác thực độc lập về hiệu quả ESG thực tế của nhà bán hàng.

## Dashboard Streamlit

Repository có tích hợp dashboard Streamlit với tên **Seller ESG Dashboard**. Dashboard hỗ trợ:

- Hiển thị điểm ESG tổng hợp của từng nhà bán hàng.
- Hiển thị điểm thành phần Environmental, Social và Governance.
- Xếp hạng nhà bán hàng theo điểm ESG.
- Phân tích đóng góp của từng chỉ báo.
- So sánh nhiều nhà bán hàng.
- Gợi ý các khía cạnh cần ưu tiên cải thiện.

File chính:

```text
streamlit_app.py
```

## Cài Đặt Môi Trường

Clone repository:

```bash
git clone https://github.com/NguynnThah/CTD2026_DT049.git
cd CTD2026_DT049
```

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

Kích hoạt môi trường ảo trên macOS/Linux:

```bash
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install -r requirements_streamlit.txt
```

Các thư viện chính:

```text
streamlit>=1.37
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
```

## Chạy Dashboard

Chạy ứng dụng Streamlit từ thư mục gốc của project:

```bash
streamlit run streamlit_app.py
```

Ứng dụng cần thư mục dữ liệu đã xử lý tại:

```text
Dataset/processed/
```

## Thứ Tự Chạy Notebook

Để tái lập toàn bộ pipeline, chạy notebook theo thứ tự:

```text
00_merge_dataset.ipynb
01_preEDA_cleaning_preprocessing.ipynb
02_EDA.ipynb
03_build_esg_knowledge_base.ipynb
04_esg_operationalization.ipynb
05_feature_engineering.ipynb
06_seller_esg_modeling.ipynb
```

## Các Output Chính

| Output | Mô tả |
|---|---|
| `cleaned_amazon_dataset.csv` | Bộ dữ liệu Amazon đã làm sạch |
| `seller_esg_feature_dataset.csv` | Bộ dữ liệu đặc trưng ESG ở cấp nhà bán hàng |
| `seller_esg_reference_dataset.csv` | Bộ dữ liệu có điểm ESG tham chiếu |
| `seller_esg_model.pkl` | Mô hình dự báo điểm ESG đã lưu |
| `model_performance.csv` | Kết quả đánh giá mô hình |
| `feature_importance.csv` | Kết quả đóng góp hoặc tầm quan trọng của đặc trưng |
| `model_metadata.json` | Metadata phục vụ tái lập mô hình |

## Hạn Chế

Đây là một prototype nghiên cứu. Điểm `ESG_reference_score` được xây dựng nội sinh từ bộ chỉ báo và trọng số AHP, chưa phải điểm ESG độc lập từ chuyên gia hoặc tổ chức xếp hạng bên ngoài.

Ngoài ra, một số yếu tố ESG quan trọng như phát thải trong sản xuất, điều kiện lao động, nguồn năng lượng, kiểm toán chuỗi cung ứng hoặc cơ chế quản trị nội bộ không thể quan sát trực tiếp từ dữ liệu sản phẩm và review. Vì vậy, mô hình nên được hiểu là phương pháp khai thác tín hiệu ESG gián tiếp từ dữ liệu thương mại điện tử, không thay thế hoàn toàn cho kiểm toán ESG chính thức.

## Hướng Phát Triển

Các hướng phát triển tiếp theo gồm:

- Mở rộng dữ liệu sang các sàn thương mại điện tử tại Việt Nam như Shopee, Lazada, TikTok Shop hoặc Tiki.
- Bổ sung nhãn ESG độc lập từ chuyên gia, khảo sát người tiêu dùng hoặc chứng nhận bên thứ ba.
- Ứng dụng các mô hình NLP nâng cao cho dữ liệu review tiếng Việt.
- Bổ sung SHAP hoặc các phương pháp giải thích mô hình để tăng tính minh bạch.
- Phát triển dashboard thành hệ thống theo dõi ESG cho nhà bán hàng theo thời gian.

## Ghi Chú

Repository này phục vụ mục đích nghiên cứu khoa học sinh viên. Các kết quả mô hình cần được diễn giải trong phạm vi dữ liệu và phương pháp nghiên cứu đã nêu.
