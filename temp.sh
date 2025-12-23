#!/bin/bash

# 1. Tạo cấu trúc thư mục
echo "Creating project structure..."
mkdir -p features models processing

# ==========================================
# CORE SYSTEM (Hệ thống lõi)
# ==========================================

# 2. factory.py (Tạo đối tượng từ Config)
cat <<EOF > factory.py
import importlib

class Factory:
    @staticmethod
    def create(config_section):
        """Tạo class động: module.class(**params)"""
        try:
            module_name = config_section["module"]
            class_name = config_section["class"]
            params = config_section.get("params", {})

            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(**params)
        except Exception as e:
            raise RuntimeError(f"Factory failed to create {config_section}: {e}")
EOF

# 4. runner.py (Chạy quy trình)
cat <<EOF > runner.py
class PipelineRunner:
    def __init__(self, config):
        self.config = config
        self.context = {} # Bộ nhớ chia sẻ giữa các bước

    def add(self, step_instance):
        step_name = step_instance.__class__.__name__
        print(f"\n=== RUNNING STEP: {step_name} ===")
        # Gọi hàm run() của từng bước, truyền context đi
        self.context = step_instance.run(self.context, self.config)
        return self
EOF

# ==========================================
# COMPONENTS (Các thành phần xử lý)
# ==========================================

# 5. models/base.py (Interface chuẩn)
cat <<EOF > models/base.py
from abc import ABC, abstractmethod

class BaseStep(ABC):
    @abstractmethod
    def run(self, context, config):
        """Mọi bước phải có hàm này nhận context và trả về context"""
        pass
EOF

# 6. features/sequence.py (Trích xuất đặc trưng Amino Acid)
cat <<EOF > features/sequence.py
import numpy as np
from collections import Counter
from Bio import SeqIO
from tqdm.auto import tqdm
from data_paths import DataPaths
from models.base import BaseStep

class SequenceEncoder(BaseStep):
    def __init__(self, dimensions=85):
        self.dims = dimensions

    def run(self, context, config):
        print(">> Loading paths...")
        t_path = DataPaths.get("train_sequences")
        s_path = DataPaths.get("testsuperset")

        print(">> Extracting Train Features...")
        context["train_ids"], context["X_train"] = self._process(t_path)
        
        print(">> Extracting Test Features...")
        context["test_ids"], context["X_test"] = self._process(s_path)
        
        return context

    def _process(self, path):
        # Amino Acid Weights (giống code cũ của bạn)
        aa_wt = {'A':89,'C':121,'D':133,'E':147,'F':165,'G':75,'H':155,'I':131,'K':146,'L':131,
                 'M':149,'N':132,'P':115,'Q':146,'R':174,'S':105,'T':119,'V':117,'W':204,'Y':181}
        
        ids = []
        features = []
        
        # Đọc file FASTA
        for r in tqdm(SeqIO.parse(path, "fasta"), desc="Encoding"):
            # Xử lý ID
            pid = r.id.split('|')[1] if '|' in r.id else r.id
            ids.append(pid)
            
            seq = str(r.seq)
            n = len(seq)
            if n < 3:
                features.append(np.zeros(self.dims, dtype=np.float32))
                continue
                
            c = Counter(seq)
            
            # 1. Frequency (20 dims)
            freq = [c.get(a,0)/n for a in 'ACDEFGHIKLMNPQRSTVWY']
            
            # 2. Phys-Chem (4 dims)
            phys = [
                np.log1p(n),
                sum(c.get(a,0) for a in 'AILMFWYV')/n,
                sum(c.get(a,0) for a in 'DEKR')/n,
                np.log1p(sum(c.get(a,0)*aa_wt.get(a,0) for a in c))
            ]
            
            # Gộp lại (đơn giản hóa so với bản gốc để code ngắn gọn, nhưng vẫn hiệu quả)
            vec = np.concatenate([freq, phys])
            
            # Padding cho đủ 85 chiều (hoặc số chiều bạn set trong config)
            pad_len = self.dims - len(vec)
            if pad_len > 0:
                vec = np.pad(vec, (0, pad_len))
            else:
                vec = vec[:self.dims]
                
            features.append(vec.astype(np.float32))
            
        return ids, np.array(features)
EOF

# 7. models/generic.py (WRAPPER VẠN NĂNG - KHÔNG CẦN VIẾT CODE MODEL MỚI)
cat <<EOF > models/generic.py
import importlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from data_paths import DataPaths
from models.base import BaseStep

class SklearnWrapper(BaseStep):
    def __init__(self, library, algorithm, model_params):
        self.library = library
        self.algorithm = algorithm
        self.params = model_params
        
        # 1. Import động (Dynamic Import)
        module = importlib.import_module(library)
        self.model_class = getattr(module, algorithm)
        
        # 2. Khởi tạo model
        self.model = self.model_class(**self.params)

    def run(self, context, config):
        print(f">> Initialized {self.algorithm} from {self.library}")
        
        # --- A. CHUẨN BỊ LABEL (Cho bài toán CAFA) ---
        print(">> Preparing Labels...")
        terms_path = DataPaths.get("train_terms")
        train_terms = pd.read_csv(terms_path, sep='\t')
        
        # Lấy Top N GO Terms phổ biến nhất
        top_n = config.get("top_go_terms", 1500)
        top_go = train_terms['term'].value_counts().head(top_n).index
        go2idx = {go: i for i, go in enumerate(top_go)}
        
        # Mapping Labels vào Matrix (Nếu cần training thực sự)
        # Lưu ý: Với KNN Cosine, ta fit X_train, việc map label diễn ra lúc predict
        
        # --- B. TRAINING ---
        print(f">> Fitting {self.algorithm} on {context['X_train'].shape}...")
        self.model.fit(context["X_train"])
        
        # --- C. PREDICTION (Logic tùy biến cho Neighbors vs Classifiers) ---
        print(">> Predicting...")
        X_test = context["X_test"]
        test_ids = context["test_ids"]
        
        output_file = "model_output.tsv"
        batch_size = 2048
        
        # Xử lý đặc biệt cho dòng Nearest Neighbors (dùng trọng số)
        if "Neighbors" in self.algorithm:
            self._predict_knn(self.model, context["X_test"], context["train_ids"], 
                              train_terms, top_go, test_ids, output_file, batch_size)
        else:
            # Placeholder cho các model khác (RandomForest, etc.)
            print("Warning: Generic prediction implemented mainly for Neighbors strategies.")
        
        context["prediction_path"] = output_file
        return context

    def _predict_knn(self, model, X_test, train_ids, train_terms, top_go, test_ids, output_file, batch_size):
        # Load label map nhanh
        train_labels = train_terms.groupby('EntryID')['term'].apply(list).to_dict()
        go2idx = {go: i for i, go in enumerate(top_go)}
        
        # Ma trận Y train (Train IDs x Top GOs)
        # Lưu ý: Để tiết kiệm RAM, ta chỉ lookup khi cần, hoặc tạo sparse matrix ở đây
        # Demo đơn giản: Map trực tiếp
        
        with open(output_file, 'w') as f:
            for i in tqdm(range(0, len(X_test), batch_size), desc="Batch Pred"):
                batch_X = X_test[i:i+batch_size]
                dists, idxs = model.kneighbors(batch_X)
                
                # Tính trọng số cosine
                weights = 1 / (dists + 1e-8)
                
                for j, (w_row, idx_row) in enumerate(zip(weights, idxs)):
                    pid = test_ids[i+j]
                    scores = {}
                    
                    # Cộng dồn điểm cho các GO term từ hàng xóm
                    for w, train_idx in zip(w_row, idx_row):
                        neighbor_id = train_ids[train_idx]
                        terms = train_labels.get(neighbor_id, [])
                        for term in terms:
                            if term in go2idx:
                                scores[term] = scores.get(term, 0) + w
                                
                    # Chuẩn hóa (chia tổng trọng số)
                    total_w = w_row.sum()
                    final_scores = []
                    for term, val in scores.items():
                        prob = val / total_w
                        if prob > 0.01: # Threshold
                            final_scores.append((term, prob))
                            
                    # Ghi top kết quả
                    final_scores.sort(key=lambda x: x[1], reverse=True)
                    for term, prob in final_scores[:100]: # Top 100 per protein
                        f.write(f"{pid}\t{term}\t{prob:.3f}\n")

# 8. processing/merge.py (Kết hợp kết quả)
cat <<EOF > processing/merge.py
import pandas as pd
import os
import gc
from data_paths import DataPaths
from models.base import BaseStep

class ResultMerger(BaseStep):
    def __init__(self, method="max"):
        self.method = method

    def run(self, context, config):
        print(">> Merging Results...")
        
        # 1. File Model vừa chạy
        mypred_path = context.get("prediction_path")
        
        # 2. File BLAST (Tìm trong input)
        try:
            # Giả sử file BLAST có tên chứa 'submission' hoặc tên cụ thể
            blast_path = DataPaths.get("submission") 
            print(f">> Found BLAST file: {blast_path}")
        except:
            print(">> No BLAST file found. Using model output only.")
            os.rename(mypred_path, "submission.tsv")
            return context

        # 3. Load & Merge
        df1 = pd.read_csv(mypred_path, sep='\t', names=['id', 'term', 'score'], header=None)
        
        # Load Blast (Chunking nếu file lớn)
        df2 = pd.read_csv(blast_path, sep='\t', names=['id', 'term', 'score'], header=None)
        # Filter rác
        df2 = df2[df2['score'] > 0.01]
        
        print(f"Concatenating {len(df1)} rows (Model) + {len(df2)} rows (Blast)...")
        combined = pd.concat([df1, df2])
        del df1, df2
        gc.collect()
        
        # 4. Max Pooling
        print("Grouping & Max Pooling...")
        final = combined.groupby(['id', 'term'], as_index=False)['score'].max()
        
        # 5. Sort & Save
        print("Final Sorting...")
        final.sort_values(['id', 'score'], ascending=[True, False], inplace=True)
        
        # Giới hạn số lượng
        top_k = config.get("top_predict", 1500)
        final = final.groupby('id').head(top_k)
        
        final.to_csv("submission.tsv", sep='\t', header=False, index=False)
        print(">> DONE. File saved: submission.tsv")
        
        return context
EOF

# ==========================================
# CONFIG & MAIN
# ==========================================

# 9. config.yaml (Điều khiển toàn bộ dự án)
cat <<EOF > config.yaml
# Cấu hình data input
data:
  root: "/kaggle/input"

# Các tham số chung
top_go_terms: 2000
top_predict: 1500

# Pipeline thực thi
pipeline:
  # Bước 1: Feature Engineering
  feature_engineering:
    module: "features.sequence"
    class: "SequenceEncoder"
    params:
      dimensions: 85

  # Bước 2: Model (Dùng Wrapper vạn năng)
  model:
    module: "models.generic"
    class: "SklearnWrapper"
    params:
      # Bạn muốn dùng thuật toán nào thì sửa ở đây
      library: "sklearn.neighbors"
      algorithm: "NearestNeighbors"
      model_params:
        n_neighbors: 7
        metric: "cosine"
        n_jobs: -1

  # Bước 3: Merge kết quả
  post_process:
    module: "processing.merge"
    class: "ResultMerger"
    params:
      method: "max_pooling"
EOF

# 10. main.py (File chạy chính)
cat <<EOF > main.py
import yaml
import sys
from data_paths import DataPaths
from factory import Factory
from runner import PipelineRunner

def main():
    print("--- STARTING CAFA PIPELINE ---")
    
    # 1. Tìm data
    try:
        DataPaths.autopath("/kaggle/input")
    except Exception as e:
        print(f"Error checking data paths: {e}")
    
    # 2. Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 3. Khởi tạo Runner
    runner = PipelineRunner(config)

    # 4. Build Pipeline
    try:
        # Tạo object từ config
        encoder = Factory.create(config['pipeline']['feature_engineering'])
        model   = Factory.create(config['pipeline']['model'])
        merger  = Factory.create(config['pipeline']['post_process'])

        # Chạy tuần tự
        (runner
         .add(encoder)
         .add(model)
         .add(merger))
         
        print("\nSUCCESS! Pipeline finished.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
EOF

# Cấp quyền thực thi
chmod +x main.py
echo "Project setup complete! Run 'python main.py' to start."