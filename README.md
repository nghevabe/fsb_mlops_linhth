# ML Ops Project - DDM501

## Hướng dẫn chạy

### Huấn luyện mô hình và log với MLflow:
```bash
python train.py
```

### Chạy Flask app:
```bash
python app.py
```

### Gửi request thử nghiệm:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"features": [0.1, 0.2, ..., 0.5]}'
```

### Docker:
```bash
docker build -t mse_ddm501 .
docker run -p 5000:5000 mse_ddm501
```