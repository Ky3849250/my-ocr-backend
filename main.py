from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rapidocr_onnxruntime import RapidOCR

# 啟動 API 伺服器與 Umi-OCR 的核心引擎
app = FastAPI()
ocr = RapidOCR()

# 允許任何網頁連線到你的伺服器 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "OCR 伺服器運行中！"}

@app.post("/api/ocr")
async def do_ocr(file: UploadFile = File(...)):
    # 讀取上傳的檔案
    contents = await file.read()
    
    # 進行 OCR 辨識
    result, _ = ocr(contents)
    
    # 整理辨識結果 (包含座標、文字、信心度)
    output_data = []
    if result:
        for item in result:
            box = item[0]   # 四個角的座標
            text = item[1]  # 辨識出的文字
            output_data.append({"text": text, "box": box})
            
    return {"status": "success", "data": output_data}
