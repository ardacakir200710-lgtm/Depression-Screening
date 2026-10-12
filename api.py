from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Hassas Psikolojik Ön Tarama API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm domainlere izin ver (Geliştirme aşaması için)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Model belleğe sunucu başlarken 1 kez yüklenir
kayitli_model = joblib.load('hassas_risk_modeli.joblib')

# 1. 14 Depresyon Sorusunun Şeması
class AnketCevaplari(BaseModel):
    Q3A: int
    Q5A: int
    Q10A: int
    Q13A: int
    Q16A: int
    Q17A: int
    Q21A: int
    Q24A: int
    Q26A: int
    Q31A: int
    Q34A: int
    Q37A: int
    Q38A: int
    Q42A: int

@app.post("/api/risk-hesapla")
def risk_hesapla(cevaplar: AnketCevaplari):
    # 2. Gelen JSON verisini, modelin beklediği Pandas DataFrame formatına çeviriyoruz
    # dict() metodu, Pydantic modelini doğrudan sözlüğe çevirir
    veri_sozlugu = cevaplar.dict()
    kullanici_verisi = pd.DataFrame([veri_sozlugu])
    
    # 3. Puanlama Düzeltmesi (1-4 aralığını 0-3 aralığına çekiyoruz)
    kullanici_verisi = kullanici_verisi - 1

    # 4. Modeli kullanarak 5 sınıfın olasılık matrisini alıyoruz
    # [0] indeksi ile sadece ilk (ve tek) kullanıcının sonuç dizisini seçiyoruz
    ihtimaller = kayitli_model.predict_proba(kullanici_verisi)[0]

    # 5. Sonuçları frontend için yüzdelik dilimlere çevirip paketliyoruz
    return {
        "durum": "basarili",
        "analiz_sonuclari": {
            "normal_ihtimali": round(ihtimaller[0] * 100, 2),
            "hafif_risk_ihtimali": round(ihtimaller[1] * 100, 2),
            "orta_risk_ihtimali": round(ihtimaller[2] * 100, 2),
            "siddetli_risk_ihtimali": round(ihtimaller[3] * 100, 2),
            "cok_siddetli_risk_ihtimali": round(ihtimaller[4] * 100, 2)
        }
    }