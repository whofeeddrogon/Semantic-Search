"""
restore_snapshot.py — Eski collection'ı silip snapshot'tan yeni data yükler.

Kullanım:
    python restore_snapshot.py

Adımlar:
    1. Mevcut 'products' collection'ını siler (varsa)
    2. Snapshot dosyasını Qdrant'a upload eder
    3. Yeni collection'ın durumunu kontrol eder
"""

import os
import sys
import time
import requests

# ── Ayarlar ─────────────────────────────────────────────
QDRANT_URL = "http://localhost:6333"
SNAPSHOT_FILE = "products-553830950559874-2026-02-06-21-11-04.snapshot"
COLLECTION_NAME = "products"  # Snapshot bu collection'dan alınmış


def wait_for_qdrant(timeout: int = 60) -> bool:
    """Qdrant hazır olana kadar bekle."""
    print(f"⏳ Qdrant'a bağlanılıyor ({QDRANT_URL})...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{QDRANT_URL}/healthz", timeout=3)
            if r.status_code == 200:
                print("✅ Qdrant çalışıyor!")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print("❌ Qdrant'a bağlanılamadı. Docker çalışıyor mu?")
    return False


def list_collections() -> list[str]:
    """Mevcut collection'ları listele."""
    r = requests.get(f"{QDRANT_URL}/collections")
    data = r.json()
    names = [c["name"] for c in data.get("result", {}).get("collections", [])]
    return names


def delete_collection(name: str) -> bool:
    """Collection'ı sil."""
    print(f"🗑️  '{name}' collection siliniyor...")
    r = requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=30)
    if r.status_code == 200:
        print(f"✅ '{name}' silindi.")
        return True
    else:
        print(f"⚠️  Silme başarısız: {r.status_code} - {r.text}")
        return False


def upload_snapshot(collection_name: str, snapshot_path: str) -> bool:
    """Snapshot dosyasını Qdrant'a yükle ve collection'ı restore et."""
    file_size_gb = os.path.getsize(snapshot_path) / (1024**3)
    print(f"📦 Snapshot yükleniyor: {snapshot_path}")
    print(f"   Dosya boyutu: {file_size_gb:.2f} GB")
    print(f"   Hedef collection: {collection_name}")
    print(f"   ⏳ Bu işlem birkaç dakika sürebilir...")

    url = f"{QDRANT_URL}/collections/{collection_name}/snapshots/upload"

    start = time.time()
    with open(snapshot_path, "rb") as f:
        # Streaming upload - büyük dosyalar için bellek dostu
        r = requests.post(
            url,
            files={"snapshot": (os.path.basename(snapshot_path), f)},
            timeout=600,  # 10 dakika timeout
        )

    elapsed = time.time() - start

    if r.status_code == 200:
        print(f"✅ Snapshot başarıyla yüklendi! ({elapsed:.1f} saniye)")
        return True
    else:
        print(f"❌ Yükleme başarısız: {r.status_code}")
        print(f"   Detay: {r.text}")
        return False


def check_collection(name: str):
    """Collection'ın durumunu kontrol et."""
    r = requests.get(f"{QDRANT_URL}/collections/{name}")
    if r.status_code == 200:
        info = r.json().get("result", {})
        points = info.get("points_count", 0)
        status = info.get("status", "unknown")
        vectors = info.get("config", {}).get("params", {}).get("vectors", {})
        print(f"\n📊 Collection Bilgileri:")
        print(f"   İsim:     {name}")
        print(f"   Durum:    {status}")
        print(f"   Nokta:    {points:,}")
        print(f"   Vektör:   {vectors}")
    else:
        print(f"⚠️  Collection bilgisi alınamadı: {r.text}")


def main():
    # 1) Qdrant bağlantısını kontrol et
    if not wait_for_qdrant():
        sys.exit(1)

    # 2) Snapshot dosyasını kontrol et
    script_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_path = os.path.join(script_dir, SNAPSHOT_FILE)

    if not os.path.exists(snapshot_path):
        print(f"❌ Snapshot dosyası bulunamadı: {snapshot_path}")
        sys.exit(1)

    # 3) Mevcut collection'ları göster
    collections = list_collections()
    print(f"\n📋 Mevcut collection'lar: {collections}")

    # 4) Eski collection'ları sil
    for col in collections:
        delete_collection(col)

    # 5) Snapshot'ı yükle
    print(f"\n{'='*50}")
    success = upload_snapshot(COLLECTION_NAME, snapshot_path)
    if not success:
        sys.exit(1)

    # 6) Sonucu doğrula
    print(f"\n{'='*50}")
    check_collection(COLLECTION_NAME)

    # 7) Güncellenen collection listesini göster
    new_collections = list_collections()
    print(f"\n📋 Güncel collection'lar: {new_collections}")
    print(f"\n🎉 Tamamlandı! '{COLLECTION_NAME}' collection'ı hazır.")
    print(f"   ⚠️  config.py'deki COLLECTION_NAME'i '{COLLECTION_NAME}' olarak güncellemeyi unutma!")


if __name__ == "__main__":
    main()
