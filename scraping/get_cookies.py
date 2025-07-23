from selenium import webdriver
import time
import json
import os

# Buat folder jika belum ada
cookie_dir = "./Scraping/LinkedIn/dump"
os.makedirs(cookie_dir, exist_ok=True)

# Jalankan browser Chrome
driver = webdriver.Chrome()
driver.get("https://www.linkedin.com/login")

# Tunggu pengguna login secara manual
print("🔑 Silakan login ke LinkedIn secara manual di jendela browser yang terbuka.")
time.sleep(60)  # Ubah jika perlu waktu login lebih panjang

# Simpan cookies setelah login berhasil
cookies = driver.get_cookies()
cookie_path = os.path.join(cookie_dir, "cookies.json")
with open(cookie_path, "w", encoding="utf-8") as f:
    json.dump(cookies, f, ensure_ascii=False, indent=4)

print(f"✅ Cookies berhasil disimpan di: {cookie_path}")
driver.quit()
