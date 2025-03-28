# Telegram Botu ile Gemini AI Entegrasyonu

## Proje Açıklaması
Bu proje, Google'ın Gemini AI modeli ile entegre edilmiş bir Telegram botudur. Kullanıcılar ile akıllı sohbetler gerçekleştirebilir ve geçmiş konuşmaları hatırlayabilir.

## Özellikler
- `/start` komutuna karşılık hoş geldin mesajı gönderir
- Metin mesajlarını Gemini AI ile işler
- FAISS vektör veritabanı kullanarak konuşma hafızasını korur
- Geçmiş konuşmalardan ilgili bağlamı alır
- Kullanıcı başına en fazla 100 ilgili mesajı hatırlar
- Hataları zarif bir şekilde yönetir

## Kurulum Talimatları
1. Python 3.7 veya üzeri yüklü olmalıdır
2. Gerekli bağımlılıkları yüklemek için:
   ```
   pip install -r requirements.txt
   ```
3. `.env` dosyasını oluşturun ve gerekli API anahtarlarını ekleyin

## API Anahtarı Yapılandırması
`.env` dosyasına aşağıdaki değişkenleri ekleyin:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

## Kullanım Kılavuzu
1. Botu başlatmak için:
   ```
   python bot.py
   ```
2. Telegram'da botunuzu bulun ve sohbet başlatın
3. `/start` komutu ile botu başlatın
4. Herhangi bir mesaj göndererek sohbet edebilirsiniz

## Bağımlılıklar
- python-telegram-bot
- google-generativeai
- faiss-cpu
- numpy

## Katkıda Bulunma
1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi pushlayın (`git push origin feature/amazing-feature`)
5. Bir Pull Request açın

## Not
API anahtarlarınızı güvende tutun ve asla herkese açık şekilde paylaşmayın.