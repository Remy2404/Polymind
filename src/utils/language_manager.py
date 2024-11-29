import utils.telegramlog as telegram_logger
from telegram import Update 
from telegram.ext import ContextTypes
class LanguageManager:
    def __init__(self):
        self.translations = {
            'en': {
                'welcome': 'Welcome to Gem Bot!',
                'help': 'Here are the available commands...'
            },
            'es': {
                'welcome': '¡Bienvenido a Gem Bot!',
                'help': 'Aquí están los comandos disponibles...'
            }
        ,
        'kh': {
            'welcome': 'សូមស្វាគមន៍មកកាន់ Gem Bot!',
            'help': 'នេះគឺជាការប្រើប្រាស់ដោយអាជ្ញាប័ណ្ណដ៏មានសំរាប់អ្នក...'
        },
        'fr': {
            'welcome': 'Bienvenue sur Gem Bot!',
            'help': 'Voici les commandes disponibles...'
        },
        'de': {
            'welcome': 'Willkommen bei Gem Bot!',
            'help': 'Hier sind die verfügbaren Befehle...'
        },
        'it': {
            'welcome': 'Benvenuto su Gem Bot!',
            'help': 'Ecco i comandi disponibili...'
        },
        'pt': {
            'welcome': 'Bem-vindo ao Gem Bot!',
            'help': 'Aqui estão os comandos disponíveis...'
        },
        'ru': {
            'welcome': 'Добро пожаловать в Gem Bot!',
            'help': 'Вот доступные команды...'
        },
        'zh': {
            'welcome': '欢迎使用 Gem Bot!',
            'help': '以下是可用命令...'
        },
        'ja': {
            'welcome': 'Gem Botへようこそ!',
            'help': '利用可能なコマンドはこちら...'
        },
        'ko': {
            'welcome': 'Gem Bot에 오신 것을 환영합니다!',
            'help': '사용 가능한 명령어는 다음과 같습니다...'
        },
        'ar': {
            'welcome': 'مرحبًا بك في Gem Bot!',
            'help': 'إليك الأوامر المتاحة...'
        },
        'hi': {
            'welcome': 'Gem Bot में आपका स्वागत है!',
            'help': 'यहाँ उपलब्ध कमांड हैं...'
        },
        'bn': {
            'welcome': 'Gem Bot এ আপনাকে স্বাগতম!',
            'help': 'এখানে উপলব্ধ কমান্ডগুলি...'
        },
        'pa': {
            'welcome': 'Gem Bot ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ!',
            'help': 'ਇੱਥੇ ਉਪਲਬਧ ਕਮਾਂਡ ਹਨ...'
        },
        'jv': {
            'welcome': 'Sugeng rawuh ing Gem Bot!',
            'help': 'Mangkene printah sing kasedhiya...'
        },
        'vi': {
            'welcome': 'Chào mừng đến với Gem Bot!',
            'help': 'Đây là các lệnh có sẵn...'
        },
        'tr': {
            'welcome': 'Gem Bot\'a hoş geldz!',
            'help': 'İşte mevcut komutlar...'
        },
        'pl': {
            'welcome': 'Witamy w Gem Bot!',
            'help': 'Oto dostępne polecenia...'
        },
        'uk': {
            'welcome': 'Ласкаво просимо до Gem Bot!',
            'help': 'Ось доступні команди...'
        },
        'ro': {
            'welcome': 'Bine ați venit la Gem Bot!',
            'help': 'Iată comenzile disponibile...'
        },
        'nl': {
            'welcome': 'Welkom bij Gem Bot!',
            'help': 'Hier zijn de beschikbare commando\'s...'
        },
        'sv': {
            'welcome': 'Välkommen till Gem Bot!',
            'help': 'Här är de tillgängliga kommandona...'
        },
        'no': {
            'welcome': 'Velkommen til Gem Bot!',
            'help': 'Her er de tilgjengelige kommandoene...'
        },
        'da': {
            'welcome': 'Velkommen til Gem Bot!',
            'help': 'Her er de tilgængelige kommandoer...'
        },
        'fi': {
            'welcome': 'Tervetuloa Gem Bot!',
            'help': 'Tässä ovat käytettävissä olevat komennot...'
        },
        'hu': {
            'welcome': 'Üdvözöljük a Gem Bot!',
            'help': 'Itt vannak az elérhető parancsok...'
        },
        'cs': {
            'welcome': 'Vítejte v Gem Bot!',
            'help': 'Zde jsou dostupné příkazy...'
        },
        'sk': {
            'welcome': 'Vitajte v Gem Bot!',
            'help': 'Tu sú dostupné príkazy...'
        },
        'el': {
            'welcome': 'Καλώς ήρθατε στο Gem Bot!',
            'help': 'Εδώ είναι οι διαθέσιμες εντολές...'
        },
        'he': {
            'welcome': 'ברוך הבא ל-Gem Bot!',
            'help': 'הנה הפקודות הזמינות...'
        },
        'th': {
            'welcome': 'ยินดีต้อนรับสู่ Gem Bot!',
            'help': 'นี่คือคำสั่งที่ใช้ได้...'
        },
        'id': {
            'welcome': 'Selamat datang di Gem Bot!',
            'help': 'Berikut adalah perintah yang tersedia...'
        },
        'ms': {
            'welcome': 'Selamat datang ke Gem Bot!',
            'help': 'Berikut adalah arahan yang tersedia...'
        },
        'ta': {
            'welcome': 'Gem Bot இல் வரவேற்கிறோம்!',
            'help': 'இங்கே கிடைக்கும் கட்டளைகள்...'
        },
        'te': {
            'welcome': 'Gem Bot కు స్వాగతం!',
            'help': 'ఇక్కడ అందుబాటులో ఉన్న ఆదేశాలు...'
        },
        'ml': {
            'welcome': 'Gem Bot ലേക്ക് സ്വാഗതം!',
            'help': 'ഇവിടെ ലഭ്യമായ കമാൻഡുകൾ...'
        },
        'mr': {
            'welcome': 'Gem Bot मध्ये आपले स्वागत आहे!',
            'help': 'येथे उपलब्ध आदेश आहेत...'
        },
        'ur': {
            'welcome': 'Gem Bot میں خوش آمدید!',
            'help': 'یہاں دستیاب کمانڈز ہیں...'
        },
        'fa': {
            'welcome': 'به Gem Bot خوش آمدید!',
            'help': 'دستورات موجود در اینجا هستند...'
        },
        'bg': {
            'welcome': 'Добре дошли в Gem Bot!',
            'help': 'Ето наличните команди...'
        },
        'sr': {
            'welcome': 'Добродошли у Gem Bot!',
            'help': 'Ово су доступне команде...'
        },
        'hr': {
            'welcome': 'Dobrodošli u Gem Bot!',
            'help': 'Ovdje su dostupne naredbe...'
        },
        'sl': {
            'welcome': 'Dobrodošli v Gem Bot!',
            'help': 'Tukaj so na voljo ukazi...'
        },
        'lt': {
            'welcome': 'Sveiki atvykę į Gem Bot!',
            'help': 'Čia yra galimos komandos...'
        },
        'lv': {
            'welcome': 'Laipni lūdzam Gem Bot!',
            'help': 'Šeit ir pieejamās komandas...'
        },
        'et': {
            'welcome': 'Tere tulemast Gem Bot!',
            'help': 'Siin on saadaval käsud...'
        },
        'ka': {
            'welcome': 'კეთილი იყოს თქვენი მობრძანება Gem Bot-ში!',
            'help': 'აქ არის ხელმისაწვდომი ბრძანებები...'
        },
        'az': {
            'welcome': 'Gem Bot-a xoş gəlmisiniz!',
            'help': 'Budur mövcud əmrlər...'
        },
        'kk': {
            'welcome': 'Gem Bot-қа қош келдіңіз!',
            'help': 'Міне қол жетімді командалар...'
        }
        }
    
    def get_text(self, key: str, lang: str = 'en') -> str:
        return self.translations.get(lang, {}).get(key, self.translations['en'].get(key, ''))

    async def set_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /language command to set user's preferred language."""
        user_id = update.effective_user.id
        args = context.args

        if not args:
            await update.message.reply_text("Please provide a language code. Example: /language en")
            return

        lang = args[0].lower()
        if lang not in self.translations:
            await update.message.reply_text("Unsupported language code. Please try again.")
            return

        # Here you should save the user's preferred language in the database or in memory
        # Example: self.user_data_manager.set_user_language(user_id, lang)
        # Assuming you have access to user_data_manager

        await update.message.reply_text(self.get_text('welcome', lang))
        telegram_logger.info(f"User {user_id} set language to {lang}.")
