"""
Пайплайн разметки микрокатегорий ремонта по тексту объявления.

  102 Штукатурные работы     105 Электрика
  103 Плиточные работы       106 Поклейка обоев и малярные работы (включая шпаклёвку)
  104 Сантехника
  (101 «Ремонт под ключ» в выводе не проставляется; признак «только полный ремонт» используется
   внутри, чтобы снять подкатегории, которые отдельно не заказать.)

Этапы:
  1. clean()    — нормализация (латинские двойники букв), вырезание промпт-инъекций,
                  служебных блоков, SEO-хвостов; сжатие длинных шаблонных текстов.
  2. analyze()  — разбор на предложения и фрагменты, поиск категорий по словарям
                  (с поправкой на типичные опечатки), учёт отрицаний, чужих мастеров,
                  «раньше делал / больше не беру», «только в составе полного ремонта».
  3. ML (опционально) — TF-IDF + логистическая регрессия по 102–106, обученная на
                  исправленном jsonl; добавляет категорию, только если правила её
                  не нашли и не запретили, а модель уверена.
Каждое решение сопровождается объяснением (фрагмент текста или причина).
"""
from __future__ import annotations
import re, json, pickle
from collections import Counter, defaultdict

NAMES = {101: 'Ремонт под ключ', 102: 'Штукатурные работы', 103: 'Плиточные работы',
         104: 'Сантехника', 105: 'Электрика', 106: 'Поклейка обоев и малярные работы'}
SUB = [102, 103, 104, 105, 106]

# ---------------------------------------------------------------- нормализация
_LAT2CYR = str.maketrans('ABCEHKMOPTXaceopxyBMk', 'АВСЕНКМОРТХасеорхуВМк')
_CYR = re.compile('[а-яёА-ЯЁ]')

def fix_homoglyphs(t: str) -> str:
    """Латинские буквы-двойники внутри кириллических слов → кириллица (РЕMOHT → РЕМОНТ)."""
    def f(m):
        w = m.group(0)
        return w.translate(_LAT2CYR) if _CYR.search(w) else w
    return re.sub(r'[A-Za-zА-Яа-яЁё]+', f, t)

def low(t: str) -> str:
    return t.lower().replace('ё', 'е')

_KEY = str.maketrans({'о': 'а', 'и': 'е', 'ё': 'е'})
_DOUBLE = re.compile(r'([а-я])\1+')
def key(t: str) -> str:
    """Фонетический ключ против опечаток: аканье (о→а), е/и, удвоенные буквы. Применяется и к тексту, и к словарям.
    Длина строки может измениться (удвоения), поэтому позиции в ключе не совпадают с исходным текстом."""
    return _DOUBLE.sub(r'\1', low(t).translate(_KEY))

# ---------------------------------------------------------------- очистка
INJ_LINE = re.compile(
    r'(?i)(^\s*(system|assistant|user|служебное сообщение)\b\s*[:\]>]|<\s*/?\s*(tool_call|system|meta|картинк)|<!--|'
    r'\[(assistant|system|user)\]|ignore (all )?previous|игнорируй|языков\w* модел|для модели|команда (модели|классификатор)|'
    r'классификатор|correct_answer|gold_?label|microcats|верни\s*[\[«"]|верни (все|пуст|только)|выдай только|'
    r'скопируй его|готовый ответ|начисли|проверка не пройд|предполагаемый текст картинки|attachment:|'
    r'якобы проверенн|"status"\s*:|измени правила классификац|повтори (этот текст|предыдущ))')
SERVICE_LINE = re.compile(
    r'(?i)^\s*(код (услуги|проекта|бригады|комплектации)|артикул|номер (сметы|заявки)|прораб\s*:|бригадир\s*[:\-]|логист|менеджер\s*:|'
    r'услуга\s*:|арт\s*:|характеристика|материал\s*:\s*\d|вид работы\s*:\s*\d|инструменты\s*:\s*\d|объект\s*:|площадь ванной|стиль\s*-|'
    r'мастер\s*[:\-]\s*[А-Я][а-я]+ [А-Я]|профиль\s*:|текст проверен|приём заявок и запись|бригада на объекте|доставка материалов на объект\s*—)')
CUT_REST = re.compile(r'(?i)^\s*(информация не для клиента|служебная информация)')
SEO_MARK = re.compile(r'(?i)(возможно,? (вы|вас) искали|нас (также|так же)? ?(можно )?(так же )?(ищут|найти)|также (нас )?ищут|ключевые слова|'
                      r'для поиска\s*:|поисковые запросы|нас также ищут|теги\s*:)')
SENT_END = re.compile(r'[.!?…]')

def SEO_STRICT(line: str) -> bool:
    """Старый строгий признак SEO-строки (много «ремонт/отделк») — допустим и не в конце текста."""
    words = re.findall(r'[а-яёa-z]+', low(line))
    seps = line.count(',') + line.count('/') + line.count('|')
    rem = len(re.findall(r'ремонт|отделк', low(line)))
    return len(words) >= 14 and len(SENT_END.findall(line)) <= 2 and (seps >= 6 or rem >= 5) and rem >= 3

def _is_keyword_blob(line: str) -> bool:
    words = re.findall(r'[а-яёa-z]+', low(line))
    if len(words) < 14:
        return False
    seps = line.count(',') + line.count('/') + line.count('|')
    ends = len(SENT_END.findall(line))
    rem = len(re.findall(r'ремонт|отдел', low(line)))
    items = [x for x in re.split(r'[,/|]', line) if x.strip()]
    short = sorted(len(x.split()) for x in items)
    median = short[len(short) // 2] if short else 99
    first_person = re.search(r'\b(делаю|делаем|выполняю|выполняем|работаю|работаем|берем|беру|предлагаем)\b', low(line))
    seoish = rem >= 2 or re.search(r'недорог|цен[аы]\b|отзыв|частник|москв|спб|петербург|область|район', low(line))
    by_shape = len(items) >= 7 and median <= 4 and ends <= 1 and not first_person and seoish
    return by_shape or (ends <= 2 and (seps >= 6 or rem >= 5) and rem >= 3)

def _dedup_templates(lines):
    """Для очень длинных текстов: убираем предложения-шаблоны, повторяющиеся много раз,
    и заголовки вида «Раздел. 017» — остаются уникальные (содержательные) фразы."""
    hdr = re.compile(r'^.{3,80}\.\s*\d{2,4}\s*$')
    def tpl(s): return re.sub(r'\d+', 'N', re.sub(r'«[^»]*»', 'Q', s))
    sents = []
    for ln in lines:
        if hdr.match(ln.strip()):
            continue
        for s in re.split(r'(?<=[.!?])\s+', ln.strip()):
            if s: sents.append(s)
    c = Counter(tpl(s) for s in sents)
    return [s for s in sents if c[tpl(s)] <= 1]

def clean(raw: str, long_threshold: int = 8000):
    notes = []
    t = (raw or '').replace('\\n', '\n').replace('\\u2028', '\n').replace(' ', '\n').replace(' ', '\n').replace('\xa0', ' ')
    t = fix_homoglyphs(t)
    out, cut = [], False
    lines = t.split('\n')
    n_content = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if CUT_REST.search(s):
            notes.append('вырезан служебный блок «не для клиента»'); cut = True; break
        if INJ_LINE.search(s):
            parts = re.split(r'(?<=[.!?])\s+', s)
            keep = [x for x in parts if not INJ_LINE.search(x)]
            notes.append('вырезана промпт-инъекция: «%s»' % ' '.join(x for x in parts if INJ_LINE.search(x))[:80])
            if not keep or len(parts) == 1: continue
            s = ' '.join(keep)
        if SERVICE_LINE.search(s):
            notes.append('вырезана служебная строка: «%s»' % s[:60]); continue
        m = SEO_MARK.search(s)
        if m:
            before = s[:m.start()].strip()
            if before and len(before) > 40: out.append(before)
            notes.append('вырезан SEO-хвост'); continue
        is_last = all(not x.strip() for x in lines[lines.index(ln) + 1:]) if ln in lines else False
        if n_content >= 2 and _is_keyword_blob(s) and (is_last or SEO_STRICT(s)):
            notes.append('вырезан SEO-хвост'); continue
        out.append(s); n_content += 1
    if sum(len(x) for x in out) > long_threshold:
        before = len(out)
        out = _dedup_templates(out)
        notes.append('длинный шаблонный текст: оставлены уникальные фразы (%d)' % len(out))
    return '\n'.join(out), notes

# ---------------------------------------------------------------- словари категорий
def _k(p):  # компиляция в пространстве фонетического ключа
    return re.compile(key(p))

LEX = {
    102: _k(r'\bштукатур|\bштукатр|\bоштукатур|\bвыравнива\w*\s+(стен|потолк|и\s+штукатур)|\bвыровня\w*\s+стен|\bвыравнивани\w*\s+стен|\bпо\s+маяк|\bмаяк'),
    103: _k(r'\bплитк|\bплиточ|\bкафел|\bкерамогранит|\bкерамагранит|\bмозаик|\bфартук|\bзатирк|\bзатира|\bклинкер|\bоблицов\w*\s+(плитк|керамогр|кафел|санузл|ванн|фартук|стен|пол)|\bоблицовыва\w*\s+(санузл|кухон|фартук)|\bложу\s+(кафел|плитк)'),
    104: _k(r'\bсантех|\bсмесител|\bунитаз|\bраковин|\bсифон|\bводоснабж|\bводопровод|\bканализац|\bтруб(?!к)|\bпротечк|\bтечь|\bтеч(и|ет)\b|\bподтека|\bкапает|\bподводк|\bстиральн\w*\s+машин|\bинсталляц|\bустановк\w*\s+ванн|\bдушев\w*\s+(кабин|систем|поддон)|\bполотенцесуш|\bразводк\w*\s+вод|\bстояк|\bводонагрев|\bбойлер'),
    105: _k(r'\bсвет(а|ом|у)?\b|\bэлектр(?!он)|проводк|\bрозет|\bвыключател|\bсветильн|\bлюстр|\bосвещени|\bподсветк|\bщит(ок|к|а|ы|ов)?\b|\bщиток|\bэлектрощит|\bавтомат(ы|ов)\b|\bкабел|\bштроб|\bштраб|\bсчетчик|\bузо\b|\bдифавтомат'),
    106: _k(r'\bобо(и|ев|ям|ями|ях)\b|\bклею\b|\bклеим\b|\bпоклейк|\bоклейк|\bпереклеи|\bпереклей|\bклеить\s+обо|\bпокрас|\bокрас|\bкрашу|\bкрасим|\bперекраш|\bперекрас|\bмаляр|\bшпакл|\bшпатл|\bводно-дисперс|\bкраск'),
}
# фразы, которые выглядят как категория, но ею не являются
EXCLUDE = {
    102: _k(r'декоративн\w*\s+штукатур\w*|венецианск\w*\s+штукатур\w*|штукатурн\w*\s+(и\s+)?потолк\w*'),
    103: _k(r'тротуарн\w*\s+плитк\w*|пвх[\s-]*плитк\w*|кварц[\s-]*винилов\w*\s+плитк\w*|потолочн\w*\s+плитк\w*|плиточн\w*\s+кле\w*|под\s+(будущую\s+)?(укладку\s+)?плитк\w*|(ровн\w*\s+)?плоскост\w*\s+под\s+плитк\w*'),
    104: _k(r'сантехническ\w*\s+люк\w*|ревизионн\w*\s+люк\w*|протечк\w*\s+(крыш|кровл)\w*|датчик\w*\s+протечк\w*|(крыш|кровл)\w*\s+(\w+\s+)?протека\w*'),
    105: _k(r'электроинструмент\w*|электрическ\w*\s+тепл\w*\s+пол\w*|при\s+другом\s+освещени\w*'),
    106: _k(r'фотообо\w*|печат\w*\s+полотн\w*|краскопульт'),
}
TURNKEY = _k(r'капитальн\w*\s+ремонт|\bкомплексн\w*\s+ремонт|евроремонт|\bремонт\w*\s+(и\s+отделк\w*\s+)?квартир|'
             r'\bремонт\w*\s+(частн\w*\s+)?дом(ов|а)\b|\bполн\w*\s+ремонт|ремонт\w*\s+всей\s+квартир|(квартир|объект|\bдом|помещени)\w*\s+целиком|(берем|беру)\s+всю\s+квартир|'
             r'весь\s+объект|от\s+голых\s+стен|един\w*\s+(проект|подряд|заказ)|от\s+демонтаж\w*\s+до|от\s+черновой\s+до\s+чистовой|'
             r'ремонт\w*\s+новострой|дизайнерск\w*\s+ремонт|отделк\w*\s+квартир|реализаци\w*\s+дизайн|по\s+дизайн[\s-]*проект')
PODKLUCH = re.compile(r'под\s+«?ключ|под\s+🔑')
PK_CTX = _k(r'ремонт|квартир|\bдом|коттедж|дач|санузл|санузел|ванн|туалет|новострой|кухн|студи|однушк|объект')
TK_EXCL = _k(r'чернов\w*\s+ремонт\w*|косметическ\w*\s+ремонт\w*|мелк\w*\s+ремонт\w*|небольш\w*\s+ремонт\w*|бытов\w*\s+ремонт\w*')
TURNKEY_WEAK = _k(r'^\s*ремонт\s+квартир\w*\s*[.!]?\s*$|^ремонт\s+квартир\w*\.\s')

# ---------------------------------------------------------------- маркеры полярности
NEG_LOCAL = re.compile(
    r'\bне\s+(делаю|делаем|выполня\w*|беру|берем|берусь|беремся|занима\w*|принима\w*|клею|клеим|кладу|кладем|ложу|укладыва\w*|'
    r'монтиру\w*|предлага\w*|оказыва\w*|предоставля\w*|штукатур\w*|крашу|красим|лезу|подойду|буду|мой|моя|мое|мои|ко\s+мне|'
    r'включ\w*|отправля\w*|требу\w*|нужно|обязательно|продаю|продаем|работаю\s+с|подключа\w*|выезжа\w*|предусмотрен\w*|становл\w*|'
    r'является|являются|означает|продает|направля\w*|меня\w*|устанавлива\w*)\b|'
    r'\bне\s+в\s+мой|не\s+мой\s+профиль|мимо\b|не\s+ко\s+мне|отказ\w*|больше\s+не|(заказ|заявк|запис|направлени|прием)\w*[^.]{0,30}закрыт|закрыт(а|о|ы)?\b[^.]{0,20}(заказ|заявк|запис|направлени)|ушел|ушли|прекрати\w*|'
    r'(?<!как )(?<!как и )\bраньше\b|\bранее\b|до\s+прошлого\s+года|от\s+старого\s+объявлен|осталась\s+от|старые\s+фотограф|'
    r'\bбез\s+(штукатур|плитк|сантех|электр|отделк|обо)|оставляем\s+(другим|сантехник|электрик|плиточник)|это\s+к\s+(электрик|сантехник|плиточник)|'
    r'\bне\s+(штукатурим|красим|клеим)|даже\s+если|\bне\s+(пере)?\w+(ю|ем|им|ет|ут|ют|ат|ят)\b|\bне\s+работаю|\bне\s+бригада|\bнет\b|\bни\s|нельзя|'
    r'без\s+замен|без\s+(полного\s+)?ремонт|начинал|работал\w*\s+\w+\s+в\s+(19|20)\d\d')
NEG_SENT = re.compile(
    r'(сантехник\w*|электрик\w*|плиточник\w*|маляр\w*|отделочник\w*|мастер\w*|специалист\w*|бригад\w*)\s+заказчика|заказчик\s+(приглашает|нанимает|организует)|в\s+(нашей\s+)?команде|у\s+нас\s+(свои|работают)|в\s+штате|сертификат|допуск|провер\w*\s+(ровност|правильност|качеств)|согласу\w*,?\s+где|ваш(ими|им|его|ему|и|е)?\s+(специалист|мастер|электрик|сантехник|плиточник|маляр|штукатур|отделочник)|'
    r'друг(ой|ие|им|ими|ому|ого)\s+(мастер|исполнител|специалист|бригад|люд)|друг\w+\s+мастер|чуж\w+|нанят\w*|наняла|нанимал|'
    r'не\s+наш\w*\s+(услуг|предложен)|а\s+не\s+(наш|как\s+наш|мои)|не\s+становятся\s+моими|упомянут\w*|упоминан\w*|встречаются|'
    r'в\s+(статье|тексте|разборе|каталоге|памятке|дневнике|справочник\w*)|описывают\s+назначение|'
    r'попросил\w*|просят|просили|спрашивают|в\s+отзыве|клиент\s+написал|хозяйк\w*|нужен\s+(плиточник|электрик|сантехник|мастер|штукатур|маляр)|^\s*про\s|перед\s+нашим\s+приездом|до\s+моего\s+приезда|к\s+моему\s+приезду|к\s+приезду|должн\w*\s+(закончить|закончиться|быть)|\bпредлагают\b|а\s+не\s+просто|вопрос\w*\s+(про|по)\s|консультац\w*\s+по\s|консультиру\w*\s+по\s|делаю\s+проект|схем\w*\s+(электр|сантех|розет)|развертк\w*|визуализац\w*|помо\w+\s+(с\s+)?(выбр|подбор|закуп|заказ)|подбор\w*\s+(материал|мебел)|комплектац|работы\s+выполнял\w*|приезжали|укладывал\s+плиточник|клеил\s+отделочник')
CANCEL = re.compile(r'больше\s+не|(заказ|заявк|запис|направлени|прием)\w*[^.]{0,30}закрыт|закрыт(а|о|ы)?\b[^.]{0,20}(заказ|заявк|запис|направлени)|ушел|ушли|прекрати\w*|от\s+старого\s+объявлен|осталась\s+от|(?<!как )(?<!как и )\bраньше\b|\bранее\b|до\s+прошлого\s+года|старые\s+фотограф|не\s+принимаются')
CURRENT_ONLY = re.compile(r'(сейчас|теперь|текущие\s+услуги|актуальн\w*|в\s+данный\s+момент|пока\s+работаю)[^.]{0,60}\bтолько\b|\bтолько\b[^.]{0,30}\b(сейчас|теперь)\b')
BUNDLE = re.compile(
    r'(?<!не )(только|лишь)\s+(вместе\s+)?(в\s+составе|в\s+рамках|при\s+заказе|как\s+един|вместе\s+с|полн\w*\s+ремонт|весь\s+проект|на\s+полный|ремонт\s+под\s+ключ|всю\s+квартир|целиком)|'
    r'по\s+отдельности\s+не|отдельн\w*\s+(заказ|заявк|работ|этап|вид)\w*[^.]{0,40}\bне\b|\bне\s+(берем|беру|принимаем)\s+отдельн|'
    r'делаем\s+только\s+ремонт|включать\s+весь\s+проект|выезд\s+ради\s+одного\s+этапа\s+не|заказ\w*\s+отдельных\s+этапов|'
    r'отдельного\s+мастера\s+на\s+конкретную\s+задачу\s+не|единым\s+заказом|только\s+полный\s+(цикл|ремонт)|заявки\s+только\s+на\s+полный|'
    r'на\s+самостоятельн\w*\s+заказ\w*[^.]{0,40}не\s+выезжа|не\s+каталог\s+услуг\s+на\s+выбор')
STANDALONE = re.compile(
    r'самостоятельн|отдельно\b|исключени|без\s+остальных|можно\s+заказать\s+без|сама\s+по\s+себе|сам\s+по\s+себе|по\s+одной\s+заявке|'
    r'на\s+отдельные\s+заявки|остается\s+доступн|можно\s+(вызвать|заказать|обратиться|пригласить)|любой\s+(вид|из)|независимо|по\s+каждому\s+направлению|'
    r'доступен\s+и\s+без|принимаем\s+как')
NOT_SERVICE = re.compile(
    r'\b(продаю|продам|прода[её]тся|продаем)\b[^.!?]{0,40}(коробк|остатк|материал|рулон|книг|учебник|игр|инструмент|плитк|керамогранит|обо[ие]|юрлиц|юридическ|компани|бизнес)|в\s+объявлении\s+только\s+товар|самовывоз|вакансия|ищем\s+\w+\s+в\s+штат|соискател|'
    r'провожу\s+обучени|продаем\s+обучени|\bучим\b|сда(ем|ю)\s[^.]{0,40}в\s+аренду|прокат\w*\s+(оборуд|инструм)|в\s+прокате|'
    r'ремонтных\s+услуг\s+не\s+оказыва|работы\s+на\s+объекте\s+не\s+выполня|ремонт\s+не\s+выполняю|услуги\s+клиентам[^.]*не\s+предлагаются|'
    r'ни\s+по\s+одному\s+направлению|прекратила\s+работу|я\s+не\s+мастер|настольн\w*\s+игр|продажа\s+компании|юридическое\s+лицо|'
    r'только\s+дизайн|монтаж\s+и\s+отделку\s+не\s+выполня|работы\s+на\s+объектах\s+заказчиков\s+не\s+выполня|'
    r'без\s+предложения\s+работ|ремонт(ом)?\s+не\s+(занима|делаю|выполня)|предложения\s+услуг\s+в\s+записи\s+нет|ремонт\s+не\s+выполня|перевожу\s+коробки')
POS_VERB = re.compile(
    r'\b(делаю|делаем|выполня\w*|клею|клеим|кладу|кладем|ложу|укладыва\w*|штукатур(ю|им)|монтиру\w*|устанавлива\w*|меняю|меняем|миняю|ставлю|'
    r'подключа\w*|развожу|разводим|прокладыва\w*|ищу|ищем|устраня\w*|устроня\w*|крашу|красим|принима\w*|берем|беру|облицовыва\w*|наносим|'
    r'выравнива\w*|восстанов\w*|заменим|поклеим|подключим|устраним|занимаюсь|занимаемся|предлагаем|оказываем|собира\w*|работаю|работаем)\b')

SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+|\s{2,}|(?<=[а-яё])\s+(?=[А-ЯЁ][а-яё]{2,}\s)|(?<=[а-я][.!?])(?=[А-Я])|\n|\s\|\s|\s•\s|^•\s|(?<=\S)\s*•\s*')
CONTRAST = re.compile(r',?\s+(?:а|но|однако|зато)\s+|;\s*')
SEG_SPLIT = re.compile(r'[,;:]|\s[—–-]\s|\s(а|но|однако)\s')

try:
    import regex as _rx
    FUZZY = {c: _rx.compile(r'(?:\b(?:%s)){e<=1}' % '|'.join(key(w) for w in ws)) for c, ws in {
        102: ['штукатур', 'оштукатур'], 103: ['плиточн', 'керамогранит'],
        104: ['сантехн', 'смесител', 'канализац', 'водопровод', 'водоснабж'],
        105: ['электромонтаж', 'электропровод', 'выключател', 'светильн'],
        106: ['покраск', 'шпаклев', 'шпатлев'],
    }.items()}
except Exception:            # пакет regex не установлен — работаем без нечёткого поиска
    FUZZY = {}

def _cats_in(seg_key: str):
    found = {}
    for c in SUB:
        s = EXCLUDE[c].sub(' ', seg_key)
        m = LEX[c].search(s)
        if m:
            found[c] = m.group(0)
        elif c in FUZZY:
            fm = FUZZY[c].search(s)
            if fm:
                found[c] = fm.group(0)
    return found

def analyze(text: str):
    """Возвращает (метки, объяснения) по очищенному тексту."""
    ev = defaultdict(list)           # категория -> фрагменты-доказательства
    neg = defaultdict(list)
    cancel, bundled, stand = set(), set(), set()
    current_only = None
    has_bundle = False
    turnkey_ev, turnkey_neg = [], []
    lt = low(text)
    if NOT_SERVICE.search(lt):
        m = NOT_SERVICE.search(lt)
        return [], {'_doc': 'не услуга ремонта (найдено: «%s»)' % m.group(0)}
    sents0 = [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]
    sents = []
    for s0 in sents0:
        parts = [x.strip() for x in CONTRAST.split(s0) if x and x.strip()]
        flags = [bool(NEG_SENT.search(low(x))) for x in parts]
        if len(parts) > 1 and not BUNDLE.search(low(s0)) and any(flags) and not all(flags):
            sents.extend(parts)      # «электрики монтируют проводку, а сантехнику пусть делает ваш мастер»
        else:
            sents.append(s0)
    last_pol = 1
    doc_ctx = bool(PK_CTX.search(TK_EXCL.sub(' ', key(text))))
    for sent in sents:
        sl = low(sent)
        if sl.rstrip().endswith('?'):
            continue                                   # вопросы — не предложение услуги
        if re.match(r'^[^а-яa-z0-9]*если\b', sl):          # условная часть «Если …,» отбрасывается
            m = re.search(r'[,—–:]', sent)
            if not m: continue
            sent = sent[m.end():]; sl = low(sent)
        sk = key(sent)
        sent_cats = _cats_in(sk)
        sent_neg = bool(NEG_SENT.search(sl))
        is_bundle = bool(BUNDLE.search(sl))
        is_stand = (not is_bundle) and bool(STANDALONE.search(sl))
        is_cancel = bool(CANCEL.search(sl))
        if CURRENT_ONLY.search(sl):
            tail = sl[sl.find('только'):] if 'только' in sl else sl
            current_only = (current_only or set()) | set(_cats_in(key(tail)))
        # 101
        tks = TK_EXCL.sub(' ', sk)
        pk_ok = PK_CTX.search(tks) or (doc_ctx and 'отделк' not in sl)   # «отделка … под ключ» без ремонта — не 101
        tk = TURNKEY.search(tks) or (PODKLUCH.search(sl) and pk_ok)
        if tk:
            if sent_neg or NEG_LOCAL.search(sl) and not is_bundle:
                turnkey_neg.append(sent[:120])
            else:
                turnkey_ev.append(sent[:120])
        if is_bundle:
            has_bundle = True
            bundled |= set(sent_cats) if sent_cats else {'*'}
            for c, w in sent_cats.items():
                ev[c].append(sent[:120])
            continue
        # фрагменты с наследованием полярности для «голых» перечислений
        segs, after_colon, last = [], [], 0
        colon_seen = False
        for m in SEG_SPLIT.finditer(sent):
            segs.append(sent[last:m.start()]); after_colon.append(colon_seen)
            if ':' in m.group(0): colon_seen = True
            last = m.end()
        segs.append(sent[last:]); after_colon.append(colon_seen)
        keep = [i for i, x in enumerate(segs) if x.strip() and x.strip() not in ('а', 'но', 'однако')]
        segs = [segs[i] for i in keep]; after_colon = [after_colon[i] for i in keep]
        pol = []
        for sg in segs:
            g = low(sg)
            if sent_neg or NEG_LOCAL.search(g):
                pol.append(-1)
            elif re.search(r'\bтоже\W*$', g) and len(g.split()) <= 4:
                pol.append(2)                          # «Электрику тоже.» — как предыдущее
            elif POS_VERB.search(g) or len(g.split()) > 4:
                pol.append(1)
            else:
                pol.append(0)                          # «голый» элемент списка
        header = next((pol[i] for i in range(len(segs) - 1, -1, -1) if not after_colon[i]), 0) if any(after_colon) else 0
        for i in range(len(pol)):
            if pol[i] == 2:
                pol[i] = pol[i - 1] if i > 0 and pol[i - 1] in (1, -1) else last_pol
        for i in range(len(pol) - 1, -1, -1):
            if pol[i] == 0:
                if after_colon[i] and header in (1, -1):
                    pol[i] = header                    # «Делаю ремонт санузлов: укладка плитки, …» — от заголовка
                else:
                    pol[i] = next((p for p in pol[i + 1:] if p != 0), 1)   # наследуем от следующего
        for sg, p in zip(segs, pol):
            cats = _cats_in(key(sg))
            for c in cats:
                if p > 0:
                    ev[c].append(sent[:120])
                    if is_stand: stand.add(c)
                else:
                    neg[c].append(sent[:120])
                    if is_cancel: cancel.add(c)
            if cats: last_pol = p
    labels, why = set(), {}
    for c in SUB:
        if current_only and c in current_only: cancel.discard(c)
        if ev[c] and c not in cancel:
            labels.add(c); why[c] = 'найдено: «%s»' % ev[c][0]
        elif c in cancel and ev[c]:
            why['-%d' % c] = 'снято: больше не выполняют («%s»)' % neg[c][0]
        elif neg[c] and not ev[c]:
            why['-%d' % c] = 'упомянуто, но не предлагается («%s»)' % neg[c][0]
    if current_only is not None and current_only:
        for c in list(labels):
            if c not in current_only:
                labels.discard(c); why['-%d' % c] = 'снято: сейчас выполняют только другое'
                why.pop(c, None)
    if has_bundle:
        drop = set(SUB) if '*' in bundled else bundled
        why['_stand'] = sorted(c for c in labels if c in stand and c != 101)   # «отдельно можно» — высокоточная конструкция
        for c in list(labels):
            if c in drop and c not in stand:
                labels.discard(c); why.pop(c, None)
                why['-%d' % c] = 'только в составе полного ремонта, отдельно не заказать'
        labels.add(101); why[101] = 'работают только полным ремонтом/единым проектом'
    elif turnkey_ev:
        labels.add(101); why[101] = 'найдено: «%s»' % turnkey_ev[0]
    elif TURNKEY_WEAK.search(key(text.strip())):
        labels.add(101); why[101] = 'объявление «Ремонт квартир» без деталей'
    elif turnkey_neg:
        why['-101'] = 'упомянуто, но не предлагается («%s»)' % turnkey_neg[0]
    return sorted(labels), why

# ---------------------------------------------------------------- ML-компонент
class MLModel:
    def __init__(self, path=None):
        self.vec = self.clfs = None
        if path:
            with open(path, 'rb') as f:
                self.vec, self.clfs = pickle.load(f)

    def fit(self, texts, label_lists, C=4.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import hstack
        self.vec = (TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200000,
                                    preprocessor=key),
                    TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000,
                                    preprocessor=key))
        X = hstack([v.fit_transform(texts) for v in self.vec]).tocsr()
        self.clfs = {}
        for c in SUB:
            y = [int(c in l) for l in label_lists]
            self.clfs[c] = LogisticRegression(C=C, max_iter=2000, class_weight='balanced').fit(X, y)
        return self

    def proba(self, texts):
        from scipy.sparse import hstack
        X = hstack([v.transform(texts) for v in self.vec]).tocsr()
        return {c: self.clfs[c].predict_proba(X)[:, 1] for c in SUB}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.vec, self.clfs), f)

# ---------------------------------------------------------------- вход для ученика «категория + текст с подсветкой»
# Более широкий словарь УПОМИНАНИЙ — только для подсветки (решение принимает модель), правила его не используют.
MENTION_EXTRA = {
    102: _k(r'заштукатур|\bвыровня\w*|\bвыравнива\w*|под\s+правил|подготовк\w*\s+(стен|поверхност)|черновая\s+отделк|декоративн\w*\s+штукатур'),
    103: _k(r'\bплит(ок|очн|ку|ка|ки|кой)|облицов|санузл|санузел|ванн\w*\s+комнат|кафельщик'),
    104: _k(r'стиралк|посудомойк|счетчик\w*\s+(на\s+)?вод|фильтр\w*|\bкран|коммуникац|вывод\w*\s+вод|\bводы\b|засор|сливн|бачок|душев'),
    105: _k(r'слаботоч|коммуникац|\bэлектрик|\bсвет\w*|розетк|\bщит|автомат'),
    106: _k(r'пошпакл|покраш|покраш|\bобо(я|ями)\b|отделочн|чистов\w*\s+отделк|косметическ|финишн|грунтовк|декоративн\w*\s+штукатур'),
}
CAT_Q = {102: 'Штукатурные работы', 103: 'Плиточные работы', 104: 'Сантехника', 105: 'Электрика',
         106: 'Обои, покраска, шпаклёвка (малярные работы)'}
MARK_L, MARK_R = '[[', ']]'


def mentions(sent: str, c: int) -> bool:
    k = key(sent)
    return bool(_cats_in(k).get(c)) or bool(MENTION_EXTRA[c].search(k))


def mark_doc(ct: str, c: int, max_chars: int = 3500) -> str:
    """Очищенный текст -> «Категория | текст», где предложения с упоминанием категории c обёрнуты в [[ ]].
    Длинный текст сокращается: остаются начало, подсвеченные предложения и их соседи."""
    lines = [ln for ln in ct.split('\n') if ln.strip()]
    sents = []                                   # (текст, номер строки) — сохраняем разбиение по строкам
    for li, ln in enumerate(lines):
        for s in SENT_SPLIT.split(ln):
            if s and s.strip():
                sents.append((s.strip(), li))
    hit = [mentions(s, c) for s, _ in sents]
    keep = list(range(len(sents)))
    if sum(len(s) for s, _ in sents) > max_chars:
        want = set(range(min(3, len(sents))))
        for i, h in enumerate(hit):
            if h: want |= {i - 1, i, i + 1}
        keep = sorted(i for i in want if 0 <= i < len(sents))
    out, prev_line, prev_i = [], None, None
    for i in keep:
        s, li = sents[i]
        if prev_i is not None and i != prev_i + 1:
            out.append('\n…\n')
        elif prev_line is not None:
            out.append('\n' if li != prev_line else ' ')
        out.append(f'{MARK_L} {s} {MARK_R}' if hit[i] else s)
        prev_line, prev_i = li, i
    body = ''.join(out)
    if len(body) > max_chars * 1.5:
        body = body[:max_chars * 1.5]
    return f'{CAT_Q[c]} | {body}'


RULE_NAME = {102: 'штукатурка', 103: 'плитка', 104: 'сантехника', 105: 'электрика', 106: 'обои/покраска'}


def rules_prefix(ct: str) -> str:
    """Решение правил по очищенному тексту — подсказка ученику («стекинг»): «правила: штукатурка нет; плитка да; …»."""
    labels, why = analyze(ct)
    return 'правила: ' + '; '.join('%s %s' % (RULE_NAME[c], 'да' if c in labels else 'нет') for c in SUB) + '\n'


# ---------------------------------------------------------------- ученик (нейросеть, ONNX на CPU)
def _evidence(text: str, c: int) -> str:
    """Первое предложение, где упомянута категория c (по словарям) — для объяснения решения модели."""
    for s in SENT_SPLIT.split(text):
        if s and s.strip() and _cats_in(key(s)).get(c):
            return s.strip()[:120]
    return ''


RULES_PRIORITY = True    # высокоточные конструкции правил важнее модели (см. _merge_student)


def _merge_student(ct, labels, why, p, thr, keep_rule_p):
    """Решение по категории принимает ученик; правила дают объяснение. Гибрид: если правила нашли категорию,
    а модель не уверена в обратном (p ≥ keep_rule_p), метку оставляем."""
    new, nwhy = [], {k: v for k, v in why.items() if not (isinstance(k, int) or (isinstance(k, str) and k[1:].isdigit()))}
    if RULES_PRIORITY and '_doc' in why:          # «не услуга ремонта» (вакансия, продажа, работы прекращены) — за правилами
        return [], nwhy
    stand = set(why.get('_stand', ())) if RULES_PRIORITY else set()
    for j, c in enumerate(SUB):
        pc = float(p[j])
        if c in stand:                            # «только под ключ, но X можно заказать отдельно» — за правилами
            new.append(c); nwhy[c] = '%s; отдельно заказать можно (правило; модель p=%.2f)' % (why.get(c, ''), pc)
        elif pc >= thr or (c in labels and pc >= keep_rule_p):
            new.append(c)
            if c in labels:
                nwhy[c] = '%s (модель p=%.2f)' % (why.get(c, ''), pc)
            else:
                ev = _evidence(ct, c)
                nwhy[c] = ('модель: «%s» (p=%.2f)' % (ev, pc)) if ev else 'модель уверена (p=%.2f), явного слова нет' % pc
        elif c in labels:
            nwhy['-%d' % c] = 'правило нашло %s, но по смыслу это не услуга автора (модель p=%.2f)' % (
                why.get(c, '').replace('найдено: ', ''), pc)
        elif ('-%d' % c) in why:
            nwhy['-%d' % c] = why['-%d' % c]
    return new, nwhy


_STUDENT = {}
def load_student(path=None, int8=True):
    """Ученик по умолчанию — models/student/default (симлинк/каталог), если есть."""
    import os
    from student import Student
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'student', 'default')
    if path not in _STUDENT:
        _STUDENT[path] = Student(path, int8=int8)
    return _STUDENT[path]


# ---------------------------------------------------------------- общий вход
def predict(texts, model: MLModel | None = None, ml_threshold: float | None = None, veto_threshold: float | None = 0.1,
            veto_min_len: int = 600, student=None, thr: float = 0.5, keep_rule_p: float = 0.2):
    """model=None — только правила. С моделью: снимает найденную правилом категорию в длинных текстах,
    если модель почти уверена, что её нет (veto). ml_threshold — разрешить модели добавлять категории (по умолчанию выкл.).
    student — ученик (src/student.Student или путь): решение по категориям за ним, правила дают объяснения."""
    cleaned = [clean(t) for t in texts]
    probs = model.proba([c[0] for c in cleaned]) if model is not None else None
    if isinstance(student, str):
        student = load_student(student)
    sp = student.proba([c[0] for c in cleaned]) if student is not None else None
    out = []
    for i, (ct, notes) in enumerate(cleaned):
        labels, why = analyze(ct)
        labels = [c for c in labels if c != 101]          # 101 не выводим
        why.pop(101, None); why.pop('-101', None)
        if sp is not None:
            if '_doc' in why:
                why['_doc'] = 'правила: ' + why['_doc']
            labels, why = _merge_student(ct, labels, why, sp[i], thr, keep_rule_p)
        if probs is not None and '_doc' not in why:
            for c in SUB:
                p = float(probs[c][i])
                if ml_threshold is not None and c not in labels and ('-%d' % c) not in why and p >= ml_threshold:
                    labels = sorted(set(labels) | {c}); why[c] = 'модель уверена (p=%.2f), явного слова нет' % p
                if veto_threshold is not None and len(ct) >= veto_min_len and c in labels and p < veto_threshold:
                    labels = [x for x in labels if x != c]
                    why['-%d' % c] = 'правило нашло «%s», но модель против (p=%.2f)' % (why.pop(c, '')[:60], p)
        expl = []
        if '_doc' in why: expl.append(why['_doc'])
        for c in labels: expl.append('%d: %s' % (c, why.get(c, '')))
        for k, v in why.items():
            if isinstance(k, str) and k.startswith('-'): expl.append('нет %s: %s' % (k[1:], v))
        expl += notes
        out.append({'mcId': labels, 'labels': [NAMES[c] for c in labels], 'explanation': ' | '.join(expl)})
    return out
