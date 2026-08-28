import sys
from PIL import Image
import torch
from lane_detector_ufld import UFLDLaneDetector
from transformers import CLIPProcessor, CLIPModel
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QVBoxLayout, QWidget, QProgressBar, QPushButton, QScrollArea)
from PyQt5.QtCore import Qt, QMimeData, QTimer
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QScreen
import tempfile

# 所有有Google街景的国家及地区（中英对照）
COUNTRIES = {
    "Albania": "阿尔巴尼亚",
    "Andorra": "安道尔",
    "Argentina": "阿根廷",
    "Australia": "澳大利亚",
    "Austria": "奥地利",
    "Bangladesh": "孟加拉国",
    "Belgium": "比利时",
    "Bolivia":"玻利维亚",
    "Botswana": "博茨瓦纳",
    "Butan":"不丹",
    "Brazil": "巴西",
    "Bulgaria": "保加利亚",
    "Cambodia": "柬埔寨",
    "Canada": "加拿大",
    "Chile": "智利",
    "Colombia": "哥伦比亚",
    "Costa Rica": "哥斯达黎加",
    "Croatia": "克罗地亚",
    "Czech Republic": "捷克",
    "Denmark": "丹麦",
    "Dominican Republic": "多米尼加共和国",
    "Ecuador": "厄瓜多尔",
    "Estonia": "爱沙尼亚",
    "Finland": "芬兰",
    "France": "法国",
    "Germany": "德国",
    "Ghana": "加纳",
    "Greece": "希腊",
    "Guatemala": "危地马拉",
    "Hong Kong": "香港",
    "Hungary": "匈牙利",
    "Iceland": "冰岛",
    "India": "印度",
    "Indonesia": "印度尼西亚",
    "Ireland": "爱尔兰",
    "Israel": "以色列",
    "Italy": "意大利",
    "Japan": "日本",
    "Jordan": "约旦",
    "Kenya": "肯尼亚",
    "Kyrgyzstan": "吉尔吉斯斯坦",
    "Laos": "老挝",
    "Latvia": "拉脱维亚",
    "Lesotho": "莱索托",
    "Liechtenstein": "列支敦士登",
    "Lithuania": "立陶宛",
    "Luxembourg": "卢森堡",
    "Lebanon": "黎巴嫩",
    "Macau": "澳门",
    "Malaysia": "马来西亚",
    "Malta": "马耳他",
    "Mexico": "墨西哥",
    "Monaco": "摩纳哥",
    "Mongolia": "蒙古",
    "Montenegro": "黑山",
    "Morocco": "摩洛哥",
    "Myanmar": "缅甸",
    "Namibia":"纳米比亚",
    "Nepal": "尼泊尔",
    "Netherlands": "荷兰",
    "New Zealand": "新西兰",
    "Nigeria": "尼日利亚",
    "North Macedonia": "北马其顿",
    "Northern Mariana Islands": "北马里亚纳群岛",
    "Norway": "挪威",
    "Omen":"阿曼",
    "Pakistan": "巴基斯坦",
    "Panama": "巴拿马",
    "Peru": "秘鲁",
    "Philippines": "菲律宾",
    "Poland": "波兰",
    "Portugal": "葡萄牙",
    "Puerto Rico": "波多黎各",
    "Romania": "罗马尼亚",
    "Russia": "俄罗斯",
    "Rwanda": "卢旺达",
    "San Marino": "圣马力诺",
    "Senegal": "塞内加尔",
    "Serbia": "塞尔维亚",
    "Singapore": "新加坡",
    "Slovakia": "斯洛伐克",
    "Slovenia": "斯洛文尼亚",
    "South Africa": "南非",
    "South Korea": "韩国",
    "Spain": "西班牙",
    "Sri Lanka": "斯里兰卡",
    "Swaziland": "斯威士兰",
    "Sweden": "瑞典",
    "Switzerland": "瑞士",
    "Taiwan": "台湾",
    "Tanzania": "坦桑尼亚",
    "Thailand": "泰国",
    "Tunisia": "突尼斯",
    "Turkey": "土耳其",
    "Uganda": "乌干达",
    "Ukraine": "乌克兰",
    "United Arab Emirates": "阿拉伯联合酋长国",
    "United Kingdom": "英国",
    "United States": "美国",
    "Uruguay": "乌拉圭",
    "Vatican City": "梵蒂冈",
    "Vietnam": "越南"
}

# 完整州/省数据（包含方位信息）
REGIONS = {
    "United States": {
        # 美国50个州完整列表
        "Alabama": ("阿拉巴马州", "东南部"),
        "Alaska": ("阿拉斯加州", "西北部"),
        "Arizona": ("亚利桑那州", "西南部"),
        "Arkansas": ("阿肯色州", "中南部"),
        "California": ("加利福尼亚州", "西部"),
        "Colorado": ("科罗拉多州", "西部"),
        "Connecticut": ("康涅狄格州", "东北部"),
        "Delaware": ("特拉华州", "东北部"),
        "Florida": ("佛罗里达州", "东南部"),
        "Georgia": ("佐治亚州", "东南部"),
        "Hawaii": ("夏威夷州", "太平洋"),
        "Idaho": ("爱达荷州", "西北部"),
        "Illinois": ("伊利诺伊州", "中西部"),
        "Indiana": ("印第安纳州", "中西部"),
        "Iowa": ("艾奥瓦州", "中西部"),
        "Kansas": ("堪萨斯州", "中西部"),
        "Kentucky": ("肯塔基州", "中南部"),
        "Louisiana": ("路易斯安那州", "南部"),
        "Maine": ("缅因州", "东北部"),
        "Maryland": ("马里兰州", "东北部"),
        "Massachusetts": ("马萨诸塞州", "东北部"),
        "Michigan": ("密歇根州", "中西部"),
        "Minnesota": ("明尼苏达州", "中西部"),
        "Mississippi": ("密西西比州", "南部"),
        "Missouri": ("密苏里州", "中西部"),
        "Montana": ("蒙大拿州", "西北部"),
        "Nebraska": ("内布拉斯加州", "中西部"),
        "Nevada": ("内华达州", "西部"),
        "New Hampshire": ("新罕布什尔州", "东北部"),
        "New Jersey": ("新泽西州", "东北部"),
        "New Mexico": ("新墨西哥州", "西南部"),
        "New York": ("纽约州", "东北部"),
        "North Carolina": ("北卡罗来纳州", "东南部"),
        "North Dakota": ("北达科他州", "中西部"),
        "Ohio": ("俄亥俄州", "中西部"),
        "Oklahoma": ("俄克拉何马州", "中南部"),
        "Oregon": ("俄勒冈州", "西北部"),
        "Pennsylvania": ("宾夕法尼亚州", "东北部"),
        "Rhode Island": ("罗德岛州", "东北部"),
        "South Carolina": ("南卡罗来纳州", "东南部"),
        "South Dakota": ("南达科他州", "中西部"),
        "Tennessee": ("田纳西州", "中南部"),
        "Texas": ("德克萨斯州", "南部"),
        "Utah": ("犹他州", "西部"),
        "Vermont": ("佛蒙特州", "东北部"),
        "Virginia": ("弗吉尼亚州", "东南部"),
        "Washington": ("华盛顿州", "西北部"),
        "West Virginia": ("西弗吉尼亚州", "东部"),
        "Wisconsin": ("威斯康星州", "中西部"),
        "Wyoming": ("怀俄明州", "西部")
    },
    "Canada": {
        # 加拿大13个省/地区完整列表
        "Alberta": ("阿尔伯塔省", "西部"),
        "British Columbia": ("不列颠哥伦比亚省", "西部"),
        "Manitoba": ("马尼托巴省", "中部"),
        "New Brunswick": ("新不伦瑞克省", "东部"),
        "Newfoundland and Labrador": ("纽芬兰与拉布拉多省", "东部"),
        "Northwest Territories": ("西北地区", "北部"),
        "Nova Scotia": ("新斯科舍省", "东部"),
        "Nunavut": ("努纳武特地区", "北部"),
        "Ontario": ("安大略省", "中部"),
        "Prince Edward Island": ("爱德华王子岛省", "东部"),
        "Quebec": ("魁北克省", "东部"),
        "Saskatchewan": ("萨斯喀彻温省", "中部"),
        "Yukon": ("育空地区", "西部")
    },
    "Japan": {
        # 日本47个都道府县完整列表
        "Hokkaido": ("北海道", "北部"),
        "Aomori": ("青森县", "东北地区"),
        "Iwate": ("岩手县", "东北地区"),
        "Miyagi": ("宫城县", "东北地区"),
        "Akita": ("秋田县", "东北地区"),
        "Yamagata": ("山形县", "东北地区"),
        "Fukushima": ("福岛县", "东北地区"),
        "Ibaraki": ("茨城县", "关东地区"),
        "Tochigi": ("栃木县", "关东地区"),
        "Gunma": ("群马县", "关东地区"),
        "Saitama": ("埼玉县", "关东地区"),
        "Chiba": ("千叶县", "关东地区"),
        "Tokyo": ("东京都", "关东地区"),
        "Kanagawa": ("神奈川县", "关东地区"),
        "Niigata": ("新潟县", "中部地区"),
        "Toyama": ("富山县", "中部地区"),
        "Ishikawa": ("石川县", "中部地区"),
        "Fukui": ("福井县", "中部地区"),
        "Yamanashi": ("山梨县", "中部地区"),
        "Nagano": ("长野县", "中部地区"),
        "Gifu": ("岐阜县", "中部地区"),
        "Shizuoka": ("静冈县", "中部地区"),
        "Aichi": ("爱知县", "中部地区"),
        "Mie": ("三重县", "近畿地区"),
        "Shiga": ("滋贺县", "近畿地区"),
        "Kyoto": ("京都府", "近畿地区"),
        "Osaka": ("大阪府", "近畿地区"),
        "Hyogo": ("兵库县", "近畿地区"),
        "Nara": ("奈良县", "近畿地区"),
        "Wakayama": ("和歌山县", "近畿地区"),
        "Tottori": ("鸟取县", "中国地区"),
        "Shimane": ("岛根县", "中国地区"),
        "Okayama": ("冈山县", "中国地区"),
        "Hiroshima": ("广岛县", "中国地区"),
        "Yamaguchi": ("山口县", "中国地区"),
        "Tokushima": ("德岛县", "四国地区"),
        "Kagawa": ("香川县", "四国地区"),
        "Ehime": ("爱媛县", "四国地区"),
        "Kochi": ("高知县", "四国地区"),
        "Fukuoka": ("福冈县", "九州地区"),
        "Saga": ("佐贺县", "九州地区"),
        "Nagasaki": ("长崎县", "九州地区"),
        "Kumamoto": ("熊本县", "九州地区"),
        "Oita": ("大分县", "九州地区"),
        "Miyazaki": ("宫崎县", "九州地区"),
        "Kagoshima": ("鹿儿岛县", "九州地区"),
        "Okinawa": ("冲绳县", "南部")
    },
    "Australia": {
        # 澳大利亚8个州/地区完整列表
        "New South Wales": ("新南威尔士州", "东南部"),
        "Victoria": ("维多利亚州", "东南部"),
        "Queensland": ("昆士兰州", "东北部"),
        "Western Australia": ("西澳大利亚州", "西部"),
        "South Australia": ("南澳大利亚州", "南部"),
        "Tasmania": ("塔斯马尼亚州", "东南部"),
        "Australian Capital Territory": ("澳大利亚首都领地", "东南部"),
        "Northern Territory": ("北领地", "北部")
    },
    "Brazil": {
        # 巴西26个州和1个联邦区完整列表
        "Acre": ("阿克州", "北部"),
        "Alagoas": ("阿拉戈斯州", "东北部"),
        "Amapa": ("阿马帕州", "北部"),
        "Amazonas": ("亚马孙州", "北部"),
        "Bahia": ("巴伊亚州", "东北部"),
        "Ceara": ("塞阿拉州", "东北部"),
        "Distrito Federal": ("联邦区", "中部"),
        "Espirito Santo": ("圣埃斯皮里图州", "东南部"),
        "Goias": ("戈亚斯州", "中部"),
        "Maranhao": ("马拉尼昂州", "东北部"),
        "Mato Grosso": ("马托格罗索州", "中部"),
        "Mato Grosso do Sul": ("南马托格罗索州", "中部"),
        "Minas Gerais": ("米纳斯吉拉斯州", "东南部"),
        "Para": ("帕拉州", "北部"),
        "Paraiba": ("帕拉伊巴州", "东北部"),
        "Parana": ("巴拉那州", "南部"),
        "Pernambuco": ("伯南布哥州", "东北部"),
        "Piaui": ("皮奥伊州", "东北部"),
        "Rio de Janeiro": ("里约热内卢州", "东南部"),
        "Rio Grande do Norte": ("北大河州", "东北部"),
        "Rio Grande do Sul": ("南大河州", "南部"),
        "Rondonia": ("朗多尼亚州", "北部"),
        "Roraima": ("罗赖马州", "北部"),
        "Santa Catarina": ("圣卡塔琳娜州", "南部"),
        "Sao Paulo": ("圣保罗州", "东南部"),
        "Sergipe": ("塞尔希培州", "东北部"),
        "Tocantins": ("托坎廷斯州", "北部")
    },
    "Germany": {
        # 德国16个州完整列表
        "Baden-Wurttemberg": ("巴登-符腾堡州", "西南部"),
        "Bavaria": ("巴伐利亚州", "东南部"),
        "Berlin": ("柏林州", "东部"),
        "Brandenburg": ("勃兰登堡州", "东部"),
        "Bremen": ("不来梅州", "西北部"),
        "Hamburg": ("汉堡州", "北部"),
        "Hesse": ("黑森州", "中部"),
        "Lower Saxony": ("下萨克森州", "西北部"),
        "Mecklenburg-Vorpommern": ("梅克伦堡-前波莫瑞州", "东北部"),
        "North Rhine-Westphalia": ("北莱茵-威斯特法伦州", "西部"),
        "Rhineland-Palatinate": ("莱茵兰-普法尔茨州", "西部"),
        "Saarland": ("萨尔州", "西南部"),
        "Saxony": ("萨克森州", "东部"),
        "Saxony-Anhalt": ("萨克森-安哈尔特州", "东部"),
        "Schleswig-Holstein": ("石勒苏益格-荷尔斯泰因州", "北部"),
        "Thuringia": ("图林根州", "中部")
    },
    "France": {
        # 法国18个大区完整列表（含海外）
        "Auvergne-Rhone-Alpes": ("奥弗涅-罗讷-阿尔卑斯大区", "东南部"),
        "Bourgogne-Franche-Comte": ("勃艮第-弗朗什-孔泰大区", "东部"),
        "Brittany": ("布列塔尼大区", "西北部"),
        "Centre-Val de Loire": ("中央-卢瓦尔河谷大区", "中部"),
        "Corsica": ("科西嘉大区", "东南部"),
        "Grand Est": ("大东部大区", "东北部"),
        "Hauts-de-France": ("上法兰西大区", "北部"),
        "Ile-de-France": ("法兰西岛大区", "北部"),
        "Normandy": ("诺曼底大区", "西北部"),
        "Nouvelle-Aquitaine": ("新阿基坦大区", "西南部"),
        "Occitanie": ("奥克西塔尼大区", "南部"),
        "Pays de la Loire": ("卢瓦尔河地区大区", "西部"),
        "Provence-Alpes-Cote d'Azur": ("普罗旺斯-阿尔卑斯-蓝色海岸大区", "东南部"),
        "Guadeloupe": ("瓜德罗普", "加勒比海"),
        "Martinique": ("马提尼克", "加勒比海"),
        "Guyane": ("法属圭亚那", "南美洲"),
        "La Reunion": ("留尼汪", "印度洋"),
        "Mayotte": ("马约特", "印度洋")
    },
    "Italy": {
        # 意大利20个大区完整列表
        "Abruzzo": ("阿布鲁佐大区", "中部"),
        "Aosta Valley": ("瓦莱达奥斯塔大区", "西北部"),
        "Apulia": ("普利亚大区", "南部"),
        "Basilicata": ("巴西利卡塔大区", "南部"),
        "Calabria": ("卡拉布里亚大区", "南部"),
        "Campania": ("坎帕尼亚大区", "南部"),
        "Emilia-Romagna": ("艾米利亚-罗马涅大区", "北部"),
        "Friuli-Venezia Giulia": ("弗留利-威尼斯朱利亚大区", "东北部"),
        "Lazio": ("拉齐奥大区", "中部"),
        "Liguria": ("利古里亚大区", "西北部"),
        "Lombardy": ("伦巴第大区", "北部"),
        "Marche": ("马尔凯大区", "中部"),
        "Molise": ("莫利塞大区", "南部"),
        "Piedmont": ("皮埃蒙特大区", "西北部"),
        "Sardinia": ("撒丁岛", "西部岛屿"),
        "Sicily": ("西西里岛", "南部岛屿"),
        "Trentino-Alto Adige": ("特伦蒂诺-上阿迪杰大区", "北部"),
        "Tuscany": ("托斯卡纳大区", "中部"),
        "Umbria": ("翁布里亚大区", "中部"),
        "Veneto": ("威尼托大区", "东北部")
    },
    "Spain": {
        # 西班牙17个自治区和2个自治市完整列表
        "Andalusia": ("安达卢西亚", "南部"),
        "Aragon": ("阿拉贡", "东北部"),
        "Asturias": ("阿斯图里亚斯", "北部"),
        "Balearic Islands": ("巴利阿里群岛", "东部岛屿"),
        "Basque Country": ("巴斯克地区", "北部"),
        "Canary Islands": ("加那利群岛", "西南部岛屿"),
        "Cantabria": ("坎塔布里亚", "北部"),
        "Castile and Leon": ("卡斯蒂利亚-莱昂", "西北部"),
        "Castilla-La Mancha": ("卡斯蒂利亚-拉曼恰", "中部"),
        "Catalonia": ("加泰罗尼亚", "东北部"),
        "Extremadura": ("埃斯特雷马杜拉", "西部"),
        "Galicia": ("加利西亚", "西北部"),
        "La Rioja": ("拉里奥哈", "北部"),
        "Madrid": ("马德里", "中部"),
        "Murcia": ("穆尔西亚", "东南部"),
        "Navarre": ("纳瓦拉", "北部"),
        "Valencian Community": ("瓦伦西亚自治区", "东部"),
        "Ceuta": ("休达", "北非"),
        "Melilla": ("梅利利亚", "北非")
    },
    "Mexico": {
        # 墨西哥32个州完整列表
        "Aguascalientes": ("阿瓜斯卡连特斯州", "中部"),
        "Baja California": ("下加利福尼亚州", "西北部"),
        "Baja California Sur": ("南下加利福尼亚州", "西北部"),
        "Campeche": ("坎佩切州", "东南部"),
        "Chiapas": ("恰帕斯州", "南部"),
        "Chihuahua": ("奇瓦瓦州", "北部"),
        "Coahuila": ("科阿韦拉州", "北部"),
        "Colima": ("科利马州", "西部"),
        "Durango": ("杜兰戈州", "西北部"),
        "Guanajuato": ("瓜纳华托州", "中部"),
        "Guerrero": ("格雷罗州", "南部"),
        "Hidalgo": ("伊达尔戈州", "东部"),
        "Jalisco": ("哈利斯科州", "西部"),
        "Mexico": ("墨西哥州", "中部"),
        "Michoacan": ("米却肯州", "西部"),
        "Morelos": ("莫雷洛斯州", "中部"),
        "Nayarit": ("纳亚里特州", "西部"),
        "Nuevo leon": ("新莱昂州", "东北部"),
        "Oaxaca": ("瓦哈卡州", "南部"),
        "Puebla": ("普埃布拉州", "东部"),
        "Queretaro": ("克雷塔罗州", "中部"),
        "Quintana Roo": ("金塔纳罗奥州", "东南部"),
        "San Luis Potosi": ("圣路易斯波托西州", "中部"),
        "Sinaloa": ("锡那罗亚州", "西北部"),
        "Sonora": ("索诺拉州", "西北部"),
        "Tabasco": ("塔巴斯科州", "东南部"),
        "Tamaulipas": ("塔毛利帕斯州", "东北部"),
        "Tlaxcala": ("特拉斯卡拉州", "东部"),
        "Veracruz": ("韦拉克鲁斯州", "东部"),
        "Yucatan": ("尤卡坦州", "东南部"),
        "Zacatecas": ("萨卡特卡斯州", "中部")
    },
    "Argentina": {
        # 阿根廷23个省和1个自治市完整列表
        "Buenos Aires": ("布宜诺斯艾利斯省", "东部"),
        "Catamarca": ("卡塔马卡省", "西北部"),
        "Chaco": ("查科省", "北部"),
        "Chubut": ("丘布特省", "南部"),
        "Cordoba": ("科尔多瓦省", "中部"),
        "Corrientes": ("科连特斯省", "东北部"),
        "Entre Rios": ("恩特雷里奥斯省", "东部"),
        "Formosa": ("福尔摩沙省", "北部"),
        "Jujuy": ("胡胡伊省", "西北部"),
        "La Pampa": ("拉潘帕省", "中部"),
        "La Rioja": ("拉里奥哈省", "西北部"),
        "Mendoza": ("门多萨省", "西部"),
        "Misiones": ("米西奥内斯省", "东北部"),
        "Neuquen": ("内乌肯省", "西部"),
        "Rio Negro": ("里奥内格罗省", "南部"),
        "Salta": ("萨尔塔省", "西北部"),
        "San Juan": ("圣胡安省", "西部"),
        "San Luis": ("圣路易斯省", "中部"),
        "Santa Cruz": ("圣克鲁斯省", "南部"),
        "Santa Fe": ("圣菲省", "东部"),
        "Santiago del Estero": ("圣地亚哥-德尔埃斯特罗省", "北部"),
        "Tierra del Fuego": ("火地岛省", "南部"),
        "Tucuman": ("图库曼省", "西北部"),
        "Ciudad Autonoma de Buenos Aires": ("布宜诺斯艾利斯自治市", "东部")
    },
    "South Africa": {
        # 南非9个省完整列表
        "Eastern Cape": ("东开普省", "东南部"),
        "Free State": ("自由邦省", "中部"),
        "Gauteng": ("豪登省", "东北部"),
        "KwaZulu-Natal": ("夸祖鲁-纳塔尔省", "东部"),
        "Limpopo": ("林波波省", "北部"),
        "Mpumalanga": ("姆普马兰加省", "东部"),
        "North West": ("西北省", "北部"),
        "Northern Cape": ("北开普省", "西部"),
        "Western Cape": ("西开普省", "西南部")
    },
    "Chile": {
        # 智利16个大区完整列表
        "Arica y Parinacota": ("阿里卡和帕里纳科塔大区", "最北部"),
        "Tarapaca": ("塔拉帕卡大区", "北部"),
        "Antofagasta": ("安托法加斯塔大区", "北部"),
        "Atacama": ("阿塔卡马大区", "北部"),
        "Coquimbo": ("科金博大区", "中北部"),
        "Valparaiso": ("瓦尔帕莱索大区", "中部"),
        "Metropolitana": ("首都大区", "中部"),
        "O'Higgins": ("奥希金斯大区", "中部"),
        "Maule": ("马乌莱大区", "中部"),
        "Nuble": ("纽布莱大区", "中部"),
        "Biobio": ("比奥比奥大区", "中南部"),
        "Araucania": ("阿劳卡尼亚大区", "南部"),
        "Los Rios": ("河流大区", "南部"),
        "Los Lagos": ("湖泊大区", "南部"),
        "Aysen": ("艾森大区", "南部"),
        "Magallanes": ("麦哲伦大区", "最南部")
    },
    "Turkey": {
        # 土耳其81个省完整列表（部分示例）
        "Adana": ("阿达纳省", "南部"),
        "Adiyaman": ("阿德亚曼省", "东南部"),
        "Afyonkarahisar": ("阿菲永卡拉希萨尔省", "西部"),
        "Agri": ("阿勒省", "东部"),
        "Aksaray": ("阿克萨赖省", "中部"),
        "Amasya": ("阿马西亚省", "北部"),
        "Ankara": ("安卡拉省", "中部"),
        "Antalya": ("安塔利亚省", "南部"),
        "Ardahan": ("阿尔达汉省", "东北部"),
        "Artvin": ("阿尔特温省", "东北部"),
        # 继续添加其他省份...
        "Istanbul": ("伊斯坦布尔省", "西北部"),
        "Izmir": ("伊兹密尔省", "西部"),
        # ...其他省份
        "Zonguldak": ("宗古尔达克省", "北部")
    },
    "Russia": {
        # 共和国（22个）
        "Adygea": ("阿迪格共和国", "西南部"),
        "Altai": ("阿尔泰共和国", "西伯利亚南部"),
        "Bashkortostan": ("巴什科尔托斯坦共和国", "欧洲部分东部"),
        "Buryatia": ("布里亚特共和国", "西伯利亚东南部"),
        "Chechnya": ("车臣共和国", "西南部高加索地区"),
        "Chuvashia": ("楚瓦什共和国", "欧洲部分中部"),
        "Crimea": ("克里米亚共和国", "西南部半岛*"),
        "Dagestan": ("达吉斯坦共和国", "西南部高加索地区"),
        "Ingushetia": ("印古什共和国", "西南部高加索地区"),
        "Kabardino-Balkaria": ("卡巴尔达-巴尔卡尔共和国", "西南部高加索地区"),
        "Kalmykia": ("卡尔梅克共和国", "西南部"),
        "Karachay-Cherkessia": ("卡拉恰伊-切尔克斯共和国", "西南部高加索地区"),
        "Karelia": ("卡累利阿共和国", "西北部"),
        "Khakassia": ("哈卡斯共和国", "西伯利亚中部"),
        "Komi": ("科米共和国", "欧洲部分东北部"),
        "Mari El": ("马里埃尔共和国", "欧洲部分中部"),
        "Mordovia": ("莫尔多瓦共和国", "欧洲部分中部"),
        "North Ossetia-Alania": ("北奥塞梯-阿兰共和国", "西南部高加索地区"),
        "Sakha (Yakutia)": ("萨哈（雅库特）共和国", "远东北部"),
        "Tatarstan": ("鞑靼斯坦共和国", "欧洲部分中部"),
        "Tuva": ("图瓦共和国", "西伯利亚南部"),
        "Udmurtia": ("乌德穆尔特共和国", "欧洲部分中部"),
        
        # 边疆区（9个）
        "Altai Krai": ("阿尔泰边疆区", "西伯利亚西南部"),
        "Kamchatka Krai": ("堪察加边疆区", "远东东部半岛"),
        "Khabarovsk Krai": ("哈巴罗夫斯克边疆区", "远东东南部"),
        "Krasnodar Krai": ("克拉斯诺达尔边疆区", "西南部"),
        "Krasnoyarsk Krai": ("克拉斯诺亚尔斯克边疆区", "西伯利亚中部"),
        "Perm Krai": ("彼尔姆边疆区", "欧洲部分东部"),
        "Primorsky Krai": ("滨海边疆区", "远东东南部"),
        "Stavropol Krai": ("斯塔夫罗波尔边疆区", "西南部"),
        "Zabaykalsky Krai": ("外贝加尔边疆区", "西伯利亚东南部"),
        
        # 州（46个）
        "Amur Oblast": ("阿穆尔州", "远东南部"),
        "Arkhangelsk Oblast": ("阿尔汉格尔斯克州", "欧洲部分北部"),
        "Astrakhan Oblast": ("阿斯特拉罕州", "欧洲部分南部"),
        "Belgorod Oblast": ("别尔哥罗德州", "欧洲部分西南部"),
        "Bryansk Oblast": ("布良斯克州", "欧洲部分西部"),
        "Chelyabinsk Oblast": ("车里雅宾斯克州", "乌拉尔地区"),
        "Irkutsk Oblast": ("伊尔库茨克州", "西伯利亚南部"),
        "Ivanovo Oblast": ("伊万诺沃州", "欧洲部分中部"),
        "Kaliningrad Oblast": ("加里宁格勒州", "欧洲最西部飞地"),
        "Kaluga Oblast": ("卡卢加州", "欧洲部分西部"),
        "Kemerovo Oblast": ("克麦罗沃州", "西伯利亚南部"),
        "Kirov Oblast": ("基洛夫州", "欧洲部分东部"),
        "Kostroma Oblast": ("科斯特罗马州", "欧洲部分中部"),
        "Kurgan Oblast": ("库尔干州", "乌拉尔地区东部"),
        "Kursk Oblast": ("库尔斯克州", "欧洲部分西南部"),
        "Leningrad Oblast": ("列宁格勒州", "欧洲部分西北部"),
        "Lipetsk Oblast": ("利佩茨克州", "欧洲部分中部"),
        "Magadan Oblast": ("马加丹州", "远东东北部"),
        "Moscow Oblast": ("莫斯科州", "欧洲部分中部"),
        "Murmansk Oblast": ("摩尔曼斯克州", "欧洲部分西北部"),
        "Nizhny Novgorod Oblast": ("下诺夫哥罗德州", "欧洲部分中部"),
        "Novgorod Oblast": ("诺夫哥罗德州", "欧洲部分西北部"),
        "Novosibirsk Oblast": ("新西伯利亚州", "西伯利亚南部"),
        "Omsk Oblast": ("鄂木斯克州", "西伯利亚西南部"),
        "Orenburg Oblast": ("奥伦堡州", "欧洲部分东南部"),
        "Oryol Oblast": ("奥廖尔州", "欧洲部分西部"),
        "Penza Oblast": ("奔萨州", "欧洲部分中部"),
        "Pskov Oblast": ("普斯科夫州", "欧洲部分西北部"),
        "Rostov Oblast": ("罗斯托夫州", "欧洲部分西南部"),
        "Ryazan Oblast": ("梁赞州", "欧洲部分中部"),
        "Sakhalin Oblast": ("萨哈林州", "远东东部岛屿"),
        "Samara Oblast": ("萨马拉州", "欧洲部分东部"),
        "Saratov Oblast": ("萨拉托夫州", "欧洲部分南部"),
        "Smolensk Oblast": ("斯摩棱斯克州", "欧洲部分西部"),
        "Sverdlovsk Oblast": ("斯维尔德洛夫斯克州", "乌拉尔地区"),
        "Tambov Oblast": ("坦波夫州", "欧洲部分中部"),
        "Tomsk Oblast": ("托木斯克州", "西伯利亚中部"),
        "Tula Oblast": ("图拉州", "欧洲部分中部"),
        "Tver Oblast": ("特维尔州", "欧洲部分西北部"),
        "Tyumen Oblast": ("秋明州", "西伯利亚西部"),
        "Ulyanovsk Oblast": ("乌里扬诺夫斯克州", "欧洲部分东部"),
        "Vladimir Oblast": ("弗拉基米尔州", "欧洲部分中部"),
        "Volgograd Oblast": ("伏尔加格勒州", "欧洲部分南部"),
        "Vologda Oblast": ("沃洛格达州", "欧洲部分北部"),
        "Voronezh Oblast": ("沃罗涅日州", "欧洲部分西南部"),
        "Yaroslavl Oblast": ("雅罗斯拉夫尔州", "欧洲部分中部"),
        
        # 联邦直辖市（3个）
        "Moscow": ("莫斯科", "欧洲部分中部"),
        "Saint Petersburg": ("圣彼得堡", "欧洲部分西北部"),
        "Sevastopol": ("塞瓦斯托波尔", "西南部半岛*"),
        
        # 自治州（1个）
        "Jewish Autonomous Oblast": ("犹太自治州", "远东南部"),
        
        # 自治区（4个）
        "Chukotka Autonomous Okrug": ("楚科奇自治区", "远东东北部"),
        "Khanty-Mansi Autonomous Okrug": ("汉特-曼西自治区", "西伯利亚西部"),
        "Nenets Autonomous Okrug": ("涅涅茨自治区", "欧洲部分北部"),
        "Yamalo-Nenets Autonomous Okrug": ("亚马尔-涅涅茨自治区", "西伯利亚西北部")
    },
    "Malaysia": {
    # 马来半岛（西马）
    "Johor": ("柔佛州", "南部"),
    "Kedah": ("吉打州", "西北部"),
    "Kelantan": ("吉兰丹州", "东北部"),
    "Malacca": ("马六甲州", "西南部"),
    "Negeri Sembilan": ("森美兰州", "中南部"),
    "Pahang": ("彭亨州", "东部"),
    "Penang": ("槟城州", "西北部"),
    "Perak": ("霹雳州", "西部"),
    "Perlis": ("玻璃市州", "最北部"),
    "Selangor": ("雪兰莪州", "中部"),
    "Terengganu": ("登嘉楼州", "东北部"),
    
    # 婆罗洲（东马）
    "Sabah": ("沙巴州", "东马北部"),
    "Sarawak": ("砂拉越州", "东马西部"),
    
    # 联邦直辖区
    "Kuala Lumpur": ("吉隆坡", "中部"),
    "Labuan": ("纳闽", "东马西部沿海"),
    "Putrajaya": ("布城", "中部")
    },
    "Indonesia": {
    # 苏门答腊
    "Aceh": ("亚齐特区", "最西北端"),
    "North Sumatra": ("北苏门答腊省", "西北部"),
    "West Sumatra": ("西苏门答腊省", "中西部"),
    "Riau": ("廖内省", "中部东岸"),
    "Riau Islands": ("廖内群岛省", "马六甲海峡东部"),
    "Jambi": ("占碑省", "中部"),
    "South Sumatra": ("南苏门答腊省", "东南部"),
    "Bengkulu": ("明古鲁省", "西南部"),
    "Lampung": ("楠榜省", "最南端"),
    
    # 爪哇
    "Banten": ("万丹省", "西北部"),
    "Jakarta": ("雅加达特区", "西北部"),
    "West Java": ("西爪哇省", "西部"),
    "Central Java": ("中爪哇省", "中部"),
    "Yogyakarta": ("日惹特区", "中南部"),
    "East Java": ("东爪哇省", "东部"),
    
    # 小巽他群岛
    "Bali": ("巴厘省", "东部岛屿"),
    "West Nusa Tenggara": ("西努沙登加拉省", "中部岛屿"),
    "East Nusa Tenggara": ("东努沙登加拉省", "东南部岛屿"),
    
    # 加里曼丹
    "West Kalimantan": ("西加里曼丹省", "西部"),
    "Central Kalimantan": ("中加里曼丹省", "中部"),
    "South Kalimantan": ("南加里曼丹省", "东南部"),
    "East Kalimantan": ("东加里曼丹省", "东北部"),
    "North Kalimantan": ("北加里曼丹省", "最北部"),
    
    # 苏拉威西
    "North Sulawesi": ("北苏拉威西省", "东北部半岛"),
    "Gorontalo": ("哥伦打洛省", "北部半岛"),
    "Central Sulawesi": ("中苏拉威西省", "中部"),
    "West Sulawesi": ("西苏拉威西省", "西部"),
    "South Sulawesi": ("南苏拉威西省", "南部"),
    "Southeast Sulawesi": ("东南苏拉威西省", "东南部"),
    
    # 马鲁古和巴布亚
    "Maluku": ("马鲁古省", "东部群岛"),
    "North Maluku": ("北马鲁古省", "东北部群岛"),
    "West Papua": ("西巴布亚省", "最东部"),
    "Papua": ("巴布亚省", "最东部")
    },"India": {
    # 北部地区
    "Jammu and Kashmir": ("查谟和克什米尔", "最北部"),
    "Ladakh": ("拉达克", "最北部高原"),
    "Himachal Pradesh": ("喜马偕尔邦", "北部山地"),
    "Punjab": ("旁遮普邦", "西北部"),
    "Uttarakhand": ("北阿坎德邦", "北部喜马拉雅山麓"),
    "Haryana": ("哈里亚纳邦", "北部平原"),
    "Delhi": ("德里首都辖区", "北部中部"),

    # 西部地区
    "Rajasthan": ("拉贾斯坦邦", "西部沙漠区"),
    "Gujarat": ("古吉拉特邦", "西部海岸"),
    "Dadra and Nagar Haveli and Daman and Diu": ("达德拉-纳加尔哈维利和达曼-第乌", "西部沿海"),

    # 中部地区
    "Madhya Pradesh": ("中央邦", "中部"),
    "Chhattisgarh": ("恰蒂斯加尔邦", "中东部"),

    # 东部地区
    "Bihar": ("比哈尔邦", "东北部平原"),
    "Jharkhand": ("贾坎德邦", "东部森林区"),
    "West Bengal": ("西孟加拉邦", "东部海岸"),
    "Odisha": ("奥里萨邦", "东海岸"),

    # 南部地区
    "Andhra Pradesh": ("安得拉邦", "东南海岸"),
    "Telangana": ("特伦甘纳邦", "南部高原"),
    "Karnataka": ("卡纳塔克邦", "西南高原"),
    "Tamil Nadu": ("泰米尔纳德邦", "最南部"),
    "Kerala": ("喀拉拉邦", "西南海岸"),

    # 东北部地区（七姊妹邦）
    "Assam": ("阿萨姆邦", "东北部"),
    "Arunachal Pradesh": ("阿鲁纳恰尔邦", "最东北部"),
    "Manipur": ("曼尼普尔邦", "东部内陆"),
    "Meghalaya": ("梅加拉亚邦", "东北部高原"),
    "Mizoram": ("米佐拉姆邦", "东南部山地"),
    "Nagaland": ("那加兰邦", "东北部山地"),
    "Tripura": ("特里普拉邦", "东部小平原"),

    # 联邦属地（其余）
    "Puducherry": ("本地治里", "东南沿海多个飞地"),
    "Andaman and Nicobar Islands": ("安达曼-尼科巴群岛", "东南海上群岛"),
    "Lakshadweep": ("拉克沙群岛", "西南阿拉伯海群岛"),
    "Chandigarh": ("昌迪加尔", "北部联合首府")
    },

    "Philippines" :{
    # 吕宋岛
    "Ilocos": ("伊罗戈斯大区", "西北部"),
    "Cagayan Valley": ("卡加延河谷大区", "东北部"),
    "Central Luzon": ("中吕宋大区", "中部"),
    "Calabarzon": ("卡拉巴松大区", "西南部"),
    "Mimaropa": ("民马罗巴大区", "中西部群岛"),
    "Bicol": ("比科尔大区", "东南部"),
    "Cordillera": ("科迪勒拉行政区", "北部山区"),
    "Metro Manila": ("国家首都区", "中部"),
    
    # 米沙鄢群岛
    "Western Visayas": ("西米沙鄢大区", "中部偏西"),
    "Central Visayas": ("中米沙鄢大区", "中部"),
    "Eastern Visayas": ("东米沙鄢大区", "中东部"),
    
    # 棉兰老岛
    "Zamboanga Peninsula": ("三宝颜半岛大区", "西部半岛"),
    "Northern Mindanao": ("北棉兰老大区", "北部"),
    "Davao": ("达沃大区", "东南部"),
    "Soccsksargen": ("索克斯萨尔根大区", "南部"),
    "Caraga": ("卡拉加大区", "东北部"),
    "Bangsamoro": ("邦萨摩洛自治区", "西南部")
    }
}

# 主要城市数据（每个国家5个主要城市）
CITIES = {
    "United States": {
        "New York": ("纽约", "纽约州", "东北部"),
        "Los Angeles": ("洛杉矶", "加利福尼亚州", "西部"),
        "Chicago": ("芝加哥", "伊利诺伊州", "中西部"),
        "Houston": ("休斯顿", "德克萨斯州", "南部"),
        "Phoenix": ("菲尼克斯", "亚利桑那州", "西南部")
    },
    "Japan": {
        "Tokyo": ("东京", "东京都", "关东地区"),
        "Yokohama": ("横滨", "神奈川县", "关东地区"),
        "Osaka": ("大阪", "大阪府", "近畿地区"),
        "Nagoya": ("名古屋", "爱知县", "中部地区"),
        "Sapporo": ("札幌", "北海道", "北部地区")
    },
    "Australia": {
        "Sydney": ("悉尼", "新南威尔士州", "东南部"),
        "Melbourne": ("墨尔本", "维多利亚州", "东南部"),
        "Brisbane": ("布里斯班", "昆士兰州", "东北部"),
        "Perth": ("珀斯", "西澳大利亚州", "西部"),
        "Adelaide": ("阿德莱德", "南澳大利亚州", "南部")
    },
    "Germany": {
        "Berlin": ("柏林", "柏林州", "东部"),
        "Hamburg": ("汉堡", "汉堡州", "北部"),
        "Munich": ("慕尼黑", "巴伐利亚州", "东南部"),
        "Cologne": ("科隆", "北莱茵-威斯特法伦州", "西部"),
        "Frankfurt": ("法兰克福", "黑森州", "中部")
    },
    "France": {
        "Paris": ("巴黎", "法兰西岛大区", "北部"),
        "Marseille": ("马赛", "普罗旺斯-阿尔卑斯- 蓝色海岸大区", "东南部"),
        "Lyon": ("里昂", "奥弗涅-罗讷-阿尔卑斯大区", "东南部"),
        "Toulouse": ("图卢兹", "奥克西塔尼大区", "南部"),
        "Nice": ("尼斯", "普罗旺斯-阿尔卑斯-蓝色海岸大区", "东南部")
    },
    "Italy": {
        "Rome": ("罗马", "拉齐奥大区", "中部"),
        "Milan": ("米兰", "伦巴第大区", "北部"),
        "Naples": ("那不勒斯", "坎帕尼亚大区", "南部"),
        "Turin": ("都灵", "皮埃蒙特大区", "西北部"),
        "Palermo": ("巴勒莫", "西西里岛", "南部")
    },
    "Spain": {
        "Madrid": ("马德里", "马德里自治区", "中部"),
        "Barcelona": ("巴塞罗那", "加泰罗尼亚", "东北部"),
        "Valencia": ("瓦伦西亚", "瓦伦西亚自治区", "东部"),
        "Seville": ("塞维利亚", "安达卢西亚", "南部"),
        "Zaragoza": ("萨拉戈萨", "阿拉贡", "东北部")
    },
    "Brazil": {
        "Sao Paulo": ("圣保罗", "圣保罗州", "东南部"),
        "Rio de Janeiro": ("里约热内卢", "里约热内卢州", "东南部"),
        "Brasilia": ("巴西利亚", "联邦区", "中部"),
        "Salvador": ("萨尔瓦多", "巴伊亚州", "东北部"),
        "Fortaleza": ("福塔莱萨", "塞阿拉州", "东北部")
    },
    "Mexico": {
        "Mexico City": ("墨西哥城", "联邦区", "中部"),
        "Guadalajara": ("瓜达拉哈拉", "哈利斯科州", "西部"),
        "Monterrey": ("蒙特雷", "新莱昂州", "东北部"),
        "Puebla": ("普埃布拉", "普埃布拉州", "东部"),
        "Tijuana": ("蒂华纳", "下加利福尼亚州", "西北部")
    },
    "Russia": {
        "Moscow": ("莫斯科", "莫斯科", "国家西部中心"),
        "Saint Petersburg": ("圣彼得堡", "圣彼得堡", "西北部"),
        "Novosibirsk": ("新西伯利亚", "新西伯利亚州", "西伯利亚南部"),
        "Yekaterinburg": ("叶卡捷琳堡", "斯维尔德洛夫斯克州", "乌拉尔地区"),
        "Kazan": ("喀山", "鞑靼斯坦共和国", "伏尔加地区中部"),
        "Nizhny Novgorod": ("下诺夫哥罗德", "下诺夫哥罗德州", "欧洲部分中部"),
        "Chelyabinsk": ("车里雅宾斯克", "车里雅宾斯克州", "乌拉尔地区"),
        "Samara": ("萨马拉", "萨马拉州", "伏尔加地区南部"),
        "Omsk": ("鄂木斯克", "鄂木斯克州", "西伯利亚西南部"),
        "Rostov-on-Don": ("顿河畔罗斯托夫", "罗斯托夫州", "西南部")
    }
}

class GeoGuessWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoGuess - 高级地理识别")
        self.setAcceptDrops(True)
        self.setFixedSize(1000, 800)  # 增大窗口尺寸
        
        self.init_ui()
        self.load_model()
        
        self.lane_detector = UFLDLaneDetector()
    def init_ui(self):
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()
        
        # 拖拽区域
        self.drop_label = QLabel("拖拽街景图片到此处", self)
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                padding: 30px;
                font-size: 16px;
                color: #666;
            }
            QLabel:hover {
                background-color: #f5f5f5;
            }
        """)
        
        # 截图按钮
        self.screenshot_btn = QPushButton("截图识别", self)
        self.screenshot_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        
        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        # 结果展示区域（可滚动）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout()
        self.result_container.setLayout(self.result_layout)
        
        self.scroll_area.setWidget(self.result_container)
        
        self.layout.addWidget(self.drop_label)
        self.layout.addWidget(self.screenshot_btn)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.scroll_area)
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def load_model(self):
        """加载模型"""
        try:
            self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
            self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=False)
            self.drop_label.setText("准备就绪，请拖拽图片")
        except Exception as e:
            self.drop_label.setText(f"模型加载失败: {str(e)}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_image(file_path)
                break

    def take_screenshot(self):
        """截取屏幕并识别"""
        self.hide()
        QTimer.singleShot(500, self._capture_screen)

    def _capture_screen(self):
        """实际执行截图操作"""
        try:
            screen = QApplication.primaryScreen()
            screenshot = screen.grabWindow(0)
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            screenshot.save(temp_file.name, "PNG")
            
            self.process_image(temp_file.name)
            self.show()
            
        except Exception as e:
            self._clear_results()
            self._add_result_label(f"截图失败: {str(e)}", "error")
            self.show()

    def process_image(self, image_path):
        """处理图片并显示结果"""
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._clear_results()
        
        try:
            image = Image.open(image_path).convert("RGB")
            # 国家识别（前5个）
            country_results = self._predict_location(image, list(COUNTRIES.keys()), "国家", top_k=5)

            # ====== 新增车道线检测 ======
            lane_info = self.detect_lane_lines(image_path)
            self._add_result_label(f"🚗 车道线检测结果：{lane_info}", "header")
            # ===========================
            # 显示国家结果
            self._add_result_label("🌍 国家识别结果 (前5名):", "header")
            for i, (country, prob) in enumerate(country_results):
                self._add_result_label(
                    f"{i+1}. {country} ({COUNTRIES[country]}) - {prob:.1f}%", 
                    "country"
                )
                
                # 如果这个国家有州/省数据，则识别其州/省
                if country in REGIONS:
                    regions = list(REGIONS[country].keys())
                    region_results = self._predict_location(image, regions, f"{COUNTRIES[country]}州/省", top_k=3)
                    
                    self._add_result_label("   🏙️ 最可能的州/省:", "subheader")
                    for j, (region, region_prob) in enumerate(region_results):
                        chn_name, location = REGIONS[country][region]
                        self._add_result_label(
                            f"     {j+1}. {region} ({chn_name}, {location}) - {region_prob:.1f}%",
                            "region"
                        )
                
                # 如果这个国家有城市数据，则识别其城市
                if country in CITIES:
                    cities = list(CITIES[country].keys())
                    city_results = self._predict_location(image, cities, f"{COUNTRIES[country]}城市", top_k=3)
                    
                    self._add_result_label("   🏢 最可能的城市:", "subheader")
                    for k, (city, city_prob) in enumerate(city_results):
                        chn_name, state, location = CITIES[country][city]
                        self._add_result_label(
                            f"     {k+1}. {city} ({chn_name}, {state}, {location}) - {city_prob:.1f}%",
                            "city"
                        )
            
        except Exception as e:
            self._add_result_label(f"错误: {str(e)}", "error")
        finally:
            self.progress.setVisible(False)

    def _predict_location(self, image, locations, location_type, top_k=3):
        """预测位置并返回结果"""
        inputs = self.processor(
            text=[f"a photo of {location}, {location_type}" for location in locations],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
        top_results = torch.topk(probs, min(top_k, len(locations)))
        
        results = []
        for i in range(top_results.indices.shape[1]):
            location = locations[top_results.indices[0][i].item()]
            prob = top_results.values[0][i].item() * 100
            results.append((location, prob))
        
        return results

    def _clear_results(self):
        """清除之前的结果"""
        for i in reversed(range(self.result_layout.count())): 
            self.result_layout.itemAt(i).widget().setParent(None)

    def _add_result_label(self, text, style_type="normal"):
        """添加结果标签"""
        label = QLabel(text)
        label.setWordWrap(True)
        
        # 根据不同类型设置不同样式
        if style_type == "header":
            label.setStyleSheet("""
                font-size: 16px; 
                font-weight: bold; 
                color: #2c3e50;
                margin-top: 10px;
                margin-bottom: 5px;
            """)
        elif style_type == "country":
            label.setStyleSheet("""
                font-size: 14px; 
                font-weight: bold;
                color: #2980b9;
                margin-left: 10px;
                margin-bottom: 3px;
            """)
        elif style_type == "subheader":
            label.setStyleSheet("""
                font-size: 13px;
                font-weight: bold;
                color: #16a085;
                margin-left: 20px;
                margin-top: 5px;
            """)
        elif style_type == "region":
            label.setStyleSheet("""
                font-size: 12px;
                color: #34495e;
                margin-left: 30px;
            """)
        elif style_type == "city":
            label.setStyleSheet("""
                font-size: 12px;
                color: #7f8c8d;
                margin-left: 30px;
                margin-bottom: 5px;
            """)
        elif style_type == "error":
            label.setStyleSheet("""
                font-size: 14px;
                color: #e74c3c;
                font-weight: bold;
            """)
        else:  # normal
            label.setStyleSheet("font-size: 13px;")
        
        self.result_layout.addWidget(label)

    def detect_lane_lines(self, image_path):
        """用UFLD检测道路线，返回类型描述"""
        try:
            result = self.lane_detector.detect(image_path)
            lane_count = result["lane_count"]
            colors = result["colors"]
            if lane_count == 0:
                return "未检测到明显道路线"
            color_str = ", ".join(colors)
            return f"检测到{lane_count}条道路线，颜色：{color_str}"
        except Exception as e:
            return f"检测失败: {e}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeoGuessWindow()
    window.show()
    sys.exit(app.exec_())
