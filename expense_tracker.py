import sys
import csv
import json
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QLineEdit,
                             QTableWidget, QTableWidgetItem, QComboBox, QLabel, QDateEdit, QMessageBox, QFileDialog,
                             QInputDialog, QStackedWidget, QTextEdit, QGroupBox, QHeaderView, QStatusBar, QSpacerItem,
                             QSizePolicy, QGridLayout, QStyle)  
from PyQt5.QtCore import QTimer, QDate, QSettings, Qt, QSize, QEvent
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QKeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import matplotlib.style as mplstyle

# Используем стиль для графиков
mplstyle.use('seaborn-v0_8-darkgrid')

DATA_FILE = "data.json"
SETTINGS_ORG = "MyCompany"
SETTINGS_APP = "ExpenseTrackerModern"

# --- Стили (QSS) ---
STYLESHEET = """
QWidget {
    background-color: #2E2E2E;
    color: #E0E0E0;
    font-size: 10pt; /* Базовый размер шрифта */
    font-family: 'Segoe UI', Arial, sans-serif; /* Более современный шрифт */
}

QMainWindow {
    background-color: #2E2E2E;
}

QStackedWidget {
    background-color: #353535;
    border: none;
}

/* ---- Стили для главного меню ---- */
MainMenu QLabel#titleLabel {
    font-size: 22pt; /* Крупнее */
    font-weight: bold;
    color: #50C878; /* Изумрудный */
    padding-bottom: 25px;
    margin-top: 20px; /* Отступ сверху */
}

MainMenu QPushButton {
    background-color: #50C878; /* Изумрудный */
    border: none;
    color: #1E1E1E; /* Темный текст на светлой кнопке */
    padding: 16px 35px;
    text-align: center;
    text-decoration: none;
    font-size: 14pt;
    font-weight: bold;
    margin: 10px 5px;
    border-radius: 10px; /* Более скругленные */
    min-width: 250px;
    transition: background-color 0.3s ease; /* Плавный переход для hover */
}

MainMenu QPushButton:hover {
    background-color: #45B86A; /* Чуть темнее при наведении */
    color: #FFFFFF; /* Белый текст при наведении */
}

/* ---- Общие стили для виджетов ввода ---- */
QLineEdit, QComboBox, QDateEdit {
    background-color: #3C3C3C;
    border: 1px solid #5A5A5A; /* Чуть светлее рамка */
    padding: 8px; /* Больше отступ */
    border-radius: 6px; /* Более скругленные */
    color: #E0E0E0;
    font-size: 10pt;
}

QLineEdit:focus, QComboBox:focus, QDateEdit:focus {
    border: 1px solid #50C878; /* Изумрудная рамка при фокусе */
    background-color: #424242; /* Слегка другой фон при фокусе */
}

QComboBox::drop-down {
    border: none;
    background-color: transparent;
    width: 20px; /* Шире область стрелки */
}

QComboBox::down-arrow {
    image: url(none); /* Убираем стандартную стрелку, если используем иконку темы */
    /* Можно вернуть свою иконку: image: url(path/to/down_arrow.png); */
    /* Или использовать символ: content: "▼"; color: # AAAAAA; */
    padding-right: 5px; /* Отступ для стрелки */

}

QDateEdit::up-button, QDateEdit::down-button {
     width: 18px; /* Немного больше кнопки */
}

/* ---- Стили для кнопок управления (общие) ---- */
QPushButton {
    background-color: #007BFF; /* Синий по умолчанию */
    color: white;
    padding: 10px 18px; /* Больше вертикальный отступ */
    border: none;
    border-radius: 6px;
    font-size: 10pt;
    font-weight: 500; /* Средняя жирность */
    transition: background-color 0.2s ease, border 0.2s ease;
    margin: 2px 0; /* Небольшой вертикальный отступ */
    text-align: left; /* Текст и иконка слева */
    padding-left: 10px; /* Отступ для иконки */
}

QPushButton:hover {
    background-color: #0056b3;
    border: 1px solid #007BFF; /* Рамка в цвет кнопки при наведении */
}

/* --- Стили для конкретных кнопок (переопределение цвета и ховера) --- */
QPushButton#addButton { background-color: #28a745; }
QPushButton#addButton:hover { background-color: #218838; border-color: #28a745; }

QPushButton#deleteButton { background-color: #dc3545; }
QPushButton#deleteButton:hover { background-color: #c82333; border-color: #dc3545; }

QPushButton#budgetButton { background-color: #17a2b8; }
QPushButton#budgetButton:hover { background-color: #138496; border-color: #17a2b8;}

QPushButton#monthlyButton { background-color: #6c757d; }
QPushButton#monthlyButton:hover { background-color: #5a6268; border-color: #6c757d;}

QPushButton#forecastButton { background-color: #ffc107; color: #333; }
QPushButton#forecastButton:hover { background-color: #e0a800; border-color: #ffc107;}

QPushButton#filterButton { background-color: #6f42c1; }
QPushButton#filterButton:hover { background-color: #5a32a3; border-color: #6f42c1;}

QPushButton#resetFilterButton { background-color: #fd7e14; }
QPushButton#resetFilterButton:hover { background-color: #e66a0a; border-color: #fd7e14;}

QPushButton#exportButton { background-color: #343a40; }
QPushButton#exportButton:hover { background-color: #23272b; border-color: #343a40;}

QPushButton#manageCategoriesButton { background-color: #20c997;}
QPushButton#manageCategoriesButton:hover { background-color: #1ba17c; border-color: #20c997;}

/* Кнопка Назад/Меню */
QPushButton#backButton {
    background-color: #6c757d; /* Серый */
    text-align: center; /* Текст по центру для этой кнопки */
    padding-left: 18px; /* Компенсируем левый padding по умолчанию*/
}
QPushButton#backButton:hover { background-color: #5a6268; border-color: #6c757d;}


/* ---- Стили для таблицы ---- */
QTableWidget {
    background-color: #3C3C3C;
    border: 1px solid #5A5A5A;
    gridline-color: #5A5A5A;
    alternate-background-color: #424242; /* Чуть светлее для чередования */
    selection-background-color: #50C878; /* Изумрудный для выделения */
    selection-color: #1E1E1E; /* Активный цвет текста выделения */
    font-size: 9.5pt; /* Чуть меньше шрифт в таблице */
}

QHeaderView::section {
    background-color: #4A4A4A;
    color: #E0E0E0;
    padding: 7px 5px; /* Больше вертикальный отступ */
    border: 1px solid #5A5A5A;
    border-bottom: 2px solid #50C878; /* Акцентная нижняя граница */
    font-weight: bold;
    font-size: 10pt;
}

/* ---- Стили для QGroupBox ---- */
QGroupBox {
    border: 1px solid #5A5A5A;
    border-radius: 8px; /* Более скругленные */
    margin-top: 12px;
    background-color: #383838;
    padding: 10px; /* Внутренний отступ для содержимого группы */
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px 15px; /* Больше горизонтальный отступ */
    margin-left: 10px; /* Отступ от левого края */
    color: #50C878; /* Изумрудный */
    font-weight: bold;
    font-size: 11pt; /* Крупнее заголовок группы */
    background-color: #404040; /* Фон для заголовка */
    border-radius: 4px;
}

/* ---- Стили для QTextEdit (Обучение, Авторы) ---- */
QTextEdit {
     background-color: #3C3C3C;
     border: 1px solid #5A5A5A;
     border-radius: 6px;
     padding: 10px;
     font-size: 10.5pt; /* Чуть крупнее текст */
}

/* ---- Стили для статус-бара ---- */
QStatusBar {
    background-color: #4A4A4A;
    color: #CCCCCC; /* Светло-серый текст */
    font-size: 9pt;
}

QStatusBar::item {
    border: none;
}

/* ---- Стили для Matplotlib Canvas ---- */
#figureCanvas { /* Добавляем objectName для точности */
    border: 1px solid #5A5A5A;
    border-radius: 6px;
    background-color: #3C3C3C; /* Явный фон */
}
"""

# Определяем константы для имен объектов (для стилей QSS)
ADD_BTN_ID = "addButton"
DEL_BTN_ID = "deleteButton"
BUDGET_BTN_ID = "budgetButton"
MONTHLY_BTN_ID = "monthlyButton"
FORECAST_BTN_ID = "forecastButton"
FILTER_BTN_ID = "filterButton"
RESET_FILTER_BTN_ID = "resetFilterButton"
EXPORT_BTN_ID = "exportButton"
MANAGE_CAT_BTN_ID = "manageCategoriesButton"
BACK_BTN_ID = "backButton"
TITLE_LABEL_ID = "titleLabel"
FIGURE_CANVAS_ID = "figureCanvas"


class MainMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MainMenuWidget")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)  # Отступы
        self.main_layout.addSpacerItem(QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.title_label = QLabel("Ассистент по расходам")
        self.title_label.setObjectName(TITLE_LABEL_ID)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        self.buttons_layout = QVBoxLayout()
        self.buttons_layout.setAlignment(Qt.AlignCenter)
        self.buttons_layout.setSpacing(15)


        style = QApplication.instance().style()
        icon_program = style.standardIcon(QStyle.SP_MediaPlay)
        icon_training = style.standardIcon(QStyle.SP_MessageBoxInformation)
        icon_authors = style.standardIcon(QStyle.SP_DialogApplyButton)

        self.program_button = QPushButton(icon_program, " Программа")
        self.training_button = QPushButton(
            icon_training, " Обучение программе")
        self.authors_button = QPushButton(icon_authors, " Авторы")

        icon_size = QSize(28, 28)  # Крупные иконки
        self.program_button.setIconSize(icon_size)
        self.training_button.setIconSize(icon_size)
        self.authors_button.setIconSize(icon_size)

        self.buttons_layout.addWidget(self.program_button)
        self.buttons_layout.addWidget(self.training_button)
        self.buttons_layout.addWidget(self.authors_button)

        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addSpacerItem(QSpacerItem(
            20, 60, QSizePolicy.Minimum, QSizePolicy.Expanding))  # Больше отступ снизу


class TrainingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        # Текст руководства
        self.welcome_message = """
        <h2 style='color: #50C878;'>Руководство по работе с "Ассистентом по расходам"</h2>

        <p>Добро пожаловать! Этот ассистент поможет вам легко управлять личными финансами.</p>

        <h3>1. Добавление расхода 📝</h3>
        <p>Чтобы начать, выберите удобную для вас <b>категорию</b> (или <a href="#add_category" style='color: #17a2b8;'>создайте свою</a>) — будь то «Еда», «Транспорт», «Развлечения» или «Другое».<br>
        Далее введите короткое, но информативное <b>описание</b>, например, "Обед в кафе".<br>
        После этого укажите <b>сумму</b> (только цифры, например, <code>250.50</code>). Выберите нужную <b>дату</b>.<br>
        Завершите процесс нажатием кнопки <b>«Добавить расход»</b> <span style='color: #28a745;'>(➕)</span>, и ваш расход моментально отобразится в таблице.</p>

        <h3>2. Установка бюджета 💰</h3>
        <p>Нажмите на кнопку <b>«Установить бюджет»</b> <span style='color: #17a2b8;'>(📊)</span>.<br>
        Выберите нужную категорию, затем введите сумму бюджета. Вы можете устанавливать бюджет для каждой категории отдельно.</p>
        <p>⚠️ <b style='color: #ffc107;'>Важно:</b> Ваш бюджет сохраняется даже при закрытии программы, и вы можете изменить его в любое время.</p>

        <h3 id="add_category">3 Управление категориями 📁</h3>
        <p>Нажмите <b>«Управление категориями»</b> <span style='color: #20c997;'>(⚙️)</span>, чтобы добавить свои собственные категории расходов, делая учет еще более персонализированным.</p>

        <h3>4. Фильтрация расходов 📅</h3>
        <p>Для быстрого анализа используйте <b>календари</b> 🗓️, чтобы выбрать <b>начальную («От»)</b> и <b>конечную («До»)</b> дату интересующего периода.<br>
        Затем нажмите <b>«Применить фильтр»</b> <span style='color: #6f42c1;'>(🔍)</span> и получите список расходов и диаграмму за выбранный диапазон.<br>
        Кнопка <b>«Сбросить фильтр»</b> <span style='color: #fd7e14;'>(❌)</span> вернет отображение всех расходов.</p>

        <h3>5. Анализ расходов (Диаграмма) 📊</h3>
        <p>Кнопка <b>«Диаграмма: Расходы за месяц»</b> <span style='color: #6c757d;'>(📅)</span> покажет суммарные траты по категориям за <i>текущий</i> календарный месяц и обновит диаграмму.<br>
        При применении <b>фильтра по датам</b> диаграмма покажет данные за выбранный период.<br>
        На диаграмме вы увидите:<br>
        <span style='color: #007BFF;'>🟦</span> Синие столбики — это ваши траты по категориям.<br>
        <span style='color: #90EE90;'>🟩</span> Зелёные столбики — отображают ваш бюджет для этих категорий.</p>

        <h3>6. Прогнозирование расходов 📈</h3>
        <p>Нажав на кнопку <b>«Прогноз расходов на месяц»</b> <span style='color: #ffc107;'>(🔮)</span>, вы получите оценку, сколько средств может быть потрачено к концу текущего месяца на основе текущих трат.<br>
        Если прогнозируемые траты окажутся выше установленного общего бюджета, система уведомит вас <span style='color: #dc3545;'>🚨</span>.<br>
        Это позволяет заранее скорректировать ваши траты и избежать перерасхода.</p>

        <h3>7. Экспорт данных 💾</h3>
        <p>Если вам необходимо провести подробный анализ или поделиться информацией, воспользуйтесь функцией <b>«Экспорт в CSV»</b> <span style='color: #343a40;'>(📄)</span>.<br>
        Все ваши расходы (с учетом текущего фильтра в таблице!) будут сохранены в файл, который можно легко открыть в Excel или Google Sheets.</p>

        <h3>8. Дополнительные возможности ✨</h3>
        <p>Интерфейс также позволяет оперативно управлять данными:<br>
        - <b>Удаление расходов:</b> Для этого достаточно выделить нужную строку (или несколько строк) в таблице и нажать кнопку <b>«Удалить»</b> <span style='color: #dc3545;'>(➖)</span>.</p>

        <h3>Советы для эффективного использования:</h3>
        <ul>
        <li>Настройте свои <b>категории</b> для удобства.</li>
        <li>Устанавливайте реалистичный <b>бюджет</b> для каждой важной категории.</li>
        <li>Регулярно добавляйте расходы, чтобы картина ваших финансов была полной.</li>
        <li>Используйте <b>фильтр</b> и <b>диаграмму</b> для анализа трат за периоды времени.</li>
        <li>Проверяйте <b>прогноз</b> расходов — это поможет вовремя скорректировать ваши траты.</li>
        <li>Периодически <b>экспортируйте</b> данные для долгосрочного анализа и резервного копирования.</li>
        </ul>

        <p style='text-align: center; font-size: 14pt; color: #50C878;'>Успешного управления финансами! 🚀</p>
        """
        self.text_edit.setHtml(self.welcome_message)
        self.layout.addWidget(self.text_edit)
        icon_back = QApplication.instance().style().standardIcon(QStyle.SP_ArrowBack)
        self.back_to_menu_button = QPushButton(icon_back, " В главное меню")
        self.back_to_menu_button.setObjectName(BACK_BTN_ID)
        self.back_to_menu_button.setIconSize(QSize(20, 20))
        self.layout.addWidget(self.back_to_menu_button,
                              alignment=Qt.AlignCenter)


class AuthorsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setAlignment(Qt.AlignCenter)
        self.authors_message = f"""
        <div style='text-align: center;'>
        <h2 style='color: #50C878; '>Авторы</h2>
        <p style='font-size: 13pt;'>Пастухов А.А.</p>
        <p style='font-size: 11pt;'>Год: 2025</p>
        <br>
        <p style='font-size: 11pt;'>Почта для обратной связи:<br>
        <a href='mailto:rstriggers@gmail.com' style='color: #17a2b8; text-decoration: none;'>rstriggers@gmail.com</a>
        </p>
        <br><br>
        <p style='font-size: 9pt; color: #999999;'>Иконки взяты из PyQt Standard Pixmaps</p>
        </div>
        """
        self.text_edit.setHtml(self.authors_message)
        self.layout.addWidget(self.text_edit)
        icon_back = QApplication.instance().style().standardIcon(QStyle.SP_ArrowBack)
        self.back_to_menu_button = QPushButton(icon_back, " В главное меню")
        self.back_to_menu_button.setObjectName(BACK_BTN_ID)
        self.back_to_menu_button.setIconSize(QSize(20, 20))
        self.layout.addWidget(self.back_to_menu_button,
                              alignment=Qt.AlignCenter)


class ExpenseTracker(QMainWindow):
    DEFAULT_CATEGORIES = ['Еда', 'Транспорт', 'Развлечения', 'Жилье',
                          'Коммунальные', 'Одежда', 'Здоровье', 'Подарки', 'Другое']

    def __init__(self):
        super().__init__()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.setWindowTitle('Ассистент по расходам 💸')
        self.setWindowIcon(QIcon.fromTheme("wallet", QIcon(
            # эххх
            ":/qt-project.org/styles/commonstyle/images/standardbutton-apply-16.png")))
        self.setGeometry(100, 100, 1250, 750)
        self.load_window_settings()

        self.central_widget = QWidget()
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.setCentralWidget(self.central_widget)

        self.setStyleSheet(STYLESHEET)

        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.budget = {}
        self.categories = self.DEFAULT_CATEGORIES[:]
        self.all_expenses_data = []

        # --- Экраны ---
        self.main_menu = MainMenu()
        self.stacked_widget.addWidget(self.main_menu)

        self.program_widget = QWidget()
        self.setup_program_widget()
        self.stacked_widget.addWidget(self.program_widget)

        self.training_widget = TrainingWidget()
        self.stacked_widget.addWidget(self.training_widget)

        self.authors_widget = AuthorsWidget()
        self.stacked_widget.addWidget(self.authors_widget)

        # --- Подключение кнопок меню ---
        self.main_menu.program_button.clicked.connect(
            lambda: self.switch_screen(self.program_widget))
        self.main_menu.training_button.clicked.connect(
            lambda: self.switch_screen(self.training_widget))
        self.main_menu.authors_button.clicked.connect(
            lambda: self.switch_screen(self.authors_widget))

        # --- Подключение кнопок возврата ---
        self.training_widget.back_to_menu_button.clicked.connect(
            lambda: self.switch_screen(self.main_menu))
        self.authors_widget.back_to_menu_button.clicked.connect(
            lambda: self.switch_screen(self.main_menu))

        icon_back = QApplication.instance().style().standardIcon(QStyle.SP_ArrowBack)
        self.back_to_menu_main_button = QPushButton(icon_back, " Меню")
        self.back_to_menu_main_button.setObjectName(
            BACK_BTN_ID)  # Стиль кнопки назад
        self.back_to_menu_main_button.setIconSize(QSize(18, 18))
        self.back_to_menu_main_button.clicked.connect(
            lambda: self.switch_screen(self.main_menu))

        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum,
                             QSizePolicy.Expanding)
        self.left_panel.addSpacerItem(spacer)
        self.left_panel.addWidget(self.back_to_menu_main_button)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Приложение готово.", 3000)

        # --- Загрузка данных и обновление UI ---
        self.load_data()
        self.update_category_input()
        self.apply_filter()  # Применяем фильтр по умолчанию при старте
        # Обновляем график для текущего месяца чуть позже, чтобы UI успел отрисоваться
        QTimer.singleShot(100, self.show_monthly_expenses)

    def switch_screen(self, widget):
        self.stacked_widget.setCurrentWidget(widget)

    def setup_program_widget(self):
        self.program_layout = QHBoxLayout(self.program_widget)
        self.program_layout.setSpacing(10)  # Отступ между панелями
        # --- Левая панель ---
        self.left_panel = QVBoxLayout()
        self.left_panel.setSpacing(10)  # Отступ между группами
        self.program_layout.addLayout(self.left_panel, 3)
        # --- Группа ввода ---
        self.input_group = QGroupBox("Добавить расход")
        self.input_layout = QGridLayout(self.input_group)
        self.input_layout.setSpacing(10)
        self.input_layout.addWidget(QLabel('Категория:'), 0, 0)


        self.category_input = QComboBox()
        self.category_input.setMinimumHeight(
            35)  # Увеличиваем минимальную высоту
        self.category_input.setSizeAdjustPolicy(
            # Автоматически подстраивает размер
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.input_layout.addWidget(self.category_input, 0, 1)

        self.input_layout.addWidget(QLabel('Описание:'), 1, 0)
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText('На что потрачены деньги?')
        self.input_layout.addWidget(self.description_input, 1, 1)
        self.input_layout.addWidget(QLabel('Сумма:'), 2, 0)
        self.amount_input = QLineEdit()
        self.amount_input.setPlaceholderText('Например: 150.99')
        self.input_layout.addWidget(self.amount_input, 2, 1)
        self.input_layout.addWidget(QLabel('Дата:'), 3, 0)
        self.date_input = QDateEdit(QDate.currentDate())
        self.date_input.setCalendarPopup(True)
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        self.input_layout.addWidget(self.date_input, 3, 1)

        style = QApplication.instance().style()

        self.add_button = QPushButton(QIcon.fromTheme(
            "list-add", style.standardIcon(QStyle.SP_DialogYesButton)), ' Добавить расход')
        self.add_button.setIconSize(QSize(20, 20))
        self.add_button.setObjectName(ADD_BTN_ID)
        self.add_button.clicked.connect(self.add_expense)
        self.input_layout.addWidget(self.add_button, 4, 0, 1, 2)
        self.left_panel.addWidget(self.input_group)
        # --- Группа управления ---
        self.manage_group = QGroupBox("Управление и Анализ")
        self.manage_layout = QVBoxLayout(self.manage_group)
        self.manage_layout.setSpacing(8)
        icon_del = style.standardIcon(QStyle.SP_TrashIcon)
        icon_budget = style.standardIcon(QStyle.SP_FileDialogDetailedView)
        icon_cat = style.standardIcon(QStyle.SP_FileDialogNewFolder)
        icon_month = style.standardIcon(QStyle.SP_ArrowForward)
        icon_forecast = style.standardIcon(QStyle.SP_MessageBoxQuestion)
        icon_export = style.standardIcon(QStyle.SP_DialogSaveButton)
        btn_icon_size = QSize(18, 18)
        self.delete_button = QPushButton(icon_del, ' Удалить выбранный расход')
        self.delete_button.setIconSize(btn_icon_size)
        self.delete_button.setObjectName(DEL_BTN_ID)
        self.delete_button.clicked.connect(self.delete_expense)
        self.manage_layout.addWidget(self.delete_button)
        self.budget_button = QPushButton(
            icon_budget, ' Установить бюджет по категориям')
        self.budget_button.setIconSize(btn_icon_size)
        self.budget_button.setObjectName(BUDGET_BTN_ID)
        self.budget_button.clicked.connect(self.set_budget)
        self.manage_layout.addWidget(self.budget_button)
        self.manage_categories_button = QPushButton(
            icon_cat, ' Управление категориями')
        self.manage_categories_button.setIconSize(btn_icon_size)
        self.manage_categories_button.setObjectName(MANAGE_CAT_BTN_ID)
        self.manage_categories_button.clicked.connect(self.manage_categories)
        self.manage_layout.addWidget(self.manage_categories_button)
        self.monthly_expenses_button = QPushButton(
            icon_month, ' Диаграмма: Расходы за месяц')
        self.monthly_expenses_button.setIconSize(btn_icon_size)
        self.monthly_expenses_button.setObjectName(MONTHLY_BTN_ID)
        self.monthly_expenses_button.clicked.connect(
            self.show_monthly_expenses)
        self.manage_layout.addWidget(self.monthly_expenses_button)
        self.forecast_button = QPushButton(
            icon_forecast, ' Прогноз расходов на месяц')
        self.forecast_button.setIconSize(btn_icon_size)
        self.forecast_button.setObjectName(FORECAST_BTN_ID)
        self.forecast_button.clicked.connect(self.predict_monthly_total)
        self.manage_layout.addWidget(self.forecast_button)
        self.export_button = QPushButton(
            icon_export, ' Экспорт в CSV (по фильтру)')
        self.export_button.setIconSize(btn_icon_size)
        self.export_button.setObjectName(EXPORT_BTN_ID)
        self.export_button.clicked.connect(self.export_to_csv)
        self.manage_layout.addWidget(self.export_button)
        self.left_panel.addWidget(self.manage_group)
        # --- Группа фильтрации ---
        self.filter_group = QGroupBox("Фильтр по дате")
        self.filter_layout = QGridLayout(self.filter_group)
        self.filter_layout.setSpacing(10)
        self.filter_layout.addWidget(QLabel('От:'), 0, 0)
        self.date_filter_start = QDateEdit(QDate.currentDate().addMonths(-1))
        self.date_filter_start.setCalendarPopup(True)
        self.date_filter_start.setDisplayFormat("yyyy-MM-dd")
        self.filter_layout.addWidget(self.date_filter_start, 0, 1)
        self.filter_layout.addWidget(QLabel('До:'), 1, 0)
        self.date_filter_end = QDateEdit(QDate.currentDate())
        self.date_filter_end.setCalendarPopup(True)
        self.date_filter_end.setDisplayFormat("yyyy-MM-dd")
        self.filter_layout.addWidget(self.date_filter_end, 1, 1)
        icon_filter = style.standardIcon(QStyle.SP_DialogApplyButton)
        icon_reset = style.standardIcon(QStyle.SP_DialogCancelButton)
        self.filter_button = QPushButton(icon_filter, ' Применить')
        self.filter_button.setIconSize(btn_icon_size)
        self.filter_button.setObjectName(FILTER_BTN_ID)
        self.filter_button.clicked.connect(self.apply_filter)
        self.filter_layout.addWidget(self.filter_button, 2, 0)
        self.reset_filter_button = QPushButton(icon_reset, ' Сбросить')
        self.reset_filter_button.setIconSize(btn_icon_size)
        self.reset_filter_button.setObjectName(RESET_FILTER_BTN_ID)
        self.reset_filter_button.clicked.connect(self.reset_filter)
        self.filter_layout.addWidget(self.reset_filter_button, 2, 1)
        self.left_panel.addWidget(self.filter_group)
        # --- Правая панель ---
        self.right_panel = QVBoxLayout()
        self.right_panel.setSpacing(10)
        self.program_layout.addLayout(self.right_panel, 7)
        # Таблица расходов
        self.expense_table = QTableWidget()
        self.expense_table.setColumnCount(4)
        self.expense_table.setHorizontalHeaderLabels(
            ['Описание', 'Сумма', 'Категория', 'Дата'])
        self.expense_table.setSortingEnabled(True)
        self.expense_table.setAlternatingRowColors(True)
        self.expense_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.expense_table.setSelectionBehavior(QTableWidget.SelectRows)
        # Возможность выбора нескольких строк
        self.expense_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.expense_table.verticalHeader().setVisible(False)
        self.expense_table.setShowGrid(True)  # Показываем сетку
        header = self.expense_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setMinimumSectionSize(100)  # Минимальная ширина колонки
        header.setDefaultAlignment(Qt.AlignLeft)  # Выравнивание заголовков
        # Таблица занимает 3/4 правой панели ешки пашки волосатые кудрящки
        self.right_panel.addWidget(self.expense_table, 3)
        # График расходов
        self.figure = plt.figure(facecolor='#3C3C3C')
        self.ax = self.figure.add_subplot(111, facecolor='#3C3C3C')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName(FIGURE_CANVAS_ID)  # ID для стилей
        # График занимает 1/4 правой панели
        self.right_panel.addWidget(self.canvas, 2)

    # --- Методы управления данными (Add, Delete, Budget, Category) ---

    def add_expense(self):
        description = self.description_input.text().strip()
        amount_str = self.amount_input.text().strip().replace(',', '.')
        category = self.category_input.currentText()
        date_str = self.date_input.date().toString("yyyy-MM-dd")

        if not description:
            QMessageBox.warning(self, 'Ошибка ввода',
                                'Пожалуйста, введите описание расхода.')
            return
        if not category:
            QMessageBox.warning(self, 'Ошибка ввода',
                                'Пожалуйста, выберите категорию.')
            return

        try:
            amount_value = float(amount_str)
            if amount_value <= 0:
                QMessageBox.warning(self, 'Ошибка ввода',
                                    'Сумма должна быть положительным числом.')
                return
        except ValueError:
            QMessageBox.warning(self, 'Ошибка ввода',
                                'Некорректное значение суммы. Введите число.')
            return

        # Проверка превышения бюджета
        today = QDate.currentDate()
        first_day_of_month = QDate(today.year(), today.month(), 1)
        monthly_expenses = 0.0

        # Считаем текущие расходы по этой категории за текущий месяц
        for expense in self.all_expenses_data:
            try:
                expense_date = QDate.fromString(expense["date"], "yyyy-MM-dd")
                if (expense["category"] == category and
                        first_day_of_month <= expense_date <= today):
                    monthly_expenses += float(expense["amount"])
            except (ValueError, TypeError):
                continue

        # Получаем бюджет для этой категории
        category_budget = self.budget.get(category, 0.0)

        # Проверяем, будет ли превышен бюджет после добавления нового расхода
        if category_budget > 0 and (monthly_expenses + amount_value) > category_budget:
            reply = QMessageBox.question(
                self,
                'Превышение бюджета',
                f'Бюджет для категории "{category}" ({category_budget:.2f} руб.) будет превышен!\n'
                f'Текущие расходы: {monthly_expenses:.2f} руб.\n'
                f'После добавления: {monthly_expenses + amount_value:.2f} руб.\n\n'
                'Вы уверены, что хотите добавить этот расход?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        new_expense = {
            "description": description,
            "amount": f"{amount_value:.2f}",
            "category": category,
            "date": date_str
        }

        self.all_expenses_data.insert(0, new_expense)

        self.description_input.clear()
        self.amount_input.clear()

        self.status_bar.showMessage(
            f"✅ Расход '{description}' добавлен.", 3000)
        self.apply_filter()
        self.save_data()

    def delete_expense(self):
        selected_rows = self.expense_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, 'Удаление не выбрано',
                                'Пожалуйста, выберите расход(ы) для удаления в таблице.')
            return
        reply = QMessageBox.question(self, 'Подтверждение удаления',
                                     f'Вы уверены, что хотите удалить выбранные {len(selected_rows)} записей?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            rows_to_delete_indices = sorted(
                [index.row() for index in selected_rows], reverse=True)
            visible_expenses_in_table = []
            current_row_count = self.expense_table.rowCount()
            for r in range(current_row_count):
                description = self.expense_table.item(r, 0).text()
                amount_str = self.expense_table.item(r, 1).text().replace(' ', '').replace(
                    ',', '.')  # Убираем пробелы и заменяем запятую на точку
                category = self.expense_table.item(r, 2).text()
                date_str = self.expense_table.item(r, 3).text()
                visible_expenses_in_table.append({
                    "description": description,
                    # Преобразуем в формат с двумя знаками после запятой
                    "amount": f"{float(amount_str):.2f}",
                    "category": category,
                    "date": date_str
                })

            expenses_to_remove_from_all_data = []
            for row_index in rows_to_delete_indices:
                if 0 <= row_index < len(visible_expenses_in_table):
                    expenses_to_remove_from_all_data.append(
                        visible_expenses_in_table[row_index])

            initial_len = len(self.all_expenses_data)
            new_all_expenses_data = []
            removed_count = 0

            set_to_remove = {tuple(sorted(d.items()))
                             for d in expenses_to_remove_from_all_data}
            for exp in self.all_expenses_data:
                if tuple(sorted(exp.items())) not in set_to_remove:
                    new_all_expenses_data.append(exp)
                else:
                    removed_count += 1
            self.all_expenses_data = new_all_expenses_data
            if removed_count > 0:
                self.status_bar.showMessage(
                    f"🗑️ Удалено {removed_count} записей.", 3000)
                self.apply_filter()
                self.save_data()
            else:
                # Это может произойти, если данные в таблице и all_expenses_data рассинхронизированы
                self.status_bar.showMessage(
                    "⚠️ Не удалось удалить записи (возможна рассинхронизация).", 5000)
                print("Debug: Data to remove not found in all_expenses_data:",
                      expenses_to_remove_from_all_data)

    def set_budget(self):
        if not self.categories:
            QMessageBox.warning(
                self, "Нет категорий", "Сначала добавьте категории в разделе 'Управление категориями'.")
            return

        category, ok = QInputDialog.getItem(self, 'Установка бюджета', 'Выберите категорию:',
                                            self.categories, 0, False)
        if ok and category:
            current_budget = self.budget.get(category, 0.0)
            # Используем форматирование для отображения текущего значения
            amount_str, ok = QInputDialog.getText(self, 'Установка бюджета',
                                                  f'Введите сумму бюджета для "{category}":\n(Текущий: {current_budget:.2f})',
                                                  QLineEdit.Normal, f"{current_budget:.2f}")
            if ok:
                try:
                    amount = float(amount_str.replace(',', '.'))
                    if amount < 0:
                        amount = 0  # Бюджет не может быть отрицательным
                    self.budget[category] = amount
                    QMessageBox.information(
                        self, 'Успех', f'Бюджет для "{category}" установлен в {amount:.2f} руб.')
                    self.status_bar.showMessage(
                        f'💰 Бюджет для "{category}" обновлен.', 3000)
                    self.save_data()
                    self.update_chart()
                except ValueError:
                    QMessageBox.warning(
                        self, 'Ошибка ввода', 'Некорректное значение суммы. Введите число.')

    def manage_categories(self):
        current_categories_sorted = sorted(self.categories)
        current_categories_str = "\n".join(current_categories_sorted)
        text, ok = QInputDialog.getMultiLineText(self, 'Управление категориями',
                                                 'Список категорий (каждая на новой строке):',
                                                 current_categories_str)
        if ok:
            # Получаем новые категории, удаляем пустые строки и дубликаты
            new_categories_set = set(line.strip()
                                     for line in text.splitlines() if line.strip())
            new_categories = sorted(
                list(new_categories_set))  # Сразу сортируем

            if not new_categories:
                QMessageBox.warning(
                    self, "Ошибка", "Список категорий не может быть пустым.")
                return

            # Проверяем, используются ли удаляемые категории
            removed_categories = set(self.categories) - set(new_categories)
            if removed_categories:
                used_removed_categories = set()
                for expense in self.all_expenses_data:
                    if expense["category"] in removed_categories:
                        used_removed_categories.add(expense["category"])

                if used_removed_categories:
                    reply = QMessageBox.question(self, 'Подтверждение удаления категорий',
                                                 f"Следующие категории используются в ваших расходах:\n{', '.join(sorted(used_removed_categories))}\n\n"
                                                 "Если вы их удалите, они не будут использоваться для подсчета общих расходов, "
                                                 "их бюджет сбросится, но все траты останутся в таблице(если вы захотите добавить категорию обратно).\n\n"
                                                 "Продолжить обновление списка категорий?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.No:
                        return  # Отмена операции

            self.categories = new_categories
            self.update_category_input()  # Обновляем выпадающий список
            # Очищаем бюджет для удаленных категорий
            self.budget = {cat: self.budget[cat]
                           for cat in self.budget if cat in self.categories}
            self.save_data()
            self.status_bar.showMessage("⚙️ Список категорий обновлен.", 3000)
            self.apply_filter()  # Обновляем отображение

    # --- Методы фильтрации и отображения ---

    def apply_filter(self):
        self.update_expense_table()
        self.update_chart()
        self.status_bar.showMessage("Фильтр применен.", 2000)

    def reset_filter(self):
        self.date_filter_start.setDate(QDate.currentDate().addDays(-30))
        self.date_filter_end.setDate(QDate.currentDate())
        self.update_expense_table()
        # Обновляем график для текущего месяца после обновления таблицы
        QTimer.singleShot(50, self.show_monthly_expenses)
        self.status_bar.showMessage(
            "Фильтр сброшен. Показаны расходы за последние 30 дней.", 2000)

    def update_expense_table(self):
        self.expense_table.setSortingEnabled(False)
        self.expense_table.setRowCount(0)

        start_date = self.date_filter_start.date().toString("yyyy-MM-dd")
        end_date = self.date_filter_end.date().toString("yyyy-MM-dd")

        filtered_expenses = [
            exp for exp in self.all_expenses_data
            if start_date <= exp["date"] <= end_date
        ]

        for expense in filtered_expenses:
            row_position = self.expense_table.rowCount()
            self.expense_table.insertRow(row_position)

            # Описание
            item_desc = QTableWidgetItem(expense["description"])
            self.expense_table.setItem(row_position, 0, item_desc)

            # Сумма
            try:
                amount_val = float(expense['amount'])
                amount_item = QTableWidgetItem(
                    f"{amount_val:,.2f}".replace(',', ' '))
                amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            except ValueError:
                amount_item = QTableWidgetItem(str(expense["amount"]))
                amount_item.setForeground(QColor('red'))
                amount_item.setToolTip("Некорректное значение суммы")
            self.expense_table.setItem(row_position, 1, amount_item)

            # Категория
            item_cat = QTableWidgetItem(expense["category"])
            self.expense_table.setItem(row_position, 2, item_cat)

            # Дата
            item_date = QTableWidgetItem(expense["date"])
            item_date.setTextAlignment(Qt.AlignCenter)
            self.expense_table.setItem(row_position, 3, item_date)

        self.expense_table.setSortingEnabled(True)

    def update_category_input(self):
        self.category_input.clear()
        if self.categories:
            self.category_input.addItems(self.categories)
        else:
            self.category_input.addItem("Нет доступных категорий")
            self.category_input.setEnabled(False)

    # --- Методы работы с графиком ---

    def update_chart(self):
        self.ax.clear()
        # --- Настройка внешнего вида осей и текста ---
        axis_color = '#B0B0B0'  # Цвет осей
        text_color = '#E0E0E0'  # Цвет текста (заголовки, метки)
        self.ax.tick_params(axis='x', colors=axis_color)
        self.ax.tick_params(axis='y', colors=axis_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.title.set_color(text_color)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color(axis_color)
        self.ax.spines['bottom'].set_color(axis_color)

        expenses = {cat: 0 for cat in self.categories}
        visible_categories_in_period = set()

        # Собираем данные из видимых строк таблицы
        for row in range(self.expense_table.rowCount()):
            category = self.expense_table.item(row, 2).text()
            try:
                # Сумму берем из данных
                amount_str = self.expense_table.item(
                    row, 1).text().replace(' ', '').replace(',', '.')
                amount = float(amount_str)
                if category in expenses:
                    expenses[category] += amount
                    visible_categories_in_period.add(category)
            except ValueError:
                pass

        # Оставляем только те категории, по которым были расходы > 0
        active_categories = sorted(
            [cat for cat in self.categories if expenses.get(cat, 0) > 0])

        if not active_categories:
            self.ax.set_title(
                "Нет данных для отображения за выбранный период", color=text_color)
            self.canvas.draw()
            return

        spent_amounts = [expenses[cat] for cat in active_categories]
        budget_amounts = [self.budget.get(category, 0)
                          for category in active_categories]

        x = np.arange(len(active_categories))
        width = 0.4

        rects1 = self.ax.bar(x - width/2, spent_amounts, width,
                             label='Потрачено', color='#007BFF', zorder=3)
        # Показываем бюджет только если он больше 0
        valid_budget_indices = [
            i for i, b in enumerate(budget_amounts) if b > 0]
        if valid_budget_indices:
            x_budget = x[valid_budget_indices]
            budget_amounts_valid = [budget_amounts[i]
                                    for i in valid_budget_indices]
            rects2 = self.ax.bar(x_budget + width/2, budget_amounts_valid,
                                 width, label='Бюджет', color='#90EE90', alpha=0.7, zorder=3)

        self.ax.set_ylabel('Сумма (руб)', color=text_color, fontsize=10)
        start_date_str = self.date_filter_start.date().toString("dd.MM.yy")
        end_date_str = self.date_filter_end.date().toString("dd.MM.yy")
        date_range = f"{start_date_str} - {end_date_str}"
        if start_date_str == end_date_str:
            date_range = start_date_str
        self.ax.set_title(
            f'Расходы и бюджет ({date_range})', color=text_color, fontsize=12, fontweight='bold')

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(
            active_categories, rotation=40, ha="right", color=text_color, fontsize=9)
        self.ax.grid(axis='y', linestyle='--', alpha=0.5,
                     color=axis_color, zorder=0)

        legend = self.ax.legend(
            facecolor='#4A4A4A', edgecolor=axis_color, labelcolor=text_color, fontsize=9)

        for rect in rects1:
            height = rect.get_height()
            if height > 0:
                self.ax.annotate(f'{height:,.0f}'.replace(',', ' '),
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=8, color=text_color)

        self.figure.tight_layout()
        self.canvas.draw()

    def show_monthly_expenses(self):
        today = QDate.currentDate()
        first_day = QDate(today.year(), today.month(), 1)
        last_day = QDate(today.year(), today.month(), today.daysInMonth())

        self.date_filter_start.setDate(first_day)
        self.date_filter_end.setDate(last_day)

        self.apply_filter()
        self.status_bar.showMessage(
            f"📊 Показаны расходы за {today.toString('MMMM yyyy')}", 3000)

    # --- Методы сохранения/загрузки ---
    def save_data(self):
        self.all_expenses_data.sort(key=lambda x: x.get(
            'date', '0000-00-00'), reverse=True)

        data_to_save = {
            "expenses": self.all_expenses_data,
            "budget": self.budget,
            "categories": self.categories
        }
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as file:
                json.dump(data_to_save, file, ensure_ascii=False, indent=4)
            # self.status_bar.showMessage("💾 Данные сохранены.", 2000)
        except Exception as e:
            QMessageBox.critical(
                self, 'Ошибка сохранения', f'Не удалось сохранить данные в файл {DATA_FILE}:\n{e}')
            self.status_bar.showMessage("⚠️ Ошибка сохранения данных!", 5000)

    def load_data(self):
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, "r", encoding="utf-8") as file:
                    loaded_data = json.load(file)

                    
                    raw_expenses = loaded_data.get("expenses", [])
                    self.all_expenses_data = []
                    for exp in raw_expenses:
                        # Проверяем наличие ключей и базовый формат
                        if all(k in exp for k in ["description", "amount", "category", "date"]):
                            try:
                                
                                amount_str = str(exp['amount']).replace(
                                    ',', '.').replace(' ', '')
                                exp['amount'] = f"{float(amount_str):.2f}"
                                # Проверка формата даты
                                QDate.fromString(exp['date'], "yyyy-MM-dd")
                                self.all_expenses_data.append(exp)
                            except (ValueError, TypeError):
                                print(
                                    f"Пропуск некорректной записи при загрузке: {exp}")
                        else:
                            print(
                                f"Пропуск неполной записи при загрузке: {exp}")

                    # Сортируем загруженные данные
                    self.all_expenses_data.sort(key=lambda x: x.get(
                        'date', '0000-00-00'), reverse=True)

                    self.budget = loaded_data.get("budget", {})
                    loaded_categories = loaded_data.get("categories", [])
                    # Используем загруженные категории, если они не пусты, иначе дефолтные
                    self.categories = loaded_categories if loaded_categories else self.DEFAULT_CATEGORIES[
                        :]
                    # Очищаем бюджет от категорий, которых больше нет
                    self.budget = {
                        cat: self.budget[cat] for cat in self.budget if cat in self.categories}

                    self.status_bar.showMessage(
                        f"Данные из {DATA_FILE} загружены.", 2000)
            else:
                self.status_bar.showMessage(
                    "Файл данных не найден. Начните добавлять расходы.", 3000)
                self.categories = self.DEFAULT_CATEGORIES[:]
                # Устанавливаем фильтр по умолчанию на последний месяц
                self.date_filter_start.setDate(
                    QDate.currentDate().addMonths(-1))
                self.date_filter_end.setDate(QDate.currentDate())

        except json.JSONDecodeError as e:
            # Предлагаем создать резервную копию поврежденного файла
            reply = QMessageBox.critical(self, 'Ошибка загрузки данных',
                                         f"Файл данных '{DATA_FILE}' поврежден или имеет неверный формат.\n{e}\n\n"
                                         f"Создать резервную копию файла '{DATA_FILE}.bak' и начать с чистого листа?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes and os.path.exists(DATA_FILE):
                try:
                    os.rename(DATA_FILE, f"{DATA_FILE}.bak")
                    self.status_bar.showMessage(
                        "Создана резервная копия поврежденного файла. Начните заново.", 5000)
                    # Сбрасываем данные в приложении к дефолтным
                    self.all_expenses_data = []
                    self.budget = {}
                    self.categories = self.DEFAULT_CATEGORIES[:]
                except OSError as os_err:
                    QMessageBox.critical(
                        self, 'Ошибка', f'Не удалось создать резервную копию файла:\n{os_err}')
            else:
                self.status_bar.showMessage(
                    "⚠️ Ошибка загрузки данных! Проверьте файл.", 5000)

        except Exception as e:
            QMessageBox.critical(
                self, 'Неизвестная ошибка загрузки', f'Не удалось загрузить данные:\n{e}')
            self.status_bar.showMessage("⚠️ Ошибка загрузки данных!", 5000)

        # Финальная проверка
        if not self.categories:
            self.categories = self.DEFAULT_CATEGORIES[:]
            self.category_input.setEnabled(True)

    def load_window_settings(self):
        try:
            self.restoreGeometry(self.settings.value(
                "geometry", self.saveGeometry()))
            self.restoreState(self.settings.value(
                "windowState", self.saveState()))
        except Exception as e:
            print(f"Не удалось загрузить настройки окна: {e}")

    def save_window_settings(self):
        try:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
        except Exception as e:
            print(f"Не удалось сохранить настройки окна: {e}")

    def closeEvent(self, event):
        self.save_window_settings()
        super().closeEvent(event)

    # --- Экспорт и Прогноз ---

    def export_to_csv(self):
        if self.expense_table.rowCount() == 0:
            QMessageBox.warning(self, 'Экспорт невозможен',
                                'Нет данных для экспорта в таблице (проверьте фильтр).')
            return

        start_date_str = self.date_filter_start.date().toString("yyyyMMdd")
        end_date_str = self.date_filter_end.date().toString("yyyyMMdd")
        default_filename = f"расходы_{start_date_str}-{end_date_str}.csv"

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить файл CSV", default_filename, "CSV Files (*.csv);;All Files (*)")

        if file_name:
            try:
                with open(file_name, 'w', newline='', encoding='utf-8-sig') as file:  # utf-8-sig для Excel
                    writer = csv.writer(
                        file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['Описание', 'Сумма', 'Категория', 'Дата'])
                    for row in range(self.expense_table.rowCount()):
                        # Получаем данные напрямую из виджетов таблицы
                        desc = self.expense_table.item(row, 0).text()
                        amount = self.expense_table.item(row, 1).text().replace(
                            ' ', '')  # Убираем пробелы из суммы
                        cat = self.expense_table.item(row, 2).text()
                        date = self.expense_table.item(row, 3).text()
                        # Сумму пишем с запятой, как принято в CSV для Excel
                        writer.writerow([desc, amount, cat, date])
                self.status_bar.showMessage(
                    f"📄 Данные экспортированы в {os.path.basename(file_name)}", 4000)
            except Exception as e:
                QMessageBox.critical(
                    self, 'Ошибка экспорта', f'Не удалось экспортировать данные в файл:\n{e}')
                self.status_bar.showMessage("⚠️ Ошибка экспорта данных!", 5000)

    def predict_monthly_total(self):
        today = QDate.currentDate()
        first_day_of_month = QDate(today.year(), today.month(), 1)
        last_day_of_month_num = today.daysInMonth()
        current_day_num = today.day()

        # Собираем траты 
        daily_spending = [0.0] * (last_day_of_month_num + 1)  # Индексы 1-31
        current_total_spent = 0.0

        for expense in self.all_expenses_data:
            try:
                expense_date = QDate.fromString(expense["date"], "yyyy-MM-dd")
                if first_day_of_month <= expense_date <= today:
                    day = expense_date.day()
                    amount = float(expense["amount"])
                    daily_spending[day] += amount
                    current_total_spent += amount
            except (ValueError, TypeError):
                continue

        # Подготовка данных для модели:
        X = np.array([[d] for d in range(1, current_day_num + 1)])
        y = np.array([daily_spending[d]
                     for d in range(1, current_day_num + 1)])

        # Если данных недостаточно (меньше 3 дней), используем среднее дневных трат
        if len(X) < 3:
            avg_daily = np.mean(y) if y.size > 0 else 0
            remaining_days = last_day_of_month_num - current_day_num
            predicted_total = current_total_spent + avg_daily * remaining_days
        else:
            try:
                # Функция для оценки модели
                def evaluate_model(model, X, y):
                    scores = cross_val_score(
                        model, X, y, cv=3, scoring='neg_mean_squared_error')
                    return -np.mean(scores)

                # Линейная регрессия
                linear_model = LinearRegression()
                linear_score = evaluate_model(linear_model, X, y)

                # Полиномиальная регрессия с разными степенями
                best_poly_degree = None
                best_poly_score = float('inf')
                for degree in range(2, 6):  # Проверяем степени от 2 до 5
                    poly = PolynomialFeatures(degree=degree)
                    X_poly = poly.fit_transform(X)
                    # Используем Ridge
                    poly_model = Ridge(alpha=1.0)
                    poly_score = evaluate_model(poly_model, X_poly, y)
                    if poly_score < best_poly_score:
                        best_poly_score = poly_score
                        best_poly_degree = degree

                # Выбираем лучшую модель
                if best_poly_degree is not None and best_poly_score < linear_score:
                    # Полиномиальная регрессия с лучшей степенью
                    poly = PolynomialFeatures(degree=best_poly_degree)
                    X_poly = poly.fit_transform(X)
                    model = Ridge(alpha=1.0)
                    model.fit(X_poly, y)

                    # Преобразуем будущие дни для прогноза
                    future_days = np.array(
                        [[d] for d in range(current_day_num + 1, last_day_of_month_num + 1)])
                    future_days_poly = poly.transform(future_days)
                    predicted_daily = model.predict(future_days_poly)
                else:
                    # Линейная регрессия
                    model = LinearRegression()
                    model.fit(X, y)
                    future_days = np.array(
                        [[d] for d in range(current_day_num + 1, last_day_of_month_num + 1)])
                    predicted_daily = model.predict(future_days)

                # Заменяем отрицательные прогнозы на 0
                predicted_daily = [max(0, x) for x in predicted_daily]
                predicted_total = current_total_spent + sum(predicted_daily)
            except Exception as e:
                print(f"Ошибка прогнозирования: {e}")
                # Если модель не сработала, используем среднее
                avg_daily = np.mean(y) if y.size > 0 else 0
                remaining_days = last_day_of_month_num - current_day_num
                predicted_total = current_total_spent + avg_daily * remaining_days

        # Обеспечиваем, чтобы прогноз был не меньше текущих трат
        predicted_total = max(predicted_total, current_total_spent)

        # Рассчитываем общий бюджет
        total_budget = sum(b for b in self.budget.values()
                           if isinstance(b, (int, float)) and b > 0)

        # Формируем сообщение с результатами
        msgBox = QMessageBox(self)
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setWindowTitle('Прогноз расходов на месяц')
        msgBox.setTextFormat(Qt.RichText)

        message = (
            f"Текущие расходы за месяц: <b>{current_total_spent:,.2f} руб.</b><br><br>"
            f"Прогнозируемые расходы к концу месяца ({last_day_of_month_num}-е число):<br>"
            f"<font size='+1'><b>≈ {predicted_total:,.2f} руб.</b></font><br><br>"
        ).replace(',', ' ')  # Форматирование

        if total_budget > 0:
            message += f"Общий бюджет по категориям: <b>{total_budget:,.2f} руб.</b><br>".replace(
                ',', ' ')
            if predicted_total > total_budget:
                overspending = predicted_total - total_budget
                message += (f"<font color='#E74C3C'><b>Внимание!</b> Прогнозируется превышение бюджета "
                            f"на <b>≈ {overspending:,.2f} руб.</b></font>").replace(',', ' ')
            else:
                remaining = total_budget - predicted_total
                message += (f"<font color='#90EE90'>Прогноз в рамках бюджета. "
                            f"Ориентировочный остаток: <b>≈ {remaining:,.2f} руб.</b></font>").replace(',', ' ')
        else:
            message += "<i>Общий бюджет по категориям не установлен.</i>"

        msgBox.setText(message)
        msgBox.exec_()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # QIcon.setThemeName("breeze")
    QIcon.setThemeName("gnome")
    tracker = ExpenseTracker()
    tracker.show()
    sys.exit(app.exec_())
