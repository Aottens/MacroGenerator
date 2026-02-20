import sys
import os
import re
import json
import logging
import pandas as pd
from PyQt5 import QtWidgets, QtCore

# --- Logging setup ---
logger = logging.getLogger('MacroGenerator')
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
log_file = os.path.join(os.getcwd(), 'macro_generator.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.info("Logger initialized; writing to %s", log_file)

# --- Configuration ---
CONFIG_FILE = os.path.expanduser('~/.macro_generator_config.json')
DEFAULT_SETTINGS = {
    'popups': {
        'INT': 901,
        'INT_MinMaxAfterScale': 902,
        'BCD': 903,
        'REAL': 904,
        'REAL_MinMaxAfterScale': 905,
        'BOOL': 906,     # single generic BOOL page
        # --- DINT (toegevoegd) ---
        'DINT': 907,
        'DINT_MinMaxAfterScale': 908,
    },
    'connection': 'HOST3',
    'interface_base': 20119,
    # Nieuw: temp-DM’s voor REAL (alleen gebruikt bij indirecte min/max)
    'temp_min_real_dm': 0,
    'temp_max_real_dm': 0,
}
CONNECTION_OPTIONS = ['HOST3', 'ETHERNET']

OFFSET = {
    'param_block':       19,
    'bool_param_block':  22,
    'start_flag':         1,
    'old_value_int':     15,
    'old_value_real':    16,
    'old_value_bool':    18,
    # --- DINT (toegevoegd) ---
    'old_value_dint':    44,
}

def load_config():
    logger.debug("Loading config from %s", CONFIG_FILE)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
        migrated = False

        # Legacy BOOL migratie
        for settings in cfg.get('machines', {}).values():
            pops = settings.get('popups', {})
            legacy = {'DBOOL', 'WBOOL', 'HBOOL', 'EBOOL'} & set(pops)
            if legacy and 'BOOL' not in pops:
                key = sorted(legacy)[0]
                pops['BOOL'] = pops[key]
                for k in legacy:
                    pops.pop(k, None)
                migrated = True
                logger.info("Migrated legacy popup keys %s → BOOL", legacy)

        # Nieuwe REAL temp-DM velden migreren
        for name, settings in cfg.get('machines', {}).items():
            if 'temp_min_real_dm' not in settings:
                settings['temp_min_real_dm'] = DEFAULT_SETTINGS['temp_min_real_dm']
                migrated = True
            if 'temp_max_real_dm' not in settings:
                settings['temp_max_real_dm'] = DEFAULT_SETTINGS['temp_max_real_dm']
                migrated = True

        if migrated:
            save_config(cfg)
        return cfg
    logger.debug("No config file found, using defaults")
    return {'machines': {}}

def save_config(cfg):
    logger.debug("Saving config to %s", CONFIG_FILE)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=4)

def decimal_to_bcd(val):
    bcd, mul = 0, 1
    while val:
        bcd += (val % 10) * mul
        mul <<= 4
        val //= 10
    return bcd

def parse_address(address):
    s = str(address).upper()
    if '_' in s:
        area, rest = s.split('_', 1)
        if '.' in rest:
            ptr_str, bit_str = rest.split('.', 1)
            bit = int(bit_str)
        else:
            ptr_str, bit = rest, None
        ptr = int(ptr_str)
    else:
        m = re.match(r"^([A-Z]+)(\d+)(?:\.(\d+))?$", s)
        if not m:
            raise ValueError(f"Invalid address: {address}")
        area, ptr = m.group(1), int(m.group(2))
        bit = int(m.group(3)) if m.group(3) else None
    return area, ptr, bit

def area_code(area: str) -> int:
    a = area.upper()
    if a.startswith('D'):   return 0
    if a.startswith('W'):   return 1
    if a.startswith('H'):   return 2
    m = re.fullmatch(r'E([0-3])', a)
    if m:
        return 3 + int(m.group(1))
    raise ValueError(f"Unsupported area: {area}")

def is_address_token(val):
    try:
        parse_address(val)
        return True
    except:
        return False

# --- Config Dialog ---
class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Machine Configuration')
        self.cfg = load_config()
        layout = QtWidgets.QVBoxLayout(self)

        self.machine_select = QtWidgets.QComboBox()
        self.machine_select.addItems(sorted(self.cfg['machines'].keys()))
        self.machine_select.currentTextChanged.connect(self.fill_fields)
        layout.addWidget(QtWidgets.QLabel('Select Machine:'))
        layout.addWidget(self.machine_select)

        self.conn_combo = QtWidgets.QComboBox()
        self.conn_combo.addItems(CONNECTION_OPTIONS)
        cf = QtWidgets.QFormLayout()
        cf.addRow('Connection type:', self.conn_combo)
        layout.addLayout(cf)

        self.base_spin = QtWidgets.QSpinBox()
        self.base_spin.setRange(0, 99999)
        bf = QtWidgets.QFormLayout()
        bf.addRow('Interface Base DM:', self.base_spin)
        layout.addLayout(bf)

        # Popups
        self.spin_boxes = {}
        popf = QtWidgets.QFormLayout()
        for key in DEFAULT_SETTINGS['popups']:
            sb = QtWidgets.QSpinBox(); sb.setRange(0, 9999)
            popf.addRow(f"{key} page:", sb)
            self.spin_boxes[key] = sb
        layout.addLayout(popf)

        # Nieuw: REAL temp DM’s
        self.temp_min_real_sb = QtWidgets.QSpinBox(); self.temp_min_real_sb.setRange(0, 65535)
        self.temp_max_real_sb = QtWidgets.QSpinBox(); self.temp_max_real_sb.setRange(0, 65535)
        tf = QtWidgets.QFormLayout()
        tf.addRow('REAL temp MIN (DM):', self.temp_min_real_sb)
        tf.addRow('REAL temp MAX (DM):', self.temp_max_real_sb)
        layout.addLayout(tf)

        btns = QtWidgets.QHBoxLayout()
        self.new_btn    = QtWidgets.QPushButton('New Machine')
        self.save_btn   = QtWidgets.QPushButton('Save')
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        btns.addWidget(self.new_btn); btns.addWidget(self.save_btn); btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

        self.new_btn.clicked.connect(self.add_machine)
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if self.machine_select.count():
            self.fill_fields(self.machine_select.currentText())

    def fill_fields(self, name):
        settings = self.cfg['machines'].get(name, DEFAULT_SETTINGS)
        self.conn_combo.setCurrentText(settings.get('connection', DEFAULT_SETTINGS['connection']))
        self.base_spin.setValue(settings.get('interface_base', DEFAULT_SETTINGS['interface_base']))
        pops = settings.get('popups', DEFAULT_SETTINGS['popups'])
        for key, sb in self.spin_boxes.items():
            sb.setValue(pops.get(key, DEFAULT_SETTINGS['popups'][key]))
        # REAL temp DM’s
        self.temp_min_real_sb.setValue(settings.get('temp_min_real_dm', DEFAULT_SETTINGS['temp_min_real_dm']))
        self.temp_max_real_sb.setValue(settings.get('temp_max_real_dm', DEFAULT_SETTINGS['temp_max_real_dm']))

    def add_machine(self):
        name, ok = QtWidgets.QInputDialog.getText(self, 'New Machine', 'Machine identifier:')
        if ok and name:
            self.cfg['machines'][name] = json.loads(json.dumps(DEFAULT_SETTINGS))
            save_config(self.cfg)
            self.machine_select.addItem(name)
            self.machine_select.setCurrentText(name)

    def accept(self):
        name = self.machine_select.currentText()
        if not name:
            QtWidgets.QMessageBox.warning(self, 'Error', 'No machine selected.')
            return
        conn = self.conn_combo.currentText()
        base = self.base_spin.value()
        pops = {k: sb.value() for k, sb in self.spin_boxes.items()}
        temp_min_real_dm = self.temp_min_real_sb.value()
        temp_max_real_dm = self.temp_max_real_sb.value()
        self.cfg['machines'][name] = {
            'connection': conn,
            'interface_base': base,
            'popups': pops,
            'temp_min_real_dm': temp_min_real_dm,
            'temp_max_real_dm': temp_max_real_dm,
        }
        save_config(self.cfg)
        super().accept()

    def get_selected(self):
        name = self.machine_select.currentText()
        settings = self.cfg['machines'].get(name, DEFAULT_SETTINGS)
        return (
            name,
            settings['connection'],
            settings['interface_base'],
            settings['popups'],
            settings.get('temp_min_real_dm', DEFAULT_SETTINGS['temp_min_real_dm']),
            settings.get('temp_max_real_dm', DEFAULT_SETTINGS['temp_max_real_dm']),
        )

# --- Single Macro Dialog ---
class SingleMacroDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Generate Single Macro")
        layout = QtWidgets.QFormLayout(self)

        self.adres_e = QtWidgets.QLineEdit()
        self.scal_e  = QtWidgets.QLineEdit()
        self.min_e   = QtWidgets.QLineEdit()
        self.max_e   = QtWidgets.QLineEdit()
        self.desc_e  = QtWidgets.QLineEdit()
        self.dt_e    = QtWidgets.QLineEdit()
        self.after_cb = QtWidgets.QCheckBox("MinMax After Scale")

        layout.addRow("Adres:", self.adres_e)
        layout.addRow("Scaling:", self.scal_e)
        layout.addRow("Min Waarde:", self.min_e)
        layout.addRow("Max Waarde:", self.max_e)
        layout.addRow("Omschrijving:", self.desc_e)
        layout.addRow("Datatype:", self.dt_e)
        layout.addRow("", self.after_cb)

        btns = QtWidgets.QHBoxLayout()
        gen  = QtWidgets.QPushButton("Generate")
        close = QtWidgets.QPushButton("Close")
        btns.addWidget(gen); btns.addWidget(close)
        layout.addRow("", btns)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addRow("Macro:", self.output)

        gen.clicked.connect(self.on_generate)
        close.clicked.connect(self.reject)

    def _parse_value(self, text, force_int=False):
        t = text.strip()
        if not t:
            return 0
        if is_address_token(t) and not force_int:
            return t
        try:
            return int(float(t.replace(',', '.')))
        except:
            return 0

    def on_generate(self):
        addr    = self.adres_e.text().strip()
        scaling = self._parse_value(self.scal_e.text(), force_int=True)
        minv    = self._parse_value(self.min_e.text())
        maxv    = self._parse_value(self.max_e.text())
        desc    = self.desc_e.text().strip()
        dt      = self.dt_e.text().strip().upper()
        after   = self.after_cb.isChecked()

        logger.info(
            "SingleMacro input → Adres=%s, Scaling=%s, Min=%s, Max=%s, Desc=%r, Datatype=%s, After=%s",
            addr, scaling, minv, maxv, desc, dt, after
        )

        try:
            area, ptr, bit = parse_address(addr)
            code           = area_code(area)
            parent         = self.parent()
            conn           = parent.connection
            base           = parent.interface_base
            pops           = parent.popup_mapping
            pb             = base + OFFSET['param_block']
            bb             = base + OFFSET['bool_param_block']
            sf             = base + OFFSET['start_flag']
            oi             = base + OFFSET['old_value_int']
            orl            = base + OFFSET['old_value_real']
            ob             = base + OFFSET['old_value_bool']
            # --- DINT: old value offset
            odi            = base + OFFSET['old_value_dint']
            tmin_real_dm   = parent.temp_min_real_dm
            tmax_real_dm   = parent.temp_max_real_dm

            if dt == 'BOOL':
                if area.startswith('D'):
                    readcode = 300
                elif area.startswith('W'):
                    readcode = 104
                elif area.startswith('H'):
                    readcode = 101
                else:
                    m = re.fullmatch(r'E([0-3])', area)
                    if not m:
                        raise ValueError(f"Unsupported BOOL area: {area}")
                    readcode = 302 + int(m.group(1))
                popup_key = 'BOOL'
                lines = [
                    f"'Adres: {area}{ptr:05}.{bit}'",
                    f"'Deze macro triggert het openen van een pop-up voor boolean invoerfunctionaliteit in het {area} gebied.'",
                    f"$W1 = {ptr}; 'Adres pointer voor {area}{ptr:05} instellen'",
                    f"$W2 = {bit}; 'Bit index instellen voor {area}{ptr:05}.{bit}'",
                    f"STRCPY($W3,\"{desc}\"); 'Schrijf beschrijving naar $W3'",
                    f"$W22={code}; 'D=0, W=1, H=2, E0=3, E1=4, E2=5, E3=6'",
                    f"WRITECMEM([{conn}:DM{bb}], $W1,22); 'Kopieerslag naar PLC (incl. omschrijving)'",
                    f"READHOSTB($B10,[{conn}],{readcode},{ptr},{bit},1); 'Lees bit'",
                    f"WRITEHOSTB([{conn}],300,{ob},0,$B10,1); 'Schrijf boolean oude waarde'",
                    f"$W30 = 1; 'Start edit vlag'",
                    f"WRITECMEM([{conn}:DM{sf}], $W30,1); 'Schrijf start edit vlag naar PLC'",
                    f"SHOWPAGE({pops.get(popup_key)}); 'Open pop-up'"
                ]
            else:
                # ---- Min/Max handling conform afspraken ----
                lines = [f"'Adres: {area}{ptr}'", f"STRCPY($W6,\"{desc}\"); 'Omschrijving'"]
                lines.append(f"$W1={scaling}; 'Scaling'")

                def append_int_bcd_minmax(v, is_min):
                    # Direct constant of indirect 1-woord read (area-aware)
                    if is_address_token(v):
                        a, p, _ = parse_address(v)
                        lines.append(f"READCMEM($W{2 if is_min else 3},[{conn}:{a}{p}],1); 'Load {'Min' if is_min else 'Max'} waarde (indirect)'")
                    else:
                        vv = decimal_to_bcd(v) if dt == 'BCD' else v
                        lines.append(f"$W{2 if is_min else 3}={vv}; '{'Min' if is_min else 'Max'} waarde'")

                def append_real_min(v, temp_dm):
                    if is_address_token(v):
                        if temp_dm == 0:
                            raise ValueError("REAL indirect: temp_min_real_dm is 0 (niet geconfigureerd).")
                        a, p, _ = parse_address(v)
                        lines.append(f"READCMEM($W50,[{conn}:{a}{p}],2); 'Load Min (REAL indirect, 2 woorden)'")
                        lines.append(f"WRITECMEM([{conn}:DM{temp_dm}], $W50,2); 'Push Min naar temp REAL DM'")
                    else:
                        lines.append(f"$W2={v}; 'Min waarde (REAL direct, 1 woord limiet)'")

                def append_real_max(v, temp_dm):
                    if is_address_token(v):
                        if temp_dm == 0:
                            raise ValueError("REAL indirect: temp_max_real_dm is 0 (niet geconfigureerd).")
                        a, p, _ = parse_address(v)
                        lines.append(f"READCMEM($W52,[{conn}:{a}{p}],2); 'Load Max (REAL indirect, 2 woorden)'")
                        lines.append(f"WRITECMEM([{conn}:DM{temp_dm}], $W52,2); 'Push Max naar temp REAL DM'")
                    else:
                        lines.append(f"$W3={v}; 'Max waarde (REAL direct, 1 woord limiet)'")

                if dt in ('INT', 'BCD'):
                    append_int_bcd_minmax(minv, True)
                    append_int_bcd_minmax(maxv, False)
                elif dt == 'REAL':
                    append_real_min(minv, tmin_real_dm)
                    append_real_max(maxv, tmax_real_dm)
                # --- DINT (toegevoegd) ---
                elif dt == 'DINT':
                    if is_address_token(minv):
                        raise ValueError("Indirecte min/max in DINT wordt niet ondersteund (min)")
                    if is_address_token(maxv):
                        raise ValueError("Indirecte min/max in DINT wordt niet ondersteund (max)")
                    lines.append(f"$W2={minv}; 'Min waarde'")
                    lines.append(f"$W3={maxv}; 'Max waarde'")
                else:
                    raise ValueError(f"Unsupported Datatype: {dt}")

                popup_key = dt + ('_MinMaxAfterScale' if after else '')
                lines.extend([
                    f"$W4={ptr}; 'Adres pointer'",
                    f"$W25={code}; 'Area code'",
                    f"WRITECMEM([{conn}:DM{pb}], $W1,25); 'Copy to PLC'",
                    f"$W30=1; 'Start edit flag'",
                    f"WRITECMEM([{conn}:DM{sf}], $W30,1); 'Start edit flag'",
                    f"READCMEM($W35,[{conn}:{area}{ptr}],{1 if dt in ('INT', 'BCD') else 2}); 'Read current value'",
                    f"WRITECMEM([{conn}:DM{oi if dt in ('INT', 'BCD') else (orl if dt=='REAL' else odi)}], $W35,{1 if dt in ('INT', 'BCD') else 2}); 'Write old value'",
                    f"SHOWPAGE({pops.get(popup_key)}); 'Open pop-up'"
                ])

            self.output.setPlainText("\n".join(lines))
            logger.info("SingleMacro: generated successfully")

        except Exception as e:
            logger.exception("SingleMacro generation failed")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

# --- Main Application ---
class MacroGeneratorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Macro Generator')
        cfg = load_config()
        self.machine        = None

        # Backward-compat: root-level (oude) velden, valt terug op defaults
        self.connection     = cfg.get('connection', DEFAULT_SETTINGS['connection'])
        self.interface_base = cfg.get('interface_base', DEFAULT_SETTINGS['interface_base'])
        self.popup_mapping  = cfg.get('popups', DEFAULT_SETTINGS['popups']).copy()
        self.temp_min_real_dm = cfg.get('temp_min_real_dm', DEFAULT_SETTINGS['temp_min_real_dm'])
        self.temp_max_real_dm = cfg.get('temp_max_real_dm', DEFAULT_SETTINGS['temp_max_real_dm'])

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("Machine:"))
        self.machine_combo = QtWidgets.QComboBox()
        self.machine_combo.addItems(sorted(cfg.get('machines', {}).keys()))
        self.machine_combo.currentTextChanged.connect(self.load_machine)
        top_bar.addWidget(self.machine_combo)
        main_layout.addLayout(top_bar)

        for text, handler in [
            ('Configure Machines...',     self.open_config),
            ('Generate from Excel...',    self.select_excel),
            ('Generate Single Macro...',  self.single_macro),
            ('Generate Excel Template...', self.generate_template),
            ('Help',                      self.show_help)
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(handler)
            main_layout.addWidget(btn)

        self.setCentralWidget(central)
        if self.machine_combo.count():
            self.machine_combo.setCurrentIndex(0)

    def load_machine(self, name):
        if not name:
            return
        machines = load_config().get('machines', {})
        settings = machines.get(name, DEFAULT_SETTINGS)
        self.connection        = settings.get('connection', self.connection)
        self.interface_base    = settings.get('interface_base', self.interface_base)
        self.popup_mapping     = settings.get('popups', self.popup_mapping)
        self.temp_min_real_dm  = settings.get('temp_min_real_dm', DEFAULT_SETTINGS['temp_min_real_dm'])
        self.temp_max_real_dm  = settings.get('temp_max_real_dm', DEFAULT_SETTINGS['temp_max_real_dm'])
        logger.info("Loaded machine '%s': connection=%s, base=%s, temp_min_real_dm=%s, temp_max_real_dm=%s",
                    name, self.connection, self.interface_base, self.temp_min_real_dm, self.temp_max_real_dm)

    def open_config(self):
        dlg = ConfigDialog(self)
        if dlg.exec_():
            machine, _, _, _, _, _ = dlg.get_selected()
            idx = self.machine_combo.findText(machine)
            if idx < 0:
                self.machine_combo.addItem(machine)
                idx = self.machine_combo.findText(machine)
            self.machine_combo.setCurrentIndex(idx)
            self.load_machine(machine)        # direct herladen van settings

    def select_excel(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Excel file', os.getcwd(),
            'Excel Files (*.xlsx);;All Files (*)'
        )
        if path:
            self.process_excel(path)

    def process_excel(self, file_path):
        logger.info("Processing Excel: %s", file_path)
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info("Rows loaded: %d", len(df))

        if 'Error Message' not in df.columns:
            df['Error Message'] = ''
        else:
            df['Error Message'] = df['Error Message'].astype(object).fillna('')

        macros, sc, ec = [], 0, 0

        for idx, row in df.iterrows():
            addr    = row['Adres']
            raw_s   = row.get('Scaling', '')
            raw_min = row.get('Min Waarde', '')
            raw_max = row.get('Max Waarde', '')
            desc    = row['Omschrijving']
            dt      = str(row['Datatype']).upper()
            after   = row.get('Min Max After Scale', False)

            logger.info("Row %d ⇒ %s, %s, %s, %s, %s, %s",
                        idx, addr, raw_s, raw_min, raw_max, desc, dt)
            try:
                # parse scaling
                try:
                    scaling = int(float(str(raw_s).replace(',', '.')))
                except:
                    scaling = 0
                # parse min/max
                if is_address_token(raw_min):
                    minv = raw_min
                else:
                    try:
                        minv = int(float(str(raw_min).replace(',', '.')))
                    except:
                        minv = 0
                if is_address_token(raw_max):
                    maxv = raw_max
                else:
                    try:
                        maxv = int(float(str(raw_max).replace(',', '.')))
                    except:
                        maxv = 0

                if len(desc or '') > 30:
                    df.at[idx, 'Error Message'] = 'Omschrijving >30 chars.'

                area, ptr, bit = parse_address(addr)
                code = area_code(area)
                base = self.interface_base
                pb   = base + OFFSET['param_block']
                bb   = base + OFFSET['bool_param_block']
                sf   = base + OFFSET['start_flag']
                oi   = base + OFFSET['old_value_int']
                orl  = base + OFFSET['old_value_real']
                ob   = base + OFFSET['old_value_bool']
                # --- DINT (toegevoegd) ---
                odi  = base + OFFSET['old_value_dint']
                tmin_real_dm = self.temp_min_real_dm
                tmax_real_dm = self.temp_max_real_dm

                if dt == 'BOOL':
                    if area.startswith('D'):
                        readcode = 300
                    elif area.startswith('W'):
                        readcode = 104
                    elif area.startswith('H'):
                        readcode = 101
                    else:
                        m = re.fullmatch(r'E([0-3])', area)
                        if not m:
                            raise ValueError(f"Unsupported BOOL area: {area}")
                        readcode = 302 + int(m.group(1))
                    popup_key = 'BOOL'
                    lines = [
                        f"'Adres: {area}{ptr:05}.{bit}'",
                        f"'Deze macro triggert het openen van een pop-up voor boolean invoerfunctionaliteit in het {area} gebied.'",
                        f"$W1 = {ptr}; 'Adres pointer voor {area}{ptr:05} instellen'",
                        f"$W2 = {bit}; 'Bit index instellen voor {area}{ptr:05}.{bit}'",
                        f"STRCPY($W3,\"{desc}\"); 'Schrijf beschrijving naar $W3'",
                        f"$W22={code}; 'D=0, W=1, H=2, E0=3, E1=4, E2=5, E3=6'",
                        f"WRITECMEM([{self.connection}:DM{bb}], $W1,22); 'Kopieerslag naar PLC (incl. omschrijving)'",
                        f"READHOSTB($B10,[{self.connection}],{readcode},{ptr},{bit},1); 'Lees bit'",
                        f"WRITEHOSTB([{self.connection}],300,{ob},0,$B10,1); 'Schrijf boolean oude waarde'",
                        f"$W30 = 1; 'Start edit vlag'",
                        f"WRITECMEM([{self.connection}:DM{sf}], $W30,1); 'Schrijf start edit vlag naar PLC'",
                        f"SHOWPAGE({self.popup_mapping.get(popup_key)}); 'Open pop-up'"
                    ]
                else:
                    lines = [
                        f"'Adres: {area}{ptr}'",
                        f"STRCPY($W6,\"{desc}\"); 'Omschrijving'",
                        f"$W1={scaling}; 'Scaling'"
                    ]

                    def append_int_bcd_minmax(v, is_min):
                        if is_address_token(v):
                            a, p, _ = parse_address(v)
                            lines.append(f"READCMEM($W{2 if is_min else 3},[{self.connection}:{a}{p}],1); 'Load {'Min' if is_min else 'Max'} waarde (indirect)'")
                        else:
                            vv = decimal_to_bcd(v) if dt == 'BCD' else v
                            lines.append(f"$W{2 if is_min else 3}={vv}; '{'Min' if is_min else 'Max'} waarde'")

                    def append_real_min(v, temp_dm):
                        if is_address_token(v):
                            if temp_dm == 0:
                                raise ValueError("REAL indirect: temp_min_real_dm is 0 (niet geconfigureerd).")
                            a, p, _ = parse_address(v)
                            lines.append(f"READCMEM($W50,[{self.connection}:{a}{p}],2); 'Load Min (REAL indirect, 2 woorden)'")
                            lines.append(f"WRITECMEM([{self.connection}:DM{temp_dm}], $W50,2); 'Push Min naar temp REAL DM'")
                        else:
                            lines.append(f"$W2={v}; 'Min waarde (REAL direct, 1 woord limiet)'")

                    def append_real_max(v, temp_dm):
                        if is_address_token(v):
                            if temp_dm == 0:
                                raise ValueError("REAL indirect: temp_max_real_dm is 0 (niet geconfigureerd).")
                            a, p, _ = parse_address(v)
                            lines.append(f"READCMEM($W52,[{self.connection}:{a}{p}],2); 'Load Max (REAL indirect, 2 woorden)'")
                            lines.append(f"WRITECMEM([{self.connection}:DM{temp_dm}], $W52,2); 'Push Max naar temp REAL DM'")
                        else:
                            lines.append(f"$W3={v}; 'Max waarde (REAL direct, 1 woord limiet)'")

                    if dt in ('INT', 'BCD'):
                        append_int_bcd_minmax(minv, True)
                        append_int_bcd_minmax(maxv, False)

                        popup_key = dt + ('_MinMaxAfterScale' if after else '')
                        lines.extend([
                            f"$W4={ptr}; 'Adres pointer'",
                            f"$W25={code}; 'Area code'",
                            f"WRITECMEM([{self.connection}:DM{pb}], $W1,25); 'Copy to PLC'",
                            f"$W30=1; 'Start edit flag'",
                            f"WRITECMEM([{self.connection}:DM{sf}], $W30,1); 'Start edit flag'",
                            f"READCMEM($W35,[{self.connection}:{area}{ptr}],1); 'Read current value'",
                            f"WRITECMEM([{self.connection}:DM{oi}], $W35,1); 'Write old value'",
                            f"SHOWPAGE({self.popup_mapping.get(popup_key)}); 'Open pop-up'"
                        ])

                    elif dt == 'REAL':
                        append_real_min(minv, tmin_real_dm)
                        append_real_max(maxv, tmax_real_dm)

                        popup_key = dt + ('_MinMaxAfterScale' if after else '')
                        lines.extend([
                            f"$W4={ptr}; 'Adres pointer'",
                            f"$W25={code}; 'Area code'",
                            f"WRITECMEM([{self.connection}:DM{pb}], $W1,25); 'Copy to PLC'",
                            f"$W30=1; 'Start edit flag'",
                            f"WRITECMEM([{self.connection}:DM{sf}], $W30,1); 'Start edit flag'",
                            f"READCMEM($W35,[{self.connection}:{area}{ptr}],2); 'Read current value'",
                            f"WRITECMEM([{self.connection}:DM{orl}], $W35,2); 'Write old value'",
                            f"SHOWPAGE({self.popup_mapping.get(popup_key)}); 'Open pop-up'"
                        ])

                    # --- DINT (toegevoegd) ---
                    elif dt == 'DINT':
                        if is_address_token(minv):
                            raise ValueError("Indirecte min/max in DINT wordt niet ondersteund (min)")
                        if is_address_token(maxv):
                            raise ValueError("Indirecte min/max in DINT wordt niet ondersteund (max)")
                        lines.append(f"$W2={minv}; 'Min waarde'")
                        lines.append(f"$W3={maxv}; 'Max waarde'")

                        popup_key = dt + ('_MinMaxAfterScale' if after else '')
                        lines.extend([
                            f"$W4={ptr}; 'Adres pointer'",
                            f"$W25={code}; 'Area code'",
                            f"WRITECMEM([{self.connection}:DM{pb}], $W1,25); 'Copy to PLC'",
                            f"$W30=1; 'Start edit flag'",
                            f"WRITECMEM([{self.connection}:DM{sf}], $W30,1); 'Start edit flag'",
                            f"READCMEM($W35,[{self.connection}:{area}{ptr}],2); 'Read current value'",
                            f"WRITECMEM([{self.connection}:DM{odi}], $W35,2); 'Write old value'",
                            f"SHOWPAGE({self.popup_mapping.get(popup_key)}); 'Open pop-up'"
                        ])

                    else:
                        raise ValueError(f"Unsupported Datatype: {dt}")

                df.at[idx, 'Error Message'] = ''
                macros.append("\n".join(lines))
                sc += 1
                logger.info("Row %d: generated successfully", idx)

            except Exception as e:
                df.at[idx, 'Error Message'] = str(e)
                ec += 1
                logger.exception("Row %d: failed", idx)

        df.to_excel(file_path, index=False, engine='openpyxl')
        out = os.path.splitext(file_path)[0] + '.txt'
        with open(out, 'w') as f:
            f.write("\n\n".join(macros) + "\n")
        logger.info("Batch done: Success=%d, Errors=%d → %s", sc, ec, out)

        msg = QtWidgets.QMessageBox()
        if ec:
            msg.warning(self, 'Done with errors', f"Success: {sc}, Errors: {ec}")
        else:
            msg.information(self, 'Done', f"All {sc} macros generated successfully.")

    def single_macro(self):
        dlg = SingleMacroDialog(self)
        dlg.exec_()

    def generate_template(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Template', os.getcwd(),
            'Excel Files (*.xlsx);;All Files (*)'
        )
        if path:
            pd.DataFrame(columns=[
                'Adres', 'Scaling', 'Min Waarde', 'Max Waarde',
                'Omschrijving', 'Datatype', 'Min Max After Scale'
            ]).to_excel(path, index=False)
            QtWidgets.QMessageBox.information(self, 'Template Generated', f'Saved: {path}')

    def show_help(self):
        help_txt = (
            "1. Configure connection, interface base, and pop-up pages per machine.\n"
            "   BOOL uses a single generic page; area code mapping:\n"
            "   D=0, W=1, H=2, E0=3, E1=4, E2=5, E3=6\n"
            "2. Batch-generate macros from Excel.\n"
            "3. Single-macro support—remains open after Generate.\n"
            "4. Use the template to get started.\n"
            "5. REAL indirect min/max → temp DM’s (configure in machine settings). INT/BCD unchanged.\n"
        )
        QtWidgets.QMessageBox.information(self, 'Help', help_txt)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MacroGeneratorApp()
    win.show()
    sys.exit(app.exec_())
