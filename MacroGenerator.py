import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from PyQt5 import QtCore, QtWidgets


# --- Logging setup ---
logger = logging.getLogger("MacroGenerator")
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
log_file = os.path.join(os.getcwd(), "macro_generator.log")
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.info("Logger initialized; writing to %s", log_file)


CONFIG_FILE = os.path.expanduser("~/.macro_generator_config.json")
CONNECTION_OPTIONS = ["HOST3", "ETHERNET"]


DEFAULT_SETTINGS: Dict[str, Any] = {
    "popups": {
        "INT": 901,
        "INT_MinMaxAfterScale": 902,
        "BCD": 903,
        "REAL": 904,
        "REAL_MinMaxAfterScale": 905,
        "BOOL": 906,
        "DINT": 907,
        "DINT_MinMaxAfterScale": 908,
    },
    "connection": "HOST3",
    "interface_base": 20119,
    "temp_min_real_dm": 0,
    "temp_max_real_dm": 0,
    "pending_changes_bit": "",
    "commit_changes_bit": "",
    "rollback_changes_bit": "",
}

OFFSET = {
    "param_block": 19,
    "bool_param_block": 22,
    "start_flag": 1,
    "old_value_int": 15,
    "old_value_real": 16,
    "old_value_bool": 18,
    "old_value_dint": 44,
}


@dataclass
class MachineSettings:
    connection: str = DEFAULT_SETTINGS["connection"]
    interface_base: int = DEFAULT_SETTINGS["interface_base"]
    popups: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SETTINGS["popups"]))
    temp_min_real_dm: int = DEFAULT_SETTINGS["temp_min_real_dm"]
    temp_max_real_dm: int = DEFAULT_SETTINGS["temp_max_real_dm"]
    pending_changes_bit: str = DEFAULT_SETTINGS["pending_changes_bit"]
    commit_changes_bit: str = DEFAULT_SETTINGS["commit_changes_bit"]
    rollback_changes_bit: str = DEFAULT_SETTINGS["rollback_changes_bit"]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MachineSettings":
        return cls(
            connection=data.get("connection", DEFAULT_SETTINGS["connection"]),
            interface_base=data.get("interface_base", DEFAULT_SETTINGS["interface_base"]),
            popups=dict(data.get("popups", DEFAULT_SETTINGS["popups"])),
            temp_min_real_dm=data.get("temp_min_real_dm", DEFAULT_SETTINGS["temp_min_real_dm"]),
            temp_max_real_dm=data.get("temp_max_real_dm", DEFAULT_SETTINGS["temp_max_real_dm"]),
            pending_changes_bit=data.get("pending_changes_bit", DEFAULT_SETTINGS["pending_changes_bit"]),
            commit_changes_bit=data.get("commit_changes_bit", DEFAULT_SETTINGS["commit_changes_bit"]),
            rollback_changes_bit=data.get("rollback_changes_bit", DEFAULT_SETTINGS["rollback_changes_bit"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection": self.connection,
            "interface_base": self.interface_base,
            "popups": dict(self.popups),
            "temp_min_real_dm": self.temp_min_real_dm,
            "temp_max_real_dm": self.temp_max_real_dm,
            "pending_changes_bit": self.pending_changes_bit,
            "commit_changes_bit": self.commit_changes_bit,
            "rollback_changes_bit": self.rollback_changes_bit,
        }


def load_config() -> Dict[str, Any]:
    logger.debug("Loading config from %s", CONFIG_FILE)
    if not os.path.exists(CONFIG_FILE):
        logger.debug("No config file found, using defaults")
        return {"machines": {}}

    with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    migrated = False
    for settings in cfg.get("machines", {}).values():
        popups = settings.get("popups", {})
        legacy = {"DBOOL", "WBOOL", "HBOOL", "EBOOL"} & set(popups)
        if legacy and "BOOL" not in popups:
            chosen_key = sorted(legacy)[0]
            popups["BOOL"] = popups[chosen_key]
            for key in legacy:
                popups.pop(key, None)
            migrated = True
            logger.info("Migrated legacy popup keys %s → BOOL", legacy)

    for settings in cfg.get("machines", {}).values():
        if "temp_min_real_dm" not in settings:
            settings["temp_min_real_dm"] = DEFAULT_SETTINGS["temp_min_real_dm"]
            migrated = True
        if "temp_max_real_dm" not in settings:
            settings["temp_max_real_dm"] = DEFAULT_SETTINGS["temp_max_real_dm"]
            migrated = True
        if "pending_changes_bit" not in settings:
            settings["pending_changes_bit"] = DEFAULT_SETTINGS["pending_changes_bit"]
            migrated = True
        if "commit_changes_bit" not in settings:
            settings["commit_changes_bit"] = DEFAULT_SETTINGS["commit_changes_bit"]
            migrated = True
        if "rollback_changes_bit" not in settings:
            settings["rollback_changes_bit"] = DEFAULT_SETTINGS["rollback_changes_bit"]
            migrated = True

    if migrated:
        save_config(cfg)

    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    logger.debug("Saving config to %s", CONFIG_FILE)
    with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)


def decimal_to_bcd(value: int) -> int:
    bcd_value = 0
    multiplier = 1
    while value:
        bcd_value += (value % 10) * multiplier
        multiplier <<= 4
        value //= 10
    return bcd_value


def parse_address(address: Any) -> Tuple[str, int, Optional[int]]:
    text = str(address).upper()

    if "_" in text:
        area, rest = text.split("_", 1)
        if "." in rest:
            ptr_text, bit_text = rest.split(".", 1)
            bit = int(bit_text)
        else:
            ptr_text = rest
            bit = None
        pointer = int(ptr_text)
        return area, pointer, bit

    match = re.match(r"^([A-Z]+)(\d+)(?:\.(\d+))?$", text)
    if not match:
        raise ValueError(f"Invalid address: {address}")

    area = match.group(1)
    pointer = int(match.group(2))
    bit = int(match.group(3)) if match.group(3) else None
    return area, pointer, bit


def area_code(area: str) -> int:
    candidate = area.upper()
    if candidate.startswith("D"):
        return 0
    if candidate.startswith("W"):
        return 1
    if candidate.startswith("H"):
        return 2

    match = re.fullmatch(r"E([0-9])", candidate)
    if match:
        return 3 + int(match.group(1))

    raise ValueError(f"Unsupported area: {area}")


def is_address_token(value: Any) -> bool:
    try:
        parse_address(value)
        return True
    except Exception:
        return False


@dataclass
class MacroInput:
    address: Any
    scaling: int
    min_value: Union[int, str]
    max_value: Union[int, str]
    description: str
    datatype: str
    after_scale: Any = False
    save_workflow_enabled: bool = False


@dataclass
class MacroContext:
    connection: str
    interface_base: int
    popup_mapping: Dict[str, int]
    temp_min_real_dm: int
    temp_max_real_dm: int
    pending_changes_bit: str
    commit_changes_bit: str
    rollback_changes_bit: str


class MacroBuilder:
    def __init__(self, context: MacroContext):
        self.context = context

    def generate(self, macro_input: MacroInput) -> List[str]:
        area, pointer, bit = parse_address(macro_input.address)
        datatype = macro_input.datatype.upper()
        code = area_code(area)

        if datatype == "BOOL":
            lines = self._build_bool(area, pointer, bit, code, macro_input.description)
            if macro_input.save_workflow_enabled:
                self._append_pending_changes(lines)
            return lines

        lines = self._build_numeric(
            area=area,
            pointer=pointer,
            code=code,
            datatype=datatype,
            scaling=macro_input.scaling,
            minimum=macro_input.min_value,
            maximum=macro_input.max_value,
            description=macro_input.description,
            after_scale=macro_input.after_scale,
        )
        if macro_input.save_workflow_enabled:
            self._append_pending_changes(lines)
        return lines

    def _append_pending_changes(self, lines: List[str]) -> None:
        pending_address = (self.context.pending_changes_bit or "").strip()
        if not pending_address:
            raise ValueError(
                "Save Workflow Enabled is TRUE but machine setting 'pending_changes_bit' is empty."
            )

        area, pointer, bit = parse_address(pending_address)
        if bit is None:
            raise ValueError(
                "Machine setting 'pending_changes_bit' must include a bit index (example: D123.0)."
            )

        if area.startswith("D"):
            write_code = 300
        elif area.startswith("W"):
            write_code = 104
        elif area.startswith("H"):
            write_code = 101
        else:
            match = re.fullmatch(r"E([0-9])", area)
            if not match:
                raise ValueError(f"Unsupported pending_changes_bit area: {area}")
            write_code = 302 + int(match.group(1))

        lines.extend(
            [
                "'Save workflow: mark pending changes'",
                "$B90=1; 'Set pending buffer to TRUE'",
                f"WRITEHOSTB([{self.context.connection}],{write_code},{pointer},{bit},$B90,1); 'Set pending changes bit'",
            ]
        )

    def _build_bool(self, area: str, pointer: int, bit: Optional[int], code: int, description: str) -> List[str]:
        if area.startswith("D"):
            readcode = 300
        elif area.startswith("W"):
            readcode = 104
        elif area.startswith("H"):
            readcode = 101
        else:
            match = re.fullmatch(r"E([0-9])", area)
            if not match:
                raise ValueError(f"Unsupported BOOL area: {area}")
            readcode = 302 + int(match.group(1))


        bool_block = self.context.interface_base + OFFSET["bool_param_block"]
        start_flag = self.context.interface_base + OFFSET["start_flag"]
        old_bool = self.context.interface_base + OFFSET["old_value_bool"]

        return [
            f"'Adres: {area}{pointer:05}.{bit}'",
            f"'Deze macro triggert het openen van een pop-up voor boolean invoerfunctionaliteit in het {area} gebied.'",
            f"$W1 = {pointer}; 'Adres pointer voor {area}{pointer:05} instellen'",
            f"$W2 = {bit}; 'Bit index instellen voor {area}{pointer:05}.{bit}'",
            f"STRCPY($W3,\"{description}\"); 'Schrijf beschrijving naar $W3'",
            "$W22={}; 'D=0, W=1, H=2, E0=3 ... E9=12'".format(code),
            f"WRITECMEM([{self.context.connection}:DM{bool_block}], $W1,22); 'Kopieerslag naar PLC (incl. omschrijving)'",
            f"READHOSTB($B10,[{self.context.connection}],{readcode},{pointer},{bit},1); 'Lees bit'",
            f"WRITEHOSTB([{self.context.connection}],300,{old_bool},0,$B10,1); 'Schrijf boolean oude waarde'",
            "$W30 = 1; 'Start edit vlag'",
            f"WRITECMEM([{self.context.connection}:DM{start_flag}], $W30,1); 'Schrijf start edit vlag naar PLC'",
            f"SHOWPAGE({self.context.popup_mapping.get('BOOL')}); 'Open pop-up'",
        ]

    def _build_numeric(
        self,
        area: str,
        pointer: int,
        code: int,
        datatype: str,
        scaling: int,
        minimum: Union[int, str],
        maximum: Union[int, str],
        description: str,
        after_scale: Any,
    ) -> List[str]:
        lines = [
            f"'Adres: {area}{pointer}'",
            f"STRCPY($W6,\"{description}\"); 'Omschrijving'",
            f"$W1={scaling}; 'Scaling'",
        ]

        if datatype in ("INT", "BCD"):
            self._append_int_or_bcd_minmax(lines, minimum, datatype, True)
            self._append_int_or_bcd_minmax(lines, maximum, datatype, False)
        elif datatype == "REAL":
            self._append_real_min(lines, minimum)
            self._append_real_max(lines, maximum)
        elif datatype == "DINT":
            if is_address_token(minimum):
                raise ValueError("Indirecte min/max in DINT wordt niet ondersteund (min)")
            if is_address_token(maximum):
                raise ValueError("Indirecte min/max in DINT wordt niet ondersteund (max)")
            lines.append(f"$W2={minimum}; 'Min waarde'")
            lines.append(f"$W3={maximum}; 'Max waarde'")
        else:
            raise ValueError(f"Unsupported Datatype: {datatype}")

        parameter_block = self.context.interface_base + OFFSET["param_block"]
        start_flag = self.context.interface_base + OFFSET["start_flag"]
        old_int = self.context.interface_base + OFFSET["old_value_int"]
        old_real = self.context.interface_base + OFFSET["old_value_real"]
        old_dint = self.context.interface_base + OFFSET["old_value_dint"]

        popup_key = datatype + ("_MinMaxAfterScale" if after_scale else "")
        read_words = 1 if datatype in ("INT", "BCD") else 2
        old_value_dm = old_int if datatype in ("INT", "BCD") else (old_real if datatype == "REAL" else old_dint)

        lines.extend(
            [
                f"$W4={pointer}; 'Adres pointer'",
                f"$W25={code}; 'Area code'",
                f"WRITECMEM([{self.context.connection}:DM{parameter_block}], $W1,25); 'Copy to PLC'",
                "$W30=1; 'Start edit flag'",
                f"WRITECMEM([{self.context.connection}:DM{start_flag}], $W30,1); 'Start edit flag'",
                f"READCMEM($W35,[{self.context.connection}:{area}{pointer}],{read_words}); 'Read current value'",
                f"WRITECMEM([{self.context.connection}:DM{old_value_dm}], $W35,{read_words}); 'Write old value'",
                f"SHOWPAGE({self.context.popup_mapping.get(popup_key)}); 'Open pop-up'",
            ]
        )

        return lines

    def _append_int_or_bcd_minmax(
        self, lines: List[str], value: Union[int, str], datatype: str, is_minimum: bool
    ) -> None:
        register = "$W2" if is_minimum else "$W3"
        label = "Min" if is_minimum else "Max"

        if is_address_token(value):
            area, pointer, _ = parse_address(value)
            lines.append(
                f"READCMEM({register},[{self.context.connection}:{area}{pointer}],1); 'Load {label} waarde (indirect)'"
            )
        else:
            normalized = decimal_to_bcd(value) if datatype == "BCD" else value
            lines.append(f"{register}={normalized}; '{label} waarde'")

    def _append_real_min(self, lines: List[str], value: Union[int, str]) -> None:
        if is_address_token(value):
            if self.context.temp_min_real_dm == 0:
                raise ValueError("REAL indirect: temp_min_real_dm is 0 (niet geconfigureerd).")
            area, pointer, _ = parse_address(value)
            lines.extend(
                [
                    f"READCMEM($W50,[{self.context.connection}:{area}{pointer}],2); 'Load Min (REAL indirect, 2 woorden)'",
                    f"WRITECMEM([{self.context.connection}:DM{self.context.temp_min_real_dm}], $W50,2); 'Push Min naar temp REAL DM'",
                ]
            )
        else:
            lines.append(f"$W2={value}; 'Min waarde (REAL direct, 1 woord limiet)'")

    def _append_real_max(self, lines: List[str], value: Union[int, str]) -> None:
        if is_address_token(value):
            if self.context.temp_max_real_dm == 0:
                raise ValueError("REAL indirect: temp_max_real_dm is 0 (niet geconfigureerd).")
            area, pointer, _ = parse_address(value)
            lines.extend(
                [
                    f"READCMEM($W52,[{self.context.connection}:{area}{pointer}],2); 'Load Max (REAL indirect, 2 woorden)'",
                    f"WRITECMEM([{self.context.connection}:DM{self.context.temp_max_real_dm}], $W52,2); 'Push Max naar temp REAL DM'",
                ]
            )
        else:
            lines.append(f"$W3={value}; 'Max waarde (REAL direct, 1 woord limiet)'")


def parse_int_like(value: Any, fallback: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", ".")))
    except Exception:
        return fallback


def parse_minmax(value: Any) -> Union[int, str]:
    if is_address_token(value):
        return value
    return parse_int_like(value, fallback=0)


def parse_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().upper()
    if text in {"1", "TRUE", "YES", "Y", "JA", "ON", "X"}:
        return True
    if text in {"0", "FALSE", "NO", "N", "NEE", "OFF", ""}:
        return False
    return False


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Machine Configuration")
        self.cfg = load_config()

        layout = QtWidgets.QVBoxLayout(self)

        self.machine_select = QtWidgets.QComboBox()
        self.machine_select.addItems(sorted(self.cfg["machines"].keys()))
        self.machine_select.currentTextChanged.connect(self.fill_fields)
        layout.addWidget(QtWidgets.QLabel("Select Machine:"))
        layout.addWidget(self.machine_select)

        self.conn_combo = QtWidgets.QComboBox()
        self.conn_combo.addItems(CONNECTION_OPTIONS)
        conn_form = QtWidgets.QFormLayout()
        conn_form.addRow("Connection type:", self.conn_combo)
        layout.addLayout(conn_form)

        self.base_spin = QtWidgets.QSpinBox()
        self.base_spin.setRange(0, 99999)
        base_form = QtWidgets.QFormLayout()
        base_form.addRow("Interface Base DM:", self.base_spin)
        layout.addLayout(base_form)

        self.spin_boxes: Dict[str, QtWidgets.QSpinBox] = {}
        popup_form = QtWidgets.QFormLayout()
        for key in DEFAULT_SETTINGS["popups"]:
            spin_box = QtWidgets.QSpinBox()
            spin_box.setRange(0, 9999)
            popup_form.addRow(f"{key} page:", spin_box)
            self.spin_boxes[key] = spin_box
        layout.addLayout(popup_form)

        self.temp_min_real_sb = QtWidgets.QSpinBox()
        self.temp_min_real_sb.setRange(0, 65535)
        self.temp_max_real_sb = QtWidgets.QSpinBox()
        self.temp_max_real_sb.setRange(0, 65535)
        temp_form = QtWidgets.QFormLayout()
        temp_form.addRow("REAL temp MIN (DM):", self.temp_min_real_sb)
        temp_form.addRow("REAL temp MAX (DM):", self.temp_max_real_sb)
        layout.addLayout(temp_form)

        self.pending_bit_e = QtWidgets.QLineEdit()
        self.commit_bit_e = QtWidgets.QLineEdit()
        self.rollback_bit_e = QtWidgets.QLineEdit()
        workflow_form = QtWidgets.QFormLayout()
        workflow_form.addRow("Pending changes bit:", self.pending_bit_e)
        workflow_form.addRow("Commit changes bit:", self.commit_bit_e)
        workflow_form.addRow("Rollback changes bit:", self.rollback_bit_e)
        layout.addLayout(workflow_form)

        buttons_layout = QtWidgets.QHBoxLayout()
        self.new_btn = QtWidgets.QPushButton("New Machine")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        buttons_layout.addWidget(self.new_btn)
        buttons_layout.addWidget(self.save_btn)
        buttons_layout.addWidget(self.cancel_btn)
        layout.addLayout(buttons_layout)

        self.new_btn.clicked.connect(self.add_machine)
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if self.machine_select.count():
            self.fill_fields(self.machine_select.currentText())

    def fill_fields(self, name: str) -> None:
        settings = MachineSettings.from_dict(self.cfg["machines"].get(name, DEFAULT_SETTINGS))
        self.conn_combo.setCurrentText(settings.connection)
        self.base_spin.setValue(settings.interface_base)
        for key, spin_box in self.spin_boxes.items():
            spin_box.setValue(settings.popups.get(key, DEFAULT_SETTINGS["popups"][key]))
        self.temp_min_real_sb.setValue(settings.temp_min_real_dm)
        self.temp_max_real_sb.setValue(settings.temp_max_real_dm)
        self.pending_bit_e.setText(settings.pending_changes_bit)
        self.commit_bit_e.setText(settings.commit_changes_bit)
        self.rollback_bit_e.setText(settings.rollback_changes_bit)

    def add_machine(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "New Machine", "Machine identifier:")
        if ok and name:
            self.cfg["machines"][name] = MachineSettings().to_dict()
            save_config(self.cfg)
            self.machine_select.addItem(name)
            self.machine_select.setCurrentText(name)

    def accept(self) -> None:
        name = self.machine_select.currentText()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Error", "No machine selected.")
            return

        settings = MachineSettings(
            connection=self.conn_combo.currentText(),
            interface_base=self.base_spin.value(),
            popups={key: spin_box.value() for key, spin_box in self.spin_boxes.items()},
            temp_min_real_dm=self.temp_min_real_sb.value(),
            temp_max_real_dm=self.temp_max_real_sb.value(),
            pending_changes_bit=self.pending_bit_e.text().strip(),
            commit_changes_bit=self.commit_bit_e.text().strip(),
            rollback_changes_bit=self.rollback_bit_e.text().strip(),
        )
        self.cfg["machines"][name] = settings.to_dict()
        save_config(self.cfg)
        super().accept()

    def get_selected(self) -> Tuple[str, str, int, Dict[str, int], int, int]:
        name = self.machine_select.currentText()
        settings = MachineSettings.from_dict(self.cfg["machines"].get(name, DEFAULT_SETTINGS))
        return (
            name,
            settings.connection,
            settings.interface_base,
            settings.popups,
            settings.temp_min_real_dm,
            settings.temp_max_real_dm,
        )


class SingleMacroDialog(QtWidgets.QDialog):
    def __init__(self, parent: "MacroGeneratorApp"):
        super().__init__(parent)
        self.setWindowTitle("Generate Single Macro")

        layout = QtWidgets.QFormLayout(self)

        self.adres_e = QtWidgets.QLineEdit()
        self.scal_e = QtWidgets.QLineEdit()
        self.min_e = QtWidgets.QLineEdit()
        self.max_e = QtWidgets.QLineEdit()
        self.desc_e = QtWidgets.QLineEdit()
        self.dt_e = QtWidgets.QLineEdit()
        self.after_cb = QtWidgets.QCheckBox("MinMax After Scale")

        layout.addRow("Adres:", self.adres_e)
        layout.addRow("Scaling:", self.scal_e)
        layout.addRow("Min Waarde:", self.min_e)
        layout.addRow("Max Waarde:", self.max_e)
        layout.addRow("Omschrijving:", self.desc_e)
        layout.addRow("Datatype:", self.dt_e)
        layout.addRow("", self.after_cb)

        buttons_layout = QtWidgets.QHBoxLayout()
        generate_button = QtWidgets.QPushButton("Generate")
        close_button = QtWidgets.QPushButton("Close")
        buttons_layout.addWidget(generate_button)
        buttons_layout.addWidget(close_button)
        layout.addRow("", buttons_layout)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addRow("Macro:", self.output)

        generate_button.clicked.connect(self.on_generate)
        close_button.clicked.connect(self.reject)

    def _parse_value(self, text: str, force_int: bool = False) -> Union[int, str]:
        stripped = text.strip()
        if not stripped:
            return 0
        if is_address_token(stripped) and not force_int:
            return stripped
        return parse_int_like(stripped, fallback=0)

    def on_generate(self) -> None:
        address = self.adres_e.text().strip()
        scaling = self._parse_value(self.scal_e.text(), force_int=True)
        min_value = self._parse_value(self.min_e.text())
        max_value = self._parse_value(self.max_e.text())
        description = self.desc_e.text().strip()
        datatype = self.dt_e.text().strip().upper()
        after_scale = self.after_cb.isChecked()

        logger.info(
            "SingleMacro input → Adres=%s, Scaling=%s, Min=%s, Max=%s, Desc=%r, Datatype=%s, After=%s",
            address,
            scaling,
            min_value,
            max_value,
            description,
            datatype,
            after_scale,
        )

        try:
            parent = self.parent()
            context = MacroContext(
                connection=parent.connection,
                interface_base=parent.interface_base,
                popup_mapping=parent.popup_mapping,
                temp_min_real_dm=parent.temp_min_real_dm,
                temp_max_real_dm=parent.temp_max_real_dm,
                pending_changes_bit=parent.pending_changes_bit,
                commit_changes_bit=parent.commit_changes_bit,
                rollback_changes_bit=parent.rollback_changes_bit,
            )
            builder = MacroBuilder(context)
            lines = builder.generate(
                MacroInput(
                    address=address,
                    scaling=int(scaling),
                    min_value=min_value,
                    max_value=max_value,
                    description=description,
                    datatype=datatype,
                    after_scale=after_scale,
                    save_workflow_enabled=False,
                )
            )
            self.output.setPlainText("\n".join(lines))
            logger.info("SingleMacro: generated successfully")
        except Exception as exc:
            logger.exception("SingleMacro generation failed")
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))


class MacroGeneratorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Macro Generator")

        cfg = load_config()
        self.machine = None

        self.connection = cfg.get("connection", DEFAULT_SETTINGS["connection"])
        self.interface_base = cfg.get("interface_base", DEFAULT_SETTINGS["interface_base"])
        self.popup_mapping = cfg.get("popups", DEFAULT_SETTINGS["popups"]).copy()
        self.temp_min_real_dm = cfg.get("temp_min_real_dm", DEFAULT_SETTINGS["temp_min_real_dm"])
        self.temp_max_real_dm = cfg.get("temp_max_real_dm", DEFAULT_SETTINGS["temp_max_real_dm"])
        self.pending_changes_bit = cfg.get("pending_changes_bit", DEFAULT_SETTINGS["pending_changes_bit"])
        self.commit_changes_bit = cfg.get("commit_changes_bit", DEFAULT_SETTINGS["commit_changes_bit"])
        self.rollback_changes_bit = cfg.get("rollback_changes_bit", DEFAULT_SETTINGS["rollback_changes_bit"])

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("Machine:"))
        self.machine_combo = QtWidgets.QComboBox()
        self.machine_combo.addItems(sorted(cfg.get("machines", {}).keys()))
        self.machine_combo.currentTextChanged.connect(self.load_machine)
        top_bar.addWidget(self.machine_combo)
        main_layout.addLayout(top_bar)

        buttons = [
            ("Configure Machines...", self.open_config),
            ("Generate from Excel...", self.select_excel),
            ("Generate Single Macro...", self.single_macro),
            ("Generate Excel Template...", self.generate_template),
            ("Help", self.show_help),
        ]
        for text, handler in buttons:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(handler)
            main_layout.addWidget(button)

        self.setCentralWidget(central)
        if self.machine_combo.count():
            self.machine_combo.setCurrentIndex(0)

    def _macro_context(self) -> MacroContext:
        return MacroContext(
            connection=self.connection,
            interface_base=self.interface_base,
            popup_mapping=self.popup_mapping,
            temp_min_real_dm=self.temp_min_real_dm,
            temp_max_real_dm=self.temp_max_real_dm,
            pending_changes_bit=self.pending_changes_bit,
            commit_changes_bit=self.commit_changes_bit,
            rollback_changes_bit=self.rollback_changes_bit,
        )

    def load_machine(self, name: str) -> None:
        if not name:
            return

        settings = MachineSettings.from_dict(load_config().get("machines", {}).get(name, DEFAULT_SETTINGS))
        self.connection = settings.connection
        self.interface_base = settings.interface_base
        self.popup_mapping = settings.popups
        self.temp_min_real_dm = settings.temp_min_real_dm
        self.temp_max_real_dm = settings.temp_max_real_dm
        self.pending_changes_bit = settings.pending_changes_bit
        self.commit_changes_bit = settings.commit_changes_bit
        self.rollback_changes_bit = settings.rollback_changes_bit

        logger.info(
            "Loaded machine '%s': connection=%s, base=%s, temp_min_real_dm=%s, temp_max_real_dm=%s",
            name,
            self.connection,
            self.interface_base,
            self.temp_min_real_dm,
            self.temp_max_real_dm,
        )

    def open_config(self) -> None:
        dialog = ConfigDialog(self)
        if dialog.exec_():
            machine, _, _, _, _, _ = dialog.get_selected()
            index = self.machine_combo.findText(machine)
            if index < 0:
                self.machine_combo.addItem(machine)
                index = self.machine_combo.findText(machine)
            self.machine_combo.setCurrentIndex(index)
            self.load_machine(machine)

    def select_excel(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Excel file",
            os.getcwd(),
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if path:
            self.process_excel(path)

    def process_excel(self, file_path: str) -> None:
        logger.info("Processing Excel: %s", file_path)
        dataframe = pd.read_excel(file_path, engine="openpyxl")
        logger.info("Rows loaded: %d", len(dataframe))

        if "Error Message" not in dataframe.columns:
            dataframe["Error Message"] = ""
        else:
            dataframe["Error Message"] = dataframe["Error Message"].astype(object).fillna("")

        macros: List[str] = []
        success_count = 0
        error_count = 0
        builder = MacroBuilder(self._macro_context())

        for idx, row in dataframe.iterrows():
            address = row["Adres"]
            raw_scaling = row.get("Scaling", "")
            raw_min = row.get("Min Waarde", "")
            raw_max = row.get("Max Waarde", "")
            description = row["Omschrijving"]
            datatype = str(row["Datatype"]).upper()
            after_scale = row.get("Min Max After Scale", False)

            logger.info(
                "Row %d ⇒ %s, %s, %s, %s, %s, %s, SaveWorkflow=%s",
                idx,
                address,
                raw_scaling,
                raw_min,
                raw_max,
                description,
                datatype,
                row.get("Save Workflow Enabled", False),
            )

            try:
                scaling = parse_int_like(raw_scaling, fallback=0)
                min_value = parse_minmax(raw_min)
                max_value = parse_minmax(raw_max)
                save_workflow_enabled = parse_bool_like(row.get("Save Workflow Enabled", False))

                if len(description or "") > 30:
                    dataframe.at[idx, "Error Message"] = "Omschrijving >30 chars."

                lines = builder.generate(
                    MacroInput(
                        address=address,
                        scaling=scaling,
                        min_value=min_value,
                        max_value=max_value,
                        description=description,
                        datatype=datatype,
                        after_scale=after_scale,
                        save_workflow_enabled=save_workflow_enabled,
                    )
                )

                dataframe.at[idx, "Error Message"] = ""
                macros.append("\n".join(lines))
                success_count += 1
                logger.info("Row %d: generated successfully", idx)
            except Exception as exc:
                dataframe.at[idx, "Error Message"] = str(exc)
                error_count += 1
                logger.exception("Row %d: failed", idx)

        dataframe.to_excel(file_path, index=False, engine="openpyxl")
        output_txt = os.path.splitext(file_path)[0] + ".txt"
        with open(output_txt, "w", encoding="utf-8") as handle:
            handle.write("\n\n".join(macros) + "\n")

        logger.info("Batch done: Success=%d, Errors=%d → %s", success_count, error_count, output_txt)

        if error_count:
            QtWidgets.QMessageBox().warning(
                self, "Done with errors", f"Success: {success_count}, Errors: {error_count}"
            )
        else:
            QtWidgets.QMessageBox().information(
                self, "Done", f"All {success_count} macros generated successfully."
            )

    def single_macro(self) -> None:
        SingleMacroDialog(self).exec_()

    def generate_template(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Template",
            os.getcwd(),
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if path:
            pd.DataFrame(
                columns=[
                    "Adres",
                    "Scaling",
                    "Min Waarde",
                    "Max Waarde",
                    "Omschrijving",
                    "Datatype",
                    "Min Max After Scale",
                    "Save Workflow Enabled",
                ]
            ).to_excel(path, index=False)
            QtWidgets.QMessageBox.information(self, "Template Generated", f"Saved: {path}")

    def show_help(self) -> None:
        help_txt = (
            "1. Configure connection, interface base, and pop-up pages per machine.\n"
            "   BOOL uses a single generic page; area code mapping:\n"
            "   D=0, W=1, H=2, E0=3 ... E9=12\n"
            "2. Batch-generate macros from Excel.\n"
            "3. Single-macro support—remains open after Generate.\n"
            "4. Use the template to get started.\n"
            "5. REAL indirect min/max → temp DM’s (configure in machine settings). INT/BCD unchanged.\n"
            "6. Save workflow per row: set 'Save Workflow Enabled' in Excel and configure pending/commit/rollback bits per machine.\n"
        )
        QtWidgets.QMessageBox.information(self, "Help", help_txt)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MacroGeneratorApp()
    window.show()
    sys.exit(app.exec_())
