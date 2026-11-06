"""Загрузка быстрой части правил на C (fast.c). Библиотека собирается при первом импорте,
своя для каждой ширины доски в 64-битных словах: 64 клетки — одно слово, 256 — четыре."""
import ctypes
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "fast.c")

CODE_NONE, CODE_WIN, CODE_DRAW, CODE_LOST, CODE_FORCED, CODE_VCF = range(6)

_libraries = {}


def _build(words, target):
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=HERE)
    os.close(fd)
    try:
        subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", f"-DWORDS={words}", "-o", tmp, SOURCE], check=True)
        os.replace(tmp, target)  # атомарно: процессы могут собирать одновременно
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _library(words):
    if words not in _libraries:
        target = os.path.join(HERE, f"_fast{words}.so")
        if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(SOURCE):
            _build(words, target)
        lib = ctypes.CDLL(target)
        lib.qubic_new.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p)
        lib.qubic_new.restype = ctypes.c_void_p
        lib.qubic_find_vcf.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
        lib.qubic_find_vcf.restype = ctypes.c_int
        lib.qubic_analyse.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        lib.qubic_analyse.restype = ctypes.c_int
        _libraries[words] = lib
    return _libraries[words]


class Rules:
    """Правила одной геометрии доски. Камни передаются как целые числа Python."""

    def __init__(self, line_masks, line_len, cells, words):
        self.size = words * 8
        self.lib = _library(words)
        masks = b"".join(m.to_bytes(self.size, "little") for m in line_masks)
        self.handle = self.lib.qubic_new(len(line_masks), line_len, cells, masks)

    def analyse(self, own, opp, budget=0):
        size = self.size
        return self.lib.qubic_analyse(self.handle, own.to_bytes(size, "little"), opp.to_bytes(size, "little"), budget)

    def find_vcf(self, own, opp, budget=400, depth=16):
        """Ищет выигрыш серией шахов: каждый ход создаёт угрозу, ответ соперника вынужден.

        Возвращает первый ход такой серии или None.
        """
        size = self.size
        cell = self.lib.qubic_find_vcf(self.handle, own.to_bytes(size, "little"), opp.to_bytes(size, "little"), budget, depth)
        return cell if cell >= 0 else None
