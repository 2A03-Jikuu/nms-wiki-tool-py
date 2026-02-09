"""
No Man's Sky Multi-Tool Wiki Generator for Jupyter/Colab Notebooks.

This module provides a complete graphical interface for creating wiki pages for
Multi-Tools in the game No Man's Sky. It features:
- A tabbed form for entering all Multi-Tool details
- Procedural region name generation from portal glyphs using game logic
- Coordinate conversion from glyphs to standard galactic format
- Validation and template rendering for the No Man's Sky Wiki

Key Classes:
    ByteUtils: Handles low-level byte operations for procedural generation
    Generator: Generates procedural names using game algorithms
    RegionNameGenerator: Creates region names from galactic coordinates
    NMSGalaxyMap: Converts between portal glyphs and galactic coordinates
    NMSData: Loads and manages game data from external sources
    NMSWikiMultiToolGenerator: Main application with GUI and wiki generation

The tool is designed to run in Jupyter Notebook or Google Colab environments
with widget support for interactive form filling.
"""

import json
import re
import struct
from dataclasses import dataclass, fields
from html import escape as html_escape
from typing import Dict, List, Optional

import arrow
import requests
from IPython.display import Javascript, clear_output, display
from ipywidgets import (
    Button, Checkbox, Combobox, DatePicker, Dropdown, FloatText, HBox,
    HTML, IntText, Label, Layout, Output, Tab, Text, Textarea, VBox, Widget
)
from jinja2 import Environment
from pydantic import BaseModel, ConfigDict, Field, ValidationError


# Constants for byte operations and procedural generation
BYTE_MASK = 0xFF          # Mask to get only the lowest 8 bits (one byte)
BYTE_SHIFT = 8            # Number of bits to shift to move to next byte
SIGNED_SHORT_OFFSET = 32768  # Offset to convert between signed/unsigned 16-bit numbers
UNSIGNED_SHORT_MAX = 65536   # Maximum value for an unsigned 16-bit number
MAX_NAME_LENGTH = 64          # Maximum length for generated region names
SAFETY_LIMIT = 50             # Safety limit for infinite loop prevention
VOWELS = "aeiou"              # Standard vowels for name validation
VOWELS_WITH_Y = "aeiouy"      # Vowels including 'y' for consonant checking
DEFAULT_IMAGE = "NmsMisc NotAvailable.png"  # Default image when none provided
DEFAULT_CIVILIZATION = "Alliance of Galactic Travellers"  # Default civilization
DEFAULT_RESEARCH_TEAM = "AGT Bureau of Multi-Tool Registration"  # Default research team
DEFAULT_CIV_STUB = "{{AGT Notice}}"          # Default civilization notice template
DEFAULT_CIV_IMAGE = "AGT-BMultiToolResearch01.png"  # Default civilization badge
DEFAULT_RELEASE = "Breach"    # Default game release version
DEFAULT_MODE = "Normal"       # Default game mode


class ByteUtils:
    """
    Provides low-level byte manipulation utilities for procedural generation.

    This class implements the exact byte operations used by No Man's Sky for
    procedural generation, including arithmetic, logical operations, and seed
    management. It works with bytes as lists of integers (0-255).

    Attributes:
        SEED_MULTIPLIER (List[int]): Constant multiplier used in seed updates
    """

    SEED_MULTIPLIER = [0x99, 0xF8, 0x76, 0x5A]  # Game's constant for seed mixing

    @staticmethod
    def parse(val: str, little_endian: bool = True) -> List[int]:
        """
        Converts a hex string to a list of byte values.

        Args:
            val (str): Hexadecimal string (e.g., "1A2B3C")
            little_endian (bool): If True, reverse byte order (least significant first)

        Returns:
            List[int]: List of byte values (0-255)

        Example:
            >>> ByteUtils.parse("1A2B", True)
            [0x2B, 0x1A]
        """
        val = val.strip()
        if not val:
            return []
        if len(val) % 2 != 0:
            val = "0" + val
        res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]
        if little_endian:
            res.reverse()
        return res

    @staticmethod
    def format_short(byte_list: List[int]) -> List[int]:
        """
        Ensures a byte list has at least 2 bytes (16 bits).

        Pads with zeros if the list is shorter than 2 bytes.

        Args:
            byte_list (List[int]): Input byte list

        Returns:
            List[int]: Padded byte list with at least 2 bytes

        Example:
            >>> ByteUtils.format_short([0x01])
            [0x01, 0x00]
        """
        res = list(byte_list)
        while len(res) < 2:
            res.append(0x00)
        return res

    @staticmethod
    def add(op1: List[int], op2: List[int]) -> List[int]:
        """
        Adds two byte arrays together with carry propagation.

        Works like big integer addition but with 8-bit bytes.

        Args:
            op1 (List[int]): First operand (byte list)
            op2 (List[int]): Second operand (byte list)

        Returns:
            List[int]: Result of addition as byte list

        Example:
            >>> ByteUtils.add([0xFF, 0x01], [0x01, 0x00])
            [0x00, 0x02]  # 0x01FF + 0x0001 = 0x0200
        """
        result = list(op2)
        for i in range(len(op1)):
            result = ByteUtils._add_single(op1[i], result, i)
        return result

    @staticmethod
    def _add_single(val: int, target_list: List[int], index: int) -> List[int]:
        """
        Helper for adding a single byte with carry handling.

        Adds one byte to a specific position in the byte list, propagating
        any overflow (carry) to higher bytes.

        Args:
            val (int): Byte value to add (0-255)
            target_list (List[int]): Byte list to modify
            index (int): Position to add the byte

        Returns:
            List[int]: Modified byte list with carry propagated
        """
        if index < len(target_list):
            total = val + target_list[index]
            target_list[index] = total & BYTE_MASK
            rem = (total >> BYTE_SHIFT) & BYTE_MASK
            if rem != 0:
                target_list = ByteUtils._add_single(rem, target_list, index + 1)
        else:
            target_list.append(val)
        return target_list

    @staticmethod
    def sub(op1: List[int], op2: List[int]) -> List[int]:
        """
        Subtracts op1 from op2 (op2 - op1) with borrow propagation.

        Args:
            op1 (List[int]): Bytes to subtract
            op2 (List[int]): Bytes to subtract from

        Returns:
            List[int]: Result of subtraction as byte list
        """
        result = list(op2)
        for i in range(len(op1)):
            result = ByteUtils._sub_single(op1[i], result, i)
        return result

    @staticmethod
    def _sub_single(val: int, target_list: List[int], index: int) -> List[int]:
        """
        Helper for subtracting a single byte with borrow handling.

        Args:
            val (int): Byte value to subtract
            target_list (List[int]): Byte list to modify
            index (int): Position to subtract from

        Returns:
            List[int]: Modified byte list with borrow handled
        """
        if index < len(target_list):
            diff = val - target_list[index]
            target_list[index] = diff & BYTE_MASK
            rem = (diff >> BYTE_SHIFT) & BYTE_MASK
            if rem != 0:
                target_list = ByteUtils._sub_single(rem, target_list, index + 1)
        else:
            target_list.append(val)
        return target_list

    @staticmethod
    def multiply(op1: List[int], op2: List[int]) -> List[int]:
        """
        Multiplies two byte arrays using game's multiplication logic.

        Implements the exact multiplication algorithm used by No Man's Sky
        for procedural generation, handling signed intermediate results.

        Args:
            op1 (List[int]): First factor
            op2 (List[int]): Second factor

        Returns:
            List[int]: Product as byte list
        """
        result = []
        for i in range(len(op1)):
            rem = 0
            for j in range(len(op2)):
                raw_prod = (op1[i] * op2[j]) + rem
                signed_prd = (raw_prod + SIGNED_SHORT_OFFSET) % UNSIGNED_SHORT_MAX - SIGNED_SHORT_OFFSET
                rem = (signed_prd >> BYTE_SHIFT) & BYTE_MASK
                res = signed_prd & BYTE_MASK
                idx = i + j
                if idx < len(result):
                    result = ByteUtils._add_single(res, result, idx)
                else:
                    result.append(res)
            if rem > 0:
                idx = i + len(op2)
                if idx < len(result):
                    result = ByteUtils._add_single(rem, result, idx)
                else:
                    result.append(rem)
        return result

    @staticmethod
    def shl(op1: List[int], shift: int) -> List[int]:
        """
        Shifts bytes left (removes lower bytes).

        Similar to bitwise left shift but operates on whole bytes.

        Args:
            op1 (List[int]): Input byte list
            shift (int): Number of bytes to shift

        Returns:
            List[int]: Shifted byte list or [0x00] if empty
        """
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1: List[int], shift: int) -> List[int]:
        """
        Shifts bytes right (removes higher bytes).

        Args:
            op1 (List[int]): Input byte list
            shift (int): Number of bytes to shift

        Returns:
            List[int]: Shifted byte list or [0x00] if empty
        """
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1: List[int], roll: int) -> List[int]:
        """
        Rotates bytes left (moves bytes from start to end).

        Args:
            op1 (List[int]): Input byte list
            roll (int): Number of positions to rotate

        Returns:
            List[int]: Rotated byte list
        """
        if not op1:
            return op1
        r = roll % len(op1)
        return op1[r:] + op1[:r]

    @staticmethod
    def zxd(op1: List[int], extend: int) -> List[int]:
        """
        Zero-extends a byte array to specified length.

        Adds zero bytes at the end to reach the desired length.

        Args:
            op1 (List[int]): Input byte list
            extend (int): Desired total length

        Returns:
            List[int]: Extended byte list padded with zeros
        """
        return list(op1) + [0x00] * (extend - len(op1))

    @staticmethod
    def sxd(op1: List[int], extend: int) -> List[int]:
        """
        Sign-extends a byte array to specified length.

        Adds 0xFF or 0x00 bytes depending on the sign bit of the last byte.

        Args:
            op1 (List[int]): Input byte list
            extend (int): Desired total length

        Returns:
            List[int]: Sign-extended byte list
        """
        result = list(op1)
        val = 0xFF if (len(op1) > 0 and (op1[-1] >> 7) == 1) else 0x00
        for _ in range(extend - len(op1)):
            result.append(val)
        return result

    @staticmethod
    def logical_op(op1: List[int], op2: List[int], mode: int) -> List[int]:
        """
        Performs bitwise logical operations on two byte arrays.

        Args:
            op1 (List[int]): First operand
            op2 (List[int]): Second operand
            mode (int): 0=AND, 1=OR, 2=XOR

        Returns:
            List[int]: Result of logical operation

        Raises:
            ValueError: If mode is not 0, 1, or 2
        """
        l1, l2 = len(op1), len(op2)
        if l1 > l2:
            longer = list(op1)
            shorter = list(op2) + [0x00] * (l1 - l2)
        else:
            longer = list(op2)
            shorter = list(op1) + [0x00] * (l2 - l1)
        res = []
        for i in range(len(longer)):
            if mode == 0:
                res.append(longer[i] & shorter[i])
            elif mode == 1:
                res.append(longer[i] | shorter[i])
            else:
                res.append(longer[i] ^ shorter[i])
        return res

    @staticmethod
    def xor(op1: List[int], op2: List[int]) -> List[int]:
        """
        Bitwise XOR of two byte arrays.

        Args:
            op1 (List[int]): First operand
            op2 (List[int]): Second operand

        Returns:
            List[int]: XOR result
        """
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1: List[int], op2: List[int]) -> List[int]:
        """
        Bitwise AND of two byte arrays.

        Args:
            op1 (List[int]): First operand
            op2 (List[int]): Second operand

        Returns:
            List[int]: AND result
        """
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1: List[int], op2: List[int]) -> List[int]:
        """
        Bitwise OR of two byte arrays.

        Args:
            op1 (List[int]): First operand
            op2 (List[int]): Second operand

        Returns:
            List[int]: OR result
        """
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache: List[List[int]], steps: int = 1) -> List[List[int]]:
        """
        Updates the procedural generation seed using game's algorithm.

        This is the core seed mixing algorithm used by No Man's Sky for
        procedural generation. It evolves the seed state deterministically.

        Args:
            cache (List[List[int]]): Seed cache with two byte arrays
            steps (int): Number of times to update the seed

        Returns:
            List[List[int]]: Updated seed cache
        """
        for _ in range(steps):
            step1 = ByteUtils.multiply(cache[0], ByteUtils.SEED_MULTIPLIER)
            result = ByteUtils.add(step1, cache[1])
            cache[0] = ByteUtils.shl(result, 4)
            cache[1] = ByteUtils.shr(result, 4)
        return cache

    @staticmethod
    def _unpack(arr: List[int], offset: int, size: int, fmt: str) -> int:
        """
        Helper to unpack bytes into a numeric value.

        Args:
            arr (List[int]): Source byte array
            offset (int): Starting position in array
            size (int): Number of bytes to read
            fmt (str): struct format string (like '<I' for unsigned int)

        Returns:
            int: Unpacked numeric value
        """
        chunk = arr[offset:offset + size]
        while len(chunk) < size:
            chunk.append(0)
        return struct.unpack(fmt, bytes(chunk))[0]

    @staticmethod
    def to_uint32(arr: List[int], offset: int = 0) -> int:
        """
        Converts 4 bytes to unsigned 32-bit integer (little-endian).

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position

        Returns:
            int: Unsigned 32-bit integer
        """
        return ByteUtils._unpack(arr, offset, 4, '<I')

    @staticmethod
    def to_int32(arr: List[int], offset: int = 0) -> int:
        """
        Converts 4 bytes to signed 32-bit integer (little-endian).

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position

        Returns:
            int: Signed 32-bit integer
        """
        return ByteUtils._unpack(arr, offset, 4, '<i')

    @staticmethod
    def to_int16(arr: List[int], offset: int = 0) -> int:
        """
        Converts 2 bytes to signed 16-bit integer (little-endian).

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position

        Returns:
            int: Signed 16-bit integer
        """
        return ByteUtils._unpack(arr, offset, 2, '<h')

    @staticmethod
    def to_double(arr: List[int], offset: int = 0) -> float:
        """
        Converts 8 bytes to double-precision float (little-endian).

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position

        Returns:
            float: Double-precision floating point number
        """
        return ByteUtils._unpack(arr, offset, 8, '<d')

    @staticmethod
    def to_single(arr: List[int], offset: int = 0) -> float:
        """
        Converts 4 bytes to single-precision float (little-endian).

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position

        Returns:
            float: Single-precision floating point number
        """
        return ByteUtils._unpack(arr, offset, 4, '<f')

    @staticmethod
    def get_bytes_uint32(val: int) -> List[int]:
        """
        Converts unsigned 32-bit integer to 4-byte list (little-endian).

        Args:
            val (int): Unsigned 32-bit integer

        Returns:
            List[int]: 4-byte representation
        """
        return list(struct.pack('<I', val))


class StringExtensions:
    """
    String manipulation utilities for the procedural generator.
    """

    @staticmethod
    def short_to_formatted_hex(val: int, trunc: int) -> str:
        """
        Converts a 16-bit value to hexadecimal string with truncation.

        Args:
            val (int): 16-bit value (0-65535)
            trunc (int): Number of hex digits to keep (2-4)

        Returns:
            str: Hexadecimal string, truncated from the right

        Example:
            >>> StringExtensions.short_to_formatted_hex(0x1234, 2)
            "34"
        """
        val = val & 0xFFFF
        hex_str = f"{val:04X}"
        return hex_str[-trunc:]


class Generator:
    """
    Generates procedural names using No Man's Sky algorithms.

    This class implements the exact name generation logic used by the game
    for creating region names, system names, and other procedural text.
    """

    TINY_DOUBLE = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]  # Small constant for scaling

    @staticmethod
    def generate_name(cache0: List[List[int]], cache1: List[List[int]]) -> str:
        """
        Generates a procedural name from seed caches.

        This is the main name generation algorithm that creates pronounceable
        names using weighted character tables and linguistic rules.

        Args:
            cache0 (List[List[int]]): Primary seed cache
            cache1 (List[List[int]]): Secondary seed cache with generation parameters

        Returns:
            str: Generated name, or empty string if generation fails

        Raises:
            RuntimeError: If safety limit exceeded (infinite loop detection)
        """
        # Step 1: Get initial characters from the alphabet set
        name = Generator.get_characters_from_alphaset(cache0, cache1)
        if name == "__EMPTY__":
            return ""

        # Step 2: Update seed and determine character selection method
        ByteUtils.update_seed(cache0)
        check_op = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        alternate_char_getter = (ByteUtils.to_int16(check_op) != 0)
        ByteUtils.update_seed(cache0)

        # Step 3: Calculate how many additional characters to generate
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])
        limit = ByteUtils.to_int16(ByteUtils.sxd(register0, 2))

        # Step 4: Generate additional characters using weighted probabilities
        if 0 < limit:
            i = 0
            safety = 0
            while i < limit:
                ByteUtils.update_seed(cache0)
                sub_str = name[i: i + 3]
                alphaset_idx = cache1[0][0] if cache1[0] else 0
                char_weights = Generator.get_string_weights(sub_str, alphaset_idx)
                val_u32 = ByteUtils.to_uint32(cache0[0])
                tiny_dbl = ByteUtils.to_double(Generator.TINY_DOUBLE)
                target = float(val_u32 * tiny_dbl)

                if char_weights is None:
                    i -= 1
                    safety += 1
                    if safety > SAFETY_LIMIT:
                        break
                else:
                    safety = 0
                    index = 0
                    if alternate_char_getter:
                        target *= (len(char_weights) - 1)
                        b_tgt = list(struct.pack('<f', target))
                        op_and = ByteUtils.and_op(b_tgt, [0x00, 0x00, 0x00, 0x80])
                        op = ByteUtils.or_op(op_and, [0x00, 0x00, 0x00, 0x3F])
                        index = int(ByteUtils.to_single(op) + target)
                    else:
                        weight = 0.0
                        j = 0
                        for cw in char_weights:
                            weight += cw[1]
                            if weight >= target:
                                break
                            j += 1
                        index = j
                    if index < len(char_weights):
                        name += char_weights[index][0]
                if len(name) > MAX_NAME_LENGTH - 1:
                    name = name[:MAX_NAME_LENGTH]
                i += 1

        if not name or len(name) < 2:
            return name

        # Step 5: Apply linguistic rules for valid consonant clusters
        first = name[0]
        second = name[1]
        is_valid_cluster = False

        # Check if first two characters form a valid consonant cluster
        if (first not in VOWELS) and (second not in VOWELS):
            cond1 = first != 's' or second not in "hklmnprtwy"
            cond2 = (second == 'h' and first in "ctw")
            cond3 = (second == 'l' and first in "bcfgps")
            cond4 = (second == 'r' and first in "bcdfgkpt")
            cond5 = (second == 'w' and first in "dgt")
            cond6 = (second == 'y' and first in "hmr")
            if cond1:
                if cond2 or cond3 or cond4 or cond5 or cond6:
                    is_valid_cluster = True
                if not is_valid_cluster:
                    name = Generator.insert_vowel(name, cache0, 1)

        # Step 6: Check and fix invalid ending clusters
        ult = name[-1]
        penult = name[-2]

        if len(name) > 1 and (penult != 'g' or ult in VOWELS):
            idx = len(name) - 1
            c1 = (ult == 'b' and penult in "gn")
            c2 = (ult == 'd' and penult in "bdfghkmpst")
            c3 = (ult == 'g' and penult == 'l')
            c4 = (ult == 'p' and penult in "bdhkt")
            c5 = (ult == 'r' and penult in "bfg")
            c6 = (ult == 't' and penult == 'g')
            c7 = (ult == 'w' and penult not in VOWELS)
            if c1 or c2 or c3 or c4 or c5 or c6 or c7:
                name = Generator.insert_vowel(name, cache0, idx)

        # Step 7: Check for too many consecutive consonants
        consecutive_count = Generator.get_consecutive_consonants(name)
        if consecutive_count != -1:
            ByteUtils.update_seed(cache0)
            mult = ByteUtils.multiply(cache0[0], [0x03])
            shr = ByteUtils.shr(mult, 4)
            add = ByteUtils.add(shr, [0x01])
            offset = ByteUtils.to_int32(ByteUtils.zxd(add, 4))
            name = Generator.insert_vowel(name, cache0, consecutive_count + offset)
        return name

    @staticmethod
    def get_characters_from_alphaset(cache0: List[List[int]], cache1: List[List[int]]) -> str:
        """
        Gets initial characters from the alphabet set based on seed.

        Args:
            cache0 (List[List[int]]): Seed cache for randomness
            cache1 (List[List[int]]): Cache containing alphabet set index

        Returns:
            str: 3-character starting string, or "__EMPTY__" if no alphabet set
        """
        ByteUtils.update_seed(cache0)
        idx = cache1[0][0] if cache1[0] else 0
        if idx >= len(NMSData.ALPHASETS):
            idx = 0
        alphaset_str = NMSData.ALPHASETS[idx]
        if not alphaset_str:
            return "__EMPTY__"
        length_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        register0 = ByteUtils.multiply(cache0[0], length_bytes)
        shr_reg = ByteUtils.shr(register0, 4)
        register1 = ByteUtils.format_short(ByteUtils.multiply(shr_reg, [0x03]))
        start = ByteUtils.to_int16(register1)
        end = ByteUtils.to_int16(ByteUtils.add(register1, [0x03]))
        return alphaset_str[start:end]

    @staticmethod
    def get_string_weights(s: str, alphaset: int) -> Optional[List[tuple]]:
        """
        Gets weighted character choices for the next character in a string.

        Args:
            s (str): Current string context (last few characters)
            alphaset (int): Index of alphabet set to use

        Returns:
            Optional[List[tuple]]: List of (character, weight) tuples, or None if not found
        """
        if not NMSData.LETTER_MAP or alphaset not in NMSData.LETTER_MAP:
            return None
        subset = NMSData.LETTER_MAP[alphaset]
        if not s or s[0] not in subset:
            return None
        return Generator.recursive_search(subset[s[0]], s)

    @staticmethod
    def recursive_search(arr: List, s: str) -> Optional[List[tuple]]:
        """
        Recursively searches the letter map for character weights.

        Args:
            arr (List): Nested structure from letter map
            s (str): String to search for

        Returns:
            Optional[List[tuple]]: Character weights if found, None otherwise
        """
        for item in arr:
            if len(item) > 2:
                type_code, val = item[2], item[0]
                if type_code == "ja":
                    s_bytes = ByteUtils.zxd(list(s.encode('utf-8')), 4)
                    val_b = ByteUtils.zxd(list(str(val).encode('utf-8')), 4)
                    if ByteUtils.to_int32(s_bytes) > ByteUtils.to_int32(val_b):
                        result = Generator.recursive_search(item[1], s)
                        if result is not None:
                            return result
                elif type_code == "jz" and str(val) == s:
                    weights = [(w.get("Item1"), float(w.get("Item2", 0))) for w in item[1]]
                    return weights
        return None

    @staticmethod
    def insert_vowel(name: str, seed: List[List[int]], index: int) -> str:
        """
        Inserts a vowel into a name at specified position.

        Args:
            name (str): Original name
            seed (List[List[int]]): Seed for random vowel selection
            index (int): Position to insert vowel

        Returns:
            str: Name with vowel inserted
        """
        ByteUtils.update_seed(seed)
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)
        if calc and calc[0] < 5:
            if index <= len(name):
                return name[:index] + VOWELS[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name: str) -> int:
        """
        Finds position where too many consecutive consonants occur.

        Args:
            name (str): Name to check

        Returns:
            int: Position where 4+ consecutive consonants start, or -1 if valid
        """
        consecutive_count = 0
        for i in range(len(name)):
            if consecutive_count < 3:
                if name[i] not in VOWELS:
                    consecutive_count += 1
                else:
                    consecutive_count = 0
            else:
                if name[i] not in VOWELS_WITH_Y:
                    return i - 3
                else:
                    consecutive_count = 0
        return -1


class RegionNameGenerator:
    """
    Generates procedural region names from galactic coordinates.

    This class implements the exact algorithm used by No Man's Sky to create
    region names like "Sea of Pihend" or "Ukkibush Expanse" from coordinates.
    """

    PROC_ADORNMENTS = [
        "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
        "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
        "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
        "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux",
        "%NAME% Instability", "Sea of %NAME%", "The Arm of %NAME%",
        "%NAME% Spur", "%NAME% Shallows"
    ]  # Suffixes/prefixes added to region names

    @staticmethod
    def create_region_seed(x: int, y: int, z: int, galaxy: int) -> List[int]:
        """
        Creates a seed byte array from galactic coordinates.

        Args:
            x (int): X coordinate in galactic voxel space
            y (int): Y coordinate in galactic voxel space
            z (int): Z coordinate in galactic voxel space
            galaxy (int): Galaxy index (0=Euclid, 1=Hilbert, etc.)

        Returns:
            List[int]: Byte array seed for name generation
        """
        s_gal = StringExtensions.short_to_formatted_hex(galaxy, 2)
        s_y = StringExtensions.short_to_formatted_hex(y, 2)
        s_z = StringExtensions.short_to_formatted_hex(z, 3)
        s_x = StringExtensions.short_to_formatted_hex(x, 3)
        hex_str = s_gal + s_y + s_z + s_x
        return ByteUtils.parse(hex_str)

    @staticmethod
    def format_name(seed: List[int]) -> str:
        """
        Generates a region name from a seed byte array.

        Args:
            seed (List[int]): Byte array seed from create_region_seed()

        Returns:
            str: Generated region name

        Example:
            >>> seed = RegionNameGenerator.create_region_seed(100, 200, 300, 0)
            >>> RegionNameGenerator.format_name(seed)
            "Sea of Pihend"
        """
        # Initialize caches for the generator
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # Step 1: Initial seed processing with XOR and multiplication
        register0 = ByteUtils.shr(seed, 4)
        if register0:
            register0[0] //= 2
        xor_res = ByteUtils.xor(register0, seed)
        mult_arr_1 = [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64]
        register0 = ByteUtils.multiply(xor_res, mult_arr_1)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)
        mult_arr_2 = [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3]
        register0 = ByteUtils.multiply(xor2, mult_arr_2)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        register0 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)

        # Step 2: Prepare caches for the name generator
        shl4 = ByteUtils.shl(register0, 4)
        xor_mid = ByteUtils.xor(ByteUtils.rol(shl4, 2), ByteUtils.shr(register0, 4))
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])

        ByteUtils.update_seed(cache0)
        calc_len = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4)
        cache1[2] = ByteUtils.add(calc_len, [0x06])

        # Step 3: Generate the base name
        name = Generator.generate_name(cache0, cache1)
        if not name:
            return "Unknown Region"
        name = name[0].upper() + name[1:]  # Capitalize first letter

        # Step 4: Decide whether to add an adornment (prefix/suffix)
        ByteUtils.update_seed(cache0)
        mult_check = ByteUtils.multiply(cache0[0], [0x64])
        should_adorn = ByteUtils.shr(mult_check, 4)[0] < 0x50  # 50% chance

        if should_adorn:
            ByteUtils.update_seed(cache0)
            idx_cal = ByteUtils.multiply(cache0[0], [0x14])
            idx = ByteUtils.shr(idx_cal, 4)[0]
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                adornment = RegionNameGenerator.PROC_ADORNMENTS[idx]
                name = adornment.replace("%NAME%", name)
        return name


class NMSGalaxyMap:
    """
    Converts between portal glyphs and galactic coordinates.

    This class handles the coordinate system transformations used in
    No Man's Sky to convert 12-digit portal glyphs to XYZ coordinates
    in galactic space.
    """

    # Constants for coordinate shifting (game's coordinate system)
    SHIFT_POS_XZ = 2049  # Threshold for positive X/Z shift
    SHIFT_NEG_XZ = 2047  # Amount to shift negative X/Z coordinates
    SHIFT_POS_Y = 129    # Threshold for positive Y shift
    SHIFT_NEG_Y = 127    # Amount to shift negative Y coordinates
    CENTER_X = 2047      # Center point of X axis
    CENTER_Y = 127       # Center point of Y axis
    CENTER_Z = 2047      # Center point of Z axis

    @staticmethod
    def _shift_coordinate(value: int, pos_threshold: int, pos_shift: int, neg_shift: int) -> int:
        """
        Applies coordinate shifting used by No Man's Sky.

        The game uses a centered coordinate system where values above a
        threshold are negative when shifted.

        Args:
            value (int): Raw coordinate value
            pos_threshold (int): Threshold for positive shift
            pos_shift (int): Amount to subtract if above threshold
            neg_shift (int): Amount to add if below threshold

        Returns:
            int: Shifted coordinate value
        """
        return value - pos_shift if value >= pos_threshold else value + neg_shift

    @classmethod
    def glyphs_to_voxels(cls, glyphs: str) -> Optional[Dict[str, int]]:
        """
        Converts 12-digit portal glyphs to galactic voxel coordinates.

        Args:
            glyphs (str): 12-character hexadecimal portal code

        Returns:
            Optional[Dict[str, int]]: Dictionary with x, y, z coordinates,
                                     or None if invalid input

        Example:
            >>> NMSGalaxyMap.glyphs_to_voxels("0123456789AB")
            {'x': 100, 'y': 200, 'z': 300}
        """
        if not glyphs:
            return None
        g = glyphs.strip().upper()
        if len(g) != 12:
            return None
        try:
            # Parse glyphs: positions 4-6 are Y, 6-9 are Z, 9-12 are X
            y_hex = int(g[4:6], 16)
            z_hex = int(g[6:9], 16)
            x_hex = int(g[9:12], 16)
        except ValueError:
            return None

        # Apply coordinate shifting
        cx = cls._shift_coordinate(x_hex, cls.SHIFT_POS_XZ, cls.SHIFT_POS_XZ, cls.SHIFT_NEG_XZ)
        cz = cls._shift_coordinate(z_hex, cls.SHIFT_POS_XZ, cls.SHIFT_POS_XZ, cls.SHIFT_NEG_XZ)
        cy = cls._shift_coordinate(y_hex, cls.SHIFT_POS_Y, cls.SHIFT_POS_Y, cls.SHIFT_NEG_Y)

        return {'x': cx, 'y': cy, 'z': cz}


class NMSData:
    """
    Loads and manages game data for the wiki generator.

    This class fetches data from external sources including galaxy lists,
    Multi-Tool types, colors, and procedural generation data. It provides
    fallback data if network connections fail.

    Attributes:
        LETTER_MAP (Dict): Character weights for procedural name generation
        ALPHASETS (List[str]): Alphabet sets for different language styles
        URL_BASE (str): Base URL for data files
        GALAXIES (List[str]): List of galaxy names
        GALAXY_MAP (Dict[str, int]): Mapping from galaxy name to index
        MT_TYPES (List[str]): Available Multi-Tool archetypes
        MT_COLORS (List[str]): Available Multi-Tool colors
        LOCATION_TYPES (List[str]): Where Multi-Tools can be found
        PLATFORMS (List[str]): Supported game platforms
        GAME_MODES (List[str]): Available game modes
        CLASSES (List[str]): Multi-Tool classes (S, A, B, C)
    """

    # Class variables for procedural generation data
    LETTER_MAP = {}
    ALPHASETS = []

    # URLs for external data files
    URL_BASE = "https://raw.githubusercontent.com/2A03-Jikuu/nms-wiki-tool-py/refs/heads/main/datalist"
    URL_GALAXIES = f"{URL_BASE}/galaxies.json"
    URL_CONFIG = f"{URL_BASE}/multitool_data.json"
    URL_LETTER_MAP = f"{URL_BASE}/letter_map.json"
    URL_ALPHASETS = f"{URL_BASE}/alphasets.json"
    ALPHASET_FALLBACK_SIZE = 8  # Default size if alphasets fail to load

    def __init__(self):
        """Initializes the data manager and loads all required data."""
        self.GALAXIES: List[str] = []
        self.GALAXY_MAP: Dict[str, int] = {}
        self.MT_TYPES: List[str] = []
        self.MT_COLORS: List[str] = []
        self.LOCATION_TYPES: List[str] = []
        self.PLATFORMS: List[str] = []
        self.GAME_MODES: List[str] = []
        self.CLASSES: List[str] = []
        self._load_data()

    def _load_data(self):
        """
        Loads all required data from external sources or fallbacks.

        This method attempts to load data from GitHub repositories first,
        then falls back to hardcoded data if network requests fail.
        """
        print("Loading NMS Data...")

        # Load galaxy data
        try:
            resp = requests.get(self.URL_GALAXIES, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            sorted_galaxies = sorted(data, key=lambda x: x.get('index', 0))
            self.GALAXIES = [g['name'] for g in sorted_galaxies]
            self.GALAXY_MAP = {g['name']: g['index'] for g in sorted_galaxies}
            print(f" - Loaded {len(self.GALAXIES)} Galaxies.")
        except Exception as e:
            print(f" ! Error loading Galaxies: {e}. Using fallback.")
            self.GALAXIES = ['Euclid', 'Hilbert Dimension', 'Calypso']
            self.GALAXY_MAP = {'Euclid': 0, 'Hilbert Dimension': 1, 'Calypso': 2}

        # Load Multi-Tool configuration
        try:
            resp = requests.get(self.URL_CONFIG, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self.MT_TYPES = sorted(data.get('mt_types', []))
            self.MT_COLORS = sorted(data.get('mt_colors', []))
            self.LOCATION_TYPES = sorted(data.get('location_types', []))
            self.PLATFORMS = sorted(data.get('platforms', []))
            self.GAME_MODES = sorted(data.get('game_modes', []))
            self.CLASSES = data.get('classes', ['S', 'A', 'B', 'C'])
            print(" - Loaded Multi-Tool Configuration.")
        except Exception:
            self._set_fallbacks()

        # Load alphasets for procedural generation
        try:
            resp = requests.get(self.URL_ALPHASETS, timeout=15)
            resp.raise_for_status()
            NMSData.ALPHASETS = resp.json()
        except Exception as e:
            print(f" ! Error loading Alphasets: {e}")
            NMSData.ALPHASETS = [""] * self.ALPHASET_FALLBACK_SIZE

        # Load letter map for procedural name generation
        try:
            resp = requests.get(self.URL_LETTER_MAP, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            NMSData.LETTER_MAP = {int(k): v for k, v in raw.items()}
            print(" - Loaded Region Name Logic.")
        except Exception as e:
            print(f" ! Error loading Letter Map: {e}")
            NMSData.LETTER_MAP = {}

    def _set_fallbacks(self):
        """
        Sets fallback data when network loading fails.

        Provides minimal data to keep the application functional even
        without internet connection.
        """
        print(" ! Using fallback Multi-Tool configuration.")
        self.MT_TYPES = ['Alien', 'Experimental', 'Pistol', 'Rifle']
        self.MT_COLORS = ['Blue', 'Red', 'Yellow']
        self.LOCATION_TYPES = ['Space Station', 'Minor Settlement']
        self.PLATFORMS = ['PC', 'PS', 'Xbox']
        self.GAME_MODES = ['Normal']
        self.CLASSES = ['S', 'A', 'B', 'C']


@dataclass
class AppWidgets:
    """
    Container for all UI widgets used in the application.

    This dataclass organizes all the form widgets into a single object
    for easy access and management. Each field corresponds to a widget
    in the user interface.
    """
    name: Text
    type: Dropdown
    item_class: Dropdown
    slots: IntText
    cost: Text
    crystals: Checkbox
    horns: Checkbox
    glowtubes: Checkbox
    galaxy: Combobox
    region: Text
    system: Text
    planet: Text
    location: Dropdown
    portalglyphs: Text
    axes: Text
    coordinates: Text
    damage: FloatText
    mining: FloatText
    scanner: FloatText
    primarycolor: Dropdown
    secondarycolor: Dropdown
    accent: Dropdown
    discovered: Text
    discoveredlink: Text
    discoverydate: DatePicker
    agt_stardate: Text
    release: Text
    platform: Dropdown
    mode: Dropdown
    civilized: Text
    researchteam: Text
    civstub: Text
    civimage: Text
    civimagelabel: Text
    image: Text
    gallery: Textarea
    summarynote: Textarea
    collect: Textarea
    locationnote: Textarea
    description: Textarea
    addnote: Textarea
    footertitle: Text
    footer: Textarea
    btn_preview: Button
    btn_gen: Button
    btn_copy: Button
    btn_download: Button
    btn_clear: Button
    btn_example: Button
    status_bar: HTML
    output_area: Output

    def __iter__(self):
        """
        Allows iteration over all widget fields.

        Yields:
            Widget: Each widget in the dataclass
        """
        for f in fields(self):
            yield getattr(self, f.name)


class WikiDataModel(BaseModel):
    """
    Data model for validating and storing Multi-Tool information.

    This Pydantic model validates all user inputs and ensures they
    meet the requirements for wiki template generation.

    Attributes:
        model_config: Pydantic configuration for field name aliases
        All other attributes: Correspond to form fields with default values
    """
    model_config = ConfigDict(populate_by_name=True)
    name: str = ""
    image: str = Field(default=DEFAULT_IMAGE)
    galaxy: str = ""
    region: str = ""
    system: str = ""
    planet: str = ""
    location: str = ""
    axes: str = ""
    coordinates: str = ""
    portalglyphs: str = ""
    type: str = ""
    crystals: str = ""
    horns: str = ""
    glowtubes: str = ""
    item_class: str = Field(default="", alias="class")
    slots: str = ""
    cost: str = ""
    civstub: str = ""
    civilized: str = ""
    discovered: str = ""
    discoveredlink: str = ""
    mode: str = ""
    platform: str = ""
    release: str = ""
    damage: str = ""
    mining: str = ""
    scanner: str = ""
    researchteam: str = ""
    civimage: str = ""
    civimagelabel: str = ""
    summarynote: str = ""
    locationnote: str = ""
    description: str = ""
    discoverydate: str = ""
    agt_stardate: str = ""
    addnote: str = ""
    primarycolor: str = ""
    secondarycolor: str = ""
    accent: str = ""
    collect: str = ""
    footertitle: str = ""
    footer: str = ""
    gallery: str = ""


class NMSWikiMultiToolGenerator:
    """
    Main application class for the Multi-Tool wiki generator.

    This class creates the complete user interface with tabbed forms,
    handles user interactions, validates data, and generates wiki code.

    Attributes:
        WIKI_TEMPLATE (str): Jinja2 template for wiki output
        GLYPH_PATTERN (re.Pattern): Regex for validating portal glyphs
        FILENAME_SANITIZE_PATTERN (re.Pattern): Regex for safe filenames
        data (NMSData): Game data manager
        widgets (AppWidgets): All UI widgets
        jinja_env (Environment): Template rendering environment
        wiki_template (Template): Compiled wiki template
        _generated_wikitext (str): Most recently generated wiki code
        galaxy_map (NMSGalaxyMap): Coordinate converter
        _dropdown_placeholders (Dict): Track dropdown placeholder values
    """

    # Wiki template with all possible fields
    WIKI_TEMPLATE: str = """
{{ '{{' }}PAGEMultitool
| name = {{ name | default('', true) }}
| image = {{ image | default('NmsMisc NotAvailable.png', true) }}
| galaxy = {{ galaxy | default('', true) }}
| region = {{ region | default('', true) }}
| system = {{ system | default('', true) }}
| planet = {{ planet | default('', true) }}
| location = {{ location | default('', true) }}
| axes = {{ axes | default('', true) }}
| coordinates = {{ coordinates | default('', true) }}
| portalglyphs = {{ portalglyphs | default('', true) }}
| type = {{ type | default('', true) }}
| crystals = {{ crystals | default('', true) }}
| horns = {{ horns | default('', true) }}
| glowtubes = {{ glowtubes | default('', true) }}
| class = {{ item_class | default('', true) }}
| slots = {{ slots | default('', true) }}
| cost = {{ cost | default('', true) }}
| civstub = {{ civstub | default('', true) }}
| civilized = {{ civilized | default('', true) }}
| discovered = {{ discovered | default('', true) }}
| discoveredlink = {{ discoveredlink | default('', true) }}
| mode = {{ mode | default('', true) }}
| platform = {{ platform | default('', true) }}
| release = {{ release | default('', true) }}
| damage = {{ damage | default('', true) }}
| mining = {{ mining | default('', true) }}
| scanner = {{ scanner | default('', true) }}
| researchteam = {{ researchteam | default('', true) }}
| civimage = {{ civimage | default('', true) }}
| civimagelabel = {{ civimagelabel | default('', true) }}
| summarynote = {{ summarynote | default('', true) }}
| locationnote = {{ locationnote | default('', true) }}
| description = {{ description | default('', true) }}
| discoverydate = {{ discoverydate | default('', true) }}
| AGTstardate = {{ agt_stardate | default('', true) }}
| addnote = {{ addnote | default('', true) }}
| primarycolor = {{ primarycolor | default('', true) }}
| secondarycolor = {{ secondarycolor | default('', true) }}
| accent = {{ accent | default('', true) }}
| collect = {{ collect | default('', true) }}
| footertitle = {{ footertitle | default('', true) }}
| footer = {{ footer | default('', true) }}
{{ '}}' }}

==Gallery==
<gallery>
{{ gallery | default('', true) }}
</gallery>

==AGT Galactic Archives==
{{ '{{' }}AGT Galactic Archive Sync{{ '}}' }}
"""

    # Regular expressions for validation and sanitization
    GLYPH_PATTERN = re.compile(r"^[0-9A-F]{12}$")  # Exactly 12 hex digits
    FILENAME_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9_\-]")  # Remove unsafe chars

    def __init__(self):
        """
        Initializes the application, loads data, and creates UI.

        Sets up the data manager, creates all UI components, connects
        event handlers, and displays the interface.
        """
        self.data = NMSData()  # Load game data
        self.widgets: AppWidgets = None  # Will hold all widgets after creation
        self.jinja_env = Environment()  # Template rendering engine
        self.wiki_template = self.jinja_env.from_string(self.WIKI_TEMPLATE)
        self._generated_wikitext: str = ""  # Stores last generated wiki code
        self.galaxy_map = NMSGalaxyMap()  # Coordinate conversion utility
        self._dropdown_placeholders = {}  # Track placeholder values in dropdowns

        # Set up UI styling and create interface
        self._define_styles_and_layouts()
        self._setup_ui()
        self._connect_events()

        # Initialize calculated fields
        self._update_stardate(None)
        self._update_coordinates(None)

    def _define_styles_and_layouts(self):
        """
        Defines CSS styles and layout configurations for UI widgets.

        These styles create a consistent, visually appealing interface
        with clear section headers and organized form fields.
        """
        # CSS styles for headers and descriptions
        self.HEADER_STYLE = (
            "font-weight:bold; font-size:16px; margin-top:20px; "
            "border-bottom:2px solid #00ACC1; padding-bottom:5px; "
            "color:#006064;"
        )
        self.DESC_STYLE = (
            "font-style:italic; font-size:12px; color:#555; "
            "margin-bottom:12px; line-height:1.4em; background-color:#E0F7FA; "
            "padding:8px; border-left:4px solid #00BCD4; border-radius:4px;"
        )

        # Widget layout configurations
        self.LABEL_STYLE = {'description_width': '140px'}  # Width for field labels
        self.WIDGET_LAYOUT = Layout(width='98%')  # Standard widget width
        self.TEXT_AREA_LAYOUT = Layout(width='98%', height='80px')  # Textarea size
        self.GALLERY_LAYOUT = Layout(width='98%', height='120px')  # Gallery textarea
        self.COL_LAYOUT = Layout(width='50%')  # Two-column layout width
        self.FULL_ROW = Layout(width='100%', margin='5px 0')  # Full-width rows

    def _setup_ui(self):
        """
        Creates the complete user interface with all tabs and widgets.

        Builds the tabbed interface by creating each tab's content,
        then displays the entire interface in the notebook.
        """
        self._temp_widgets = {}  # Temporary storage during widget creation

        # Create all tab content
        tabs_content = [
            self._create_tab_basic_info(),
            self._create_tab_location(),
            self._create_tab_visuals_stats(),
            self._create_tab_discovery(),
            self._create_tab_media(),
            self._create_tab_generate()
        ]

        # Create AppWidgets instance from temporary storage
        self.widgets = AppWidgets(**self._temp_widgets)
        del self._temp_widgets  # Clean up temporary storage

        # Create tab container and set tab titles
        self.tabs = Tab(children=tabs_content)
        headers = ['Basic Info', 'Location', 'Stats & Colors', 'Discovery', 'Media & Notes', 'Generate']
        for i, h in enumerate(headers):
            self.tabs.set_title(i, h)

        # Display the interface in the notebook
        display(self.tabs)

    def _create_tab_basic_info(self) -> VBox:
        """
        Creates the 'Basic Info' tab with Multi-Tool identity fields.

        Returns:
            VBox: Vertical box containing all widgets for this tab
        """
        return VBox([
            self._header('Identity & Class'),
            self._desc("Define the Multi-Tool's name, its primary archetype, and its class."),
            self._two_col_row(
                self._create_widget(Text, 'name', description='Multi-Tool Name', placeholder='e.g., The Vesper Shard'),
                self._create_widget(Dropdown, 'type', description='Archetype', options=self.data.MT_TYPES, placeholder='Select Type...')
            ),
            self._two_col_row(
                self._create_widget(Dropdown, 'item_class', description='Class', options=self.data.CLASSES, placeholder='Select Class...')
            ),
            self._header('Configuration'),
            self._desc("Specify the number of open slots and the cost in units. Leave as 0 or empty if not applicable."),
            self._two_col_row(
                self._create_widget(IntText, 'slots', description='Slot Count', value=0),
                self._create_widget(Text, 'cost', description='Cost (Units)', placeholder='e.g. 3,250,000')
            ),
            self._header('Special Features (Alien/Experimental Only)'),
            self._desc("Check any distinct visual features. These are typically found only on Alien or Experimental models."),
            HBox([
                self._create_widget(Checkbox, 'crystals', description='Crystals', layout=Layout(width='auto', margin='0 20px 0 160px'), style={'description_width': 'initial'}),
                self._create_widget(Checkbox, 'horns', description='Horns', layout=Layout(width='auto', margin='0 20px 0 0'), style={'description_width': 'initial'}),
                self._create_widget(Checkbox, 'glowtubes', description='Glow Tubes', layout=Layout(width='auto'), style={'description_width': 'initial'})
            ])
        ], layout=Layout(padding='20px'))

    def _create_tab_location(self) -> VBox:
        """
        Creates the 'Location' tab with galactic coordinate fields.

        Returns:
            VBox: Vertical box containing all widgets for this tab
        """
        return VBox([
            self._header('Galactic Coordinates'),
            self._desc("Enter the Portal Glyphs to automatically generate the Region Name and Galactic Coordinates."),
            self._two_col_row(
                self._create_widget(Combobox, 'galaxy', description='Galaxy', options=self.data.GALAXIES, placeholder='Start typing or select...'),
                self._create_widget(Text, 'portalglyphs', description='Portal Glyphs (Hex)', placeholder='e.g., 1A2B3C4D5E6F')
            ),
            self._two_col_row(
                self._create_widget(Text, 'region', description='Region Name', placeholder='Auto-calculated...', disabled=True),
                self._create_widget(Text, 'system', description='Star System Name', placeholder='e.g., Okinami')
            ),
            self._header('Navigation & Acquisition'),
            self._desc("Specify where the tool is housed. The long-form coordinates are generated automatically from the glyphs above."),
            self._two_col_row(
                self._create_widget(Text, 'planet', description='Planet / Moon', placeholder='Required unless found on station'),
                self._create_widget(Dropdown, 'location', description='Structure Type', options=self.data.LOCATION_TYPES, placeholder='Select Location...')
            ),
            self._two_col_row(
                self._create_widget(Text, 'coordinates', description='Galactic Coords', disabled=True, placeholder='Auto-calculated (XXXX:YYYY:ZZZZ:SSSS)'),
                self._create_widget(Text, 'axes', description='Lat/Long Coords', placeholder='+12.34, -56.78')
            )
        ], layout=Layout(padding='20px'))

    def _create_tab_visuals_stats(self) -> VBox:
        """
        Creates the 'Stats & Colors' tab with statistics and color selection.

        Returns:
            VBox: Vertical box containing all widgets for this tab
        """
        return VBox([
            self._header('Core Statistics'),
            self._desc("Enter the tool's base statistics as shown in the Analysis Visor, before any upgrades are installed."),
            self._two_col_row(
                self._create_widget(FloatText, 'damage', description='Damage Potential', value=0.0),
                self._create_widget(FloatText, 'mining', description='Mining Bonus', value=0.0)
            ),
            self._two_col_row(
                self._create_widget(FloatText, 'scanner', description='Scanner Range', value=0.0)
            ),
            self._header('Color Palette'),
            self._desc("Select the primary (main body), secondary (trim/details), and accent (decals/lights) colors."),
            self._two_col_row(
                self._create_widget(Dropdown, 'primarycolor', description='Primary Color', options=self.data.MT_COLORS, placeholder='Select Main Color...'),
                self._create_widget(Dropdown, 'secondarycolor', description='Secondary Color', options=self.data.MT_COLORS, placeholder='Select Trim Color...')
            ),
            self._two_col_row(
                self._create_widget(Dropdown, 'accent', description='Accent Color', options=self.data.MT_COLORS, placeholder='Select Accent...')
            )
        ], layout=Layout(padding='20px'))

    def _create_tab_discovery(self) -> VBox:
        """
        Creates the 'Discovery' tab with discoverer and civilization info.

        Returns:
            VBox: Vertical box containing all widgets for this tab
        """
        return VBox([
            self._header('Credit'),
            self._desc("Enter your in-game username. If you have a wiki account, add your username to create a profile link."),
            self._two_col_row(
                self._create_widget(Text, 'discovered', description='Discoverer Alias', placeholder='Your In-Game Name'),
                self._create_widget(Text, 'discoveredlink', description='Wiki Username', placeholder='Your Wiki Username (Optional)')
            ),
            self._header('Time & Platform'),
            self._desc("Set the date of discovery and the game version. The AGT Stardate is calculated automatically."),
            self._two_col_row(
                self._create_widget(DatePicker, 'discoverydate', description='Discovery Date', value=arrow.now().date()),
                self._create_widget(Text, 'agt_stardate', description='AGT Stardate', disabled=True)
            ),
            self._two_col_row(
                self._create_widget(Text, 'release', description='Game Version', value=DEFAULT_RELEASE, placeholder='e.g., Orbital, Omega'),
                self._create_widget(Dropdown, 'platform', description='Platform', options=self.data.PLATFORMS, placeholder='Select Platform...')
            ),
            self._two_col_row(
                self._create_widget(Dropdown, 'mode', description='Game Mode', options=self.data.GAME_MODES, value=DEFAULT_MODE)
            ),
            self._header('Civilization'),
            self._desc("If found in civilized space, provide details here. Defaults are set for the Alliance of Galactic Travellers (AGT)."),
            self._two_col_row(
                self._create_widget(Text, 'civilized', description='Civilization Name', value=DEFAULT_CIVILIZATION),
                self._create_widget(Text, 'researchteam', description='Research Team', value=DEFAULT_RESEARCH_TEAM)
            ),
            self._two_col_row(
                self._create_widget(Text, 'civstub', description='Notice Template', value=DEFAULT_CIV_STUB, placeholder='{{Notice Template}}'),
                self._create_widget(Text, 'civimage', description='Civ Badge Image', value=DEFAULT_CIV_IMAGE)
            ),
            self._two_col_row(
                self._create_widget(Text, 'civimagelabel', description='Civ Badge Label', value=DEFAULT_RESEARCH_TEAM)
            )
        ], layout=Layout(padding='20px'))

    def _create_tab_media(self) -> VBox:
        """
        Creates the 'Media & Notes' tab for images and wiki content.

        Returns:
            VBox: Vertical box containing all widgets for this tab
        """
        return VBox([
            self._header('Images'),
            self._desc("Provide the filename for the main infobox image and any gallery images (one per line)."),
            self._create_widget(Text, 'image', description='Main Infobox Image', placeholder='ExampleTool.png'),
            self._create_widget(Textarea, 'gallery', description='Gallery Images', layout=self.GALLERY_LAYOUT, placeholder='File:ToolSide.png|Side\nFile:ToolBack.png|Rear'),
            self._header('Wiki Page Content'),
            self._desc("Write the main sections of the wiki page here. Be as detailed as possible."),
            self._create_widget(Textarea, 'summarynote', description='Summary Paragraph', placeholder='e.g., The Soul of the Ancients is an Alien...'),
            self._create_widget(Textarea, 'collect', description='Acquisition Guide', placeholder='e.g., Fly to the Station. Cabinet on right.'),
            self._create_widget(Textarea, 'locationnote', description='Location Note', placeholder='Optional extra details about the location.'),
            self._create_widget(Textarea, 'description', description='Description Note', placeholder='Optional descriptive text.'),
            self._create_widget(Textarea, 'addnote', description='Additional Note', placeholder='Optional final note.'),
            self._create_widget(Text, 'footertitle', description='Footer Title', placeholder='Optional: Title for a new final section.'),
            self._create_widget(Textarea, 'footer', description='Footer Content', placeholder='Optional: Content for the new final section.')
        ], layout=Layout(padding='20px'))

    def _create_tab_generate(self) -> VBox:
        """
        Creates the 'Generate' tab with action buttons and output display.

        Returns:
            VBox: Vertical box containing all widgets for this tab
        """
        return VBox([
            self._header('Finalization'),
            self._desc("Use the buttons below to preview, generate, copy, or download the wiki code."),
            HBox([
                self._create_widget(Button, 'btn_preview', description='Preview Code', button_style='info', icon='eye'),
                self._create_widget(Button, 'btn_gen', description='Generate & Save', button_style='success', icon='code'),
                self._create_widget(Button, 'btn_copy', description='Copy Code', button_style='primary', icon='copy', disabled=True),
                self._create_widget(Button, 'btn_download', description='Download File', button_style='primary', icon='download', disabled=True),
                self._create_widget(Button, 'btn_example', description='Load Example', button_style='warning', icon='flask'),
                self._create_widget(Button, 'btn_clear', description='Reset Form', button_style='danger', icon='trash')
            ], layout=Layout(justify_content='center', margin='15px 0')),
            self._create_widget(HTML, 'status_bar', value="<div style='text-align:center; margin:10px; color:#555;'><i>Ready...</i></div>"),
            self._header('Code Output'),
            self._create_widget(Output, 'output_area', layout={'border': '1px solid #ccc', 'height': '400px', 'overflow_y': 'scroll', 'padding': '10px'})
        ], layout=Layout(padding='20px'))

    def _create_widget(self, widget_class, key: str, **kwargs):
        """
        Creates a widget with standardized styling and stores it.

        Args:
            widget_class: Class of widget to create (Text, Dropdown, etc.)
            key (str): Name for storing in widgets dictionary
            **kwargs: Additional parameters for widget constructor

        Returns:
            Widget: Created widget instance

        Note:
            Widgets are stored temporarily in self._temp_widgets during UI creation
        """
        params = kwargs

        # Apply standard label style if widget has description
        if 'description' in params:
            params.setdefault('style', self.LABEL_STYLE)

        ph = params.pop('placeholder', None)  # Extract placeholder if provided
        opts = params.pop('options', [])      # Extract options list

        # Handle different widget types specially
        if widget_class == Dropdown:
            # Add placeholder as first option if provided
            params['options'] = [ph] + opts if ph else opts
            if ph and params['options']:
                params['value'] = ph
                self._dropdown_placeholders[key] = ph  # Track placeholder
        elif widget_class == Combobox:
            params['options'] = opts
            params['ensure_option'] = False
            if ph:
                params['placeholder'] = ph
        elif ph:
            params['placeholder'] = ph

        # Apply appropriate layouts based on widget type
        if widget_class is Textarea:
            if key == 'gallery':
                params.setdefault('layout', self.GALLERY_LAYOUT)
            else:
                params.setdefault('layout', self.TEXT_AREA_LAYOUT)
        else:
            params.setdefault('layout', self.WIDGET_LAYOUT)

        # Create widget and store in temporary dictionary
        widget_instance = widget_class(**params)
        self._temp_widgets[key] = widget_instance
        return widget_instance

    def _header(self, text: str) -> HTML:
        """
        Creates a styled header HTML widget.

        Args:
            text (str): Header text

        Returns:
            HTML: Widget displaying the header
        """
        return HTML(f"<div style='{self.HEADER_STYLE}'>{text}</div>")

    def _desc(self, text: str) -> HTML:
        """
        Creates a styled description HTML widget.

        Args:
            text (str): Description text

        Returns:
            HTML: Widget displaying the description
        """
        return HTML(f"<div style='{self.DESC_STYLE}'>{text}</div>")

    def _two_col_row(self, w1: Widget, w2: Optional[Widget] = None) -> HBox:
        """
        Arranges two widgets in a horizontal two-column layout.

        Args:
            w1 (Widget): First widget (left column)
            w2 (Optional[Widget]): Second widget (right column), optional

        Returns:
            HBox: Horizontal box containing the widgets
        """
        return HBox([VBox([w1], layout=self.COL_LAYOUT),
                    VBox([w2] if w2 else [], layout=self.COL_LAYOUT)],
                   layout=self.FULL_ROW)

    def _connect_events(self):
        """
        Connects event handlers to widget events.

        Sets up observers for interactive fields (date, glyphs) and
        click handlers for action buttons.
        """
        # Connect field change handlers
        self.widgets.discoverydate.observe(self._update_stardate, names='value')
        self.widgets.portalglyphs.observe(self._update_coordinates, names='value')
        self.widgets.galaxy.observe(self._update_coordinates, names='value')

        # Connect button click handlers
        self.widgets.btn_preview.on_click(lambda b: self._handle_generate('preview'))
        self.widgets.btn_gen.on_click(lambda b: self._handle_generate('full'))
        self.widgets.btn_clear.on_click(self._handle_clear)
        self.widgets.btn_example.on_click(self._handle_load_example)
        self.widgets.btn_copy.on_click(self._handle_copy)
        self.widgets.btn_download.on_click(self._handle_download)

    def _update_stardate(self, change):
        """
        Updates AGT Stardate when discovery date changes.

        AGT Stardate is calculated as: (Year + 1716).MM.DD
        This follows the fictional calendar used by the Alliance of Galactic Travellers.

        Args:
            change: Widget change event (not used, required by observer pattern)
        """
        d = self.widgets.discoverydate.value
        if d:
            try:
                arr_date = arrow.get(d)
                self.widgets.agt_stardate.value = f"{arr_date.year + 1716}.{arr_date.format('MM.DD')}"
            except Exception:
                self.widgets.agt_stardate.value = ""
        else:
            self.widgets.agt_stardate.value = ""

    def _update_coordinates(self, change):
        """
        Updates galactic coordinates and region name when glyphs change.

        When portal glyphs are entered, this method:
        1. Validates the glyph format (12 hex digits)
        2. Converts to galactic coordinates (XXXX:YYYY:ZZZZ:SSSS)
        3. Generates procedural region name

        Args:
            change: Widget change event (not used, required by observer pattern)
        """
        raw_glyphs = self._get_widget_value('portalglyphs').upper()
        if not raw_glyphs:
            self.widgets.coordinates.value = ""
            self.widgets.region.value = ""
            return

        # Validate glyph format (12 hex digits)
        if not self.GLYPH_PATTERN.match(raw_glyphs):
            if len(raw_glyphs) not in (0, 12):
                self.widgets.coordinates.value = "Enter exactly 12 hex digits"
                self.widgets.region.value = ""
            else:
                self.widgets.coordinates.value = ""
                self.widgets.region.value = ""
            return

        try:
            # Convert glyphs to voxel coordinates
            voxels = self.galaxy_map.glyphs_to_voxels(raw_glyphs)
            s_hex = raw_glyphs[1:4]  # System index from glyphs (positions 2-4)
            if voxels:
                # Format as standard galactic coordinates
                coords = f"{voxels['x']:04X}:{voxels['y']:04X}:{voxels['z']:04X}:{int(s_hex, 16):04X}"
                self.widgets.coordinates.value = coords
                self._calculate_region_name(raw_glyphs)  # Generate region name
            else:
                self.widgets.coordinates.value = "Calculation Error"
                self.widgets.region.value = ""
        except (ValueError, IndexError):
            self.widgets.coordinates.value = "Invalid Hex"
            self.widgets.region.value = ""

    def _calculate_region_name(self, glyphs):
        """
        Generates procedural region name from portal glyphs.

        Uses the game's exact algorithm to create region names like
        "Sea of Pihend" from galactic coordinates.

        Args:
            glyphs (str): 12-digit portal glyphs
        """
        galaxy_name = self.widgets.galaxy.value
        if not galaxy_name or galaxy_name not in self.data.GALAXY_MAP:
            return

        # Check if procedural data is loaded
        if not NMSData.ALPHASETS or not NMSData.ALPHASETS[0]:
            self.widgets.region.value = "Logic data missing"
            self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:orange;'>⚠️ Region name data not loaded.</div>"
            return

        # Get galaxy index and voxel coordinates
        gal_index = self.data.GALAXY_MAP[galaxy_name]
        voxels = self.galaxy_map.glyphs_to_voxels(glyphs)
        if not voxels:
            return

        # Convert to centered coordinates (relative to galaxy center)
        x = voxels['x'] - NMSGalaxyMap.CENTER_X
        y = voxels['y'] - NMSGalaxyMap.CENTER_Y
        z = voxels['z'] - NMSGalaxyMap.CENTER_Z

        try:
            # Generate region name using game's algorithm
            seed = RegionNameGenerator.create_region_seed(x, y, z, gal_index)
            name = RegionNameGenerator.format_name(seed)
            self.widgets.region.value = name
        except Exception as e:
            self.widgets.region.value = ""
            self.widgets.status_bar.value = f"<div style='text-align:center; margin:10px; color:orange;'>⚠️ Region calculation failed: {html_escape(str(e))}</div>"

    def _handle_generate(self, mode: str):
        """
        Handles wiki code generation (preview or full).

        Args:
            mode (str): 'preview' for display only, 'full' for save/copy enabled

        Steps:
        1. Collect and validate all form data
        2. Render wiki template with validated data
        3. Display in output area
        4. If 'full' mode, enable copy/download buttons and save to file
        """
        self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:orange;'>⏳ Validating and generating...</div>"

        # Collect all widget values
        raw_data = {f.name: self._get_widget_value(f.name)
                   for f in fields(AppWidgets)
                   if hasattr(getattr(self.widgets, f.name), 'value')}
        raw_data['coordinates'] = self.widgets.coordinates.value
        raw_data['class'] = raw_data.pop('item_class')  # Rename for template

        try:
            # Validate data using Pydantic model
            validated_data = WikiDataModel.model_validate(raw_data)
        except ValidationError as e:
            # Format validation error for display
            err_str = str(e).replace('Value error,', '')
            error_msg = f"❌ <b>Validation Error:</b><br/>{html_escape(err_str)}"
            self.widgets.status_bar.value = f"<div style='text-align:center; margin:10px; color:red;'>{error_msg}</div>"
            with self.widgets.output_area:
                clear_output(wait=True)
                print(f"VALIDATION FAILED:\n{e}")
            return

        # Render wiki template with validated data
        self._generated_wikitext = self.wiki_template.render(
            validated_data.model_dump(by_alias=True)
        )

        # Display generated code
        with self.widgets.output_area:
            clear_output(wait=True)
            print(self._generated_wikitext)

        if mode == 'full':
            # Enable copy/download buttons for full generation
            self.widgets.btn_copy.disabled = False
            self.widgets.btn_download.disabled = False
            self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:green;'>✅ <b>Success!</b> Code generated and ready.</div>"

            # Save to file
            filename = self._make_safe_filename(validated_data.name or 'Unnamed_MT')
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self._generated_wikitext)
            except Exception as e:
                self.widgets.status_bar.value = f"<div style='text-align:center; margin:10px; color:orange;'>⚠️ Code generated but file save failed: {html_escape(str(e))}</div>"
        else:
            # Preview mode only
            self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:blue;'>ℹ️ Preview generated below.</div>"

    def _make_safe_filename(self, name_str: str) -> str:
        """
        Creates a safe filename from Multi-Tool name.

        Removes unsafe characters and adds wiki suffix.

        Args:
            name_str (str): Original Multi-Tool name

        Returns:
            str: Safe filename ending with _Wiki.txt
        """
        safe_name = self.FILENAME_SANITIZE_PATTERN.sub('', name_str.replace(' ', '_'))
        return f"{safe_name}_Wiki.txt"

    def _handle_clear(self, b):
        """
        Resets all form fields to default values.

        Args:
            b: Button click event (not used, required by handler)
        """
        for widget in self.widgets:
            if not hasattr(widget, 'value'):
                continue

            # Reset based on widget type
            if isinstance(widget, (Text, Textarea)):
                widget.value = ""
            elif isinstance(widget, (IntText, FloatText)):
                widget.value = 0
            elif isinstance(widget, Checkbox):
                widget.value = False
            elif isinstance(widget, DatePicker):
                widget.value = arrow.now().date()
            elif isinstance(widget, Dropdown) and widget.options:
                widget.value = widget.options[0]
            elif isinstance(widget, Combobox):
                widget.value = ""

        # Reset to default values
        self.widgets.release.value = DEFAULT_RELEASE
        self.widgets.civilized.value = DEFAULT_CIVILIZATION
        self.widgets.researchteam.value = DEFAULT_RESEARCH_TEAM
        self.widgets.civstub.value = DEFAULT_CIV_STUB
        self.widgets.civimage.value = DEFAULT_CIV_IMAGE
        self.widgets.civimagelabel.value = DEFAULT_RESEARCH_TEAM
        if DEFAULT_MODE in self.widgets.mode.options:
            self.widgets.mode.value = DEFAULT_MODE

        # Clear output and disable buttons
        self.widgets.output_area.clear_output()
        self.widgets.btn_copy.disabled = True
        self.widgets.btn_download.disabled = True
        self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:#555;'>🗑️ Form has been reset.</div>"

    def _handle_load_example(self, b):
        """
        Loads example data for demonstration and testing.

        Args:
            b: Button click event (not used, required by handler)
        """
        self._handle_clear(None)  # Clear form first
        self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:#E65100;'>📙 Loading example data...</div>"

        # Example Multi-Tool data
        example_data = {
            'name': "Gleaming Chrono Translator",
            'type': "Pistol",
            'item_class': "C",
            'slots': 12,
            'cost': "680,000",
            'damage': 443.3,
            'scanner': 226.2,
            'mining': 180.1,
            'crystals': False,
            'horns': False,
            'galaxy': "Euclid",
            'system': "AGT Sphenis Primaris",
            'location': "Space Station",
            'portalglyphs': "0006FD20194B",
            'primarycolor': 'Red',
            'secondarycolor': 'Pink',
            'discovered': "FlyingPenguin",
            'discoveredlink': "",
            'discoverydate': arrow.get(2025, 12, 15).date(),
            'platform': 'PC',
            'civstub': DEFAULT_CIV_STUB,
            'civimage': DEFAULT_CIV_IMAGE,
            'civimagelabel': DEFAULT_RESEARCH_TEAM,
            'image': 'AGT Sphenis Primaris-(BR-MT-01a.jpg',
            'gallery': 'File:AGT Sphenis Primaris-(BR-MT-02.jpg|A view of the tool in the cabinet'
        }

        # Apply example data to widgets
        for key, value in example_data.items():
            widget = getattr(self.widgets, key, None)
            if widget is None:
                continue
            if isinstance(widget, (Dropdown, Combobox)):
                if value in widget.options:
                    widget.value = value
                elif widget.options:
                    widget.value = widget.options[0]
            else:
                widget.value = value

        # Update calculated fields
        self._update_stardate(None)
        self._update_coordinates(None)
        self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:#E65100;'>📙 Example data loaded.</div>"

    def _handle_copy(self, b):
        """
        Copies generated wiki code to clipboard.

        Uses JavaScript for clipboard access in notebook environment.

        Args:
            b: Button click event (not used, required by handler)
        """
        if not self._generated_wikitext:
            self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:red;'>❌ Nothing to copy. Please generate code first.</div>"
            return

        # Use JavaScript to copy to clipboard in notebook
        escaped_content = json.dumps(self._generated_wikitext)
        display(Javascript(f'navigator.clipboard.writeText({escaped_content})'))
        self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:green;'>✅ Code copied to clipboard!</div>"

    def _handle_download(self, b):
        """
        Initiates download of generated wiki code (Colab only).

        Args:
            b: Button click event (not used, required by handler)
        """
        if not self._generated_wikitext:
            self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:red;'>❌ Nothing to download. Please generate code first.</div>"
            return

        try:
            # Google Colab specific download function
            from google.colab import files
            name = self._get_widget_value('name') or "Unnamed_MT"
            filename = self._make_safe_filename(name)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self._generated_wikitext)
            files.download(filename)
            self.widgets.status_bar.value = f"<div style='text-align:center; margin:10px; color:green;'>✅ Download initiated for <b>{filename}</b>.</div>"
        except ImportError:
            # Not running in Colab
            self.widgets.status_bar.value = "<div style='text-align:center; margin:10px; color:red;'>❌ Download function is only available in Google Colab.</div>"

    def _get_widget_value(self, widget_name: str) -> str:
        """
        Gets the value from a widget, handling special cases.

        Args:
            widget_name (str): Name of widget to get value from

        Returns:
            str: Widget value as string, empty string for placeholders/defaults
        """
        if not hasattr(self.widgets, widget_name):
            return ""

        widget = getattr(self.widgets, widget_name)
        val = widget.value

        if val is None:
            return ""

        # Convert boolean to 'Y' or empty string
        if isinstance(val, bool):
            return 'Y' if val else ''

        # Convert date to formatted string
        if hasattr(val, 'strftime'):
            return arrow.get(val).format('DD-MMM-YYYY')

        # Don't show zero values for numeric fields
        if isinstance(widget, (FloatText, IntText)) and val == 0:
            return ''

        val_str = str(val).strip()

        # Handle dropdown placeholders (don't save placeholder text)
        if isinstance(widget, Dropdown) and widget_name in self._dropdown_placeholders and val_str == self._dropdown_placeholders[widget_name]:
            return ""

        # Handle combobox placeholders
        if isinstance(widget, Combobox) and widget.placeholder is not None and val_str == widget.placeholder:
            return ""

        return val_str


if __name__ == '__main__':
    """
    Main entry point for the application.

    Creates and runs the Multi-Tool wiki generator when the script
    is executed directly (not imported as a module).
    """
    app = NMSWikiMultiToolGenerator()