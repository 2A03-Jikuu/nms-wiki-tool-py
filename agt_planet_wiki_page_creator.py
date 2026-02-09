"""
No Man's Sky Wiki Form Generator - A Tool for Creating Planet and Moon Wiki Pages

This tool helps players of the video game No Man's Sky generate properly formatted
wiki pages for planets and moons. It handles the complex mathematics needed for
coordinate conversions, generates procedural names in the game's unique style,
and provides an easy-to-use form interface to collect all necessary information.

Key Components:
- Procedural Engine: Generates game-style names using the same algorithms as No Man's Sky
- Galaxy Map: Converts between different coordinate systems used in the game
- Data Loader: Fetches up-to-date game information from external sources
- Form Controller: Manages the user interface and data validation
- Template System: Formats the collected data into wiki-ready text

The main class is NMSWikiFormCreator which builds the complete user interface.
"""

import io
import csv
import math
import re
import struct
import time
import functools
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union

import arrow
import requests
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript
from jinja2 import Environment, Template
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    constr,
    GetCoreSchemaHandler,
    model_validator
)
from pydantic_core import core_schema


# ------------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ------------------------------------------------------------------------------

# Base URL for external game data files stored on GitHub
BASE_URL = "https://raw.githubusercontent.com/2A03-Jikuu/nms-wiki-tool-py/refs/heads/main/datalist"

# All configuration values organized for easy maintenance
CONSTANTS = {
    "URLS": {
        "PLANET": f"{BASE_URL}/planet_data.json",
        "GALAXY": f"{BASE_URL}/galaxies.json",
        "LETTER": f"{BASE_URL}/letter_map.json",
        "ALPHA": f"{BASE_URL}/alphasets.json"
    },
    "STYLES": {
        "HEADER": "font-weight:bold; font-size:16px; margin-top:20px; border-bottom:2px solid #00ACC1; padding-bottom:5px; color:#006064;",
        "DESC": "font-style:italic; font-size:13px; color:#444; margin-bottom:15px; line-height:1.5em; background-color:#E0F7FA; padding:12px; border-left:5px solid #00BCD4; border-radius:4px;",
        "SUB": "font-weight:bold; font-size:14px; margin-top:10px; color:#00838F; border-bottom: 1px dashed #ccc;",
        "POI_HEADER": "font-weight:bold; font-size:14px; color:#333; margin-top:10px; margin-bottom:5px; text-decoration:underline;",
        "ERR": "color: #D32F2F; font-weight: bold;",
        "WARN": "color: #F57C00; font-weight: bold;",
        "SUCCESS": "color: #388E3C; font-weight: bold;"
    },
    "MATH": {
        # The coordinate system center is 2047 in hexadecimal (0x7FF)
        # This is used to convert glyph coordinates to galactic coordinates
        "CENTER": 0x7FF
    },
    "DEFAULTS": {
        "CIV": "Alliance of Galactic Travellers",
        "RELEASE": "Voyagers",
        "REGION_PLACEHOLDER": "Enter Galaxy & Glyphs to generate...",
        "COORDS_PLACEHOLDER": "0000:0000:0000:0000"
    }
}


# ------------------------------------------------------------------------------
# BLOCK 1: PROCEDURAL ENGINE - NAME GENERATION
# ------------------------------------------------------------------------------


class ByteUtils:
    """
    A toolkit for working with byte arrays (lists of numbers from 0 to 255).

    No Man's Sky uses special mathematical operations on byte arrays to create
    procedural names for planets, systems, and regions. This class replicates
    those low-level operations exactly as the game does them.

    Attributes:
        None - all methods are static (don't need an instance to use them)
    """

    @staticmethod
    def parse(val: str, little_endian: bool = True) -> List[int]:
        """
        Converts a hexadecimal string into a list of bytes.

        This is like converting "A1B2" to [177, 161] (since A1=161, B2=177).

        Args:
            val: A string containing hexadecimal characters (0-9, A-F)
            little_endian: If True, reverses the byte order. No Man's Sky uses
                          little-endian format, so this is usually True.

        Returns:
            A list of integers where each integer represents one byte (0-255)

        Example:
            >>> ByteUtils.parse("A1B2", little_endian=True)
            [177, 161]
        """
        # If the string is empty, return an empty list
        if not val:
            return []

        # Hexadecimal works in pairs of characters (each pair = one byte)
        # If length is odd, add a leading zero to make it even
        if len(val) % 2 != 0:
            val = "0" + val

        try:
            # Convert each pair of hex characters to a decimal number
            # Example: "A1" becomes 161, "B2" becomes 177
            res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]

            # Reverse if little-endian (game's format)
            if little_endian:
                res.reverse()
            return res
        except ValueError:
            # If conversion fails (invalid hex), return empty list
            return []

    @staticmethod
    def format_short(op1: List[int]) -> List[int]:
        """
        Ensures a byte list has at least 2 bytes by adding zeros if needed.

        Some operations in the game require at least 2 bytes, so this method
        pads the list with zeros until it reaches that minimum length.

        Args:
            op1: The byte list to check

        Returns:
            A new list with at least 2 bytes
        """
        res = list(op1)
        while len(res) < 2:
            res.append(0x00)  # Add zero bytes
        return res

    @staticmethod
    def add(op1: List[int], op2: List[int]) -> List[int]:
        """
        Adds two byte lists together, handling carry-over between bytes.

        This works like adding normal numbers, but each "digit" is a byte
        (0-255). When a byte exceeds 255, the extra carries to the next byte.

        Args:
            op1: First byte list (like the first number)
            op2: Second byte list (like the second number)

        Returns:
            The sum as a byte list
        """
        # Start with a copy of the second operand
        result = list(op2)
        carry = 0  # Extra value that carries to the next byte
        max_len = max(len(op1), len(op2))

        # Process each byte position
        for i in range(max_len + 1):
            # Stop if we've processed all bytes and no carry remains
            if i >= len(result) and carry == 0 and i >= len(op1):
                break

            # Get byte values from each list (use 0 if list is shorter)
            val1 = op1[i] if i < len(op1) else 0
            val2 = result[i] if i < len(result) else 0

            # Add bytes plus any carry from previous position
            total = val1 + val2 + carry

            # Keep only the lowest 8 bits (0-255) for current byte
            res_byte = total & 0xFF

            # Calculate carry for next position (bits beyond first 8)
            carry = (total >> 8) & 0xFF

            # Store result byte
            if i < len(result):
                result[i] = res_byte
            else:
                result.append(res_byte)

        return result

    @staticmethod
    def sub(op1: List[int], op2: List[int]) -> List[int]:
        """
        Subtracts one byte list from another, handling borrowing.

        Similar to subtraction with borrowing in regular math, but with bytes.

        Args:
            op1: The byte list to subtract from (minuend)
            op2: The byte list to subtract (subtrahend)

        Returns:
            The difference as a byte list
        """
        result = list(op2)
        borrow = 0  # Track if we need to borrow from next byte
        max_len = max(len(op1), len(op2))

        for i in range(max_len + 1):
            # Stop if processed all bytes and no borrow remains
            if i >= len(result) and borrow == 0 and i >= len(op1):
                break

            # Get byte values
            val1 = op1[i] if i < len(op1) else 0
            val2 = result[i] if i < len(result) else 0

            # Subtract bytes and any borrow
            diff = val1 - val2 - borrow

            # Keep only lowest 8 bits
            res_byte = diff & 0xFF

            # Calculate if we need to borrow for next byte
            borrow = (diff >> 8) & 0xFF

            # Convert to 1 if any borrowing happened
            if borrow != 0:
                borrow = 1

            # Store result
            if i < len(result):
                result[i] = res_byte
            else:
                result.append(val1)

        return result

    @staticmethod
    def multiply(op1: List[int], op2: List[int]) -> List[int]:
        """
        Multiplies two byte lists using long multiplication.

        This implements multiplication of large numbers stored as byte arrays,
        similar to how you multiply multi-digit numbers by hand.

        Args:
            op1: First byte list (multiplicand)
            op2: Second byte list (multiplier)

        Returns:
            The product as a byte list
        """
        # Handle empty inputs
        if not op1 or not op2:
            return [0x00]

        result = []

        # For each byte in first number
        for i, v1 in enumerate(op1):
            rem = 0  # Remainder from previous multiplication

            # Multiply with each byte in second number
            for j, v2 in enumerate(op2):
                # Multiply bytes and add any remainder
                raw_prod = (v1 * v2) + rem

                # Convert to signed 16-bit (game's requirement)
                signed_prd = (raw_prod + 32768) % 65536 - 32768

                # Extract high and low bytes
                rem = (signed_prd >> 8) & 0xFF
                res = signed_prd & 0xFF

                # Position in result where this partial product goes
                idx = i + j

                # Ensure result list is long enough
                while len(result) <= idx:
                    result.append(0)

                # Add to existing value at this position
                current_val = result[idx]
                total = res + current_val
                result[idx] = total & 0xFF

                # Calculate carry for next position
                carry = (total >> 8) & 0xFF

                # Propagate carry through higher positions
                k = idx + 1
                while carry != 0:
                    if len(result) <= k:
                        result.append(0)
                    total_next = result[k] + carry
                    result[k] = total_next & 0xFF
                    carry = (total_next >> 8) & 0xFF
                    k += 1

            # Handle any remaining remainder after inner loop
            if rem > 0:
                idx = i + len(op2)
                while len(result) <= idx:
                    result.append(0)
                total = result[idx] + rem
                result[idx] = total & 0xFF
                carry = (total >> 8) & 0xFF
                k = idx + 1
                while carry != 0:
                    if len(result) <= k:
                        result.append(0)
                    total_next = result[k] + carry
                    result[k] = total_next & 0xFF
                    carry = (total_next >> 8) & 0xFF
                    k += 1

        return result

    @staticmethod
    def shl(op1: List[int], shift: int) -> List[int]:
        """
        Shifts bits left by removing bytes from the start.

        In byte lists, shifting left means discarding the first few bytes.
        This is different from bitwise shifting but works similarly for the game.

        Args:
            op1: Byte list to shift
            shift: Number of bytes to remove from the front

        Returns:
            Shifted byte list, or [0x00] if shift exceeds length
        """
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1: List[int], shift: int) -> List[int]:
        """
        Shifts bits right by removing bytes from the start.

        This removes bytes from the beginning of the list, shifting the
        remaining bytes "right" in position.

        Args:
            op1: Byte list to shift
            shift: Number of bytes to remove from the front

        Returns:
            Shifted byte list, or [0x00] if shift exceeds length
        """
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1: List[int], roll: int) -> List[int]:
        """
        Rotates the byte list: items moved from front are added to the back.

        Example: [1, 2, 3] rotated by 1 becomes [2, 3, 1]

        Args:
            op1: Byte list to rotate
            roll: Number of positions to rotate

        Returns:
            Rotated byte list
        """
        if not op1:
            return op1

        # Use modulo to handle rolls larger than list length
        r = roll % len(op1)
        return op1[r:] + op1[:r]

    @staticmethod
    def zxd(op1: List[int], extend: int) -> List[int]:
        """
        Extends the list with zero bytes to reach the desired length.

        Args:
            op1: Byte list to extend
            extend: Desired total length

        Returns:
            Extended byte list padded with zeros
        """
        return list(op1) + [0x00] * max(0, extend - len(op1))

    @staticmethod
    def sxd(op1: List[int], extend: int) -> List[int]:
        """
        Extends the list, preserving the sign (repeats 0xFF for negative).

        In signed numbers, negative values are extended with 0xFF bytes
        (like sign-extension in two's complement).

        Args:
            op1: Byte list to extend
            extend: Desired total length

        Returns:
            Sign-extended byte list
        """
        result = list(op1)

        # Check if number is negative (sign bit in last byte is 1)
        val = 0xFF if (len(op1) > 0 and (op1[-1] >> 7) == 1) else 0x00

        # Add the appropriate extension bytes
        result.extend([val] * max(0, extend - len(op1)))
        return result

    @staticmethod
    def logical_op(op1: List[int], op2: List[int], mode: int) -> List[int]:
        """
        Performs bitwise operations (AND, OR, XOR) on two byte lists.

        Args:
            op1: First byte list
            op2: Second byte list
            mode: 0 for AND, 1 for OR, 2 for XOR

        Returns:
            Result of bitwise operation as byte list

        Raises:
            None, but invalid mode will produce empty result
        """
        # Make both lists same length by padding with zeros
        l1, l2 = len(op1), len(op2)
        target_len = max(l1, l2)
        o1 = op1 + [0x00] * (target_len - l1)
        o2 = op2 + [0x00] * (target_len - l2)

        res = []
        for i in range(target_len):
            if mode == 0:
                res.append(o1[i] & o2[i])  # AND operation
            elif mode == 1:
                res.append(o1[i] | o2[i])  # OR operation
            else:
                res.append(o1[i] ^ o2[i])  # XOR operation
        return res

    @staticmethod
    def xor(op1: List[int], op2: List[int]) -> List[int]:
        """Performs bitwise XOR on two byte lists."""
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1: List[int], op2: List[int]) -> List[int]:
        """Performs bitwise AND on two byte lists."""
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1: List[int], op2: List[int]) -> List[int]:
        """Performs bitwise OR on two byte lists."""
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache: List[List[int]], move: int = 1) -> List[List[int]]:
        """
        Updates the random seed used for procedural generation.

        This advances the game's random number generator to the next state.
        The game uses a specific mathematical formula to generate sequences
        of numbers that look random but are actually predictable.

        Args:
            cache: Current seed state as [seed_part1, seed_part2]
            move: How many steps to advance the seed

        Returns:
            Updated seed state
        """
        # Fixed multiplier used by the game's random number generator
        multiplier = [0x99, 0xF8, 0x76, 0x5A]

        # Advance the seed the requested number of steps
        for _ in range(move):
            step1 = ByteUtils.multiply(cache[0], multiplier)
            result = ByteUtils.add(step1, cache[1])
            cache[0] = ByteUtils.shl(result, 4)
            cache[1] = ByteUtils.shr(result, 4)
        return cache

    @staticmethod
    def to_uint32(arr: List[int], offset: int = 0) -> int:
        """
        Converts 4 bytes to an unsigned 32-bit integer.

        Args:
            arr: Byte list to convert from
            offset: Starting position in the list

        Returns:
            Unsigned integer value
        """
        return struct.unpack('<I', bytes(ByteUtils.zxd(arr[offset:offset + 4], 4)))[0]

    @staticmethod
    def to_int32(arr: List[int], offset: int = 0) -> int:
        """
        Converts 4 bytes to a signed 32-bit integer (can be negative).

        Args:
            arr: Byte list to convert from
            offset: Starting position in the list

        Returns:
            Signed integer value
        """
        return struct.unpack('<i', bytes(ByteUtils.zxd(arr[offset:offset + 4], 4)))[0]

    @staticmethod
    def to_int16(arr: List[int], offset: int = 0) -> int:
        """
        Converts 2 bytes to a signed 16-bit integer (can be negative).

        Args:
            arr: Byte list to convert from
            offset: Starting position in the list

        Returns:
            Signed integer value
        """
        return struct.unpack('<h', bytes(ByteUtils.zxd(arr[offset:offset + 2], 2)))[0]

    @staticmethod
    def to_double(arr: List[int], offset: int = 0) -> float:
        """
        Converts 8 bytes to a double-precision floating point number.

        Args:
            arr: Byte list to convert from
            offset: Starting position in the list

        Returns:
            Double precision float value
        """
        return struct.unpack('<d', bytes(ByteUtils.zxd(arr[offset:offset + 8], 8)))[0]

    @staticmethod
    def to_single(arr: List[int], offset: int = 0) -> float:
        """
        Converts 4 bytes to a single-precision floating point number.

        Args:
            arr: Byte list to convert from
            offset: Starting position in the list

        Returns:
            Single precision float value
        """
        return struct.unpack('<f', bytes(ByteUtils.zxd(arr[offset:offset + 4], 4)))[0]

    @staticmethod
    def get_bytes_uint32(val: int) -> List[int]:
        """
        Converts an unsigned 32-bit integer into a list of 4 bytes.

        Args:
            val: Integer to convert (0 to 4,294,967,295)

        Returns:
            Byte list representation
        """
        return list(struct.pack('<I', val & 0xFFFFFFFF))


class StringExtensions:
    """Helper methods for string formatting and hexadecimal conversions."""

    @staticmethod
    def short_to_formatted_hex(val: int, trunc: int) -> str:
        """
        Formats an integer as a hexadecimal string and truncates it.

        Args:
            val: Integer to format
            trunc: Number of hexadecimal characters to keep

        Returns:
            Truncated hex string

        Example:
            >>> StringExtensions.short_to_formatted_hex(255, 2)
            "FF"
        """
        return f"{val & 0xFFFF:04X}"[-trunc:]


class Generator:
    """
    Generates procedural names (planets, systems) using No Man's Sky algorithms.

    This class uses 'seeds' (numbers derived from coordinates) to select letters
    from pre-defined alphabets, creating the alien-sounding names you see in-game.

    Class Attributes:
        MULTIPLIER: Fixed multiplier used in random number generation
        TINY_DOUBLE: Small double value used for probability calculations
        ALPHASETS: Lists of 3-letter combinations that form name beginnings
        LETTER_MAP: Probability maps for which letters follow others
    """

    # These class attributes are loaded from external data files
    MULTIPLIER = [0x99, 0xF8, 0x76, 0x5A]
    TINY_DOUBLE = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]
    ALPHASETS = []
    LETTER_MAP = {}

    @staticmethod
    def generate_name(cache0: List[List[int]], cache1: List[List[int]]) -> str:
        """
        Generates a procedural name based on the provided random seeds.

        The algorithm:
        1. Gets starting 3 letters from an alphabet set
        2. Determines name length based on seed values
        3. Adds letters probabilistically based on letter transition maps
        4. Applies linguistic fixes to ensure names are pronounceable

        Args:
            cache0: First seed state part (controls random selection)
            cache1: Second seed state part (determines alphabet set)

        Returns:
            Generated name string

        Example:
            Might return names like "Xylophos" or "Geknip Prime"
        """
        # Check if data is loaded
        if not Generator.ALPHASETS:
            return "Data Error"

        # Get the first 3 letters of the name from the selected alphabet set
        name = Generator.get_characters_from_alphaset(cache0, cache1)
        if name == "__EMPTY__":
            return ""

        # Advance the random seed
        ByteUtils.update_seed(cache0)

        # Determine which method to use for selecting next characters
        check_op = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        alternate_char_getter = (ByteUtils.to_int16(check_op) != 0)
        ByteUtils.update_seed(cache0)

        # --- Determine name length ---
        # This complex formula calculates how long the name should be
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])
        limit = ByteUtils.to_int16(ByteUtils.sxd(register0, 2))

        # --- Build the rest of the name ---
        if limit > 0:
            i, safety = 0, 0
            while i < limit:
                ByteUtils.update_seed(cache0)
                sub_str = name[i: i + 3]

                # Get possible next letters and their probabilities
                char_weights = Generator.get_string_weights(sub_str, cache1[0][0] if cache1[0] else 0)

                # Convert current seed to a random probability value
                val_u32 = ByteUtils.to_uint32(cache0[0])
                target = float(val_u32 * ByteUtils.to_double(Generator.TINY_DOUBLE))

                # Handle cases where no valid next letters exist
                if char_weights is None:
                    i -= 1  # Try previous position again
                    safety += 1
                    if safety > 50:  # Prevent infinite loops
                        break
                    else:
                        safety = 0
                else:
                    safety = 0

                # Select a letter based on the random probability
                index = 0
                if alternate_char_getter:
                    # Use floating-point method for selection
                    target *= (len(char_weights) - 1)
                    b_tgt = list(struct.pack('<f', target))
                    op = ByteUtils.or_op(ByteUtils.and_op(b_tgt, [0x00, 0x00, 0x00, 0x80]), [0x00, 0x00, 0x00, 0x3F])
                    index = int(ByteUtils.to_single(op) + target)
                else:
                    # Use cumulative probability method
                    weight = 0.0
                    for j, cw in enumerate(char_weights):
                        weight += cw[1]
                        if weight >= target:
                            index = j
                            break

                # Add selected letter to the name
                if index < len(char_weights):
                    name += char_weights[index][0]

                # Prevent names from getting too long
                if len(name) > 63:
                    name = name[:64]
                i += 1

        if not name:
            return ""

        # --- LINGUISTIC FIXES ---
        # Ensure the name looks pronounceable like real alien words

        # Fix 1: Avoid consonant clusters at the start
        vowels = "aeiou"
        if len(name) > 1 and name[0] not in vowels and name[1] not in vowels:
            if name[0] != 's' or name[1] not in "hklmnprtwy":
                if name[1] == 'h' and name[0] in "ctw":
                    pass  # These combinations are allowed (ch, th, wh)
                elif name[1] == 'l' and name[0] in "bcfgps":
                    pass  # These combinations are allowed (bl, cl, etc.)
                elif name[1] == 'r' and name[0] in "bcdfgkpt":
                    pass  # These combinations are allowed (br, cr, etc.)
                elif name[1] == 'w' and name[0] in "dgt":
                    pass  # These combinations are allowed
                elif name[1] == 'y' and name[0] in "hmr":
                    pass  # These combinations are allowed
                else:
                    # Insert a vowel to break up the consonants
                    name = Generator.insert_vowel(name, cache0, 1)

        # Fix 2: Avoid awkward endings
        ult, penult = name[-1], (name[-2] if len(name) > 1 else '')
        if len(name) > 1 and (penult != 'g' or ult in vowels):
            idx = len(name) - 1
            bad_end = False

            # Check for known bad ending patterns
            if ult == 'b' and penult in "gn":
                bad_end = True
            elif ult == 'd' and penult in "bdfghkmpst":
                bad_end = True
            elif ult == 'g' and penult == 'l':
                bad_end = True
            elif ult == 'p' and penult in "bdhkt":
                bad_end = True
            elif ult == 'r' and penult in "bfg":
                bad_end = True
            elif ult == 't' and penult == 'g':
                bad_end = True
            elif ult == 'w' and penult not in vowels:
                bad_end = True

            # Insert vowel if ending is awkward
            if bad_end:
                name = Generator.insert_vowel(name, cache0, idx)

        # Fix 3: Avoid too many consecutive consonants
        cons = Generator.get_consecutive_consonants(name)
        if cons != -1:
            ByteUtils.update_seed(cache0)
            mult = ByteUtils.multiply(cache0[0], [0x03])
            offset = ByteUtils.to_int32(ByteUtils.zxd(ByteUtils.add(ByteUtils.shr(mult, 4), [0x01]), 4))
            name = Generator.insert_vowel(name, cache0, cons + offset)

        return name

    @staticmethod
    def get_characters_from_alphaset(cache0, cache1):
        """
        Selects the starting 3 letters for the name from an alphabet set.

        The game has different alphabet sets (like "Gek", "Korvax", "Vy'keen"
        languages). This method picks which set to use and then selects
        3 starting letters from it.

        Args:
            cache0: Random seed for selection
            cache1: Contains index of which alphabet set to use

        Returns:
            3-character string or "__EMPTY__" if no set available
        """
        ByteUtils.update_seed(cache0)

        # Determine which alphabet set to use
        idx = cache1[0][0] if cache1[0] else 0
        if idx >= len(Generator.ALPHASETS):
            idx = 0
        alphaset_str = Generator.ALPHASETS[idx]

        if not alphaset_str:
            return "__EMPTY__"

        # Calculate starting position within the alphabet set
        reg0 = ByteUtils.multiply(cache0[0], ByteUtils.get_bytes_uint32(len(alphaset_str) // 3))
        reg1 = ByteUtils.format_short(ByteUtils.multiply(ByteUtils.shr(reg0, 4), [0x03]))
        start = ByteUtils.to_int16(reg1)

        # Return 3 characters starting at the calculated position
        return alphaset_str[start: ByteUtils.to_int16(ByteUtils.add(reg1, [0x03]))]

    @staticmethod
    def get_string_weights(s, alphaset):
        """
        Retrieves the probability of letters following the current string.

        The game uses probability maps (Markov chains) to determine which
        letters are likely to follow others, creating language-like patterns.

        Args:
            s: Current string (usually 1-3 characters)
            alphaset: Which alphabet set to use

        Returns:
            List of (character, probability) tuples, or None if not found
        """
        if not Generator.LETTER_MAP or alphaset not in Generator.LETTER_MAP:
            return None

        subset = Generator.LETTER_MAP[alphaset]
        if not s or s[0] not in subset:
            return None

        return Generator.recursive_search(subset[s[0]], s)

    @staticmethod
    def recursive_search(arr, s):
        """
        Navigates the letter map tree to find next possible characters.

        The letter map is organized as a tree structure where each level
        represents the next character position in the string.

        Args:
            arr: Current level of the letter map tree
            s: String to search for

        Returns:
            List of (character, probability) tuples for next characters
        """
        s_val_32 = ByteUtils.to_int32(ByteUtils.zxd(list(s.encode('utf-8')), 4))
        for item in arr:
            if len(item) > 2:
                if item[2] == "ja":
                    # Jump if above - navigate deeper in the tree
                    val_b = ByteUtils.zxd(list(str(item[0]).encode('utf-8')), 4)
                    if s_val_32 > ByteUtils.to_int32(val_b):
                        return Generator.recursive_search(item[1], s)
                elif item[2] == "jz" and str(item[0]) == s:
                    # Found the matching string - return its probability list
                    return [(w.get("Item1"), float(w.get("Item2", 0))) for w in item[1]]
        return None

    @staticmethod
    def insert_vowel(name, seed, index):
        """
        Inserts a vowel to fix unpronounceable consonant clusters.

        Args:
            name: Current name being built
            seed: Random seed for vowel selection
            index: Position to insert vowel at

        Returns:
            Name with vowel inserted
        """
        ByteUtils.update_seed(seed)
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)

        # Insert one of the vowels (a, e, i, o, u) based on random calculation
        if calc and calc[0] < 5 and index <= len(name):
            return name[:index] + "aeiou"[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name):
        """
        Finds the start of a sequence of 3 or more consonants.

        Args:
            name: String to check

        Returns:
            Position where 3+ consonants start, or -1 if none found
        """
        consonance = 0
        for i, char in enumerate(name):
            if char not in "aeiou":
                consonance += 1
                if consonance >= 3:
                    if char not in "aeiouy":
                        return i - 3
            else:
                consonance = 0
        return -1


class RegionNameGenerator:
    """
    Generates names for regions (e.g., "The Arm of Vektis").

    Regions are large areas of space in the galaxy, named based on
    spatial coordinates using a different algorithm than planet names.

    Class Attributes:
        PROC_ADORNMENTS: List of suffixes and prefixes to add to region names
                         (like "Void", "Expanse", "The Arm of")
    """

    # These are the decorative additions to region names
    PROC_ADORNMENTS = [
        "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
        "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
        "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
        "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux",
        "%NAME% Instability", "Sea of %NAME%", "The Arm of %NAME%",
        "%NAME% Spur", "%NAME% Shallows"
    ]

    @staticmethod
    def create_region_seed(x: int, y: int, z: int, galaxy: int) -> List[int]:
        """
        Creates a seed number from X, Y, Z coordinates and galaxy index.

        The seed is a byte array created by concatenating hexadecimal
        representations of the coordinates in a specific order.

        Args:
            x: X coordinate in galactic space
            y: Y coordinate in galactic space (vertical axis)
            z: Z coordinate in galactic space
            galaxy: Index of the galaxy (0=Euclid, 1=Hilbert, etc.)

        Returns:
            Byte array seed for region name generation
        """
        # Convert coordinates to hex and combine them in specific order
        # Format: galaxy(2) + y(2) + z(3) + x(3) = 10 hex characters total
        s = (StringExtensions.short_to_formatted_hex(galaxy, 2) +
             StringExtensions.short_to_formatted_hex(y, 2) +
             StringExtensions.short_to_formatted_hex(z, 3) +
             StringExtensions.short_to_formatted_hex(x, 3))
        return ByteUtils.parse(s)

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def format_name_cached(seed_tuple: Tuple[int]) -> str:
        """
        Cached wrapper for format_name to improve performance.

        Since region names are calculated from coordinates and coordinates
        are often repeated, caching saves computation time.

        Args:
            seed_tuple: Seed converted to a tuple (for caching compatibility)

        Returns:
            Generated region name
        """
        return RegionNameGenerator.format_name(list(seed_tuple))

    @staticmethod
    def format_name(seed: List[int]) -> str:
        """
        Calculates the region name from the seed using No Man's Sky math.

        This uses a multi-step mathematical transformation on the seed
        to create random-looking but deterministic region names.

        Args:
            seed: Byte array seed created from coordinates

        Returns:
            Generated region name (e.g., "The Arm of Vektis")
        """
        # Initialize the random number generator states
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # --- Transform the seed through multiple mathematical operations ---
        # This complex sequence creates the initial random state
        reg0 = ByteUtils.shr(seed, 4)
        if reg0:
            reg0[0] //= 2
        xor1 = ByteUtils.xor(reg0, seed)
        reg0 = ByteUtils.multiply(xor1, [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64])[:8]
        val1 = ByteUtils.to_uint32(ByteUtils.shr(reg0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val1), reg0)
        reg0 = ByteUtils.multiply(xor2, [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3])[:8]
        val2 = ByteUtils.to_uint32(ByteUtils.shr(reg0, 4)) // 2
        reg0 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val2), reg0)
        shl4 = ByteUtils.shl(reg0, 4)
        xor_mid = ByteUtils.xor(ByteUtils.rol(shl4, 2), ByteUtils.shr(reg0, 4))
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure seed is not zero (would cause issues)
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])

        # Generate the base name using the same generator as planets
        ByteUtils.update_seed(cache0)
        cache1[2] = ByteUtils.add(ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4), [0x06])

        name = Generator.generate_name(cache0, cache1)
        if not name or "[" in name:
            return name or "Unknown Region"

        # Capitalize first letter
        name = name[0].upper() + name[1:]

        # 50% chance to add an adornment (like "Void" or "Expanse")
        ByteUtils.update_seed(cache0)
        if ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x64]), 4)[0] < 0x50:
            ByteUtils.update_seed(cache0)
            idx = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x14]), 4)[0]
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                name = RegionNameGenerator.PROC_ADORNMENTS[idx].replace("%NAME%", name)

        return name


# ------------------------------------------------------------------------------
# BLOCK 2: GALAXY MAP - COORDINATE CONVERSIONS
# ------------------------------------------------------------------------------


class NMSGalaxyMap:
    """
    Handles coordinate conversions between different systems in No Man's Sky.

    The game uses several coordinate systems:
    1. Portal Glyphs: 12 hexadecimal symbols shown at portals
    2. Galactic Coordinates: X:Y:Z:Portal format used on the wiki
    3. Internal Coordinates: Raw numbers used for calculations

    This class converts between these systems.
    """

    def __init__(self):
        """Initializes the galaxy map with a regex for validating glyphs."""
        # Regular expression to validate 12-character hex strings
        self.re_glyph = re.compile(r'^[0-9A-Fa-f]{12}$')

    def glyphs_to_coords(self, glyphs: str) -> Optional[Dict[str, int]]:
        """
        Converts a 12-character hex glyph string to coordinate integers.

        Portal glyphs encode position information in a specific pattern:
        - First character: Planet index in system
        - Next 3 characters: Solar system index
        - Next 2 characters: Y coordinate (vertical)
        - Next 3 characters: Z coordinate
        - Last 3 characters: X coordinate

        Args:
            glyphs: 12-character hexadecimal string like "0123456789AB"

        Returns:
            Dictionary with keys 'x', 'y', 'z', 's' (solar system index),
            or None if conversion fails

        Example:
            >>> map = NMSGalaxyMap()
            >>> map.glyphs_to_coords("0123456789AB")
            {'x': 123, 'y': 45, 'z': 678, 's': 291}
        """
        # Clean and validate input
        g = glyphs.strip().upper()
        if not self.re_glyph.match(g):
            return None

        try:
            # Extract coordinate parts from the glyph string
            # Positions are hardcoded based on how glyphs encode coordinates
            s_hex = int(g[1:4], 16)   # Solar system index (characters 1-3)
            y_hex = int(g[4:6], 16)   # Y coordinate (characters 4-5)
            z_hex = int(g[6:9], 16)   # Z coordinate (characters 6-8)
            x_hex = int(g[9:12], 16)  # X coordinate (characters 9-11)
        except ValueError:
            return None

        def convert(val, is_y=False):
            """
            Converts glyph coordinate to galactic coordinate.

            Glyph coordinates are offset from the galactic center:
            - Values above the midpoint are negative
            - Values below the midpoint are positive

            Args:
                val: Raw coordinate value from glyphs
                is_y: Whether this is the Y coordinate (different limits)
            """
            # Different limits for Y vs X/Z coordinates
            limit = 0x80 if is_y else 0x800
            shift_pos = 0x81 if is_y else 0x801
            shift_neg = 0x7F if is_y else 0x7FF

            if val > limit:
                return val - shift_pos  # Negative region
            else:
                return val + shift_neg  # Positive region

        return {
            'x': convert(x_hex),
            'y': convert(y_hex, is_y=True),
            'z': convert(z_hex),
            's': s_hex
        }


# ------------------------------------------------------------------------------
# BLOCK 3: DATA LOADER - FETCHING GAME INFORMATION
# ------------------------------------------------------------------------------


class NMSData:
    """
    Fetches and stores game data from external sources.

    This class downloads JSON files containing up-to-date information about
    the game, including biomes, resources, weather types, and naming data.
    All data is loaded from GitHub to ensure it stays current with game updates.

    Attributes:
        Various lists and dictionaries containing game data, populated
        from external JSON files when the class is initialized.
    """

    def __init__(self):
        """Initializes the data loader and fetches all external data."""
        # Initialize all data containers as empty
        self.GALAXIES_MAP, self.GALAXIES_LIST = {}, []
        self.PLATFORMS, self.MODES, self.BODY_TYPES, self.TEMP_UNITS, self.SKY_COLOR_OPTIONS = [], [], [], [], []
        self.RESOURCES, self.RAW_INGREDIENTS = [], []
        self.RESOURCE_MAP, self.BIOME_DESCRIPTIONS_MAP, self.WEATHER_CONDITION_MAP = {}, {}, {}
        self.WEATHER_OPTIONS, self.SENTINEL_OPTIONS, self.FLORA_FAUNA_OPTIONS = [], [], []
        self.GEOLOGY_OPTIONS, self.ARCHETYPE_OPTIONS, self.TERRAIN_OPTIONS = [], [], []
        self.LAND_TYPE_OPTIONS, self.POI_TYPES, self.ECONOMY_OPTIONS = [], [], []

        # Load all data from external sources
        self._load_external_data()

    def _fetch(self, url: str, retries: int = 3) -> Dict:
        """
        Downloads JSON data from a URL with retry logic.

        Args:
            url: The URL to fetch JSON data from
            retries: How many times to retry if download fails

        Returns:
            Parsed JSON data as dictionary, or empty dict if all retries fail

        Raises:
            No explicit raises, but prints error message on failure
        """
        for i in range(retries):
            try:
                # Try to download the data with timeout
                r = requests.get(url, timeout=10)
                r.raise_for_status()  # Check for HTTP errors
                return r.json()
            except requests.RequestException:
                # Wait longer after each failed attempt
                time.sleep(1 + i)
                print(f"Failed to fetch {url}")
        return {}  # Return empty dict if all retries fail

    def _load_external_data(self):
        """Populates all data lists from the remote JSON files."""
        print("Fetching NMS data...")

        # --- Load galaxy data ---
        gal_data = self._fetch(CONSTANTS["URLS"]["GALAXY"])
        if gal_data:
            # Create mapping from index to name, and sorted name list
            self.GALAXIES_MAP = {g['index']: g['name'] for g in gal_data}
            self.GALAXIES_LIST = sorted([g['name'] for g in gal_data])

        # --- Load planet data (biomes, resources, weather, etc.) ---
        p_data = self._fetch(CONSTANTS["URLS"]["PLANET"])
        if p_data:
            # General information
            gen = p_data.get('general', {})
            self.PLATFORMS = sorted(gen.get('platforms', []))
            self.MODES = sorted(gen.get('modes', []))
            self.BODY_TYPES = sorted(gen.get('body_types', []))
            self.TEMP_UNITS = sorted(gen.get('temp_units', []))
            self.SKY_COLOR_OPTIONS = sorted(gen.get('sky_colors', []))

            # Resources and ingredients
            res = p_data.get('resources', {})
            self.RESOURCES = sorted(res.get('list', []))
            self.RAW_INGREDIENTS = sorted(res.get('raw_ingredients', []))
            self.RESOURCE_MAP = res.get('map', {})

            # Biome descriptions organized by biome type
            self.BIOME_DESCRIPTIONS_MAP = {k: sorted(v) for k, v in p_data.get('biomes', {}).items()}

            # Weather conditions grouped by type
            w_groups = p_data.get('weather', {})
            self.WEATHER_CONDITION_MAP = {cond: gname for gname, conds in w_groups.items() for cond in conds}
            self.WEATHER_OPTIONS = sorted(self.WEATHER_CONDITION_MAP.keys())

            # Various dropdown options
            opts = p_data.get('options', {})
            self.SENTINEL_OPTIONS = sorted(opts.get('sentinels', []))
            self.FLORA_FAUNA_OPTIONS = sorted(opts.get('flora_fauna', []))
            self.GEOLOGY_OPTIONS = sorted(opts.get('geology', []))
            self.ARCHETYPE_OPTIONS = sorted(opts.get('archetypes', []))
            self.TERRAIN_OPTIONS = sorted(opts.get('terrain', []))
            self.LAND_TYPE_OPTIONS = sorted(opts.get('land_type', []))
            self.POI_TYPES = sorted(opts.get('poi_types', []))
            self.ECONOMY_OPTIONS = sorted(opts.get('economy', []))

        # --- Load procedural name generation data ---
        Generator.ALPHASETS = self._fetch(CONSTANTS["URLS"]["ALPHA"])
        lm = self._fetch(CONSTANTS["URLS"]["LETTER"])
        Generator.LETTER_MAP = {int(k): v for k, v in lm.items()} if lm else {}

        print("Data loaded.")


# ------------------------------------------------------------------------------
# BLOCK 4: UI MODELS - DATA STRUCTURES FOR THE USER INTERFACE
# ------------------------------------------------------------------------------


@dataclass
class AppWidgets:
    """
    Container for all UI widgets (inputs, buttons, displays).

    Using a dataclass keeps all widgets organized in one place and
    makes them easy to access throughout the application.

    Attributes:
        Each attribute corresponds to a widget in the user interface.
        The field(init=False) means they're created after initialization.
    """

    # Basic Information widgets
    name: widgets.Text = field(init=False)
    original_name: widgets.Text = field(init=False)
    body_type: widgets.Dropdown = field(init=False)
    system: widgets.Text = field(init=False)
    galaxy: widgets.Combobox = field(init=False)
    platform: widgets.Dropdown = field(init=False)
    discovery_date: widgets.DatePicker = field(init=False)
    discovered_by: widgets.Text = field(init=False)
    discovered_by_link: widgets.Text = field(init=False)
    agt_stardate: widgets.Text = field(init=False)
    civilized: widgets.Text = field(init=False)
    mode: widgets.Dropdown = field(init=False)
    release: widgets.Text = field(init=False)

    # Orbital Information widgets
    region: widgets.Text = field(init=False)
    glyphs: widgets.Text = field(init=False)
    coordinates: widgets.Text = field(init=False)
    parent_planet: widgets.Text = field(init=False)
    moon_count: widgets.IntText = field(init=False)
    moons: widgets.Textarea = field(init=False)

    # Planetary Details widgets
    biome: widgets.Dropdown = field(init=False)
    biome_description: widgets.Dropdown = field(init=False)
    sky_color: widgets.Dropdown = field(init=False)
    atmosphere: widgets.Textarea = field(init=False)
    weather: widgets.Combobox = field(init=False)
    sentinel_level: widgets.Combobox = field(init=False)
    flora_level: widgets.Combobox = field(init=False)
    fauna_level: widgets.Combobox = field(init=False)

    # Life & Resources widgets
    fauna_count: widgets.IntText = field(init=False)
    resource_checkboxes: List[widgets.Checkbox] = field(default_factory=list)
    ingredient_checkboxes: List[widgets.Checkbox] = field(default_factory=list)

    # Geology widgets
    geo_terrain: widgets.Dropdown = field(init=False)
    geo_landtype: widgets.Dropdown = field(init=False)
    geo_archetype: widgets.Dropdown = field(init=False)
    geo_age: widgets.Text = field(init=False)
    geo_geology: widgets.Combobox = field(init=False)
    geo_core: widgets.Text = field(init=False)
    tunit: widgets.Dropdown = field(init=False)

    # Environment widgets (many temperature/radiation readings)
    daytemp: widgets.FloatText = field(init=False)
    nighttemp: widgets.FloatText = field(init=False)
    radnorm: widgets.FloatText = field(init=False)
    toxicnorm: widgets.FloatText = field(init=False)
    daystormtemp: widgets.FloatText = field(init=False)
    nightstormtemp: widgets.FloatText = field(init=False)
    radstorm: widgets.FloatText = field(init=False)
    toxicstorm: widgets.FloatText = field(init=False)
    daycavetemp: widgets.FloatText = field(init=False)
    nightcavetemp: widgets.FloatText = field(init=False)
    radcave: widgets.FloatText = field(init=False)
    toxiccave: widgets.FloatText = field(init=False)
    cavestormrad: widgets.FloatText = field(init=False)
    cavestormtoxic: widgets.FloatText = field(init=False)
    daywatertemp: widgets.FloatText = field(init=False)
    nightwatertemp: widgets.FloatText = field(init=False)
    waterrad: widgets.FloatText = field(init=False)
    watertoxic: widgets.FloatText = field(init=False)
    daystormwatertemp: widgets.FloatText = field(init=False)
    nightstormwatertemp: widgets.FloatText = field(init=False)
    waterstormrad: widgets.FloatText = field(init=False)
    waterstormtoxic: widgets.FloatText = field(init=False)

    # Points of Interest & Gallery widgets
    poi_list_csv: widgets.Textarea = field(init=False)
    documenter_name: widgets.Text = field(init=False)
    economy_type: widgets.Dropdown = field(init=False)
    additional_info: widgets.Textarea = field(init=False)
    image: widgets.Text = field(init=False)
    gallery_images: widgets.Textarea = field(init=False)

    # Button widgets
    preview_btn: widgets.Button = field(init=False)
    generate_btn: widgets.Button = field(init=False)
    copy_btn: widgets.Button = field(init=False)
    download_btn: widgets.Button = field(init=False)
    clear_btn: widgets.Button = field(init=False)
    load_planet_btn: widgets.Button = field(init=False)
    load_moon_btn: widgets.Button = field(init=False)

    # Status display widgets
    status_label: widgets.Label = field(init=False)
    output_area: widgets.Output = field(init=False)


class ArrowType(arrow.Arrow):
    """
    Custom Pydantic validator for Arrow date objects.

    This allows Pydantic to properly validate and handle Arrow date objects
    in the form data model, converting strings to Arrow objects when needed.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Tells Pydantic how to validate Arrow objects."""
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v: Any) -> arrow.Arrow:
        """
        Converts input to an Arrow date object if it isn't already one.

        Args:
            v: Input value (could be string, datetime, or Arrow)

        Returns:
            Arrow date object

        Example:
            >>> ArrowType.validate("2023-10-05")
            <Arrow [2023-10-05T00:00:00+00:00]>
        """
        return v if isinstance(v, arrow.Arrow) else arrow.get(v)


class POI(BaseModel):
    """
    Data model for a Point of Interest (location marker).

    Points of Interest are notable locations on a planet like trading posts,
    portals, or crashed ships that players might want to document.

    Attributes:
        type: Kind of POI (e.g., "Portal", "Trade Post")
        lon: Longitude coordinate on the planet
        lat: Latitude coordinate on the planet
        name: Custom name for the POI
        note: Additional notes about the POI
        date: When the POI was documented
    """
    type: str
    lon: str
    lat: str
    name: str
    note: str
    date: str


class NMSFormData(BaseModel):
    """
    Main data model using Pydantic for validation.

    This model defines all the data fields needed for a planet/moon wiki page
    and ensures they meet the game's requirements before generating wiki text.

    Pydantic automatically validates data types and can convert between formats.

    Attributes correspond to form fields, with descriptions in the docstrings.
    """

    # --- Basic Information ---
    name: constr(min_length=1)  # Current name of the celestial body
    original_name: Optional[str] = ""  # Original procedural name
    body_type: str  # "Planet" or "Moon"
    system: constr(min_length=1)  # Name of the star system
    galaxy: constr(min_length=1)  # Name of the galaxy
    platform: str  # Gaming platform (PC, PS4, Xbox, etc.)
    discovery_date: ArrowType  # When the planet was discovered
    discovered_by: constr(min_length=1)  # Username of discoverer
    discovered_by_link: Optional[str] = ""  # Wiki profile link
    civilized: Optional[str] = CONSTANTS["DEFAULTS"]["CIV"]  # Player civilization
    mode: str  # Game mode (Normal, Survival, Creative, Permadeath)
    release: str  # Game version/update when discovered

    # --- Location Information ---
    region: constr(min_length=1)  # Galactic region name
    glyphs: constr(pattern=r'^[0-9A-Fa-f]{12}$')  # 12-character portal glyphs
    coordinates: Optional[str] = ""  # Galactic coordinates in X:Y:Z:Portal format
    parent_planet: Optional[str] = ""  # For moons: which planet they orbit
    moon_count: int = 0  # How many moons this planet has
    moons_list: List[str] = []  # Names of the moons

    # --- Planetary Details ---
    biome: str  # Primary biome type (Lush, Toxic, Frozen, etc.)
    biome_description: str  # Specific biome description from scanner
    sky_color: str  # Color of the sky during day
    atmosphere: Optional[str] = ""  # Atmospheric composition
    weather: str  # Weather conditions
    sentinel_level: str  # Sentinel activity level
    flora_level: str  # Abundance of plant life
    fauna_level: str  # Abundance of animal life
    fauna_count: int = 0  # Number of fauna species
    resources_list: List[str] = []  # Resources found on the planet
    raw_ingredients_list: List[str] = []  # Harvestable ingredients

    # --- Geology Information ---
    geo_terrain: Optional[str] = ""  # Terrain structure
    geo_landtype: Optional[str] = ""  # Type of landforms
    geo_archetype: Optional[str] = ""  # Geological archetype
    geo_age: Optional[str] = ""  # Planetary age in billions of years
    geo_geology: Optional[str] = ""  # Geological characteristics
    geo_core: Optional[str] = ""  # Core composition
    tunit: Optional[str] = ""  # Temperature unit (Celsius, Fahrenheit, Kelvin)

    # --- Environmental Readings (all optional) ---
    # Normal surface conditions
    daytemp: Optional[float] = None
    nighttemp: Optional[float] = None
    radnorm: Optional[float] = None
    toxicnorm: Optional[float] = None

    # Storm conditions
    daystormtemp: Optional[float] = None
    nightstormtemp: Optional[float] = None
    radstorm: Optional[float] = None
    toxicstorm: Optional[float] = None

    # Cave conditions
    daycavetemp: Optional[float] = None
    nightcavetemp: Optional[float] = None
    radcave: Optional[float] = None
    toxiccave: Optional[float] = None
    cavestormrad: Optional[float] = None
    cavestormtoxic: Optional[float] = None

    # Water conditions
    daywatertemp: Optional[float] = None
    nightwatertemp: Optional[float] = None
    waterrad: Optional[float] = None
    watertoxic: Optional[float] = None
    daystormwatertemp: Optional[float] = None
    nightstormwatertemp: Optional[float] = None
    waterstormrad: Optional[float] = None
    waterstormtoxic: Optional[float] = None

    # --- Documentation Information ---
    poi_list: List[POI] = []  # List of Points of Interest
    documenter_name: Optional[str] = ""  # Person documenting this entry
    economy_type: Optional[str] = ""  # Economic system type
    additional_info: Optional[str] = ""  # Any extra notes
    image: Optional[str] = ""  # Main image filename
    gallery_list: List[str] = []  # List of gallery image filenames

    @field_validator('body_type', 'platform', 'mode', 'biome', 'weather', 'sentinel_level', 'flora_level', 'fauna_level')
    @classmethod
    def check_select(cls, v: str) -> str:
        """
        Ensures dropdown fields are not left on the placeholder value.

        Args:
            v: Field value to check

        Returns:
            Validated value

        Raises:
            ValueError: If field contains placeholder text
        """
        if not v or 'select' in v.lower() or '...' in v:
            raise ValueError("Selection required.")
        return v

    @model_validator(mode='after')
    def validate_logic(self):
        """
        Validates logical dependencies between fields.

        For example, moons must have a parent planet specified, and a planet
        with no fauna shouldn't have a fauna count greater than zero.

        Returns:
            Self if validation passes

        Raises:
            ValueError: If logical inconsistencies are found
        """
        if self.body_type == "Moon" and not self.parent_planet:
            raise ValueError("Moons must have a Parent Planet specified.")
        if self.fauna_level == "None" and self.fauna_count > 0:
            raise ValueError("Fauna Level 'None' cannot have Fauna Count > 0.")
        return self


# ------------------------------------------------------------------------------
# BLOCK 5: TEMPLATE - WIKI OUTPUT FORMATTING
# ------------------------------------------------------------------------------


# Jinja2 template for generating wiki markup
# Double curly braces are escaped with {{ '{{' }} to avoid Jinja2 interpreting them
WIKI_TEMPLATE = """{{ '{{' }}Version|{{ data.release }}{{ '}}' }}
{{ '{{' }}AGT Notice{{ '}}' }}
{{ '{{' }}Planet infobox
| name = {{ data.name }}
| image = {{ data.image or '' }}
| region = {{ data.region }}
| galaxy = {{ data.galaxy }}
| system = {{ data.system }}
| moon = {{ data.moon_count }}
| coordinates = {{ data.coordinates }}
| type = {{ data.biome }}
| description = {{ data.biome_description }}
| atmosphere = {{ data.atmosphere or '' }}
| terrain = {{ data.geo_terrain or '' }}
| weather = {{ data.weather }}
| resources = {{ resources_infobox or '' }}
| sentinel = {{ data.sentinel_level }}
| flora = {{ data.flora_level }}
| fauna = {{ data.fauna_level }}
| garden = {{ garden or '' }}
| civilized = {{ data.civilized }}
| discovered = {{ data.discovered_by }}
| discoveredlink = {{ data.discovered_by_link or '' }}
| mode = {{ data.mode }}
| platform = {{ data.platform }}
| release = {{ data.release }}
| researchteam = {{ research_team }}
}}
==Summary==
{% if data.body_type.lower() == 'planet' -%}
'''{{ data.name }}''' is a [[planet]] in the [[{{ data.system }}]] [[star system]].
It has planet ID {{ planet_id }} in this system.
{% else -%}
'''{{ data.name }}''' is a [[moon]] orbiting the planet [[{{ data.parent_planet }}]] in the [[{{ data.system }}]] [[star system]].
It has celestial bodies ID {{ planet_id }} in this system.
{%- endif %}
==Alias Names==
{{ '{{' }}aliasc|text=Original|name={{ data.original_name or data.name }}{{ '}}' }}
{{ '{{' }}aliasc|text={{ data.release }}|name={{ data.name }}{{ '}}' }}
==Discovery==
Discovered by {{ data.platform }} explorer ''{{ data.discovered_by }}'' during [[AGT Stardate]] {{ agt_stardate }} ({{ data.discovery_date.format('MMMM DD, YYYY') }}). This {{ data.body_type.lower() }} is claimed by the [[Alliance of Galactic Travellers]].
=={{ data.body_type }} Type==
This {{ data.body_type.lower() }} has a {{ '{{' }}Biome|{{ data.biome }}{{ '}}' }} [[biome]] type.
The [[atmosphere]] in daylight appears {{ data.sky_color.lower() }}.
The atmosphere composition is {{ data.atmosphere or '' }}.
===Geology===
{{ '{{' }}PlanetGeology
| terrain = {{ data.geo_terrain or '' }}
| landtype = {{ data.geo_landtype or '' }}
| archetype = {{ data.geo_archetype or '' }}
| age = {{ data.geo_age or '' }}
| geology = {{ data.geo_geology or '' }}
| core = {{ data.geo_core or '' }}
}}
===Environmental Conditions===
The surface has {{ data.weather }} [[weather]] conditions.
The weather conditions are {{ weather_condition }}.
{{ '{{' }}PlanetWeather
| tunit = {{ data.tunit or '' }}
| daytemp = {{ data.daytemp or '' }}
| nighttemp = {{ data.nighttemp or '' }}
| radnorm = {{ data.radnorm or '' }}
| toxicnorm = {{ data.toxicnorm or '' }}
| daystormtemp = {{ data.daystormtemp or '' }}
| nightstormtemp = {{ data.nightstormtemp or '' }}
| radstorm = {{ data.radstorm or '' }}
| toxicstorm = {{ data.toxicstorm or '' }}
| daycavetemp = {{ data.daycavetemp or '' }}
| nightcavetemp = {{ data.nightcavetemp or '' }}
| radcave = {{ data.radcave or '' }}
| toxiccave = {{ data.toxiccave or '' }}
| cavestormrad = {{ data.cavestormrad or '' }}
| cavestormtoxic = {{ data.cavestormtoxic or '' }}
| daywatertemp = {{ data.daywatertemp or '' }}
| nightwatertemp = {{ data.nightwatertemp or '' }}
| waterrad = {{ data.waterrad or '' }}
| watertoxic = {{ data.watertoxic or '' }}
| daystormwatertemp = {{ data.daystormwatertemp or '' }}
| nightstormwatertemp = {{ data.nightstormwatertemp or '' }}
| waterstormrad = {{ data.waterstormrad or '' }}
| waterstormtoxic = {{ data.waterstormtoxic or '' }}
}}
{%- if data.body_type.lower() == 'planet' %}
===Moons===
{% if data.moons_list -%}
This planet has {{ data.moons_list|length }} moons:
{% for moon in data.moons_list %}
* [[{{ moon }}]]
{%- endfor %}
{% else -%}
There are no [[moon]]s.
{%- endif %}
{%- endif %}
==Location==
{{ '{{' }}Gl|{{ data.glyphs }}{{ '}}' }}
==Documented Bases==
{{ '{{' }}CARGOBasesPlanet|{{ data.name }}{{ '}}' }}
==Documented Multi-Tool Sites==
The following [[Multi-Tool]]s were found in tool cabinet(s) on {{ data.name }}:
{{ '{{' }}CARGOMTPlanetShort|planet={{ data.name }}|galaxy={{ data.galaxy }}{{ '}}' }}
{%- if data.poi_list %}
==Notable Locations / Waypoints==
{| class="article-table"
! Longitude !! Latitude !! POI Type !! Area Name !! Notes !! Survey<br>Date !! Surveyor
|-
| colspan=7 style="background-color:white; padding:0.5px" |
{%- for poi in data.poi_list %}
{{ '{{' }}PlanetPOI|type={{ poi.type }}|yy={{ poi.lat }}|xx={{ poi.lon }}|name={{ poi.name }}|note={{ poi.note }}|date={{ poi.date }}|surveyor={{ data.documenter_name }}|release={{ data.release }}{{ '}}' }}
{%- endfor %}
|}
{%- endif %}
{%- if data.fauna_count > 0 %}
==Life==
===Fauna===
{{ data.fauna_count }} species of fauna are currently known to exist. The following table identifies those catalogued (which may include extinct fauna):
{{ '{{' }}CARGOFaunaPlanet|{{ data.name }}{{ '}}' }}
{%- endif %}
==Sentinels==
[[Sentinel]] activity is classified as: {{ data.sentinel_level }}.[[File:PLANETDATA.SENTINELS.png|40px|link=]]
{%- if data.resources_list or data.raw_ingredients_list %}
==Resources==
{%- if data.resources_list %}
===Primary Resources===
The following [[resource]]s can be found here:
{%- for res in data.resources_list %}
* {{ '{{' }}Resource2icon|{{ res }}{{ '}}' }} [[{{ res }}]]
{%- endfor %}
{%- endif %}
{%- if data.raw_ingredients_list %}
{% if data.resources_list %}
{% endif %}
===Raw Ingredients===
The following [[Raw Ingredient]]s can be found here:
{%- for ing in data.raw_ingredients_list %}
* {{ '{{' }}ilink|{{ ing }}{{ '}}' }}
{%- endfor %}
{%- endif %}
{%- endif %}
{%- if agt_stardate or research_team or data.documenter_name or data.economy_type or data.additional_info %}
==Additional Information==
{%- if agt_stardate %}
* Documentation based on a site survey conducted during AGT Stardate {{ agt_stardate }} ({{ data.discovery_date.format('D-MMM-YYYY') }}).
{%- endif %}
{%- if research_team %}
* Research information provided by {{ research_team }} research team.
{%- endif %}
{%- if data.documenter_name %}
* Most recent surveyor on record: ''{{ '{{' }}profile|{{ data.documenter_name }}{{ '}}' }}''
{%- endif %}
{%- if data.economy_type %}
* {{ data.economy_type }} based economy
{%- endif %}
{%- if data.additional_info %}
{{ data.additional_info }}
{%- endif %}
{%- endif %}
{%- if data.gallery_list %}
==Gallery==
<gallery>
{%- for item in data.gallery_list %}
{{ item }}
{%- endfor %}
</gallery>
{%- endif %}
==AGT Galactic Archives==
{{ '{{' }}AGT Galactic Archive Sync{{ '}}' }}
"""


# ------------------------------------------------------------------------------
# BLOCK 6: UI CONTROLLER - MAIN APPLICATION
# ------------------------------------------------------------------------------


class NMSWikiFormCreator:
    """
    The main application class that builds and manages the complete user interface.

    This class:
    1. Builds the form with tabs for different information sections
    2. Handles user interactions and button clicks
    3. Validates input data using Pydantic models
    4. Generates the final wiki text using templates
    5. Provides example data loading and form clearing

    Usage:
        >>> app = NMSWikiFormCreator()
        >>> app.display()  # Shows the form in a Jupyter notebook
    """

    def __init__(self):
        """Initializes the application with all necessary components."""
        self.data = NMSData()  # Load game data
        self.widgets = AppWidgets()  # Create widget container
        self.jinja_env = Environment(autoescape=False)  # Template engine
        self.wiki_template = self.jinja_env.from_string(WIKI_TEMPLATE)  # Load template
        self.galaxy_math = NMSGalaxyMap()  # Coordinate converter
        self.generated_content = ""  # Stores final generated wiki text

        # Build the user interface
        self._setup_ui()
        self._connect_events()
        self._set_defaults()

    def _set_defaults(self):
        """Sets initial disabled states and default values for widgets."""
        # Update biome dropdown based on initial selection
        self.on_biome_change(None)

        # Calculate initial AGT stardate from current date
        self._on_date_change(None)

        # Disable auto-generated fields (users shouldn't edit these)
        self.widgets.region.disabled = True
        self.widgets.coordinates.disabled = True
        self.widgets.agt_stardate.disabled = True

    def _create_input(self, cls, key, desc, **kwargs):
        """
        Factory method to create and style input widgets consistently.

        Args:
            cls: Widget class (Text, Dropdown, etc.)
            key: Attribute name to store widget under
            desc: Description label for the widget
            **kwargs: Additional arguments for the widget constructor

        Returns:
            Created widget instance
        """
        # Default layout: 98% width with some padding
        layout = kwargs.pop('layout', widgets.Layout(width='98%'))

        # Style: Description on left with fixed width
        style = {'description_width': '190px'}
        args = {'description': desc, 'style': style, 'layout': layout}

        # Handle dropdown and combobox options
        if 'options' in kwargs:
            opts = list(kwargs.pop('options'))
            if cls == widgets.Dropdown:
                # Dropdowns get a placeholder as first option
                ph = kwargs.get('placeholder', f"Select {desc}...")
                args['options'] = [ph] + opts
                args['value'] = ph
            elif cls == widgets.Combobox:
                # Comboboxes show suggestions as you type
                args['options'] = opts
                args['ensure_option'] = False

        # Pass through any placeholder
        if 'placeholder' in kwargs:
            args['placeholder'] = kwargs['placeholder']

        # Add any remaining keyword arguments
        args.update(kwargs)

        # Create widget and store it in the widgets container
        w = cls(**args)
        setattr(self.widgets, key, w)
        return w

    def _html(self, text, style_key="DESC"):
        """
        Creates a styled HTML label for section headers and descriptions.

        Args:
            text: HTML content to display
            style_key: Which style from CONSTANTS to use

        Returns:
            HTML widget with applied styling
        """
        return widgets.HTML(f"<div style='{CONSTANTS['STYLES'].get(style_key, '')}'>{text}</div>")

    def _row(self, *ws):
        """
        Arranges widgets in a horizontal layout (2 per row).

        Args:
            *ws: Widgets to arrange (typically 2 per row)

        Returns:
            HBox container with widgets in two columns
        """
        return widgets.HBox([widgets.VBox([w], layout=widgets.Layout(width='50%')) for w in ws],
                            layout=widgets.Layout(width='100%', margin='5px 0'))

    def _setup_ui(self):
        """Constructs the full user interface with tabs."""
        # Tab 1: Basic Information
        tab1 = widgets.VBox([
            self._html('Location Context', 'HEADER'),
            self._html("Identify the star system's location. Enter the <b>Galaxy</b> and <b>Portal Glyphs</b> first; the <b>Region Name</b> and <b>Galactic Coordinates</b> will be calculated automatically for you."),
            self._row(
                self._create_input(widgets.Combobox, 'galaxy', 'Galaxy', options=self.data.GALAXIES_LIST, placeholder='Select or Type Galaxy Name (e.g., Euclid)'),
                self._create_input(widgets.Text, 'glyphs', 'Portal Glyphs (Hexadecimal)', placeholder='e.g., 000000000000')
            ),
            self._row(
                self._create_input(widgets.Text, 'coordinates', 'Galactic Coordinates', placeholder='Auto-calculated (e.g., 0000:0000:0000:0000)', disabled=True),
                self._create_input(widgets.Text, 'region', 'Region Name', value=CONSTANTS["DEFAULTS"]["REGION_PLACEHOLDER"])
            ),
            self._row(
                self._create_input(widgets.Text, 'system', 'Star System Name', placeholder='e.g., Ocopadica'),
                widgets.VBox([], layout=widgets.Layout(width='50%'))
            ),
            self._html('Identity', 'HEADER'),
            self._html("Provide the primary details of the celestial body. If the name has changed since discovery, provide the current name here and the procedural name below."),
            self._row(
                self._create_input(widgets.Text, 'name', 'Current Name', placeholder='e.g., AGT New Lennon'),
                self._create_input(widgets.Text, 'original_name', 'Original Procedural Name', placeholder='e.g., Glinaxos Tanag')
            ),
            self._row(self._create_input(widgets.Dropdown, 'body_type', 'Celestial Body Type', options=self.data.BODY_TYPES)),
            self._html('Discovery Metadata', 'HEADER'),
            self._html("Credit the original discoverer. The <b>AGT Stardate</b> is automatically calculated based on the selected <b>Discovery Date</b>."),
            self._row(
                self._create_input(widgets.Dropdown, 'platform', 'Gaming Platform', options=self.data.PLATFORMS),
                self._create_input(widgets.DatePicker, 'discovery_date', 'Discovery Date', value=arrow.now().date())
            ),
            self._row(
                self._create_input(widgets.Text, 'discovered_by', 'Discoverer Name', placeholder='In-game username'),
                self._create_input(widgets.Text, 'discovered_by_link', 'Wiki Profile Link', placeholder='e.g., User:Traveler123')
            ),
            self._row(self._create_input(widgets.Text, 'agt_stardate', 'AGT Stardate', placeholder='(Calculated automatically)')),
            self._html('Game Information', 'HEADER'),
            self._html("Specify the game mode and version active at the time of discovery."),
            self._row(
                self._create_input(widgets.Text, 'civilized', 'Civilization / Hub', value=CONSTANTS["DEFAULTS"]["CIV"], placeholder='e.g., Galactic Hub'),
                self._create_input(widgets.Dropdown, 'mode', 'Game Mode', options=self.data.MODES)
            ),
            self._row(self._create_input(widgets.Text, 'release', 'Game Version / Update', value=CONSTANTS["DEFAULTS"]["RELEASE"], placeholder='e.g., Worlds Part I'))
        ], layout=widgets.Layout(padding='20px'))

        # Tab 2: Orbital Information
        tab2 = widgets.VBox([
            self._html('Orbital Position Details', 'HEADER'),
            self._html("Define the orbital hierarchy. If documenting a <b>Moon</b>, the <b>Parent Planet</b> name is mandatory. If documenting a <b>Planet</b>, list its moons (if any) to cross-reference them."),
            self._row(
                self._create_input(widgets.Text, 'parent_planet', 'Parent Planet Name', placeholder='Required only if Body Type is Moon'),
                self._create_input(widgets.IntText, 'moon_count', 'Moon Count', value=0)
            ),
            self._create_input(widgets.Textarea, 'moons', 'Moon Names List', placeholder='Enter one moon name per line...', layout=widgets.Layout(width='98%', height='100px'))
        ], layout=widgets.Layout(padding='20px'))

        # Tab 3: Planetary Details
        tab3 = widgets.VBox([
            self._html('Biological Classification', 'HEADER'),
            self._html("Classify the planetary biome. Select the broad <b>Biome Type</b> first to populate the specific <b>Biome Description</b> list found in the analysis visor."),
            self._row(
                self._create_input(widgets.Dropdown, 'biome', 'Biome Type', options=sorted(self.data.BIOME_DESCRIPTIONS_MAP.keys())),
                self._create_input(widgets.Dropdown, 'biome_description', 'Biome Description', placeholder='Select Biome Type first...')
            ),
            self._html('Atmosphere', 'HEADER'),
            self._html("Describe the visual and chemical makeup of the atmosphere as seen from the surface."),
            self._row(self._create_input(widgets.Dropdown, 'sky_color', 'Sky Color', options=self.data.SKY_COLOR_OPTIONS)),
            self._create_input(widgets.Textarea, 'atmosphere', 'Atmosphere Composition', placeholder='e.g., 52% Carbon Monoxide, 25% Nitrogen...', layout=widgets.Layout(width='98%', height='80px')),
            self._html('Hazards & Environment', 'HEADER'),
            self._html("Record the environmental hazards. Use the in-game text for <b>Weather</b> and <b>Sentinel</b> levels (autocomplete available)."),
            self._row(
                self._create_input(widgets.Combobox, 'weather', 'Weather Condition', options=self.data.WEATHER_OPTIONS, placeholder='Type to search weather...'),
                self._create_input(widgets.Combobox, 'sentinel_level', 'Sentinel Activity Level', options=self.data.SENTINEL_OPTIONS, placeholder='Type to search level...')
            ),
            self._html('Life Density', 'HEADER'),
            self._html("Indicate the density of life. These values are found on the planet summary screen when landing."),
            self._row(
                self._create_input(widgets.Combobox, 'flora_level', 'Flora Abundance', options=self.data.FLORA_FAUNA_OPTIONS, placeholder='Type to search abundance...'),
                self._create_input(widgets.Combobox, 'fauna_level', 'Fauna Abundance', options=self.data.FLORA_FAUNA_OPTIONS, placeholder='Type to search abundance...')
            )
        ], layout=widgets.Layout(padding='20px'))

        # Tab 4: Life & Resources
        def _cb_grid(items, storage):
            """
            Creates a grid of checkboxes (3 columns).

            Args:
                items: List of item names for checkboxes
                storage: List to store checkbox references in

            Returns:
                HBox container with checkbox grid
            """
            cbs = []
            for item in items:
                cb = widgets.Checkbox(value=False, description=item, layout=widgets.Layout(width='auto', margin='0 5px 0 0'))
                storage.append(cb)
                cbs.append(cb)
            # Split into 3 columns for better layout
            return widgets.HBox([widgets.VBox(cbs[i::3]) for i in range(3)], layout=widgets.Layout(width='98%'))

        tab4 = widgets.VBox([
            self._html('Resource Extraction', 'HEADER'),
            self._html("Select all <b>Primary Resources</b> visible in the planet info panel (e.g., Copper, Paraffinium)."),
            _cb_grid(self.data.RESOURCES, self.widgets.resource_checkboxes),
            self._html('Cooking Ingredients', 'HEADER'),
            self._html("Select all <b>Raw Ingredients</b> harvestable from flora (e.g., Star Bulb, Solanium)."),
            _cb_grid(self.data.RAW_INGREDIENTS, self.widgets.ingredient_checkboxes),
            self._html('Fauna Census', 'HEADER'),
            self._html("Enter the total number of fauna species listed in the discoveries tab."),
            self._row(self._create_input(widgets.IntText, 'fauna_count', 'Total Species Count', value=0))
        ], layout=widgets.Layout(padding='20px'))

        # Tab 5: Geology & Environment
        # Group environment fields by category for accordion layout
        env_map = {
            'Surface': [('daytemp', 'Daytime Temperature'), ('nighttemp', 'Nighttime Temperature'), ('radnorm', 'Radiation (Normal)'), ('toxicnorm', 'Toxicity (Normal)')],
            'Storm': [('daystormtemp', 'Daytime Temp (Storm)'), ('nightstormtemp', 'Nighttime Temp (Storm)'), ('radstorm', 'Radiation (Storm)'), ('toxicstorm', 'Toxicity (Storm)')],
            'Cave': [('daycavetemp', 'Cave Temp (Day)'), ('nightcavetemp', 'Cave Temp (Night)'), ('radcave', 'Radiation (Cave)'), ('toxiccave', 'Toxicity (Cave)'), ('cavestormrad', 'Cave Rad (Storm)'), ('cavestormtoxic', 'Cave Tox (Storm)')],
            'Water': [('daywatertemp', 'Water Temp (Day)'), ('nightwatertemp', 'Water Temp (Night)'), ('waterrad', 'Radiation (Water)'), ('watertoxic', 'Toxicity (Water)')],
            'Water Storm': [('daystormwatertemp', 'Water Temp (Storm Day)'), ('nightstormwatertemp', 'Water Temp (Storm Night)'), ('waterstormrad', 'Radiation (Water Storm)'), ('waterstormtoxic', 'Toxicity (Water Storm)')]
        }

        # Build accordion sections for environment fields
        acc_children, acc_titles = [], []
        for grp, fields in env_map.items():
            inputs = [self._create_input(widgets.FloatText, k, d, value=0.0) for k, d in fields]
            # Arrange in pairs (2 per row)
            pairs = [widgets.HBox(inputs[i:i + 2], layout=widgets.Layout(width='100%')) for i in range(0, len(inputs), 2)]
            acc_children.append(widgets.VBox(pairs, layout=widgets.Layout(padding='10px')))
            acc_titles.append(grp)

        # Create collapsible accordion for environment readings
        env_acc = widgets.Accordion(children=acc_children)
        for i, t in enumerate(acc_titles):
            env_acc.set_title(i, t)

        tab5 = widgets.VBox([
            self._html('Geological Analysis', 'HEADER'),
            self._html("Describe the physical terrain. Use the dropdowns to classify the land structure and geological age."),
            self._row(
                self._create_input(widgets.Dropdown, 'geo_terrain', 'Terrain Structure', options=self.data.TERRAIN_OPTIONS),
                self._create_input(widgets.Dropdown, 'geo_landtype', 'Land Type', options=self.data.LAND_TYPE_OPTIONS)
            ),
            self._row(
                self._create_input(widgets.Dropdown, 'geo_archetype', 'Archetype', options=self.data.ARCHETYPE_OPTIONS),
                self._create_input(widgets.Text, 'geo_age', 'Planetary Age (Billions)', placeholder='e.g., 3.4 Billion Years')
            ),
            self._row(
                self._create_input(widgets.Combobox, 'geo_geology', 'Geology', options=self.data.GEOLOGY_OPTIONS, placeholder='Type to search geology...'),
                self._create_input(widgets.Text, 'geo_core', 'Core Composition', placeholder='e.g., Molten Iron')
            ),
            self._html('Environmental Conditions', 'HEADER'),
            self._html("Input the specific temperature, radiation, and toxicity levels observed via the Analysis Visor. Leave fields blank if data is unavailable."),
            self._row(self._create_input(widgets.Dropdown, 'tunit', 'Temperature Unit', options=self.data.TEMP_UNITS)),
            env_acc
        ], layout=widgets.Layout(padding='20px'))

        # Tab 6: Points of Interest
        # Show available POI types for reference
        poi_help = widgets.HTML(f"<div style='{CONSTANTS['STYLES']['POI_HEADER']}'>Valid POI Types</div><div style='font-size:12px; color:#555;'>{', '.join(self.data.POI_TYPES)}</div>")

        tab6 = widgets.VBox([
            self._html('Points of Interest', 'HEADER'),
            self._html("Log notable landmarks using the CSV format below. Ensure <b>Longitude</b> and <b>Latitude</b> are accurate to ensure map placement."),
            self._create_input(widgets.Textarea, 'poi_list_csv', 'POI CSV Data', placeholder='Example:\nPortal,162.10,-19.31,Ancient Portal,,1-Oct-2025\nTrade Post,55.1,-2.4,Trading Post GBS-22,', layout=widgets.Layout(width='98%', height='200px')),
            poi_help
        ], layout=widgets.Layout(padding='20px'))

        # Tab 7: Galleries & Notes
        tab7 = widgets.VBox([
            self._html('Economy & Notes', 'HEADER'),
            self._html("Record the system economy type and surveyor details."),
            self._row(
                self._create_input(widgets.Text, 'documenter_name', 'Surveyor Name', placeholder='Name of the person documenting this'),
                self._create_input(widgets.Dropdown, 'economy_type', 'Economy Type', options=self.data.ECONOMY_OPTIONS)
            ),
            self._create_input(widgets.Textarea, 'additional_info', 'Additional Notes', placeholder='Any extra details about the planet...', layout=widgets.Layout(width='98%', height='100px')),
            self._html('Imagery', 'HEADER'),
            self._html("Provide filenames for the wiki. The <b>Main Image</b> appears in the infobox, while <b>Gallery Images</b> appear at the bottom of the page."),
            self._row(self._create_input(widgets.Text, 'image', 'Main Infobox Image Filename', placeholder='e.g., MyPlanet_Main.jpg')),
            self._create_input(widgets.Textarea, 'gallery_images', 'Gallery Images List', placeholder='File:Image1.jpg|Description\nFile:Image2.jpg|Description', layout=widgets.Layout(width='98%', height='150px'))
        ], layout=widgets.Layout(padding='20px'))

        # Tab 8: Output Generation
        # Create action buttons
        self.widgets.preview_btn = widgets.Button(description='Preview Code', button_style='info', icon='eye')
        self.widgets.generate_btn = widgets.Button(description='Generate Wiki Code', button_style='success', icon='cogs')
        self.widgets.copy_btn = widgets.Button(description='Copy to Clipboard', button_style='primary', icon='copy', disabled=True)
        self.widgets.download_btn = widgets.Button(description='Download File', button_style='primary', icon='download', disabled=True)
        self.widgets.clear_btn = widgets.Button(description='Reset Form', button_style='danger', icon='trash')
        self.widgets.load_planet_btn = widgets.Button(description='Load Example: Planet', button_style='warning', icon='globe')
        self.widgets.load_moon_btn = widgets.Button(description='Load Example: Moon', button_style='warning', icon='moon')

        # Status display and output area
        self.widgets.status_label = widgets.Label('Ready to generate.', style={'font_weight': 'bold', 'color': '#006064'})
        self.widgets.output_area = widgets.Output(layout={'border': '1px solid #ccc', 'height': '400px', 'overflow_y': 'scroll', 'padding': '10px'})

        # Arrange buttons in rows
        btns1 = widgets.HBox([self.widgets.preview_btn, self.widgets.generate_btn, self.widgets.copy_btn, self.widgets.download_btn], layout=widgets.Layout(justify_content='center'))
        btns2 = widgets.HBox([self.widgets.clear_btn, self.widgets.load_planet_btn, self.widgets.load_moon_btn], layout=widgets.Layout(justify_content='center'))

        tab8 = widgets.VBox([
            self._html('Finalization', 'HEADER'),
            self._html("Finalize the entry. Click <b>Generate</b> to create the code, then use <b>Copy</b> to paste it into the NMS Wiki."),
            btns1, btns2,
            self._html('Output', 'HEADER'),
            self.widgets.status_label,
            self.widgets.output_area
        ], layout=widgets.Layout(padding='20px'))

        # Create tab container with all tabs
        self.tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8])
        titles = ['Basic Info', 'Orbital Position', 'Details', 'Life & Resources', 'Geology & Environment', 'POIs', 'Galleries', 'Generate']
        for i, t in enumerate(titles):
            self.tabs.set_title(i, t)

        # Main container holds everything
        self.main_container = widgets.VBox([self.tabs])

    def _connect_events(self):
        """Links buttons and field changes to their handler functions."""
        # Connect field change observers
        self.widgets.discovery_date.observe(self._on_date_change, names='value')
        self.widgets.biome.observe(self.on_biome_change, names='value')
        self.widgets.galaxy.observe(self._on_location_change, names='value')
        self.widgets.glyphs.observe(self._on_location_change, names='value')

        # Connect button click handlers
        self.widgets.preview_btn.on_click(lambda b: self._generate_content("preview"))
        self.widgets.generate_btn.on_click(lambda b: self._generate_content("full"))
        self.widgets.copy_btn.on_click(self.on_copy_click)
        self.widgets.download_btn.on_click(self.on_download_click)
        self.widgets.clear_btn.on_click(self.on_clear_click)
        self.widgets.load_planet_btn.on_click(self.on_load_planet_click)
        self.widgets.load_moon_btn.on_click(self.on_load_moon_click)

    def _on_date_change(self, change):
        """
        Calculates the AGT Stardate when the discovery date is picked.

        AGT Stardate is a custom date format used by the Alliance of Galactic
        Travellers community. Formula: (Year + 1716).Day.Month

        Args:
            change: Widget change event (not used but required by observer)
        """
        try:
            val = self.widgets.discovery_date.value
            if not val:
                self.widgets.agt_stardate.value = "No date"
            else:
                d = arrow.get(val)
                # Apply AGT Stardate formula
                self.widgets.agt_stardate.value = f"{d.year + 1716}.{d.day}.{d.month:02d}"
        except:
            self.widgets.agt_stardate.value = "Error"

    def on_biome_change(self, change):
        """
        Updates the description options when a biome is selected.

        Each biome type has specific descriptions available in-game.
        This method populates the biome description dropdown based on
        the selected biome type.

        Args:
            change: Widget change event (not used but required by observer)
        """
        val = self.widgets.biome.value
        opts = self.data.BIOME_DESCRIPTIONS_MAP.get(val, [])
        self.widgets.biome_description.options = ['Select Description...'] + opts
        self.widgets.biome_description.value = 'Select Description...'

    def _on_location_change(self, change):
        """
        Triggered when Galaxy or Glyphs change.

        Calculates coordinates and region name automatically using the
        procedural generation algorithms.

        Args:
            change: Widget change event (not used but required by observer)
        """
        # Show calculation in progress
        self.widgets.region.value = "Calculating..."

        # Get and clean input values
        glyphs = self.widgets.glyphs.value.strip().upper()
        g_name = self.widgets.galaxy.value.strip()

        # Validate inputs
        if len(glyphs) != 12 or not re.fullmatch(r"^[0-9A-F]+$", glyphs) or not g_name:
            self.widgets.region.value = CONSTANTS["DEFAULTS"]["REGION_PLACEHOLDER"]
            self.widgets.coordinates.value = ""
            return

        # --- Find galaxy index from name ---
        g_idx = -1
        for idx, name in self.data.GALAXIES_MAP.items():
            if name.lower() == g_name.lower():
                g_idx = idx
                break

        if g_idx == -1:
            return

        # --- Convert glyphs to coordinates ---
        coords = self.galaxy_math.glyphs_to_coords(glyphs)
        if not coords:
            self.widgets.region.value = "Invalid Glyphs"
            self.widgets.coordinates.value = ""
            return

        # Format coordinates for display
        self.widgets.coordinates.value = f"{coords['x']:04X}:{coords['y']:04X}:{coords['z']:04X}:{coords['s']:04X}"

        # --- Generate Region Name ---
        # Convert to coordinates relative to galactic center
        rx = coords['x'] - CONSTANTS["MATH"]["CENTER"]
        ry = coords['y'] - 0x7F  # Y has different center (127 instead of 2047)
        rz = coords['z'] - CONSTANTS["MATH"]["CENTER"]

        try:
            # Create seed and generate region name
            seed = RegionNameGenerator.create_region_seed(rx, ry, rz, g_idx)
            self.widgets.region.value = RegionNameGenerator.format_name_cached(tuple(seed))
        except Exception as e:
            self.widgets.region.value = "Gen Error"

    def on_copy_click(self, b):
        """
        Copies the generated text to the clipboard using JavaScript.

        This works in Jupyter notebook environments by injecting
        JavaScript code to copy text to the system clipboard.

        Args:
            b: Button click event (not used but required)
        """
        # Escape special characters for JavaScript
        safe = self.generated_content.replace('`', '\\`').replace('\\', '\\\\')

        # JavaScript to create temporary textarea, copy from it, then remove it
        js = f"const e=document.createElement('textarea');e.value=`{safe}`;document.body.appendChild(e);e.select();document.execCommand('copy');document.body.removeChild(e);"

        # Execute JavaScript in notebook
        display(Javascript(js))
        self.widgets.status_label.value = "Copied to clipboard!"

    def on_download_click(self, b):
        """
        Downloads the content as a text file (works in Google Colab).

        Args:
            b: Button click event (not used but required)
        """
        # Create filename with timestamp
        fname = f"nms_wiki_{int(time.time())}.txt"
        try:
            # Try to use Google Colab's download function
            from google.colab import files
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(self.generated_content)
            files.download(fname)
            self.widgets.status_label.value = f"Downloaded {fname}"
        except ImportError:
            # Not in Colab - show message
            self.widgets.status_label.value = "Download available in Colab only."

    def on_clear_click(self, b):
        """
        Resets all form fields to default/empty states.

        Args:
            b: Button click event (not used but required)
        """
        # Reset each widget based on its type
        for k in self.widgets.__dataclass_fields__:
            w = getattr(self.widgets, k, None)

            if isinstance(w, (widgets.Text, widgets.Textarea, widgets.Combobox)):
                w.value = ''
            elif isinstance(w, (widgets.IntText, widgets.FloatText)):
                w.value = 0
            elif isinstance(w, widgets.Dropdown) and w.options:
                w.index = 0  # Reset to first option (placeholder)
            elif isinstance(w, widgets.DatePicker):
                w.value = arrow.now().date()
            elif isinstance(w, widgets.Checkbox):
                w.value = False

        # Reset checkbox lists
        for cb in self.widgets.resource_checkboxes + self.widgets.ingredient_checkboxes:
            cb.value = False

        # Set default values
        self.widgets.release.value = CONSTANTS["DEFAULTS"]["RELEASE"]
        self.widgets.civilized.value = CONSTANTS["DEFAULTS"]["CIV"]
        self.widgets.region.value = CONSTANTS["DEFAULTS"]["REGION_PLACEHOLDER"]

        # Clear output
        with self.widgets.output_area:
            clear_output()

        # Disable output buttons
        self.widgets.copy_btn.disabled = True
        self.widgets.download_btn.disabled = True
        self.widgets.status_label.value = "Cleared."

    def on_load_planet_click(self, b):
        """
        Populates the form with example planet data.

        Provides sample data for a planet to help users understand
        how to fill out the form.

        Args:
            b: Button click event (not used but required)
        """
        self.on_clear_click(None)

        # Example planet data
        self._load_data({
            'body_type': "Planet", 'name': "AGT Xobeurindj Fake Nexus", 'original_name': "Glinaxos Tanag",
            'system': "Eiginei", 'galaxy': "Xobeurindj", 'discovered_by': "celab99", 'discovered_by_link': "celab99",
            'glyphs': "005FF3545C3E", 'moon_count': 0, 'biome': "Toxic", 'biome_description': "Caustic",
            'sky_color': "Green", 'atmosphere': "52% Carbon Monoxide, 25% Nitrogen", 'weather': "Stinging Atmosphere",
            'sentinel_level': "Require Obedience", 'flora_level': "Abundant", 'fauna_level': "Fair", 'fauna_count': 7,
            'discovery_date': arrow.get("2025-10-01").date(), 'documenter_name': "celab99", 'economy_type': "Mercantile",
            'image': "AGT Xobeurindj Fake Nexus-(VY)-scn-01.jpg", 'geo_terrain': "Pangean", 'geo_landtype': "rocky/hilly",
            'geo_archetype': 'Rugged', 'geo_age': "3.3", 'geo_geology': "Kinematic", 'geo_core': "Calium", 'tunit': 'C',
            'nighttemp': 18.2, 'radnorm': 1.8, 'toxicnorm': 67.6, 'platform': 'PC', 'mode': 'Normal',
            'poi_list_csv': 'Portal,162.10,-19.31,Ancient Portal,,1-Oct-2025\nTrade Post,55.1,-2.4,Trading Post GBS-22,',
            'gallery_images': "File:AGT Xobeurindj Fake Nexus-(VY)-04.jpg\nFile:AGT Xobeurindj Fake Nexus-(VY)-05.jpg"
        }, res=['Silver', 'Ammonia', 'Copper', 'Fungal Mould'])

        self.widgets.status_label.value = "Planet Loaded."

    def on_load_moon_click(self, b):
        """
        Populates the form with example moon data.

        Provides sample data for a moon to help users understand
        how to fill out the form differently for moons vs planets.

        Args:
            b: Button click event (not used but required)
        """
        self.on_clear_click(None)

        # Example moon data
        self._load_data({
            'body_type': "Moon", 'name': "New Eden I", 'original_name': "Abyssal Moon", 'system': "New Eden",
            'galaxy': "Eissentam", 'discovered_by': "celab99", 'discovered_by_link': "celab99", 'glyphs': "1234567890AB",
            'parent_planet': "New Eden Prime", 'biome': "Dead", 'biome_description': "Airless", 'sky_color': "Blue",
            'atmosphere': "None", 'weather': "Utterly Still", 'sentinel_level': "Absent", 'flora_level': "None",
            'fauna_level': "None", 'fauna_count': 0, 'discovery_date': arrow.get("2022-11-25").date(),
            'documenter_name': "celab99", 'economy_type': "Mining", 'image': "NewEdenI.jpg", 'geo_terrain': "Oceanic",
            'platform': 'PC', 'mode': 'Normal', 'geo_landtype': "rocky/hilly", 'geo_archetype': "Craters", 'geo_age': "4.1",
            'geo_geology': "Bombarded", 'geo_core': "Iron", 'tunit': 'C', 'nighttemp': -120.5, 'daytemp': -110.0,
            'poi_list_csv': 'Travellers Grave,1.1,2.2,Fallen Traveller,,25-Nov-2022'
        }, res=['Silver', 'Gold', 'Cobalt'])

        self.widgets.status_label.value = "Moon Loaded."

    def _load_data(self, d, res=None, ing=None):
        """
        Helper to load data dictionary into widgets.

        Args:
            d: Dictionary mapping widget names to values
            res: List of resource names to check (for checkboxes)
            ing: List of ingredient names to check (for checkboxes)
        """
        # Set widget values from dictionary
        for k, v in d.items():
            if hasattr(self.widgets, k):
                getattr(self.widgets, k).value = v

        # Set resource checkboxes
        if res:
            for cb in self.widgets.resource_checkboxes:
                cb.value = cb.description in res

        # Set ingredient checkboxes
        if ing:
            for cb in self.widgets.ingredient_checkboxes:
                cb.value = cb.description in ing

        # Update dependent fields
        self.on_biome_change(None)
        self._on_date_change(None)

    def _collect_data(self) -> dict:
        """
        Gathers data from all widgets into a dictionary for validation.

        Converts widget values to appropriate formats, parses CSV data,
        and prepares everything for Pydantic validation.

        Returns:
            Dictionary ready for NMSFormData validation
        """
        d = {}

        # Collect data from each widget
        for k in self.widgets.__dataclass_fields__:
            # Skip non-data widgets (buttons, status displays)
            if k in ['resource_checkboxes', 'ingredient_checkboxes', 'preview_btn',
                    'generate_btn', 'copy_btn', 'download_btn', 'clear_btn',
                    'load_planet_btn', 'load_moon_btn', 'status_label', 'output_area']:
                continue

            w = getattr(self.widgets, k)
            v = w.value

            # Convert placeholder selections to empty strings
            if isinstance(v, str) and ('Select' in v or '...' in v):
                v = ""

            # Convert zero float values to None (not provided)
            if isinstance(w, widgets.FloatText) and v == 0.0:
                v = None

            d[k] = v

        # Collect checkbox values
        d['resources_list'] = [cb.description for cb in self.widgets.resource_checkboxes if cb.value]
        d['raw_ingredients_list'] = [cb.description for cb in self.widgets.ingredient_checkboxes if cb.value]

        # Parse multiline text fields into lists
        d['moons_list'] = [x.strip() for x in d.get('moons', '').split('\n') if x.strip()]
        d['gallery_list'] = [x.strip() for x in d.get('gallery_images', '').split('\n') if x.strip()]

        # Ensure glyphs are uppercase and padded if empty
        d['glyphs'] = (d.get('glyphs', '') or '0' * 12).upper()

        # --- Parse POI CSV Data ---
        pois = []
        if d['poi_list_csv'].strip():
            try:
                # Use CSV reader to handle quoted values and commas in notes
                reader = csv.reader(io.StringIO(d['poi_list_csv'].strip()), skipinitialspace=True)
                for row in reader:
                    if not row or row[0].startswith('#'):
                        continue  # Skip empty lines and comments

                    # Ensure row has exactly 6 elements
                    while len(row) < 6:
                        row.append('')

                    p_date = row[5]

                    # Try to parse and format the date
                    try:
                        if not p_date and d.get('discovery_date'):
                            # Use discovery date if no POI date provided
                            p_date = arrow.get(d['discovery_date']).format("D-MMM-YYYY")
                        else:
                            # Try multiple date formats
                            p_date = arrow.get(p_date, ["D-MMM-YYYY", "YYYY-MM-DD", "M/D/YYYY"]).format("D-MMM-YYYY")
                    except:
                        # Fall back to discovery date or original string
                        if d.get('discovery_date'):
                            p_date = arrow.get(d['discovery_date']).format("D-MMM-YYYY")
                        else:
                            p_date = row[5] or ""

                    # Create POI dictionary
                    pois.append({'type': row[0], 'lon': row[1], 'lat': row[2],
                                'name': row[3], 'note': row[4], 'date': p_date})
            except Exception as e:
                print(f"CSV Parse Error: {e}")

        d['poi_list'] = pois
        return d

    def _generate_content(self, mode):
        """
        Main generation logic. Validates data and renders the wiki template.

        Args:
            mode: "preview" shows first 80 lines, "full" shows everything
        """
        with self.widgets.output_area:
            clear_output(wait=True)
            self.widgets.status_label.value = "Validating..."

            try:
                # Step 1: Collect and validate data
                raw_data = self._collect_data()
                valid_data = NMSFormData(**raw_data)
                self.widgets.status_label.value = "Rendering..."

                # Step 2: Prepare template context
                ctx = {
                    'data': valid_data,
                    'research_team': CONSTANTS["DEFAULTS"]["CIV"],
                    'planet_id': valid_data.glyphs[0] if valid_data.glyphs else '0',
                    'weather_condition': self.data.WEATHER_CONDITION_MAP.get(valid_data.weather, 'Clear').lower(),
                    'agt_stardate': f"{valid_data.discovery_date.year + 1716}.{valid_data.discovery_date.day}.{valid_data.discovery_date.month:02d}",
                    'resources_infobox': ', '.join([f"[[{self.data.RESOURCE_MAP.get(r, r)}]]" for r in valid_data.resources_list]),
                    'garden': ''  # Currently unused but kept for template compatibility
                }

                # Step 3: Render template
                self.generated_content = self.wiki_template.render(ctx)

                # Step 4: Display output
                if mode == "preview":
                    print("--- PREVIEW (Top 80 Lines) ---")
                    print('\n'.join(self.generated_content.split('\n')[:80]))
                else:
                    print(self.generated_content)

                # Enable output buttons
                self.widgets.copy_btn.disabled = False
                self.widgets.download_btn.disabled = False
                self.widgets.status_label.value = "Success."

            except ValidationError as e:
                # Show validation errors in a user-friendly format
                self.widgets.status_label.value = "Validation Failed."
                html = "<ul style='color:red'>" + "".join([f"<li><b>{err['loc'][0]}</b>: {err['msg']}</li>" for err in e.errors()]) + "</ul>"
                display(widgets.HTML(html))
            except Exception as e:
                # Show any other errors
                self.widgets.status_label.value = "Error."
                display(widgets.HTML(f"<b style='color:red'>{str(e)}</b>"))

    def display(self):
        """Displays the complete user interface."""
        display(self.main_container)


# ------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create and display the application when run directly
    app = NMSWikiFormCreator()
    app.display()