"""
No Man's Sky Mineral Wiki Generator

This module provides a complete graphical interface for creating wiki pages for minerals
discovered in No Man's Sky. It features automated region name generation from portal glyphs,
data validation, and template-based wikitext generation.

Key components:
1. NMSData - Fetches and stores game data from external repositories
2. ByteUtils, Generator, RegionNameGenerator - Procedural generation logic for region names
3. AppWidgets - Container for all UI widgets
4. FormInputModel - Validates user input using Pydantic
5. NMSMineralWikiGenerator - Main application with tabbed interface

The application runs in Jupyter notebooks and Google Colab, providing an easy way for
players to document their mineral discoveries for the No Man's Sky wiki.
"""

import math
import re
import struct
import textwrap
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any

import arrow
import jinja2
import requests
from IPython.display import Javascript, clear_output, display
from ipywidgets import (Button, Combobox, DatePicker, Dropdown, HBox, HTML,
                        IntText, Layout, Output, Tab, Text, Textarea, VBox)
from pydantic import (BaseModel, ConfigDict, ValidationError, field_validator)
from traitlets import TraitError


class NMSData:
    """
    Fetches and stores static game data needed for the application.

    This class downloads lists of galaxies, mineral formations, analysis notes,
    and procedural generation data from external repositories. It acts as a
    central data source that other parts of the application can use.

    Attributes:
        GALAXIES (List[str]): Sorted list of all galaxy names
        FORMATION_LIST (List[str]): List of mineral formation types
        NOTES_LIST (List[str]): List of analysis note types
        ELEMENT_LIST (List[str]): List of harvestable elements
        GALAXY_INDEX_MAP (Dict[str, int]): Maps galaxy names to their numerical indices
        LETTER_MAP (Dict[int, Any]): Character weight data for name generation
        ALPHASETS (List[str]): Character sets for procedural name generation
    """

    BASE_URL = "https://raw.githubusercontent.com/2A03-Jikuu/nms-wiki-tool-py/refs/heads/main/datalist"
    MINERAL_DATA_URL = f"{BASE_URL}/mineral_data.json"
    GALAXY_DATA_URL = f"{BASE_URL}/galaxies.json"
    LETTER_MAP_URL = f"{BASE_URL}/letter_map.json"
    ALPHASETS_URL = f"{BASE_URL}/alphasets.json"

    def __init__(self):
        """Initialize empty data containers then fetch all external data."""
        # These lists will be populated from external JSON files
        self.GALAXIES: List[str] = []
        self.FORMATION_LIST: List[str] = []
        self.NOTES_LIST: List[str] = []
        self.ELEMENT_LIST: List[str] = []

        # These dictionaries store mapping data for procedural generation
        self.GALAXY_INDEX_MAP: Dict[str, int] = {}
        self.LETTER_MAP: Dict[int, Any] = {}
        self.ALPHASETS: List[str] = []

        # Load data immediately when object is created
        self._fetch_external_data()

    def _fetch_external_data(self):
        """
        Fetch JSON data from external repositories and populate the data lists.

        This method makes HTTP requests to GitHub to get updated game data.
        If any request fails, it uses fallback data to keep the application working.

        Steps:
        1. Fetch galaxy list and create name-to-index mapping
        2. Fetch mineral data (formations, notes, elements)
        3. Fetch letter map for procedural name generation
        4. Fetch alphasets for procedural name generation
        """
        print("Fetching external NMS data...", end=" ")

        # Fetch galaxy list - each galaxy has an index and name
        try:
            response = requests.get(self.GALAXY_DATA_URL)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            # Extract just the galaxy names and sort them alphabetically
            self.GALAXIES = sorted([e['name'] for e in data if 'name' in e])
            # Create a dictionary to quickly find a galaxy's index by name
            self.GALAXY_INDEX_MAP = {
                e['name']: e['index'] for e in data if 'name' in e
            }
        except Exception as e:
            # If the request fails, use a minimal fallback list
            print(f"\n[Error] Failed to load Galaxies: {e}")
            self.GALAXIES = ["Euclid", "Hilbert Dimension"]
            self.GALAXY_INDEX_MAP = {"Euclid": 0, "Hilbert Dimension": 1}

        # Fetch mineral data (formations, notes, elements)
        try:
            response = requests.get(self.MINERAL_DATA_URL)
            response.raise_for_status()
            data = response.json()
            # Sort lists alphabetically for easier user selection
            self.FORMATION_LIST = sorted(data.get("FORMATION_LIST", ["Volcanic"]))
            self.NOTES_LIST = sorted(data.get("NOTES_LIST", ["Igneous"]))
            self.ELEMENT_LIST = data.get("ELEMENT_LIST", ["None", "Carbon"])
        except Exception as e:
            print(f"\n[Error] Failed to load Mineral Data: {e}")
            self.FORMATION_LIST = ["Volcanic"]
            self.NOTES_LIST = ["Igneous"]
            self.ELEMENT_LIST = ["None"]

        # Fetch letter map for procedural name generation
        try:
            response = requests.get(self.LETTER_MAP_URL)
            response.raise_for_status()
            raw = response.json()
            # JSON keys are strings; convert to integers for our logic
            self.LETTER_MAP = {int(k): v for k, v in raw.items()}
        except Exception as e:
            print(f"\n[Error] Failed to load Letter Map: {e}")

        # Fetch alphasets for procedural name generation
        try:
            response = requests.get(self.ALPHASETS_URL)
            response.raise_for_status()
            self.ALPHASETS = response.json()
        except Exception as e:
            print(f"\n[Error] Failed to load Alphasets: {e}")
            # Create empty alphasets as fallback
            self.ALPHASETS = [""] * 8

        print("Done.")


class ByteUtils:
    """
    Emulates C++ byte operations for procedural name generation.

    This class handles low-level byte manipulation that mimics how No Man's Sky
    internally generates names. It works with bytes as lists of integers (0-255)
    and performs operations like addition, multiplication, and bit shifting.

    Attributes:
        SEED_MULTIPLIER (List[int]): Special multiplier used in seed updates
    """

    # This special multiplier is used in the seed update algorithm
    SEED_MULTIPLIER = [0x99, 0xF8, 0x76, 0x5A]

    @staticmethod
    def _unpack(fmt, arr, offset, size):
        """
        Convert bytes to a Python number using struct.unpack.

        Args:
            fmt (str): Format string for struct.unpack (like '<I' for unsigned int)
            arr (List[int]): List of byte values (0-255)
            offset (int): Starting position in the array
            size (int): Number of bytes to read

        Returns:
            The unpacked number (int or float depending on format)
        """
        # Extract the requested bytes, pad with zeros if needed
        chunk = list(arr[offset:offset + size])
        while len(chunk) < size:
            chunk.append(0)
        return struct.unpack(fmt, bytes(chunk))[0]

    @staticmethod
    def parse(val, little_endian=True):
        """
        Convert a hexadecimal string to a list of byte values.

        Args:
            val (str): Hexadecimal string like "1A2B3C"
            little_endian (bool): If True, reverse the byte order (LSB first)

        Returns:
            List[int]: List of byte values (0-255)

        Example:
            >>> ByteUtils.parse("1A2B", little_endian=True)
            [43, 26]  # 0x2B, 0x1A
        """
        # Ensure the hex string has even length by adding leading zero if needed
        if len(val) % 2 != 0:
            val = "0" + val
        # Convert each pair of hex digits to a byte
        res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]
        # Reverse bytes for little-endian format (least significant byte first)
        if little_endian:
            res.reverse()
        return res

    @staticmethod
    def format_short(op1):
        """
        Ensure a byte list is at least 2 bytes long by padding with zeros.

        Args:
            op1 (List[int]): Input byte list

        Returns:
            List[int]: Padded list with at least 2 bytes
        """
        res = list(op1)
        while len(res) < 2:
            res.append(0x00)
        return res

    @staticmethod
    def add(op1, op2):
        """
        Add two byte lists together, handling carry between bytes.

        Args:
            op1 (List[int]): First byte list (addend)
            op2 (List[int]): Second byte list (addend)

        Returns:
            List[int]: Result of addition as byte list
        """
        result = list(op2)
        for i in range(len(op1)):
            result = ByteUtils._add_single(op1[i], result, i)
        return result

    @staticmethod
    def _add_single(val, target_list, index):
        """
        Add a single byte to a specific position in a byte list, handling carry.

        Args:
            val (int): Byte value to add (0-255)
            target_list (List[int]): Byte list to modify
            index (int): Position in the list to add to

        Returns:
            List[int]: Modified byte list with carry handled
        """
        if index < len(target_list):
            # Add the values and keep only the lower 8 bits (byte)
            total = val + target_list[index]
            target_list[index] = total & 0xFF
            # Calculate carry (bits that overflowed beyond 8 bits)
            rem = (total >> 8) & 0xFF
            # If there's carry, add it to the next position
            if rem != 0:
                target_list = ByteUtils._add_single(rem, target_list, index + 1)
        else:
            # If beyond current length, append the value
            target_list.append(val)
        return target_list

    @staticmethod
    def sub(op1, op2):
        """
        Subtract op2 from op1, handling borrowing between bytes.

        Args:
            op1 (List[int]): Byte list to subtract from (minuend)
            op2 (List[int]): Byte list to subtract (subtrahend)

        Returns:
            List[int]: Result of subtraction as byte list
        """
        result = list(op2)
        for i in range(len(op1)):
            result = ByteUtils._sub_single(op1[i], result, i)
        return result

    @staticmethod
    def _sub_single(val, target_list, index):
        """
        Subtract a single byte from a specific position, handling borrow.

        Args:
            val (int): Byte value to subtract
            target_list (List[int]): Byte list to modify
            index (int): Position in the list to subtract from

        Returns:
            List[int]: Modified byte list with borrow handled
        """
        if index < len(target_list):
            diff = val - target_list[index]
            target_list[index] = diff & 0xFF
            # Calculate borrow (negative overflow)
            rem = (diff >> 8) & 0xFF
            if rem != 0:
                target_list = ByteUtils._sub_single(rem, target_list, index + 1)
        else:
            target_list.append(val)
        return target_list

    @staticmethod
    def multiply(op1, op2):
        """
        Multiply two byte lists using long multiplication algorithm.

        Args:
            op1 (List[int]): First byte list (multiplicand)
            op2 (List[int]): Second byte list (multiplier)

        Returns:
            List[int]: Product as byte list

        Note:
            This implements long multiplication where each byte of op1 is multiplied
            by each byte of op2, similar to how you multiply large numbers by hand.
        """
        result = []
        # For each byte in the first number
        for i in range(len(op1)):
            rem = 0  # Carry from previous multiplication
            # Multiply with each byte in the second number
            for j in range(len(op2)):
                # Multiply bytes and add previous carry
                raw_prod = (op1[i] * op2[j]) + rem
                # Handle signed overflow (wrap around 16-bit boundary)
                signed_prd = (raw_prod + 32768) % 65536 - 32768
                # Extract new carry (high byte) and result (low byte)
                rem = (signed_prd >> 8) & 0xFF
                res = signed_prd & 0xFF
                # Position in result array = i + j
                idx = i + j
                # Add the result byte to the appropriate position
                if idx < len(result):
                    result = ByteUtils._add_single(res, result, idx)
                else:
                    result.append(res)
            # Add any remaining carry after processing all bytes of op2
            if rem > 0:
                idx = i + len(op2)
                if idx < len(result):
                    result = ByteUtils._add_single(rem, result, idx)
                else:
                    result.append(rem)
        return result

    @staticmethod
    def shl(op1, shift):
        """
        Shift bytes left (remove bytes from the beginning).

        Args:
            op1 (List[int]): Input byte list
            shift (int): Number of bytes to shift left

        Returns:
            List[int]: Byte list with first 'shift' bytes removed

        Example:
            >>> ByteUtils.shl([1, 2, 3, 4], 2)
            [3, 4]
        """
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1, shift):
        """
        Shift bytes right (remove bytes from the end).

        Args:
            op1 (List[int]): Input byte list
            shift (int): Number of bytes to shift right

        Returns:
            List[int]: Byte list with last 'shift' bytes removed

        Example:
            >>> ByteUtils.shr([1, 2, 3, 4], 2)
            [1, 2]
        """
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1, roll):
        """
        Rotate bytes left (move bytes from beginning to end).

        Args:
            op1 (List[int]): Input byte list
            roll (int): Number of positions to rotate

        Returns:
            List[int]: Rotated byte list

        Example:
            >>> ByteUtils.rol([1, 2, 3, 4], 1)
            [2, 3, 4, 1]
        """
        if not op1:
            return op1
        # Handle roll values larger than list length
        r = roll % len(op1)
        return op1[r:] + op1[:r]

    @staticmethod
    def zxd(op1, extend):
        """
        Zero-extend a byte list to specified length.

        Args:
            op1 (List[int]): Input byte list
            extend (int): Desired total length

        Returns:
            List[int]: Extended list padded with zeros

        Example:
            >>> ByteUtils.zxd([1, 2], 4)
            [1, 2, 0, 0]
        """
        return list(op1) + [0x00] * (extend - len(op1))

    @staticmethod
    def sxd(op1, extend):
        """
        Sign-extend a byte list to specified length.

        Sign extension means: if the most significant bit of the last byte is 1
        (indicating a negative number in two's complement), pad with 0xFF bytes.
        Otherwise, pad with 0x00 bytes.

        Args:
            op1 (List[int]): Input byte list
            extend (int): Desired total length

        Returns:
            List[int]: Sign-extended byte list
        """
        result = list(op1)
        # Check if the number is negative (MSB of last byte is 1)
        val = 0xFF if (len(op1) > 0 and (op1[-1] >> 7) == 1) else 0x00
        for _ in range(extend - len(op1)):
            result.append(val)
        return result

    @staticmethod
    def logical_op(op1, op2, mode):
        """
        Perform bitwise AND, OR, or XOR on two byte lists.

        Args:
            op1 (List[int]): First byte list
            op2 (List[int]): Second byte list
            mode (int): 0 for AND, 1 for OR, 2 for XOR

        Returns:
            List[int]: Result of bitwise operation

        Note:
            The shorter list is padded with zeros to match the longer list.
        """
        l1, l2 = len(op1), len(op2)
        # Pad the shorter list with zeros to match lengths
        if l1 > l2:
            longer, shorter = list(op1), list(op2) + [0x00] * (l1 - l2)
        else:
            longer, shorter = list(op2), list(op1) + [0x00] * (l2 - l1)

        res = []
        # Perform operation byte by byte
        for i in range(len(longer)):
            if mode == 0:
                res.append(longer[i] & shorter[i])
            elif mode == 1:
                res.append(longer[i] | shorter[i])
            else:
                res.append(longer[i] ^ shorter[i])
        return res

    @staticmethod
    def xor(op1, op2):
        """
        Bitwise XOR of two byte lists.

        Args:
            op1 (List[int]): First byte list
            op2 (List[int]): Second byte list

        Returns:
            List[int]: XOR result as byte list
        """
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1, op2):
        """
        Bitwise AND of two byte lists.

        Args:
            op1 (List[int]): First byte list
            op2 (List[int]): Second byte list

        Returns:
            List[int]: AND result as byte list
        """
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1, op2):
        """
        Bitwise OR of two byte lists.

        Args:
            op1 (List[int]): First byte list
            op2 (List[int]): Second byte list

        Returns:
            List[int]: OR result as byte list
        """
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache, move=1):
        """
        Update the procedural generation seed using a special algorithm.

        This mimics how No Man's Sky updates its random number generator seed
        to produce deterministic but seemingly random names.

        Args:
            cache (List[List[int]]): Seed cache with two parts [cache0, cache1]
            move (int): Number of times to update the seed

        Returns:
            List[List[int]]: Updated seed cache
        """
        for _ in range(move):
            # Multiply cache0 by the special multiplier
            step1 = ByteUtils.multiply(cache[0], ByteUtils.SEED_MULTIPLIER)
            # Add cache1 to the result
            result = ByteUtils.add(step1, cache[1])
            # Shift and update both cache parts
            cache[0] = ByteUtils.shl(result, 4)
            cache[1] = ByteUtils.shr(result, 4)
        return cache

    @staticmethod
    def to_uint32(arr, offset=0):
        """
        Convert 4 bytes to an unsigned 32-bit integer.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            int: Unsigned 32-bit integer
        """
        return ByteUtils._unpack('<I', arr, offset, 4)

    @staticmethod
    def to_int32(arr, offset=0):
        """
        Convert 4 bytes to a signed 32-bit integer.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            int: Signed 32-bit integer
        """
        return ByteUtils._unpack('<i', arr, offset, 4)

    @staticmethod
    def to_int16(arr, offset=0):
        """
        Convert 2 bytes to a signed 16-bit integer.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            int: Signed 16-bit integer
        """
        return ByteUtils._unpack('<h', arr, offset, 2)

    @staticmethod
    def to_double(arr, offset=0):
        """
        Convert 8 bytes to a double-precision floating point number.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            float: Double-precision floating point number
        """
        return ByteUtils._unpack('<d', arr, offset, 8)

    @staticmethod
    def to_single(arr, offset=0):
        """
        Convert 4 bytes to a single-precision floating point number.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            float: Single-precision floating point number
        """
        return ByteUtils._unpack('<f', arr, offset, 4)

    @staticmethod
    def get_bytes_uint32(val):
        """
        Convert an unsigned 32-bit integer to 4 bytes.

        Args:
            val (int): Unsigned 32-bit integer

        Returns:
            List[int]: 4-byte representation of the integer
        """
        return list(struct.pack('<I', val))


class StringExtensions:
    """Helper methods for string manipulation in procedural generation."""

    @staticmethod
    def short_to_formatted_hex(val, trunc):
        """
        Convert a number to hexadecimal and truncate to specified length.

        Args:
            val (int): Number to convert (will be masked to 16 bits)
            trunc (int): Number of hex digits to keep from the end

        Returns:
            str: Truncated hexadecimal string

        Example:
            >>> StringExtensions.short_to_formatted_hex(0x1234, 2)
            '34'
        """
        # Mask to 16 bits to ensure we only work with 2 bytes
        val = val & 0xFFFF
        # Convert to 4-digit hexadecimal with leading zeros
        hex_str = f"{val:04X}"
        # Return only the last 'trunc' digits
        return hex_str[-trunc:]


class Generator:
    """
    Core procedural name generator for No Man's Sky region names.

    This class implements the algorithm that No Man's Sky uses to generate
    region names from coordinates and galaxy indices. It uses weighted
    character transitions to create pronounceable, sci-fi sounding names.

    Attributes:
        TINY_DOUBLE (List[int]): Small double value used in probability calculations
        MAX_BACKTRACK_ATTEMPTS (int): Safety limit to prevent infinite loops
        MAX_NAME_LENGTH (int): Maximum allowed length for generated names
        VOWELS (str): Standard vowels used for syllable checking
        VOWELS_WITH_Y (str): Vowels including 'y' for consonant run detection
    """

    # A very small double value (approximately 0.000244140625) used as a multiplier
    TINY_DOUBLE = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]
    # Safety limit to prevent infinite loops during name generation
    MAX_BACKTRACK_ATTEMPTS = 50
    # No Man's Sky has a maximum name length for regions
    MAX_NAME_LENGTH = 64
    # Standard vowels for syllable structure
    VOWELS = "aeiou"
    # Vowels including 'y' for detecting long consonant runs
    VOWELS_WITH_Y = "aeiouy"

    @staticmethod
    def generate_name(cache0, cache1, data_ref: NMSData):
        """
        Generate a procedural name using the current seed state.

        This is the main name generation algorithm that:
        1. Gets initial characters from the alphaset
        2. Builds the name using weighted character transitions
        3. Applies linguistic rules to improve readability
        4. Adds optional adornments (like "Expanse" or "Nebula")

        Args:
            cache0 (List[List[int]]): First part of the seed cache
            cache1 (List[List[int]]): Second part of the seed cache
            data_ref (NMSData): Reference to game data for alphasets and letter maps

        Returns:
            str: Generated region name, or empty string if generation failed
        """
        # Step 1: Get initial 3-character seed from alphaset
        name = Generator.get_characters_from_alphaset(cache0, cache1, data_ref)
        if name == "__EMPTY__":
            return ""

        # Update seed for next operation
        ByteUtils.update_seed(cache0)
        # Decide which character selection method to use
        check_op = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        alternate_char_getter = (ByteUtils.to_int16(check_op) != 0)
        ByteUtils.update_seed(cache0)

        # Calculate how many additional characters to generate
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])
        limit = ByteUtils.to_int16(ByteUtils.sxd(register0, 2))

        # Generate additional characters using weighted transitions
        if 0 < limit:
            i = 0
            safety = 0
            while i < limit:
                ByteUtils.update_seed(cache0)
                # Look at last 3 characters to determine next character probabilities
                sub_str = name[i: i + 3]
                alphaset_idx = cache1[0][0] if cache1[0] else 0
                char_weights = Generator.get_string_weights(
                    sub_str, alphaset_idx, data_ref
                )

                # Generate random value for character selection
                val_u32 = ByteUtils.to_uint32(cache0[0])
                tiny_dbl = ByteUtils.to_double(Generator.TINY_DOUBLE)
                target = float(val_u32 * tiny_dbl)

                if char_weights is None:
                    # No valid transitions - backtrack and try again
                    i -= 1
                    safety += 1
                    if safety > Generator.MAX_BACKTRACK_ATTEMPTS:
                        break
                else:
                    safety = 0
                    index = 0
                    if alternate_char_getter:
                        # Alternate selection method using floating point math
                        target *= (len(char_weights) - 1)
                        b_tgt = list(struct.pack('<f', target))
                        op_and = ByteUtils.and_op(b_tgt, [0x00, 0x00, 0x00, 0x80])
                        op = ByteUtils.or_op(op_and, [0x00, 0x00, 0x00, 0x3F])
                        index = int(ByteUtils.to_single(op) + target)
                    else:
                        # Standard weighted random selection
                        weight = 0.0
                        j = 0
                        for cw in char_weights:
                            weight += cw[1]
                            if weight >= target:
                                break
                            j += 1
                        index = j
                    # Add selected character to name
                    if index < len(char_weights):
                        name += char_weights[index][0]
                # Enforce maximum name length
                if len(name) >= Generator.MAX_NAME_LENGTH:
                    name = name[:Generator.MAX_NAME_LENGTH]
                i += 1

        if not name:
            return ""
        if len(name) < 2:
            return name

        # Apply linguistic rules to improve name readability

        # Rule 1: Ensure name doesn't start with two consecutive consonants
        first, second = name[0], name[1]
        if (first not in Generator.VOWELS) and (second not in Generator.VOWELS):
            # Exception: 's' followed by certain consonants is allowed
            cond1 = first != 's' or second not in "hklmnprtwy"
            if cond1:
                name = Generator.insert_vowel(name, cache0, 1)

        # Rule 2: Fix certain consonant clusters at the end of names
        ult, penult = name[-1], name[-2]
        if len(name) > 1 and (penult != 'g' or ult in Generator.VOWELS):
            c1 = (ult == 'b' and penult in "gn")
            c2 = (ult == 'd' and penult in "bdfghkmpst")
            if c1 or c2:
                name = Generator.insert_vowel(name, cache0, len(name) - 1)

        # Rule 3: Break up long runs of consonants
        consonance = Generator.get_consecutive_consonants(name)
        if consonance != -1:
            ByteUtils.update_seed(cache0)
            # Randomly choose where to insert a vowel
            mult = ByteUtils.multiply(cache0[0], [0x03])
            shr = ByteUtils.shr(mult, 4)
            add = ByteUtils.add(shr, [0x01])
            offset = ByteUtils.to_int32(ByteUtils.zxd(add, 4))
            name = Generator.insert_vowel(name, cache0, consonance + offset)

        return name

    @staticmethod
    def get_characters_from_alphaset(cache0, cache1, data_ref: NMSData):
        """
        Get initial 3-character seed from the alphaset.

        Alphasets are pre-defined character sets that vary by galaxy.
        This method selects a random 3-character substring from the alphaset.

        Args:
            cache0 (List[List[int]]): Seed cache for randomness
            cache1 (List[List[int]]): Contains alphaset index
            data_ref (NMSData): Reference to alphasets data

        Returns:
            str: 3-character seed, or "__EMPTY__" if no alphaset available
        """
        ByteUtils.update_seed(cache0)
        # Get which alphaset to use (based on galaxy index)
        idx = max(0, cache1[0][0]) if cache1[0] else 0
        if idx >= len(data_ref.ALPHASETS):
            idx = 0
        alphaset_str = data_ref.ALPHASETS[idx]
        if not alphaset_str:
            return "__EMPTY__"

        # Calculate random position within the alphaset
        length_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        register0 = ByteUtils.multiply(cache0[0], length_bytes)
        shr_reg = ByteUtils.shr(register0, 4)
        register1 = ByteUtils.format_short(ByteUtils.multiply(shr_reg, [0x03]))

        # Extract 3-character substring
        start = ByteUtils.to_int16(register1)
        end = ByteUtils.to_int16(ByteUtils.add(register1, [0x03]))
        return alphaset_str[start:end]

    @staticmethod
    def get_string_weights(substr, alphaset, data_ref: NMSData):
        """
        Get weighted character transitions for a given substring.

        The letter map contains probabilities for which character should follow
        a given sequence of characters, creating natural-sounding names.

        Args:
            substr (str): Current character sequence (up to 3 chars)
            alphaset (int): Which alphaset (galaxy) to use
            data_ref (NMSData): Reference to letter map data

        Returns:
            List[Tuple[str, float]]: List of (character, weight) pairs,
            or None if no transitions found
        """
        if not data_ref.LETTER_MAP or alphaset not in data_ref.LETTER_MAP:
            return None
        subset = data_ref.LETTER_MAP[alphaset]
        if not substr or substr[0] not in subset:
            return None
        return Generator.recursive_search(subset[substr[0]], substr)

    @staticmethod
    def recursive_search(arr, substr):
        """
        Recursively search the letter map tree for character weights.

        The letter map is organized as a tree where each node contains:
        - A comparison value
        - A type code ("ja" for jump-if-above, "jz" for jump-if-zero)
        - Child nodes or weight lists

        Args:
            arr (List): Tree structure from letter map
            substr (str): Character sequence to search for

        Returns:
            List[Tuple[str, float]]: Weighted character transitions,
            or None if not found
        """
        result, i = None, 0
        # Search through array until found or exhausted
        while result is None and i < len(arr):
            item = arr[i]
            if len(item) > 2:
                type_code, val = item[2], item[0]
                if type_code == "ja":
                    # Jump if above: continue search in child nodes
                    s_bytes = ByteUtils.zxd(list(substr.encode('utf-8')), 4)
                    val_b = ByteUtils.zxd(list(str(val).encode('utf-8')), 4)
                    if ByteUtils.to_int32(s_bytes) > ByteUtils.to_int32(val_b):
                        result = Generator.recursive_search(item[1], substr)
                elif type_code == "jz" and str(val) == substr:
                    # Found exact match: return weight list
                    weights = [
                        (w.get("Item1"), float(w.get("Item2", 0)))
                        for w in item[1]
                    ]
                    return weights
            i += 1
        return result

    @staticmethod
    def insert_vowel(name, seed, index):
        """
        Insert a vowel at a specific position in the name.

        Args:
            name (str): Current name
            seed (List[List[int]]): Seed for random vowel selection
            index (int): Position to insert vowel

        Returns:
            str: Name with vowel inserted
        """
        ByteUtils.update_seed(seed)
        # Choose which vowel to insert (a, e, i, o, or u)
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)
        if calc and calc[0] < 5:
            if index <= len(name):
                return name[:index] + Generator.VOWELS[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name):
        """
        Find position where too many consecutive consonants occur.

        Args:
            name (str): Name to check

        Returns:
            int: Position where a 4-consonant run starts, or -1 if none found

        Note:
            'y' is counted as a vowel in this check (using VOWELS_WITH_Y).
        """
        consonance = 0
        for i in range(len(name)):
            if consonance < 3:
                # Count consecutive consonants
                if name[i] not in Generator.VOWELS:
                    consonance += 1
                else:
                    consonance = 0
            else:
                # Found 3 consonants in a row, check if next is also consonant
                if name[i] not in Generator.VOWELS_WITH_Y:
                    # Found 4+ consonants, return start position of the run
                    return i - 3
                else:
                    consonance = 0
        return -1


class RegionNameGenerator:
    """
    Generates region names from galactic coordinates.

    This class converts (x, y, z) coordinates and galaxy index into
    a deterministic but seemingly random region name using the
    procedural generation algorithm.

    Attributes:
        PROC_ADORNMENTS (List[str]): Suffixes like "Expanse" or "Nebula"
        FALLBACK_NAME (str): Name used when generation fails
        SCRAMBLE_MULT_1 (List[int]): First scrambling multiplier
        SCRAMBLE_MULT_2 (List[int]): Second scrambling multiplier
    """

    # These adornments are randomly appended to some region names
    PROC_ADORNMENTS = [
        "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
        "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
        "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
        "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux",
        "%NAME% Instability", "Sea of %NAME%", "The Arm of %NAME%",
        "%NAME% Spur", "%NAME% Shallows"
    ]
    # Default name if generation fails
    FALLBACK_NAME = "Unknown Region"
    # Constants used in the seed scrambling algorithm
    SCRAMBLE_MULT_1 = [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64]
    SCRAMBLE_MULT_2 = [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3]

    @staticmethod
    def create_region_seed(x, y, z, galaxy):
        """
        Create a seed value from coordinates and galaxy index.

        Args:
            x (int): X coordinate in voxel space
            y (int): Y coordinate in voxel space
            z (int): Z coordinate in voxel space
            galaxy (int): Galaxy index (0 for Euclid, 1 for Hilbert, etc.)

        Returns:
            List[int]: 8-byte seed for name generation
        """
        # Convert each component to hex with specific formatting
        s_gal = StringExtensions.short_to_formatted_hex(galaxy, 2)
        s_y = StringExtensions.short_to_formatted_hex(y, 2)
        s_z = StringExtensions.short_to_formatted_hex(z, 3)
        s_x = StringExtensions.short_to_formatted_hex(x, 3)
        # Combine into single hex string and parse as bytes
        hex_str = s_gal + s_y + s_z + s_x
        return ByteUtils.parse(hex_str)

    @staticmethod
    def format_name(seed, data_ref: NMSData):
        """
        Generate a region name from a seed value.

        This is the main entry point for region name generation.
        It sets up the seed cache, runs the generation algorithm,
        and optionally adds adornments.

        Args:
            seed (List[int]): 8-byte seed from create_region_seed
            data_ref (NMSData): Reference to game data

        Returns:
            str: Generated region name
        """
        # Initialize seed cache with default values
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # Scramble the seed using multiple operations to ensure good randomness
        register0 = ByteUtils.shr(seed, 4)
        if register0:
            register0[0] //= 2
        xor_res = ByteUtils.xor(register0, seed)
        register0 = ByteUtils.multiply(xor_res, RegionNameGenerator.SCRAMBLE_MULT_1)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)
        register0 = ByteUtils.multiply(xor2, RegionNameGenerator.SCRAMBLE_MULT_2)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        register0 = ByteUtils.xor(
            ByteUtils.get_bytes_uint32(val_u32), register0
        )
        # Set up the seed cache for generation
        shl4 = ByteUtils.shl(register0, 4)
        xor_mid = ByteUtils.xor(
            ByteUtils.rol(shl4, 2), ByteUtils.shr(register0, 4)
        )
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure cache0 is not zero (would cause generation issues)
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])

        # Update seed and calculate name length parameters
        ByteUtils.update_seed(cache0)
        calc_len = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4)
        cache1[2] = ByteUtils.add(calc_len, [0x06])

        # Generate the base name
        name = Generator.generate_name(cache0, cache1, data_ref)
        if not name or "[" in name:  # "[" indicates generation error
            return RegionNameGenerator.FALLBACK_NAME
        # Capitalize first letter
        name = name[0].upper() + name[1:]

        # 50% chance to add an adornment (like "Expanse" or "Nebula")
        ByteUtils.update_seed(cache0)
        mult_check = ByteUtils.multiply(cache0[0], [0x64])
        should_adorn = ByteUtils.shr(mult_check, 4)[0] < 0x50  # 0x50 = 80 in decimal = 80% of 100

        if should_adorn:
            ByteUtils.update_seed(cache0)
            # Randomly select which adornment to use
            idx_cal = ByteUtils.multiply(cache0[0], [0x14])  # 0x14 = 20 in decimal
            idx = ByteUtils.shr(idx_cal, 4)[0]
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                adornment = RegionNameGenerator.PROC_ADORNMENTS[idx]
                name = adornment.replace("%NAME%", name)

        return name


class NMSGalaxyMap:
    """
    Handles coordinate conversions for No Man's Sky.

    This class converts between different coordinate systems:
    - Portal glyphs (12-digit hex)
    - Voxel coordinates (used internally by the game)
    - Region-relative coordinates (used for name generation)

    Attributes:
        SHIFT_POS_XZ (int): Wrap boundary for X and Z coordinates (positive side)
        SHIFT_NEG_XZ (int): Wrap boundary for X and Z coordinates (negative side)
        SHIFT_POS_Y (int): Wrap boundary for Y coordinates (positive side)
        SHIFT_NEG_Y (int): Wrap boundary for Y coordinates (negative side)
        REGION_CENTER_XZ (int): Center of region in X/Z dimensions
        REGION_CENTER_Y (int): Center of region in Y dimension
    """

    # These values define the coordinate wrapping boundaries
    SHIFT_POS_XZ = 2049
    SHIFT_NEG_XZ = 2047
    SHIFT_POS_Y = 129
    SHIFT_NEG_Y = 127
    # Center points used to convert to region-relative coordinates
    REGION_CENTER_XZ = 0x7FF  # 2047 in decimal
    REGION_CENTER_Y = 0x7F    # 127 in decimal

    @staticmethod
    def _wrap_coordinate(value, shift_pos, shift_neg):
        """
        Wrap a coordinate around the galactic boundaries.

        No Man's Sky uses a wrapped coordinate system where coordinates
        beyond certain bounds wrap around to the opposite side.

        Args:
            value (int): Coordinate value
            shift_pos (int): Positive wrap boundary
            shift_neg (int): Negative wrap boundary

        Returns:
            int: Wrapped coordinate
        """
        return value - shift_pos if value >= shift_pos else value + shift_neg

    def glyphs_to_coords(self, glyphs: str):
        """
        Convert portal glyphs to voxel coordinates.

        Portal glyphs are 12 hexadecimal digits representing:
        - First 4 digits: Planet index and solar system info
        - Next 2 digits: Y coordinate
        - Next 3 digits: Z coordinate
        - Last 3 digits: X coordinate

        Args:
            glyphs (str): 12-character hexadecimal string

        Returns:
            Optional[Dict[str, int]]: Dictionary with 'x', 'y', 'z' keys,
            or None if glyphs are invalid
        """
        g = glyphs.strip().upper()
        if len(g) != 12:
            return None
        try:
            # Extract coordinate components from glyph string
            # Format: [P][SSS][YY][ZZZ][XXX]
            y_hex = int(g[4:6], 16)
            z_hex = int(g[6:9], 16)
            x_hex = int(g[9:12], 16)
        except ValueError:
            return None

        # Validate coordinate ranges
        if not (0x00 <= y_hex <= 0xFF and 0x000 <= z_hex <= 0xFFF and 0x000 <= x_hex <= 0xFFF):
            return None

        # Wrap coordinates to handle galactic boundaries
        voxel_x = NMSGalaxyMap._wrap_coordinate(x_hex, self.SHIFT_POS_XZ, self.SHIFT_NEG_XZ)
        voxel_z = NMSGalaxyMap._wrap_coordinate(z_hex, self.SHIFT_POS_XZ, self.SHIFT_NEG_XZ)
        voxel_y = NMSGalaxyMap._wrap_coordinate(y_hex, self.SHIFT_POS_Y, self.SHIFT_NEG_Y)

        return {'x': voxel_x, 'y': voxel_y, 'z': voxel_z}


@dataclass
class AppWidgets:
    """
    Container for all UI widgets used in the application.

    Using a dataclass ensures all widgets are properly typed and organized.
    Each field corresponds to a specific input field in the UI.

    Fields:
        name: Current mineral name
        original_name: Original discovered name (optional)
        discoverer: Player who discovered the mineral
        discoverer_link: Wiki username of discoverer (optional)
        discovery_date: Date of discovery
        agt_stardate: Calculated AGT stardate (auto-generated)
        civilized: Civilization the discoverer belongs to
        release: Game version when discovered
        galaxy: Galaxy where mineral is found
        region: Region name (auto-calculated from glyphs)
        system: Star system name
        planet: Planet name
        moon: Moon name (optional, blank if on planet)
        glyphs: Portal glyphs (12 hex digits)
        formation: Mineral formation type
        content: Metal content percentage
        notes: Analysis notes
        polymorphic: Number of variants (1=normal, 2=dimorphic, >2=polymorphic)
        element_primary: Primary harvested element
        element_secondary: Secondary harvested element (optional)
        image: Main image filename
        gallery: Gallery image filenames with descriptions
    """

    # All fields are initialized to None and will be set during UI creation
    name: Text = field(init=False)
    original_name: Text = field(init=False)
    discoverer: Text = field(init=False)
    discoverer_link: Text = field(init=False)
    discovery_date: DatePicker = field(init=False)
    agt_stardate: Text = field(init=False)
    civilized: Text = field(init=False)
    release: Text = field(init=False)
    galaxy: Combobox = field(init=False)
    region: Text = field(init=False)
    system: Text = field(init=False)
    planet: Text = field(init=False)
    moon: Text = field(init=False)
    glyphs: Text = field(init=False)
    formation: Combobox = field(init=False)
    content: Text = field(init=False)
    notes: Combobox = field(init=False)
    polymorphic: IntText = field(init=False)
    element_primary: Dropdown = field(init=False)
    element_secondary: Dropdown = field(init=False)
    image: Text = field(init=False)
    gallery: Textarea = field(init=False)


class FormInputModel(BaseModel):
    """
    Validates user input using Pydantic's data validation.

    This model ensures all required fields are present and valid
    before generating wikitext. It automatically converts dates
    and validates format constraints.

    Attributes:
        name: Current mineral name (required, non-empty)
        original_name: Original name when discovered (optional)
        discoverer: Discoverer's in-game name (required)
        discoverer_link: Discoverer's wiki username (optional)
        discovery_date: Date of discovery as arrow object
        civilized: Civilization name (required)
        release: Game version (required)
        galaxy: Galaxy name (required)
        region: Region name (required)
        system: Star system name (required)
        planet: Planet name (required)
        moon: Moon name (optional)
        glyphs: Portal glyphs (required, 12 hex digits)
        formation: Formation type (required)
        content: Metal content (required)
        notes: Analysis notes (required)
        polymorphic: Variant count (integer, 1 or more)
        element_primary: Primary element (optional)
        element_secondary: Secondary element (optional)
        image: Main image filename (required)
        gallery: Gallery images (optional)
    """

    name: str
    original_name: Optional[str] = ""
    discoverer: str
    discoverer_link: Optional[str] = ""
    discovery_date: arrow.Arrow
    civilized: str
    release: str
    galaxy: str
    region: str
    system: str
    planet: str
    moon: Optional[str] = ""
    glyphs: str
    formation: str
    content: str
    notes: str
    polymorphic: int
    element_primary: Optional[str] = ""
    element_secondary: Optional[str] = ""
    image: str
    gallery: Optional[str] = ""

    @field_validator('discovery_date', mode='before')
    @classmethod
    def convert_date_to_arrow(cls, v):
        """
        Convert various date formats to arrow objects.

        Args:
            v: Date value (could be string, date, datetime, or arrow object)

        Returns:
            arrow.Arrow: Standardized date object

        Raises:
            ValueError: If date cannot be parsed
        """
        if v is not None:
            return arrow.get(v)
        return v

    @field_validator(
        'name', 'discoverer', 'galaxy', 'region', 'system',
        'planet', 'formation', 'content', 'notes', 'image'
    )
    @classmethod
    def field_must_not_be_empty(cls, v: str) -> str:
        """
        Validate that required fields are not empty or just whitespace.

        Args:
            v (str): Field value to validate

        Returns:
            str: Trimmed value if valid

        Raises:
            ValueError: If field is empty or whitespace only
        """
        if not v or v.isspace():
            raise ValueError('This field is required and cannot be empty.')
        return v

    @field_validator('glyphs')
    @classmethod
    def validate_glyphs(cls, v: str) -> str:
        """
        Validate portal glyphs format.

        Glyphs must be exactly 12 hexadecimal characters (0-9, A-F).

        Args:
            v (str): Glyph string to validate

        Returns:
            str: Uppercase glyph string if valid

        Raises:
            ValueError: If glyphs are not 12 hex digits
        """
        if not re.fullmatch(r'^[0-9A-Fa-f]{12}$', v):
            raise ValueError('Glyphs must be a 12-character hexadecimal string.')
        return v.upper()

    # Allow arrow.Arrow objects in the model
    model_config = ConfigDict(arbitrary_types_allowed=True)


class NMSMineralWikiGenerator:
    """
    Main application class for the No Man's Sky Mineral Wiki Generator.

    This class creates an interactive tabbed interface using ipywidgets.
    Users can fill in mineral discovery information, and the application
    automatically generates proper wikitext for the No Man's Sky wiki.

    Key features:
    - Automated region name calculation from portal glyphs
    - Real-time validation with helpful error messages
    - Template-based wikitext generation
    - Example loading and form reset
    - Copy to clipboard and download functionality

    Attributes:
        generated_content (str): Last generated wikitext
        data (NMSData): Game data container
        map_logic (NMSGalaxyMap): Coordinate conversion logic
        widgets (AppWidgets): All UI widgets
        template (jinja2.Template): Wikitext template
        DEFAULT_RELEASE (str): Default game version
    """

    # Wiki template with placeholders that will be filled with user data
    WIKI_TEMPLATE = textwrap.dedent("""\
        {{ '{{' }}Version|{{ release }}{{ '}}' }}
        {{ '{{' }}AGT Notice{{ '}}' }}
        {{ '{{' }}Mineral infobox
        | name = {{ name }}
        | image = {{ image }}
        | galaxy = {{ galaxy }}
        | region = {{ region }}
        | system = {{ system }}
        | planet = {{ planet }}
        | moon = {{ moon }}
        | content = {{ content }}
        | formation = {{ formation }}
        | notes = {{ notes }}
        | element_primary = {{ element_primary }}
        | element_secondary = {{ element_secondary }}
        | polymorphic = {{ polymorphic }}
        | civilized = {{ civilized }}
        | discovered = {{ discoverer }}
        | discoveredlink = {{ discoverer_link }}
        | discovered_on = {{ discovery_date_long }}
        | mode = Normal
        | researchteam = Alliance of Galactic Travellers
        | release = {{ release }}
        {{ '}}' }}
        '''{{ name }}''' is a variety of mineral.

        ==Summary==
        '''{{ name }}''' is a [[type]] of [[mineral]].

        ==Alias Names==
        {{ '{{' }}aliasc|text=Original|name={{ original_name_display }}{{ '}}' }}
        {{ '{{' }}aliasc|text=Current|name={{ name }}{{ '}}' }}

        ==Discovery Menu==
        * Metal Content: {{ content }}
        * Formation Process: {{ formation }}
        * Notes: {{ notes }}

        ==Location==
        It can be found {{ location_link_text }}
        {{ '{{' }}CoordGlyphConvert|{{ glyphs }}{{ '}}' }}

        ==Resources==
        {{ resource_text }}

        ==Additional Information==
        * Discovered {{ discovery_date_short }}. (AGT Stardate {{ agt_stardate }})
        * Research contributed by the Alliance of Galactic Travellers research team.

        ==Gallery==
        <gallery>
        {{ gallery_content }}
        </gallery>

        ==AGT Galactic Archives==
        {{ '{{' }}AGT Galactic Archive Sync{{ '}}' }}""")

    # Default game version for new discoveries
    DEFAULT_RELEASE = "Breach"

    def __init__(self):
        """Initialize the application, load data, and create UI."""
        self.generated_content = ""
        self.data = NMSData()
        self.map_logic = NMSGalaxyMap()  # Initialize coordinate logic
        self.widgets = AppWidgets()
        self.template = jinja2.Template(self.WIKI_TEMPLATE)

        # Set up UI components
        self._define_styles_and_layouts()
        self._setup_ui()
        self._connect_events()
        self._update_stardate_ui(None)

    def _define_styles_and_layouts(self):
        """Define CSS styles and widget layouts for consistent UI appearance."""
        # Style for section headers
        self.HEADER_STYLE = (
            "font-weight:bold; font-size:16px; margin-top:15px; "
            "border-bottom:2px solid #00ACC1; padding-bottom:5px; color:#006064;"
        )
        # Style for descriptive text below headers
        self.DESC_STYLE = (
            "font-style:italic; font-size:12px; color:#555; "
            "margin-bottom:12px; line-height:1.4em; background-color:#E0F7FA; "
            "padding:8px; border-left:4px solid #00BCD4; border-radius:4px;"
        )
        # Consistent label width for all widgets
        self.LABEL_STYLE = {'description_width': '140px'}
        # Standard width for most input widgets
        self.WIDGET_LAYOUT = Layout(width='450px')
        # Layout for text areas (like notes)
        self.TEXTAREA_LAYOUT = Layout(width='98%', height='100px')
        # Layout for gallery text area (taller)
        self.GALLERY_LAYOUT = Layout(width='98%', height='150px')
        # Layout for columns in two-column rows
        self.COL_LAYOUT = Layout(width='50%', min_width='480px')
        # Layout for full-width rows
        self.FULL_ROW_LAYOUT = Layout(width='100%', margin='5px 0')

    def _setup_ui(self):
        """Construct the main tabbed interface with all five tabs."""
        # Create each tab's content
        tab1 = self._create_tab_identity()
        tab2 = self._create_tab_location()
        tab3 = self._create_tab_geology()
        tab4 = self._create_tab_resources()
        tab5 = self._create_tab_generate()

        # Create the tab container and set tab titles
        self.tabs = Tab(children=[tab1, tab2, tab3, tab4, tab5])
        headers = ['Identity', 'Location', 'Geology', 'Resources', 'Generate']
        for i, h in enumerate(headers):
            self.tabs.set_title(i, h)
        # Display the tabbed interface
        display(self.tabs)

    def _create_tab_identity(self):
        """Create the Identity tab with mineral name and discovery info."""
        return VBox([
            self._header('Mineral Identity'),
            self._desc("Provide the official name for the mineral."),
            self._row(
                self._create_widget(Text, 'name', 'Current Name:', placeholder='e.g., M. Silicate Prime'),
                self._create_widget(Text, 'original_name', 'Original Name:', placeholder='e.g., Lopheyski XIV (Optional)')
            ),
            self._header('Discovery Information'),
            self._desc("Enter your in-game discoverer name."),
            self._row(
                self._create_widget(Text, 'discoverer', 'Discoverer Alias:', placeholder='Your In-Game Username'),
                self._create_widget(Text, 'discoverer_link', 'Wiki Username:', placeholder='Your Wiki User Name (Optional)')
            ),
            self._row(
                self._create_widget(DatePicker, 'discovery_date', 'Discovery Date:', value=arrow.now().date()),
                self._create_widget(Text, 'agt_stardate', 'AGT Stardate:', disabled=True)
            ),
            self._row(
                self._create_widget(Text, 'civilized', 'Civilization:', value='Alliance of Galactic Travellers'),
                self._create_widget(Text, 'release', 'Game Version:', value=self.DEFAULT_RELEASE)
            )
        ])

    def _create_tab_location(self):
        """Create the Location tab with galactic coordinates and auto-region calculation."""
        return VBox([
            self._header('Galactic Coordinates'),
            self._desc("Enter the Portal Glyphs to automatically calculate the Region name."),
            self._row(
                self._create_widget(Combobox, 'galaxy', 'Galaxy:', options=self.data.GALAXIES, placeholder='Select or type galaxy name...'),
                self._create_widget(Text, 'glyphs', 'Portal Glyphs:', placeholder='e.g. 0801F9801802 (12 hex digits, 0-F)')
            ),
            self._row(
                self._create_widget(Text, 'system', 'Star System:', placeholder='e.g., Oishida'),
                self._create_widget(Text, 'region', 'Region:', placeholder='(Auto-calculated from Glyphs)', disabled=True)
            ),
            self._row(
                self._create_widget(Text, 'planet', 'Planet:', placeholder='e.g., New Lennon'),
                self._create_widget(Text, 'moon', 'Moon:', placeholder='Leave blank if found on a planet')
            )
        ])

    def _create_tab_geology(self):
        """Create the Geology tab with mineral analysis information."""
        # Create the polymorphic variant widget with helper text
        variant_widget = self._create_widget(IntText, 'polymorphic', 'Variant Count:', value=1)
        right_column_content = VBox([
            variant_widget,
            HTML("<div style='font-size:11px; color:#777; margin-left:145px;'><i>Enter 2 for Dimorphic, >2 for Polymorphic.</i></div>")
        ])

        return VBox([
            self._header('Geological Analysis'),
            self._desc("Information found via Analysis Visor."),
            self._row(
                self._create_widget(Combobox, 'formation', 'Formation:', options=self.data.FORMATION_LIST, placeholder='Select or type formation...'),
                self._create_widget(Text, 'content', 'Metal Content:', placeholder='e.g., 87%')
            ),
            self._row(
                self._create_widget(Combobox, 'notes', 'Analysis Notes:', options=self.data.NOTES_LIST, placeholder='Select or type note...'),
                right_column_content
            )
        ])

    def _create_tab_resources(self):
        """Create the Resources tab with harvestable elements and media gallery."""
        return VBox([
            self._header('Resource Extraction'),
            self._desc("Select mined resources."),
            self._row(
                self._create_widget(Dropdown, 'element_primary', 'Primary Element:', options=self.data.ELEMENT_LIST),
                self._create_widget(Dropdown, 'element_secondary', 'Secondary Element:', options=self.data.ELEMENT_LIST)
            ),
            self._header('Media Gallery'),
            self._desc("Provide filenames for images."),
            self._row(
                self._create_widget(Text, 'image', 'Main Image:', placeholder='e.g., File:MyAwesomeMineral.png')
            ),
            self._create_widget(Textarea, 'gallery', 'Gallery Images:', placeholder='File:Image1.jpg|Description\nFile:Image2.png|Description')
        ])

    def _create_tab_generate(self):
        """Create the Generate tab with action buttons and output display."""
        # Create action buttons with different styles and icons
        self.btn_preview = Button(description='Preview Code', button_style='info', icon='eye')
        self.btn_generate = Button(description='Generate', button_style='success', icon='code')
        self.btn_copy = Button(description='Copy Code', button_style='primary', icon='copy', disabled=True)
        self.btn_download = Button(description='Download', button_style='primary', icon='download', disabled=True)
        self.btn_example = Button(description='Load Example', button_style='warning', icon='lightbulb')
        self.btn_clear = Button(description='Reset Form', button_style='danger', icon='trash')

        # Status display and output area
        self.status_text = HTML(value="<i style='color:#777;'>Status: Waiting for input...</i>")
        self.output = Output(layout={'border': '1px solid #ccc', 'height': '400px', 'overflow_y': 'scroll', 'padding': '10px'})

        return VBox([
            self._header('Finalize and Generate'),
            HBox([self.btn_preview, self.btn_generate, self.btn_copy, self.btn_download, self.btn_example, self.btn_clear], layout=Layout(justify_content='center', margin='15px 0')),
            VBox([self.status_text], layout=Layout(align_items='center', margin='5px 0')),
            self._header('Generated Wikitext Output'),
            self.output
        ])

    def _create_widget(self, widget_class, key, description, **kwargs):
        """
        Create a widget with consistent styling and store it in AppWidgets.

        Args:
            widget_class: Type of widget to create (Text, Dropdown, etc.)
            key (str): Attribute name in AppWidgets
            description (str): Label text for the widget
            **kwargs: Additional widget-specific parameters

        Returns:
            The created widget

        Note:
            Special handling for Dropdown and Combobox to manage options
            and placeholder text appropriately.
        """
        # Base parameters for all widgets
        params = {'description': description, 'style': self.LABEL_STYLE, 'layout': self.WIDGET_LAYOUT}

        # Adjust layout for text areas
        if widget_class == Textarea:
            params['layout'] = self.GALLERY_LAYOUT if key == 'gallery' else self.TEXTAREA_LAYOUT

        # Special handling for Dropdown and Combobox widgets
        if widget_class in [Dropdown, Combobox]:
            opts = list(kwargs.pop('options', []))
            placeholder = kwargs.pop('placeholder', None)

            if widget_class == Combobox:
                # Combobox allows typing custom values, placeholder is ghost text
                params['options'] = opts
                params['ensure_option'] = False  # Allow custom values not in options
                if placeholder:
                    params['placeholder'] = placeholder
            else:
                # Dropdown requires selection from options, first option acts as placeholder
                if placeholder:
                    params['options'] = [placeholder] + opts
                    params['value'] = placeholder
                else:
                    params['options'] = opts

        # Apply any additional parameters
        params.update(kwargs)
        widget = widget_class(**params)
        # Store widget in AppWidgets for easy access
        setattr(self.widgets, key, widget)
        return widget

    def _header(self, text):
        """Create a styled header element."""
        return HTML(f"<div style='{self.HEADER_STYLE}'>{text}</div>")

    def _desc(self, text):
        """Create a styled description element."""
        return HTML(f"<div style='{self.DESC_STYLE}'>{text}</div>")

    def _row(self, w1, w2=None):
        """Create a two-column row or single column if w2 is None."""
        return HBox([VBox([w1], layout=self.COL_LAYOUT), VBox([w2] if w2 else [], layout=self.COL_LAYOUT)], layout=self.FULL_ROW_LAYOUT)

    def _safe_set_value(self, widget, value):
        """
        Safely set widget value, handling Dropdown validation errors.

        Args:
            widget: The widget to update
            value: Value to set

        Note:
            Dropdown widgets require the value to be in their options list.
            If the value isn't valid, we fall back to the first option.
        """
        if isinstance(widget, Dropdown):
            try:
                widget.value = value
            except TraitError:
                # If value not in options, use first option as fallback
                widget.value = widget.options[0] if widget.options else None
        else:
            widget.value = value

    def _connect_events(self):
        """Connect widget events to their handler methods."""
        # Update stardate when discovery date changes
        self.widgets.discovery_date.observe(self._update_stardate_ui, names='value')
        # Auto-calculate region when glyphs or galaxy changes
        self.widgets.glyphs.observe(self._on_location_input_change, names='value')
        self.widgets.galaxy.observe(self._on_location_input_change, names='value')

        # Connect button click events
        self.btn_preview.on_click(lambda _button: self._run_generation(mode='preview'))
        self.btn_generate.on_click(lambda _button: self._run_generation(mode='full'))
        self.btn_clear.on_click(self._clear_form)
        self.btn_copy.on_click(self._copy_to_clipboard)
        self.btn_example.on_click(self._load_example)
        self.btn_download.on_click(self._download)

    def _calc_stardate(self):
        """
        Calculate AGT stardate from discovery date.

        AGT stardate format: (Year + 1716).Day.Month
        Example: May 20, 2024 becomes 3740.20.05

        Returns:
            str: Formatted stardate, or empty string if no date
        """
        if self.widgets.discovery_date.value:
            d = arrow.get(self.widgets.discovery_date.value)
            return f"{d.year + 1716}.{d.day}.{d.month:02d}"
        return ""

    def _update_stardate_ui(self, change):
        """Update the stardate widget when discovery date changes."""
        if self.widgets.discovery_date.value:
            self.widgets.agt_stardate.value = self._calc_stardate()

    def _on_location_input_change(self, change):
        """
        Calculate and update region name when glyphs or galaxy changes.

        This is triggered automatically when the user enters glyphs or
        selects a galaxy. It converts glyphs to coordinates, then uses
        procedural generation to create a region name.

        Args:
            change: Widget change event (not used directly)
        """
        glyphs = self.widgets.glyphs.value.strip().upper()
        galaxy_name = self.widgets.galaxy.value

        # Only proceed with valid 12-character hex and known galaxy
        if len(glyphs) == 12 and re.fullmatch(r'^[0-9A-F]+$', glyphs):
            if galaxy_name in self.data.GALAXY_INDEX_MAP:
                # Convert glyphs to voxel coordinates
                coords = self.map_logic.glyphs_to_coords(glyphs)
                if coords:
                    # Get galaxy index for procedural generation
                    galaxy_index = self.data.GALAXY_INDEX_MAP[galaxy_name]
                    # Adjust coordinates to region-relative space
                    x = coords['x'] - self.map_logic.REGION_CENTER_XZ
                    y = coords['y'] - self.map_logic.REGION_CENTER_Y
                    z = coords['z'] - self.map_logic.REGION_CENTER_XZ

                    try:
                        # Generate region name from coordinates
                        seed = RegionNameGenerator.create_region_seed(x, y, z, galaxy_index)
                        name = RegionNameGenerator.format_name(seed, self.data)
                        self.widgets.region.value = name
                    except Exception as e:
                        # Clear region field and show error if generation fails
                        self.widgets.region.value = ""
                        self.status_text.value = f"<b style='color:#C62828;'>ERROR:</b> Region generation failed: {e}"

    def _run_generation(self, mode):
        """
        Generate wikitext from form data with validation.

        Args:
            mode (str): 'preview' for preview only, 'full' for complete generation

        Steps:
        1. Collect data from all widgets
        2. Validate using FormInputModel
        3. Prepare template context with calculated fields
        4. Render template and display output
        5. Update status and enable/disable buttons
        """
        # Step 1: Collect widget values into a dictionary
        raw_data = {}
        for field_name in FormInputModel.model_fields:
            if hasattr(self.widgets, field_name):
                widget = getattr(self.widgets, field_name)
                val = widget.value
                # Handle Dropdown placeholders and "None" values
                if isinstance(widget, Dropdown) and (val.startswith("Select ") or val == "None"):
                    raw_data[field_name] = ""
                elif isinstance(val, str):
                    raw_data[field_name] = val.strip()
                else:
                    raw_data[field_name] = val

        try:
            # Step 2: Validate input using Pydantic model
            model = FormInputModel(**raw_data)
        except ValidationError as e:
            # Step 2a: If validation fails, show detailed error messages
            error_msg = f"<b style='color:#C62828;'>VALIDATION FAILED:</b> Check the required fields."
            errors = e.errors()
            for err in errors:
                loc = " -> ".join(map(str, err['loc']))
                error_msg += f"<br><b>{loc}:</b> {err['msg']}"
            self.status_text.value = error_msg
            self.btn_copy.disabled = True
            self.btn_download.disabled = True
            return

        # Step 3: Prepare template context with additional calculated fields
        context = model.model_dump()
        context['discovery_date_long'] = model.discovery_date.format('MMMM D, YYYY')
        context['discovery_date_short'] = model.discovery_date.format('D-MMM-YYYY')
        context['agt_stardate'] = self._calc_stardate()
        context['original_name_display'] = model.original_name or model.name

        # Generate location description text
        is_moon = bool(model.moon)
        loc_type = "moon" if is_moon else "planet"
        name_ref = model.moon if is_moon else model.planet
        context['location_link_text'] = (
            f"on the [[{loc_type}]] [[{name_ref}]] "
            f"{'orbiting the [[planet]] [[' + model.planet + ']] ' if is_moon else ''}"
            f"in the [[{model.system}]] [[star system]]."
        )

        # Generate resource extraction description
        elem_primary, elem_secondary = model.element_primary, model.element_secondary
        if elem_primary and elem_secondary:
            context['resource_text'] = f"This mineral provides the resources [[{elem_primary}]] and [[{elem_secondary}]] when mined."
        elif elem_primary:
            context['resource_text'] = f"This mineral provides the resource [[{elem_primary}]] when mined."
        else:
            context['resource_text'] = "This mineral provides no harvestable resources when mined."

        context['gallery_content'] = model.gallery if model.gallery else ""

        # Step 4: Render template and display output
        self.generated_content = self.template.render(context)
        with self.output:
            clear_output(wait=True)
            print(self.generated_content)

        # Step 5: Update UI based on generation mode
        if mode == 'full':
            self.btn_copy.disabled = False
            self.btn_download.disabled = False
            self.status_text.value = "<b style='color:#2E7D32;'>SUCCESS:</b> Code generated."
        else:
            self.btn_copy.disabled = False
            self.status_text.value = "<b style='color:#0277BD;'>INFO:</b> Preview generated successfully."

    def _clear_form(self, _button=None):
        """
        Reset all form fields to their default values.

        Args:
            _button: Button click event (ignored)
        """
        # Clear all widget values based on their type
        for f in fields(self.widgets):
            w_widget = getattr(self.widgets, f.name)
            if isinstance(w_widget, (Text, Textarea, Combobox)):
                w_widget.value = ""
            elif isinstance(w_widget, Dropdown) and w_widget.options:
                w_widget.value = w_widget.options[0]
            elif isinstance(w_widget, IntText):
                w_widget.value = 1

        # Reset special fields
        self.generated_content = ""
        self.widgets.discovery_date.value = arrow.now().date()
        self.widgets.civilized.value = "Alliance of Galactic Travellers"
        self.widgets.release.value = self.DEFAULT_RELEASE

        # Disable output buttons and clear display
        self.btn_copy.disabled = True
        self.btn_download.disabled = True
        self.output.clear_output()
        self.status_text.value = "<b style='color:#D84315;'>ACTION:</b> Form cleared."

    def _load_example(self, _button):
        """
        Load example data to demonstrate the application.

        Args:
            _button: Button click event (ignored)
        """
        # Clear form first, then populate with example data
        self._clear_form(None)

        # Set example values for each field
        self.widgets.name.value = "O. Exampleseis"
        self.widgets.original_name.value = "Rocky Cluster 9B"
        self.widgets.discoverer.value = "Example-Traveller"
        self.widgets.discoverer_link.value = "User:Example-Traveller"
        self.widgets.system.value = "Sample-Star"
        self.widgets.planet.value = "Testament"
        self.widgets.moon.value = "Datum Minor"
        self.widgets.glyphs.value = "1234ABCD5678"
        self.widgets.content.value = "91%"
        self.widgets.polymorphic.value = 2
        self.widgets.image.value = "File:Mineral_Sample.png"
        self.widgets.gallery.value = "File:Mineral_CloseUp.png|A detailed view"
        self.widgets.discovery_date.value = arrow.get("2024-05-20").date()

        # Set dropdown/combobox values with fallback handling
        self._safe_set_value(self.widgets.galaxy, "Euclid")
        self._safe_set_value(self.widgets.formation, self.data.FORMATION_LIST[0] if self.data.FORMATION_LIST else "Volcanic")
        self._safe_set_value(self.widgets.notes, self.data.NOTES_LIST[0] if self.data.NOTES_LIST else "Igneous")
        self._safe_set_value(self.widgets.element_primary, "Pure Ferrite")
        self._safe_set_value(self.widgets.element_secondary, "Cobalt")

        # Manually trigger region calculation for the example
        self._on_location_input_change(None)
        # Switch to first tab and update status
        self.tabs.selected_index = 0
        self.status_text.value = "<b style='color:#ED6C02;'>ACTION:</b> Example loaded."

    def _copy_to_clipboard(self, _button):
        """
        Copy generated wikitext to clipboard using JavaScript.

        Args:
            _button: Button click event (ignored)

        Note:
            This only works in Jupyter/Colab environments with JavaScript support.
        """
        if not self.generated_content:
            self.status_text.value = "<b style='color:#ED6C02;'>WARNING:</b> Please generate code first before copying."
            return

        # Escape special characters for JavaScript template literals
        escaped_content = self.generated_content.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${').replace('\n', '\\n').replace('\r', '\\r')

        # Use JavaScript to copy to clipboard
        display(Javascript(f"navigator.clipboard.writeText(`{escaped_content}`)"))
        self.status_text.value = "<b style='color:#0277BD;'>SUCCESS:</b> Copied to clipboard!"

    def _download(self, _button):
        """
        Download generated wikitext as a text file.

        Args:
            _button: Button click event (ignored)

        Note:
            This only works in Google Colab where files.download is available.
            In other environments, users should use the copy function instead.
        """
        if not self.generated_content:
            self.status_text.value = "<b style='color:#ED6C02;'>WARNING:</b> Please generate code first before downloading."
            return

        try:
            # Try to use Google Colab's download function
            from google.colab import files
            # Create safe filename from mineral name
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', self.widgets.name.value.replace(' ', '_'))
            filename = f"{safe_name}_Mineral.txt"

            # Write content to file and trigger download
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.generated_content)
            files.download(filename)
            self.status_text.value = "<b style='color:#0277BD;'>SUCCESS:</b> Download initiated."
        except ImportError:
            # Fallback message for non-Colab environments
            self.status_text.value = "<b style='color:#C62828;'>ERROR:</b> File download is only available in Google Colab. Use 'Copy Code' instead."


# Entry point for standalone execution
if __name__ == '__main__':
    # Create and run the application
    app = NMSMineralWikiGenerator()