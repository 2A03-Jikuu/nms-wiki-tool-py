"""
No Man's Sky Region Wiki Generator - Jupyter/Colab Widget

This tool creates wiki pages for regions in the game No Man's Sky. It takes
portal glyphs (a special code found in the game) and automatically calculates
all the information needed for a wiki page, including the region's name,
coordinates, distance from the galaxy center, and which quadrant it's in.

The tool has three main parts:
1. A user interface with forms and buttons (for Jupyter/Colab notebooks)
2. Math functions that translate game codes into usable numbers
3. Name generation that predicts what the game will call a region

Key classes:
- NMSData: Downloads game data from the internet
- ByteUtils: Does math like the game's C++ engine
- Generator: Builds procedural names
- RegionNameGenerator: Creates region names with special endings
- NMSGalaxyMap: Converts glyphs to coordinates
- NMSWikiRegionFormCreator: The main app with buttons and forms
"""

import math
import re
import struct
from dataclasses import dataclass, field, fields

import requests
from IPython.display import Javascript, clear_output, display
from ipywidgets import (
    Button,
    Combobox,
    HBox,
    HTML,
    Layout,
    Output,
    Tab,
    Text,
    VBox,
)
from jinja2 import Environment
from pydantic import BaseModel, ValidationError, field_validator


class NMSData:
    """
    Downloads and stores game data from the internet.

    Instead of storing hundreds of galaxy names inside the code, this class
    fetches them from GitHub when the tool starts. This keeps the code small
    and makes it easy to update when new galaxies are added to the game.

    Attributes:
        GALAXY_MAP (dict): Maps galaxy numbers to names (like {0: "Euclid"})
        GALAXY_OPTIONS (list): List of galaxy names for dropdown menus
        GALAXY_NAME_TO_INDEX (dict): Maps lowercase names to numbers
        LETTER_MAP (dict): Rules for which letters can follow other letters
        ALPHASETS (list): Starting strings used to build procedural names
    """
    GALAXY_MAP = {}
    GALAXY_OPTIONS = []
    GALAXY_NAME_TO_INDEX = {}
    LETTER_MAP = {}
    ALPHASETS = []

    # These URLs point to data files on GitHub
    URL_BASE = (
        "https://raw.githubusercontent.com/2A03-Jikuu/"
        "nms-wiki-tool-py/refs/heads/main/datalist"
    )
    URL_GALAXIES = f"{URL_BASE}/galaxies.json"
    URL_LETTER_MAP = f"{URL_BASE}/letter_map.json"
    URL_ALPHASETS = f"{URL_BASE}/alphasets.json"

    @classmethod
    def initialize(cls):
        """
        Downloads all needed data files from the internet.

        This should be called once when the tool starts. It gets galaxy names,
        letter rules, and starting strings for name generation.

        Raises:
            ConnectionError: If the internet connection fails (has fallback data)

        Example:
            >>> NMSData.initialize()
            # Now GALAXY_OPTIONS contains ['Euclid', 'Hilbert Dimension', ...]
        """
        # Download galaxies list first, then other data files
        cls._fetch_galaxies()
        cls._fetch_letter_map()
        cls._fetch_alphasets()

    @classmethod
    def _fetch_galaxies(cls):
        """
        Downloads the list of all galaxies in No Man's Sky.

        The game has over 250 galaxies, each with a unique name and number.
        This function gets that list and prepares it for use in dropdown menus.

        Notes:
            If the download fails, it uses a small fallback list so the tool
            still works (just with fewer galaxies).
        """
        try:
            # Try to download from GitHub
            response = requests.get(cls.URL_GALAXIES)
            # Raise an error if the download failed
            response.raise_for_status()
            data = response.json()

            # Convert the list into dictionaries for easy lookup
            cls.GALAXY_MAP = {item['index']: item['name'] for item in data}
            cls.GALAXY_OPTIONS = sorted([item['name'] for item in data])
            cls.GALAXY_NAME_TO_INDEX = {name.lower(): idx for idx, name in cls.GALAXY_MAP.items()}
        except Exception as e:
            # If download fails, use a small built-in list
            print(f"Error fetching galaxies: {e}")
            cls.GALAXY_OPTIONS = ['Euclid', 'Hilbert Dimension', 'Calypso']
            cls.GALAXY_MAP = {0: 'Euclid', 1: 'Hilbert Dimension', 2: 'Calypso'}
            cls.GALAXY_NAME_TO_INDEX = {name.lower(): idx for idx, name in cls.GALAXY_MAP.items()}

    @classmethod
    def _fetch_letter_map(cls):
        """
        Downloads probability rules for generating words.

        The game uses special rules about which letters can follow other letters
        to make generated names sound natural. This file contains those rules.

        Notes:
            JSON keys are strings but we need integers, so we convert them.
        """
        try:
            response = requests.get(cls.URL_LETTER_MAP)
            response.raise_for_status()
            raw = response.json()
            # Convert string keys to integers for math operations
            cls.LETTER_MAP = {int(k): v for k, v in raw.items()}
        except Exception as e:
            print(f"Error fetching letter_map: {e}")

    @classmethod
    def _fetch_alphasets(cls):
        """
        Downloads seed strings used to start name generation.

        These are short strings (like "ab", "ex", "ou") that serve as starting
        points for building longer procedural names.
        """
        try:
            response = requests.get(cls.URL_ALPHASETS)
            response.raise_for_status()
            cls.ALPHASETS = response.json()
        except Exception as e:
            print(f"Error fetching alphasets: {e}")
            # Use empty strings as fallback
            cls.ALPHASETS = [""] * 8


# Start downloading data immediately when this file runs
NMSData.initialize()


class ByteUtils:
    """
    Forces Python to do math like the game's C++ engine.

    No Man's Sky is written in C++, which handles numbers differently than Python.
    In C++, numbers wrap around when they get too big (like an odometer rolling
    over). This class makes Python mimic that behavior so our calculations
    match the game exactly.

    Attributes:
        SEED_MULTIPLIER (list): Special numbers used to shuffle random seeds

    Example:
        >>> ByteUtils.add([255], [1])
        [0]  # Because 255 + 1 wraps around to 0 in C++ style math
    """
    # These numbers are used to shuffle random seeds (found in game code)
    SEED_MULTIPLIER = [0x99, 0xF8, 0x76, 0x5A]

    @staticmethod
    def parse(val, little_endian=True):
        """
        Converts a hex string into a list of number bytes.

        Args:
            val (str): A hex string like "015A"
            little_endian (bool): If True, reverse the byte order (game uses this)

        Returns:
            list: A list of integers, each representing one byte (0-255)

        Example:
            >>> ByteUtils.parse("015A")
            [90, 1]  # Bytes in reverse order
        """
        # Make sure the string has an even number of characters
        if len(val) % 2 != 0:
            val = "0" + val

        # Convert every 2 characters to a byte
        res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]

        # Game reads bytes backwards, so reverse if needed
        if little_endian:
            res.reverse()
        return res

    @staticmethod
    def format_short(op1):
        """
        Ensures a byte list is at least 2 bytes long by adding zeros.

        Args:
            op1 (list): List of bytes

        Returns:
            list: Same list but with at least 2 bytes

        Example:
            >>> ByteUtils.format_short([1])
            [1, 0]
        """
        res = list(op1)
        # Add zero bytes until we have at least 2 bytes
        while len(res) < 2:
            res.append(0x00)
        return res

    @staticmethod
    def add(op1, op2):
        """
        Adds two byte lists together with carry-over, like C++ does.

        Args:
            op1 (list): First list of bytes
            op2 (list): Second list of bytes

        Returns:
            list: Result of addition as bytes

        Notes:
            This handles "carry the 1" when a byte exceeds 255, just like
            elementary school addition but with base 256.
        """
        # Start with the second operand
        result = list(op2)

        # Add each byte from first operand
        for i in range(len(op1)):
            result = ByteUtils._add_single(op1[i], result, i)
        return result

    @staticmethod
    def _add_single(val, target_list, index):
        """
        Helper function that adds one byte to a specific position.

        Args:
            val (int): Byte value to add (0-255)
            target_list (list): List of bytes being modified
            index (int): Position to add to

        Returns:
            list: Modified list with carry-over handled

        Notes:
            If the sum exceeds 255, the extra carries to the next byte position.
            This is recursive because carry might trigger another carry.
        """
        # Check if we're within the current list length
        if index < len(target_list):
            # Add the byte at this position
            total = val + target_list[index]
            # Keep only the last 8 bits (0-255 range)
            target_list[index] = total & 0xFF
            # Calculate overflow (carry) - bits beyond the first 8
            rem = (total >> 8) & 0xFF

            # If there's carry, add it to the next position
            if rem != 0:
                target_list = ByteUtils._add_single(
                    rem, target_list, index + 1
                )
        else:
            # If position doesn't exist yet, append the byte
            target_list.append(val)
        return target_list

    @staticmethod
    def sub(op1, op2):
        """
        Subtracts byte lists with borrowing, like C++ does.

        Args:
            op1 (list): Bytes to subtract
            op2 (list): Bytes to subtract from

        Returns:
            list: Result of subtraction as bytes

        Notes:
            Handles "borrowing" from next byte when a byte would go below 0.
        """
        result = list(op2)
        for i in range(len(op1)):
            result = ByteUtils._sub_single(op1[i], result, i)
        return result

    @staticmethod
    def _sub_single(val, target_list, index):
        """
        Helper function that subtracts one byte from a specific position.

        Args:
            val (int): Byte value to subtract
            target_list (list): List of bytes being modified
            index (int): Position to subtract from

        Returns:
            list: Modified list with borrowing handled

        Notes:
            If the result would be negative, borrow from the next byte.
        """
        if index < len(target_list):
            # Subtract the byte at this position
            diff = val - target_list[index]
            # Keep only the last 8 bits
            target_list[index] = diff & 0xFF
            # Calculate borrow amount
            rem = (diff >> 8) & 0xFF

            # If we borrowed, subtract from next position
            if rem != 0:
                target_list = ByteUtils._sub_single(
                    rem, target_list, index + 1
                )
        else:
            # If position doesn't exist, append the byte
            target_list.append(val)
        return target_list

    @staticmethod
    def multiply(op1, op2):
        """
        Multiplies byte lists with C++-style signed 16-bit wrapping.

        Args:
            op1 (list): First list of bytes
            op2 (list): Second list of bytes

        Returns:
            list: Result of multiplication as bytes

        Notes:
            This is complex because we need to mimic C++ behavior where
            numbers wrap around at -32768 to 32767 range (signed 16-bit).
            Without this exact behavior, random number generation fails.
        """
        result = []

        # Multiply each byte from first list with each byte from second list
        for i in range(len(op1)):
            rem = 0  # Carry from previous multiplication
            for j in range(len(op2)):
                # Multiply and add any carry
                raw_prod = (op1[i] * op2[j]) + rem

                # Force signed 16-bit wrapping: -32768 to 32767
                signed_prd = (raw_prod + 32768) % 65536 - 32768

                # Separate into current result and carry
                rem = (signed_prd >> 8) & 0xFF
                res = signed_prd & 0xFF

                # Add to the appropriate position in result
                idx = i + j
                if idx < len(result):
                    result = ByteUtils._add_single(res, result, idx)
                else:
                    result.append(res)

            # Add any remaining carry
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
        Shifts bytes left (like removing bytes from start).

        Args:
            op1 (list): List of bytes
            shift (int): How many positions to shift

        Returns:
            list: Shifted bytes or [0] if all shifted out
        """
        # Take bytes from position 'shift' to end
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1, shift):
        """
        Shifts bytes right (like removing bytes from end).

        Args:
            op1 (list): List of bytes
            shift (int): How many positions to shift

        Returns:
            list: Shifted bytes or [0] if all shifted out
        """
        # Take bytes from start to position 'length - shift'
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1, roll):
        """
        Rotates bytes left (moves bytes from start to end).

        Args:
            op1 (list): List of bytes
            roll (int): How many positions to rotate

        Returns:
            list: Rotated bytes
        """
        if not op1:
            return op1

        # Calculate effective rotation (handle roll > length)
        r = roll % len(op1)
        # Move first 'r' bytes to the end
        return op1[r:] + op1[:r]

    @staticmethod
    def zxd(op1, extend):
        """
        Extends a byte list with zeros.

        Args:
            op1 (list): List of bytes
            extend (int): Desired total length

        Returns:
            list: Extended list with zeros
        """
        # Make copy and add zero bytes to reach desired length
        return list(op1) + [0x00] * (extend - len(op1))

    @staticmethod
    def sxd(op1, extend):
        """
        Extends a byte list with sign-preserving bytes.

        Args:
            op1 (list): List of bytes
            extend (int): Desired total length

        Returns:
            list: Extended list preserving sign

        Notes:
            If the last byte has its highest bit set (negative in signed math),
            we extend with 0xFF bytes, otherwise with 0x00 bytes.
        """
        result = list(op1)

        # Check if the number is negative (highest bit of last byte is 1)
        val = 0xFF if (len(op1) > 0 and (op1[-1] >> 7) == 1) else 0x00

        # Add bytes to reach desired length
        for _ in range(extend - len(op1)):
            result.append(val)
        return result

    @staticmethod
    def logical_op(op1, op2, mode):
        """
        Performs AND, OR, or XOR operations on byte lists.

        Args:
            op1 (list): First list of bytes
            op2 (list): Second list of bytes
            mode (int): 0 for AND, 1 for OR, 2 for XOR

        Returns:
            list: Result of logical operation

        Notes:
            Lists are padded to same length with zeros before operation.
        """
        l1, l2 = len(op1), len(op2)

        # Pad the shorter list with zeros so both are same length
        if l1 > l2:
            longer = list(op1)
            shorter = list(op2) + [0x00] * (l1 - l2)
        else:
            longer = list(op2)
            shorter = list(op1) + [0x00] * (l2 - l1)

        # Perform operation on each byte pair
        res = []
        for i in range(len(longer)):
            if mode == 0:
                res.append(longer[i] & shorter[i])  # AND
            elif mode == 1:
                res.append(longer[i] | shorter[i])  # OR
            else:
                res.append(longer[i] ^ shorter[i])  # XOR
        return res

    @staticmethod
    def xor(op1, op2):
        """
        Performs XOR (exclusive OR) on two byte lists.

        Args:
            op1 (list): First list of bytes
            op2 (list): Second list of bytes

        Returns:
            list: XOR result
        """
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1, op2):
        """
        Performs AND on two byte lists.

        Args:
            op1 (list): First list of bytes
            op2 (list): Second list of bytes

        Returns:
            list: AND result
        """
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1, op2):
        """
        Performs OR on two byte lists.

        Args:
            op1 (list): First list of bytes
            op2 (list): Second list of bytes

        Returns:
            list: OR result
        """
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache, move=1):
        """
        Advances the random number generator state.

        Args:
            cache (list): Current random state [part1, part2]
            move (int): How many times to advance the state

        Returns:
            list: Updated random state

        Notes:
            Uses a special "magic number" (SEED_MULTIPLIER) found in game code
            to shuffle bits in a specific way that matches the game's random
            number generator.
        """
        for _ in range(move):
            # Multiply by magic number
            step1 = ByteUtils.multiply(cache[0], ByteUtils.SEED_MULTIPLIER)
            # Add the two parts
            result = ByteUtils.add(step1, cache[1])
            # Shift and update both parts
            cache[0] = ByteUtils.shl(result, 4)
            cache[1] = ByteUtils.shr(result, 4)
        return cache

    @staticmethod
    def _unpack(fmt, arr, offset, size):
        """
        Helper to convert bytes to Python numbers using struct.

        Args:
            fmt (str): Format string like '<I' for unsigned 32-bit
            arr (list): List of bytes
            offset (int): Where to start reading
            size (int): How many bytes to read

        Returns:
            int or float: Converted number

        Notes:
            Pads with zeros if not enough bytes available.
        """
        # Get the bytes we need (or pad with zeros)
        chunk = arr[offset:offset + size]
        while len(chunk) < size:
            chunk.append(0)
        # Convert bytes to number
        return struct.unpack(fmt, bytes(chunk))[0]

    @staticmethod
    def to_uint32(arr, offset=0):
        """
        Converts bytes to unsigned 32-bit integer.

        Args:
            arr (list): List of bytes
            offset (int): Where to start reading

        Returns:
            int: Number between 0 and 4,294,967,295
        """
        return ByteUtils._unpack('<I', arr, offset, 4)

    @staticmethod
    def to_int32(arr, offset=0):
        """
        Converts bytes to signed 32-bit integer.

        Args:
            arr (list): List of bytes
            offset (int): Where to start reading

        Returns:
            int: Number between -2,147,483,648 and 2,147,483,647
        """
        return ByteUtils._unpack('<i', arr, offset, 4)

    @staticmethod
    def to_int16(arr, offset=0):
        """
        Converts bytes to signed 16-bit integer.

        Args:
            arr (list): List of bytes
            offset (int): Where to start reading

        Returns:
            int: Number between -32,768 and 32,767
        """
        return ByteUtils._unpack('<h', arr, offset, 2)

    @staticmethod
    def to_double(arr, offset=0):
        """
        Converts bytes to 64-bit floating point number.

        Args:
            arr (list): List of bytes
            offset (int): Where to start reading

        Returns:
            float: Double precision floating point number
        """
        return ByteUtils._unpack('<d', arr, offset, 8)

    @staticmethod
    def to_single(arr, offset=0):
        """
        Converts bytes to 32-bit floating point number.

        Args:
            arr (list): List of bytes
            offset (int): Where to start reading

        Returns:
            float: Single precision floating point number
        """
        return ByteUtils._unpack('<f', arr, offset, 4)

    @staticmethod
    def get_bytes_uint32(val):
        """
        Converts unsigned 32-bit integer to bytes.

        Args:
            val (int): Number between 0 and 4,294,967,295

        Returns:
            list: List of 4 bytes representing the number
        """
        return list(struct.pack('<I', val))


class StringExtensions:
    """
    Helper class for formatting hexadecimal strings.

    Methods:
        short_to_formatted_hex: Formats a number as hex with specific length
    """
    @staticmethod
    def short_to_formatted_hex(val, trunc):
        """
        Formats a number as hexadecimal with specified length.

        Args:
            val (int): Number to format
            trunc (int): Desired length of output (truncates from right)

        Returns:
            str: Hexadecimal string of specified length

        Example:
            >>> StringExtensions.short_to_formatted_hex(255, 2)
            'FF'
            >>> StringExtensions.short_to_formatted_hex(4095, 3)
            'FFF'
        """
        # Keep only last 16 bits (0-65535 range)
        val = val & 0xFFFF
        # Convert to hex with 4 digits (like 00FF)
        hex_str = f"{val:04X}"
        # Return only the last 'trunc' characters
        return hex_str[-trunc:]


class Generator:
    """
    Builds procedural names using game's algorithm.

    This class takes random number seeds and generates pronounceable names
    that match what appears in the game. It uses probability rules (Letter Map)
    and starting strings (Alphasets) to ensure names look natural.

    Attributes:
        TINY_DOUBLE (list): Small constant used in probability calculations
        MAX_BACKTRACK_ATTEMPTS (int): How many times to try fixing bad names
        MAX_NAME_LENGTH (int): Maximum characters in a generated name
        VOWELS (str): Standard vowels (a, e, i, o, u)
        VOWELS_WITH_Y (str): Vowels including 'y'

    Example:
        >>> Generator.generate_name(cache0, cache1)
        'Ocopad'
    """
    # Tiny constant for probability calculations (0.000244140625 in decimal)
    TINY_DOUBLE = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]
    MAX_BACKTRACK_ATTEMPTS = 50
    MAX_NAME_LENGTH = 64
    VOWELS = "aeiou"
    VOWELS_WITH_Y = "aeiouy"

    @staticmethod
    def generate_name(cache0, cache1):
        """
        Main name generation algorithm.

        Steps:
        1. Get starting characters from alphaset
        2. Determine name length based on random seed
        3. Pick each letter using probability rules
        4. Fix pronunciation issues (add vowels where needed)

        Args:
            cache0 (list): First part of random state
            cache1 (list): Second part of random state with control values

        Returns:
            str: Generated name, or empty string if generation fails

        Raises:
            None: Returns empty string or fallback on failure

        Example:
            >>> cache0 = [[...], [...]]
            >>> cache1 = [[0], [6], [...]]
            >>> Generator.generate_name(cache0, cache1)
            'Ocopad'
        """
        # Step 1: Get starting characters (like "oc" or "ex")
        name = Generator.get_characters_from_alphaset(cache0, cache1)
        if name == "__EMPTY__":
            return ""

        # Advance random state for next steps
        ByteUtils.update_seed(cache0)

        # Check which letter selection method to use (game has two methods)
        check_op = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        alternate_char_getter = (ByteUtils.to_int16(check_op) != 0)

        ByteUtils.update_seed(cache0)

        # Step 2: Calculate how long the name should be
        # This formula comes from game code analysis
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])
        limit = ByteUtils.to_int16(ByteUtils.sxd(register0, 2))

        # Step 3: Build the name letter by letter
        if 0 < limit:
            i = 0
            safety = 0  # Prevents infinite loops on bad sequences
            while i < limit:
                ByteUtils.update_seed(cache0)
                # Look at last 3 characters to decide next letter
                sub_str = name[i: i + 3]
                alphaset_idx = cache1[0][0] if cache1[0] else 0

                # Get probabilities for what letter comes next
                char_weights = Generator.get_string_weights(
                    sub_str, alphaset_idx
                )

                # Convert random seed to a probability value
                val_u32 = ByteUtils.to_uint32(cache0[0])
                tiny_dbl = ByteUtils.to_double(Generator.TINY_DOUBLE)
                target = float(val_u32 * tiny_dbl)

                if char_weights is None:
                    # No valid next letter - go back one step
                    i -= 1
                    safety += 1
                    if safety > Generator.MAX_BACKTRACK_ATTEMPTS:
                        break  # Give up after too many failures
                else:
                    safety = 0
                    index = 0

                    if alternate_char_getter:
                        # Complex selection method (used 50% of the time)
                        target *= (len(char_weights) - 1)
                        b_tgt = list(struct.pack('<f', target))
                        op_and = ByteUtils.and_op(
                            b_tgt, [0x00, 0x00, 0x00, 0x80]
                        )
                        op = ByteUtils.or_op(
                            op_and, [0x00, 0x00, 0x00, 0x3F]
                        )
                        index = int(ByteUtils.to_single(op) + target)
                    else:
                        # Simple probability selection
                        weight = 0.0
                        j = 0
                        # Pick letter based on weighted probability
                        for cw in char_weights:
                            weight += cw[1]
                            if weight >= target:
                                break
                            j += 1
                        index = j

                    # Add selected letter to name
                    if index < len(char_weights):
                        name += char_weights[index][0]

                # Don't let names get too long
                if len(name) >= Generator.MAX_NAME_LENGTH:
                    name = name[:Generator.MAX_NAME_LENGTH]
                i += 1

        if not name:
            return ""

        # Step 4: Fix pronunciation issues
        # Rule 1: Fix bad starts (like "Xq" becomes "Xaq")
        if len(name) < 2:
            return name

        first, second = name[0], name[1] if len(name) > 1 else ''
        if (first not in Generator.VOWELS) and (second not in Generator.VOWELS):
            cond1 = first != 's' or second not in "hklmnprtwy"
            cond2 = (second == 'h' and first in "ctw")
            cond3 = (second == 'l' and first in "bcfgps")
            cond4 = (second == 'r' and first in "bcdfgkpt")
            cond5 = (second == 'w' and first in "dgt")
            cond6 = (second == 'y' and first in "hmr")

            if cond1:
                is_valid_cluster = cond2 or cond3 or cond4 or cond5 or cond6
                if not is_valid_cluster:
                    name = Generator.insert_vowel(name, cache0, 1)

        # Rule 2: Fix bad endings (like "bg" becomes "bag")
        ult, penult = name[-1], name[-2] if len(name) > 1 else ''
        if len(name) > 1 and (penult != 'g' or ult in Generator.VOWELS):
            idx = len(name) - 1
            c1 = (ult == 'b' and penult in "gn")
            c2 = (ult == 'd' and penult in "bdfghkmpst")
            c3 = (ult == 'g' and penult == 'l')
            c4 = (ult == 'p' and penult in "bdhkt")
            c5 = (ult == 'r' and penult in "bfg")
            c6 = (ult == 't' and penult == 'g')
            c7 = (ult == 'w' and penult not in Generator.VOWELS)

            if c1 or c2 or c3 or c4 or c5 or c6 or c7:
                name = Generator.insert_vowel(name, cache0, idx)

        # Rule 3: Fix too many consonants in a row
        consonance = Generator.get_consecutive_consonants(name)
        if consonance != -1:
            ByteUtils.update_seed(cache0)
            # Pick random position to insert vowel
            mult = ByteUtils.multiply(cache0[0], [0x03])
            shr = ByteUtils.shr(mult, 4)
            add = ByteUtils.add(shr, [0x01])
            offset = ByteUtils.to_int32(ByteUtils.zxd(add, 4))
            name = Generator.insert_vowel(name, cache0, consonance + offset)

        return name

    @staticmethod
    def get_characters_from_alphaset(cache0, cache1):
        """
        Gets starting characters for name generation.

        Args:
            cache0 (list): Random state
            cache1 (list): Control values including alphaset index

        Returns:
            str: 3 starting characters, or "__EMPTY__" if no alphaset

        Notes:
            Alphasets are strings like "abexou..." divided into 3-character
            chunks. This function picks one chunk randomly.
        """
        ByteUtils.update_seed(cache0)
        # Get which alphaset to use (0-7)
        idx = max(0, cache1[0][0]) if cache1[0] else 0
        if idx >= len(NMSData.ALPHASETS):
            idx = 0
        alphaset_str = NMSData.ALPHASETS[idx]
        if not alphaset_str:
            return "__EMPTY__"

        # Calculate which 3-character chunk to use
        length_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        register0 = ByteUtils.multiply(cache0[0], length_bytes)
        shr_reg = ByteUtils.shr(register0, 4)
        register1 = ByteUtils.format_short(
            ByteUtils.multiply(shr_reg, [0x03])
        )

        # Get start and end positions in the alphaset string
        start = ByteUtils.to_int16(register1)
        end = ByteUtils.to_int16(ByteUtils.add(register1, [0x03]))
        return alphaset_str[start:end]

    @staticmethod
    def get_string_weights(substr, alphaset):
        """
        Gets probability weights for next character.

        Args:
            substr (str): Last few characters of current name
            alphaset (int): Which alphaset we're using (0-7)

        Returns:
            list or None: List of (character, weight) pairs, or None if no match

        Notes:
            Looks up in LETTER_MAP what letters typically follow the given
            substring, and with what probabilities.
        """
        if not NMSData.LETTER_MAP or alphaset not in NMSData.LETTER_MAP:
            return None
        subset = NMSData.LETTER_MAP[alphaset]
        if not substr or substr[0] not in subset:
            return None
        return Generator.recursive_search(subset[substr[0]], substr)

    @staticmethod
    def recursive_search(arr, substr):
        """
        Searches nested LETTER_MAP structure for substring match.

        Args:
            arr (list): Nested list structure from LETTER_MAP
            substr (str): Substring to search for

        Returns:
            list or None: Probability weights if found, None otherwise

        Notes:
            LETTER_MAP is a complex nested structure for efficiency.
            This function digs through it to find the right probabilities.
        """
        result, i = None, 0
        while result is None and i < len(arr):
            item = arr[i]
            if len(item) > 2:
                type_code, val = item[2], item[0]
                if type_code == "ja":
                    # Compare substrings as numbers
                    s_bytes = ByteUtils.zxd(list(substr.encode('utf-8')), 4)
                    val_b = ByteUtils.zxd(list(str(val).encode('utf-8')), 4)
                    if ByteUtils.to_int32(s_bytes) > ByteUtils.to_int32(val_b):
                        result = Generator.recursive_search(item[1], substr)
                elif type_code == "jz" and str(val) == substr:
                    # Found exact match - extract weights
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
        Inserts a vowel into the name at specified position.

        Args:
            name (str): Current name
            seed (list): Random state for choosing which vowel
            index (int): Where to insert the vowel

        Returns:
            str: Name with vowel inserted

        Notes:
            Chooses randomly from a, e, i, o, u.
        """
        ByteUtils.update_seed(seed)
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)
        if calc and calc[0] < 5:
            if index <= len(name):
                return name[:index] + Generator.VOWELS[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name):
        """
        Finds where there are too many consonants in a row.

        Args:
            name (str): Name to check

        Returns:
            int: Position where 3+ consonants start, or -1 if none

        Notes:
            Consonants are letters that aren't a, e, i, o, u, or y.
            The game tries to avoid 3+ consonants in a row.
        """
        consonance = 0
        for i in range(len(name)):
            if consonance < 3:
                if name[i] not in Generator.VOWELS_WITH_Y:
                    consonance += 1
                else:
                    consonance = 0
            else:
                if name[i] not in Generator.VOWELS_WITH_Y:
                    return i - 3  # Found 4th consonant
                else:
                    consonance = 0
        return -1


class RegionNameGenerator:
    """
    Generates region names with special endings.

    Regions in No Man's Sky often have suffixes like "Conflux", "Void", or
    "Expanse". This class handles the special seed calculation for regions
    and adds these decorative endings.

    Attributes:
        PROC_ADORNMENTS (list): Possible suffixes for region names
        FALLBACK_NAME (str): Name to use if generation fails
        SCRAMBLE_MULT_1, SCRAMBLE_MULT_2 (list): Numbers for seed scrambling

    Example:
        >>> RegionNameGenerator.format_name(seed_bytes)
        'Ocopad Conflux'
    """
    # List of possible suffixes that can be added to region names
    PROC_ADORNMENTS = [
        "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
        "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
        "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
        "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux",
        "%NAME% Instability", "Sea of %NAME%", "The Arm of %NAME%",
        "%NAME% Spur", "%NAME% Shallows"
    ]
    FALLBACK_NAME = "Unknown Region"
    # Magic numbers for scrambling seeds (from game code analysis)
    SCRAMBLE_MULT_1 = [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64]
    SCRAMBLE_MULT_2 = [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3]

    @staticmethod
    def create_region_seed(x, y, z, galaxy):
        """
        Combines coordinates into a single seed value.

        Args:
            x (int): X coordinate
            y (int): Y coordinate
            z (int): Z coordinate
            galaxy (int): Galaxy number (0 for Euclid, 1 for Hilbert, etc.)

        Returns:
            list: Byte list representing the combined seed

        Notes:
            Format is: galaxy(2 hex) + y(2 hex) + z(3 hex) + x(3 hex)
            This exact order matters for matching game behavior.
        """
        s_gal = StringExtensions.short_to_formatted_hex(galaxy, 2)
        s_y = StringExtensions.short_to_formatted_hex(y, 2)
        s_z = StringExtensions.short_to_formatted_hex(z, 3)
        s_x = StringExtensions.short_to_formatted_hex(x, 3)
        hex_str = s_gal + s_y + s_z + s_x
        return ByteUtils.parse(hex_str)

    @staticmethod
    def format_name(seed):
        """
        Generates a full region name from seed bytes.

        Steps:
        1. Scramble the seed so similar coordinates don't get similar names
        2. Generate base name using Generator class
        3. Randomly decide whether to add a suffix (50% chance)
        4. Capitalize first letter

        Args:
            seed (list): Byte list representing region seed

        Returns:
            str: Generated region name like "Ocopad Conflux"

        Example:
            >>> seed = RegionNameGenerator.create_region_seed(100, 50, 200, 0)
            >>> RegionNameGenerator.format_name(seed)
            'Ocopad Conflux'
        """
        # Initialize random state containers
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # --- Seed Scrambling ---
        # Mix up the seed so similar coordinates give different names
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

        # Prepare final random state for name generation
        shl4 = ByteUtils.shl(register0, 4)
        xor_mid = ByteUtils.xor(
            ByteUtils.rol(shl4, 2), ByteUtils.shr(register0, 4)
        )
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure first part isn't zero
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])

        ByteUtils.update_seed(cache0)

        # Set name length parameters
        calc_len = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4)
        cache1[2] = ByteUtils.add(calc_len, [0x06])

        # --- Generate Base Name ---
        name = Generator.generate_name(cache0, cache1)
        if not name:
            return RegionNameGenerator.FALLBACK_NAME
        if "[" in name:  # Catch logic errors
            return name
        # Capitalize first letter
        name = name[0].upper() + name[1:]

        # --- Add Suffix (50% Chance) ---
        ByteUtils.update_seed(cache0)
        mult_check = ByteUtils.multiply(cache0[0], [0x64])
        should_adorn = ByteUtils.shr(mult_check, 4)[0] < 0x50  # 0x50 = 80 in decimal = 80/256 ≈ 31%

        if should_adorn:
            ByteUtils.update_seed(cache0)
            idx_cal = ByteUtils.multiply(cache0[0], [0x14])  # 0x14 = 20 in decimal
            idx = ByteUtils.shr(idx_cal, 4)[0]
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                adornment = RegionNameGenerator.PROC_ADORNMENTS[idx]
                name = adornment.replace("%NAME%", name)

        return name


class NMSGalaxyMap:
    """
    Converts portal glyphs to coordinates and calculates region info.

    Portal glyphs are 12-character hexadecimal codes found at portals in-game.
    This class decodes them into X, Y, Z coordinates, calculates distance from
    galaxy center, and determines which quadrant the region is in.

    Attributes:
        QUADRANT_MIDPOINT (int): Boundary between quadrants (2048 in hex)
        REGION_CENTER_XZ (int): Center point for X and Z coordinates
        REGION_CENTER_Y (int): Center point for Y coordinate

    Example:
        >>> map_logic = NMSGalaxyMap()
        >>> result = map_logic.glyphs_to_region_data("042F00780D56")
        >>> result['quadrant']
        'Alpha'
    """
    # Constants for coordinate calculations
    QUADRANT_MIDPOINT = 0x800  # 2048 in decimal - halfway point
    REGION_CENTER_XZ = 0x7FF   # 2047 in decimal - center for X and Z
    REGION_CENTER_Y = 0x7F     # 127 in decimal - center for Y

    def __init__(self):
        """
        Sets up coordinate conversion constants.

        Notes:
            The galaxy is 4096x256x4096 voxels (3D units) centered at
            (2047, 127, 2047). Positive wraps to negative at the boundaries.
        """
        # Center of galaxy in voxel units
        self.SHIFT_POS_XZ = 2049  # Where positive X/Z wraps to negative
        self.SHIFT_NEG_XZ = 2047  # Where negative X/Z wraps to positive
        self.SHIFT_POS_Y = 129    # Where positive Y wraps to negative
        self.SHIFT_NEG_Y = 127    # Where negative Y wraps to positive

        # Exact center point
        self.CENTER_X = 2047
        self.CENTER_Y = 127
        self.CENTER_Z = 2047

        # Conversion: 1 voxel = 400 light-years
        self.LY_SCALE = 400

    @staticmethod
    def _wrap_coordinate(value, shift_pos, shift_neg):
        """
        Wraps coordinates around the galaxy center.

        Args:
            value (int): Coordinate value
            shift_pos (int): Positive wrap point
            shift_neg (int): Negative wrap point

        Returns:
            int: Wrapped coordinate

        Notes:
            The galaxy wraps around like a torus (donut shape). Coordinates
            beyond SHIFT_POS wrap to negative, and below -SHIFT_NEG wrap positive.
        """
        return value - shift_pos if value >= shift_pos else value + shift_neg

    def glyphs_to_region_data(self, glyphs: str):
        """
        Converts portal glyphs to region information.

        Args:
            glyphs (str): 12-character hex code like "042F00780D56"

        Returns:
            dict or None: Region data dictionary, or None if glyphs invalid

        Notes:
            Glyph format: PPPSSSYYZZZXXX (P=planet, S=system, YYYZZZXXX=coords)
            We only use Y, Z, X parts for region coordinates.

        Example:
            >>> result = glyphs_to_region_data("042F00780D56")
            >>> result['coords_full']
            '042F:0078:0D56:xxxx'
        """
        g = glyphs.strip().upper()
        # Glyphs must be exactly 12 characters (0-9, A-F)
        if len(g) != 12:
            return None

        try:
            # Extract coordinate parts from glyph string
            # Format: [Planet][System][Y][Z][X]
            y_hex = int(g[4:6], 16)   # Characters 4-5: Y coordinate
            z_hex = int(g[6:9], 16)   # Characters 6-8: Z coordinate
            x_hex = int(g[9:12], 16)  # Characters 9-11: X coordinate
        except ValueError:
            return None  # Not valid hex numbers

        # Check that coordinates are within valid ranges
        if not (0x00 <= y_hex <= 0xFF) or not (0x000 <= z_hex <= 0xFFF) or not (0x000 <= x_hex <= 0xFFF):
            return None

        # Convert to voxel coordinates with wrapping
        voxel_x = self._wrap_coordinate(x_hex, self.SHIFT_POS_XZ, self.SHIFT_NEG_XZ)
        voxel_z = self._wrap_coordinate(z_hex, self.SHIFT_POS_XZ, self.SHIFT_NEG_XZ)
        voxel_y = self._wrap_coordinate(y_hex, self.SHIFT_POS_Y, self.SHIFT_NEG_Y)

        # Calculate distance from galaxy center using 3D distance formula
        distance_voxels = math.sqrt(
            (voxel_x - self.CENTER_X)**2 +
            (voxel_y - self.CENTER_Y)**2 +
            (voxel_z - self.CENTER_Z)**2
        )
        # Convert voxels to light-years
        distance_ly = int(distance_voxels * self.LY_SCALE)

        # Determine quadrant based on position relative to center
        is_right = x_hex >= self.QUADRANT_MIDPOINT  # Right of center
        is_bottom = z_hex >= self.QUADRANT_MIDPOINT  # Below center

        if not is_right and not is_bottom:
            quadrant = "Alpha"      # Top-left quadrant
        elif is_right and not is_bottom:
            quadrant = "Beta"       # Top-right quadrant
        elif not is_right and is_bottom:
            quadrant = "Gamma"      # Bottom-left quadrant
        else:
            quadrant = "Delta"      # Bottom-right quadrant

        return {
            "coords_full": f"{voxel_x:04X}:{voxel_y:04X}:{voxel_z:04X}:xxxx",
            "coords_split": {
                'x': f"{voxel_x:04X}", 'y': f"{voxel_y:04X}", 'z': f"{voxel_z:04X}"
            },
            "raw_values": {'x': voxel_x, 'y': voxel_y, 'z': voxel_z},
            "distance": f"{distance_ly:,}",  # Format with commas
            "quadrant": quadrant,
        }


class RegionInputModel(BaseModel):
    """
    Validates user input before generating wiki code.

    This class uses Pydantic to check that all required fields are filled
    and that glyphs are valid hexadecimal. It prevents bad data from being
    used in the wiki template.

    Attributes:
        name (str): Region name
        release (str): Game version when region was added
        image (str): Image filename for wiki
        civilized (str): Civilization occupying the region
        galaxy (str): Galaxy name
        portalglyphs (str): 12-character portal code
        coordinates (str): Full coordinate string
        distance (str): Distance from center in light-years
        quadrant (str): Which quadrant (Alpha, Beta, Gamma, Delta)
        coord_x, coord_y, coord_z (str): Individual coordinates

    Raises:
        ValidationError: If any field is invalid

    Example:
        >>> data = RegionInputModel(name="Test", release="Worlds", ...)
        # Raises error if any field is empty or glyphs invalid
    """
    name: str
    release: str
    image: str
    civilized: str
    galaxy: str
    portalglyphs: str
    coordinates: str
    distance: str
    quadrant: str
    coord_x: str
    coord_y: str
    coord_z: str

    @field_validator(
        'name', 'galaxy', 'coordinates', 'distance', 'quadrant'
    )
    def field_must_not_be_empty(cls, v):
        """
        Ensures required fields are not empty.

        Args:
            v (str): Field value to check

        Returns:
            str: Validated value

        Raises:
            ValueError: If value is empty or only whitespace
        """
        if not v or v.strip() == "":
            raise ValueError("This field cannot be empty.")
        return v

    @field_validator('portalglyphs')
    def validate_glyphs(cls, v):
        """
        Validates portal glyph format.

        Args:
            v (str): Glyph string to validate

        Returns:
            str: Uppercase glyph string

        Raises:
            ValueError: If not exactly 12 hex characters (0-9, A-F)
        """
        # Must be 12 characters, only hex digits allowed
        if not re.fullmatch(r"^[0-9A-F]{12}$", v.strip().upper()):
            raise ValueError("Must be a 12-character hexadecimal string.")
        return v.strip().upper()


@dataclass
class AppWidgets:
    """
    Container for all UI widgets.

    This dataclass holds references to all buttons, text boxes, and other
    interface elements so they can be easily accessed throughout the app.

    Attributes:
        Various widget fields for each UI element, organized by tab
    """
    # Tab 1: Region Data inputs
    name: Text = field(default=None)
    release: Text = field(default=None)
    image: Text = field(default=None)
    civilized: Text = field(default=None)

    # Tab 1: Calculated fields
    galaxy: Combobox = field(default=None)
    portalglyphs: Text = field(default=None)
    coordinates: Text = field(default=None)
    distance: Text = field(default=None)
    quadrant: Text = field(default=None)
    coord_x: Text = field(default=None)
    coord_y: Text = field(default=None)
    coord_z: Text = field(default=None)

    # Tab 2: Action buttons and output
    btn_preview: Button = field(default=None)
    btn_gen: Button = field(default=None)
    btn_copy: Button = field(default=None)
    btn_download: Button = field(default=None)
    btn_example: Button = field(default=None)
    btn_clear: Button = field(default=None)
    status_label: HTML = field(default=None)
    output_area: Output = field(default=None)


class NMSWikiRegionFormCreator:
    """
    Main application class with user interface.

    This class builds the entire widget interface with tabs, buttons, and
    input fields. It handles user interactions, calculates region data,
    and generates wiki code.

    Attributes:
        widgets (AppWidgets): Container for all UI elements
        map_logic (NMSGalaxyMap): Handles coordinate calculations
        jinja_env (Environment): Template engine for wiki code
        generated_content (str): Last generated wiki code
        WIKI_TEMPLATE (str): Template for wiki page markup
        DEFAULT_RELEASE (str): Default game version
        DEFAULT_IMAGE (str): Default image filename

    Example:
        >>> app = NMSWikiRegionFormCreator()
        >>> app.display_app()  # Shows the interface in Jupyter
    """
    # Wiki template with placeholders like {{ name }}, {{ coordinates }}, etc.
    WIKI_TEMPLATE = """
{{ '{{' }}Version|{{ release }}{{ '}}' }}
{{ '{{' }}Region infobox
| name = {{ name }}
| image = {{ image }}
| galaxy = {{ galaxy }}
| coordinates = {{ coordinates }}
| distance = {{ distance }}
| quadrant = {{ quadrant }}
| civilized = {{ civilized }}
| release = {{ release }}
{{ '}}' }}
'''{{ name }}''' is a region.

==Summary==
'''{{ name }}''' is a [[region]] in the [[{{ galaxy }}]] [[galaxy]].

==Regional Stats==
{{ '{{' }}CARGORegionStats|{{ name }}{{ '}}' }}

==Documented Systems==
{{ '{{' }}CARGORegionSystems|15{{ '}}' }}

==Other Systems==

==Location==
The center of this region is approximately {{ distance }} light years from the [[Galaxy Centre]] in the [[{{ quadrant }} Quadrant]].
{{ '{{' }}coords|{{ coord_x }}|{{ coord_y }}|{{ coord_z }}|xxxx{{ '}}' }}

===Adjoining Regions===
The following documented regions border {{ name }}:
{{ '{{' }}RegionNeighbours|coord={{ coordinates }}|gal={{ galaxy }}|release={{ release }}{{ '}}' }}

==Civilized Space==
{{ civilized }}

==Additional Information==
    """.strip()

    DEFAULT_RELEASE = "Breach"
    DEFAULT_IMAGE = "nmsMisc_NotAvailable.png"

    def __init__(self):
        """
        Sets up the application with UI, logic, and event handlers.
        """
        self.widgets = AppWidgets()
        self.map_logic = NMSGalaxyMap()
        self.jinja_env = Environment()
        self.generated_content = ""

        # Build the interface
        self._define_styles_and_layouts()
        self._setup_ui()
        self._connect_events()

    def _define_styles_and_layouts(self):
        """
        Defines CSS styles and layout settings for the UI.

        Notes:
            These styles make the interface look clean and organized in
            Jupyter/Colab notebooks.
        """
        # Style for section headers
        self.HEADER_STYLE = (
            "font-weight:bold; font-size:16px; margin-top:20px; "
            "border-bottom:2px solid #00ACC1; padding-bottom:5px; "
            "color:#006064;"
        )
        # Style for description text
        self.DESC_STYLE = (
            "font-style:italic; font-size:12px; color:#555; "
            "margin-bottom:12px; line-height:1.4em; background-color:#E0F7FA; "
            "padding:8px; border-left:4px solid #00BCD4; border-radius:4px;"
        )
        # Label width for form fields
        self.LABEL_STYLE = {'description_width': '140px'}
        # Layout for most widgets
        self.WIDGET_LAYOUT = Layout(width='98%')
        # Layout for two-column rows
        self.COL_LAYOUT = Layout(width='50%')
        # Layout for full-width rows
        self.FULL_ROW = Layout(width='100%', margin='5px 0')

    def _setup_ui(self):
        """
        Creates the main interface with tabs.

        Notes:
            Builds two tabs: one for data entry, one for code generation.
        """
        # Create content for each tab
        tab1 = self._create_tab_region_data()
        tab2 = self._create_tab_generate()

        # Create tab container with both tabs
        self.tabs = Tab(children=[tab1, tab2])
        self.tabs.set_title(0, 'Region Data')
        self.tabs.set_title(1, 'Export & Code')

    def _create_tab_region_data(self):
        """
        Builds the first tab with data entry fields.

        Returns:
            VBox: Container with all widgets for the first tab

        Notes:
            Organized into three sections:
            1. Location Source (galaxy and glyphs)
            2. Calculated Data (auto-filled from glyphs)
            3. Wiki Details (manual entry)
        """
        # --- SECTION 1: LOCATION SOURCE ---
        # Galaxy dropdown with search
        self._create_widget(
            Combobox, 'galaxy', 'Galaxy',
            options=NMSData.GALAXY_OPTIONS,
            placeholder='Start typing to search...'
        )
        # Portal glyphs input
        self._create_widget(
            Text, 'portalglyphs', 'Portal Glyphs',
            placeholder='e.g. 0801F9801802 (12 hex digits, 0-F)'
        )

        # Combine into first section
        section_inputs = VBox([
            self._header("Location Source"),
            self._desc(
                "Begin by selecting the galaxy and entering the 12-character "
                "Portal Glyphs found at a portal in the target region."
            ),
            self._two_col_row(self.widgets.galaxy, self.widgets.portalglyphs)
        ])

        # --- SECTION 2: AUTO-CALCULATED FIELDS ---
        # These fields are filled automatically when glyphs are entered
        self._create_widget(
            Text, 'name', 'Region Name',
            disabled=True, placeholder='e.g. Ocopad Conflux'
        )
        self._create_widget(
            Text, 'coordinates', 'Coordinates',
            disabled=True, placeholder='e.g. 042F:0078:0D56:0000'
        )
        self._create_widget(
            Text, 'distance', 'Distance (LY)',
            disabled=True, placeholder='e.g. 700,000'
        )
        self._create_widget(
            Text, 'quadrant', 'Quadrant',
            disabled=True, placeholder='e.g. Alpha'
        )
        # Hidden fields for template (accessed separately in wiki code)
        for coord in ['x', 'y', 'z']:
            self._create_widget(
                Text, f'coord_{coord}', coord.upper(),
                layout=Layout(display='none')
            )

        section_calculated = VBox([
            self._header("Regional Coordinates & Statistics"),
            self._desc(
                "Data in this section is derived mathematically from the "
                "inputs above. These fields ensure the wiki information "
                "matches in-game values."
            ),
            self._two_col_row(self.widgets.name, self.widgets.coordinates),
            self._two_col_row(self.widgets.distance, self.widgets.quadrant)
        ])

        # --- SECTION 3: WIKI DETAILS ---
        # Manual entry fields for wiki customization
        self._create_widget(
            Text, 'release', 'Release Version',
            value=self.DEFAULT_RELEASE, placeholder='e.g. Worlds Part I'
        )
        self._create_widget(
            Text, 'image', 'Infobox Image',
            value=self.DEFAULT_IMAGE, placeholder='e.g. MyRegion.png'
        )
        self._create_widget(
            Text, 'civilized', 'Occupying Civ',
            placeholder='(Optional) Name of Civ'
        )

        section_details = VBox([
            self._header("Wiki Configuration"),
            self._desc(
                "Configure visual and lore details for the infobox. These "
                "fields do not affect the mathematical location data."
            ),
            self._two_col_row(self.widgets.release, self.widgets.image),
            self._two_col_row(self.widgets.civilized, None)
        ])

        # Combine all sections into the tab
        return VBox([
            section_inputs,
            section_calculated,
            section_details
        ], layout=Layout(padding='20px'))

    def _create_tab_generate(self):
        """
        Builds the second tab with action buttons and output area.

        Returns:
            VBox: Container with buttons and output display
        """
        # Create action buttons with different styles and icons
        self.widgets.btn_preview = Button(
            description='Preview', button_style='info', icon='eye'
        )
        self.widgets.btn_gen = Button(
            description='Generate', button_style='success', icon='code'
        )
        self.widgets.btn_copy = Button(
            description='Copy Code', button_style='primary',
            icon='copy', disabled=True
        )
        self.widgets.btn_download = Button(
            description='Download File', button_style='primary',
            icon='download', disabled=True
        )
        self.widgets.btn_example = Button(
            description='Load Example', button_style='warning',
            icon='upload', tooltip='Load example region data'
        )
        self.widgets.btn_clear = Button(
            description='Reset Form', button_style='danger',
            icon='trash', tooltip='Reset all fields to defaults'
        )

        # Status display at top of tab
        self.widgets.status_label = HTML(
            "<div style='text-align:center; color:#555;'>"
            "<i>Status: Ready</i></div>"
        )

        # Output area for generated wiki code
        self.widgets.output_area = Output(
            layout={
                'border': '1px solid #ccc', 'height': '400px',
                'overflow_y': 'scroll', 'padding': '10px'
            }
        )

        # Combine everything into the tab
        return VBox([
            self._header("Export & Code Generation"),
            self._desc(
                "Generate the wiki code based on the data entered in the "
                "Region Data tab."
            ),
            # Top row of main action buttons
            HBox([
                self.widgets.btn_preview, self.widgets.btn_gen,
                self.widgets.btn_copy, self.widgets.btn_download
            ], layout=Layout(justify_content='center', margin='15px 0')),
            # Bottom row of utility buttons
            HBox([
                self.widgets.btn_example, self.widgets.btn_clear
            ], layout=Layout(justify_content='center', margin='10px 0')),
            self.widgets.status_label,
            self._header('Code Output'),
            self.widgets.output_area,
        ], layout=Layout(padding='20px'))

    def _create_widget(self, widget_class, key, description, **kwargs):
        """
        Helper to create a widget and store it in self.widgets.

        Args:
            widget_class: Type of widget (Text, Combobox, etc.)
            key (str): Attribute name in AppWidgets
            description (str): Label shown next to widget
            **kwargs: Additional widget parameters

        Notes:
            Applies consistent styling and layout to all widgets.
        """
        # Base parameters for all widgets
        params = {
            'description': description,
            'style': self.LABEL_STYLE,
            'layout': self.WIDGET_LAYOUT
        }
        params.update(kwargs)

        # Special handling for Combobox
        if widget_class is Combobox and 'options' in params:
            params.setdefault('ensure_option', False)

        # Create widget and store reference
        widget_instance = widget_class(**params)
        setattr(self.widgets, key, widget_instance)

    def _header(self, text):
        """
        Creates a styled section header.

        Args:
            text (str): Header text

        Returns:
            HTML: Styled header widget
        """
        return HTML(f"<div style='{self.HEADER_STYLE}'>{text}</div>")

    def _desc(self, text):
        """
        Creates a styled description text block.

        Args:
            text (str): Description text

        Returns:
            HTML: Styled description widget
        """
        return HTML(f"<div style='{self.DESC_STYLE}'>{text}</div>")

    def _two_col_row(self, w1, w2=None):
        """
        Arranges two widgets side by side.

        Args:
            w1: First widget
            w2: Second widget (optional)

        Returns:
            HBox: Container with both widgets in columns
        """
        c1 = VBox([w1], layout=self.COL_LAYOUT)
        c2 = VBox([w2] if w2 else [], layout=self.COL_LAYOUT)
        return HBox([c1, c2], layout=self.FULL_ROW)

    def _connect_events(self):
        """
        Connects widgets to their event handlers.

        Notes:
            Called during initialization to wire up the interface.
        """
        # Update calculations when glyphs or galaxy change
        self.widgets.portalglyphs.observe(
            self._on_glyphs_change, names='value'
        )
        self.widgets.galaxy.observe(
            self._on_glyphs_change, names='value'
        )

        # Connect buttons to their functions
        self.widgets.btn_preview.on_click(
            lambda _button: self._generate_code(mode='preview')
        )
        self.widgets.btn_gen.on_click(
            lambda _button: self._generate_code(mode='full')
        )
        self.widgets.btn_copy.on_click(self._copy_to_clipboard)
        self.widgets.btn_download.on_click(self._download_file)
        self.widgets.btn_clear.on_click(self._clear_form)
        self.widgets.btn_example.on_click(self._load_example)

    def _on_glyphs_change(self, change):
        """
        Called when portal glyphs or galaxy selection changes.

        Args:
            change: Widget change event (not directly used)

        Notes:
            Recalculates coordinates, distance, quadrant, and region name
            whenever the user changes glyphs or galaxy.
        """
        # Get current glyphs and clean them up
        glyphs = self.widgets.portalglyphs.value.strip().upper()

        # If glyphs aren't complete, clear calculated fields
        if len(glyphs) != 12:
            self.widgets.coordinates.value = ""
            self.widgets.distance.value = ""
            self.widgets.quadrant.value = ""
            self.widgets.name.value = ""
            self._update_status("Ready", "info")
            return

        try:
            # Convert glyphs to region data
            result = self.map_logic.glyphs_to_region_data(glyphs)
            if result:
                # Update calculated fields
                self.widgets.coordinates.value = result['coords_full']
                self.widgets.distance.value = result['distance']
                self.widgets.quadrant.value = result['quadrant']
                self.widgets.coord_x.value = result['coords_split']['x']
                self.widgets.coord_y.value = result['coords_split']['y']
                self.widgets.coord_z.value = result['coords_split']['z']

                # Generate region name from coordinates
                self._calculate_region_name(result['raw_values'])
            else:
                # Invalid glyphs entered
                self.widgets.coordinates.value = "Invalid Hex Code"
                self.widgets.distance.value = ""
                self.widgets.quadrant.value = ""
                self.widgets.name.value = "Invalid Hex Code"
                self._update_status("Invalid portal glyph code entered.", "warning")
        except Exception as e:
            # Clear everything on any error
            self.widgets.coordinates.value = ""
            self.widgets.distance.value = ""
            self.widgets.quadrant.value = ""
            self.widgets.name.value = ""
            self._update_status(f"Error processing glyphs: {e}", "error")

    def _calculate_region_name(self, raw_coords):
        """
        Calculates procedural region name from coordinates.

        Args:
            raw_coords (dict): Dictionary with 'x', 'y', 'z' keys

        Notes:
            Uses the same algorithm as the game to ensure names match.
        """
        # Check if data files loaded successfully
        if not NMSData.ALPHASETS[0]:
            self.widgets.name.value = "Error: Alphasets not loaded."
            return

        if not NMSData.LETTER_MAP:
            self.widgets.name.value = "Error: Letter map not loaded."
            return

        # Get galaxy number from galaxy name
        galaxy_name = self.widgets.galaxy.value
        galaxy_index = NMSData.GALAXY_NAME_TO_INDEX.get(galaxy_name.lower(), 0)

        # Adjust coordinates relative to region center
        # Game's name generation expects coordinates relative to center
        x = raw_coords['x'] - self.map_logic.REGION_CENTER_XZ
        y = raw_coords['y'] - self.map_logic.REGION_CENTER_Y
        z = raw_coords['z'] - self.map_logic.REGION_CENTER_XZ

        try:
            # Create seed from coordinates and generate name
            seed = RegionNameGenerator.create_region_seed(x, y, z, galaxy_index)
            name = RegionNameGenerator.format_name(seed)
            self.widgets.name.value = name
        except Exception as e:
            self.widgets.name.value = "Generation Error"
            self._update_status(f"Name generation failed: {e}", "error")

    def _clear_form(self, _button=None):
        """
        Resets all form fields to default values.

        Args:
            _button: Button that triggered the event (not used)

        Notes:
            Also clears the output area and resets button states.
        """
        # Clear all widget values
        for widget_field in fields(self.widgets):
            widget = getattr(self.widgets, widget_field.name)
            if hasattr(widget, 'value') and isinstance(widget.value, str):
                widget.value = ""

        # Clear hidden coordinate fields
        self.widgets.coord_x.value = ""
        self.widgets.coord_y.value = ""
        self.widgets.coord_z.value = ""

        # Reset to defaults
        self.widgets.release.value = self.DEFAULT_RELEASE
        self.widgets.image.value = self.DEFAULT_IMAGE

        # Clear output
        self.widgets.output_area.clear_output()
        self.generated_content = ""

        # Disable action buttons until new code is generated
        self.widgets.btn_download.disabled = True
        self.widgets.btn_copy.disabled = True

        self._update_status("Form cleared.", "info")

    def _load_example(self, _button=None):
        """
        Fills form with example data for testing.

        Args:
            _button: Button that triggered the event (not used)

        Notes:
            Uses real glyphs from the game to demonstrate the tool.
        """
        # Clear first, then set example values
        self._clear_form(_button)
        self.widgets.release.value = "Worlds Part II"
        self.widgets.image.value = "Duhaasen_Expanse.png"
        self.widgets.civilized.value = "Alliance of Galactic Travellers"

        # Set galaxy and glyphs (real example from game)
        self._safe_set_selection_value(self.widgets.galaxy, "Euclid")
        self.widgets.portalglyphs.value = "0801F9801802"

        # Switch to first tab to show the filled form
        self.tabs.selected_index = 0
        self._update_status("Example data loaded.", "info")

    def _copy_to_clipboard(self, _button=None):
        """
        Copies generated wiki code to clipboard using JavaScript.

        Args:
            _button: Button that triggered the event (not used)

        Notes:
            Only works in Jupyter/Colab environments with JavaScript support.
        """
        if not self.generated_content:
            self._update_status("Please generate code first before copying.", "error")
            return

        # Escape special characters for JavaScript
        js_safe_content = self.generated_content.replace(
            '\\', '\\\\'
        ).replace("`", "\\`").replace("${", "\\${").replace("\n", "\\n").replace("\r", "\\r")

        # Use JavaScript to copy to clipboard
        display(Javascript(
            f'navigator.clipboard.writeText(`{js_safe_content}`)'
        ))
        self._update_status("Success! Code copied to clipboard.", "success")

    def _download_file(self, _button=None):
        """
        Downloads generated code as a text file (Google Colab only).

        Args:
            _button: Button that triggered the event (not used)

        Notes:
            Requires Google Colab environment. Falls back to warning otherwise.
        """
        if not self.generated_content:
            self._update_status(
                "Please generate code first before downloading.", "error"
            )
            return

        try:
            # Try to import Colab files module
            from google.colab import files

            # Create safe filename from region name
            safe_name = re.sub(
                r'[^a-zA-Z0-9_\-]', '',
                self.widgets.name.value.replace(' ', '_')
            )
            filename = f"{safe_name or 'Untitled'}_Region.txt"

            # Write content to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.generated_content)

            # Trigger download in Colab
            files.download(filename)
            self._update_status(f"Downloading '{filename}'...", "success")
        except ImportError:
            # Not in Colab - show warning
            self._update_status(
                "File download is only available in Google Colab. Use 'Copy Code' instead.", "warning"
            )

    def _generate_code(self, mode):
        """
        Generates wiki code from form data.

        Args:
            mode (str): 'preview' to show only, 'full' to enable download

        Notes:
            Validates input first, then fills the wiki template.
        """
        # Get list of fields that need validation
        valid_keys = set(RegionInputModel.model_fields.keys())
        form_data = {}

        # Collect values from all relevant widgets
        for widget_field in fields(self.widgets):
            widget = getattr(self.widgets, widget_field.name)
            if widget_field.name in valid_keys and hasattr(widget, 'value'):
                form_data[widget_field.name] = widget.value

        try:
            # Validate input using Pydantic model
            model = RegionInputModel(**form_data)
        except ValidationError as e:
            # Show first error to user
            first_error = e.errors()[0]
            err_msg = (
                f"Check the Region Data tab. Validation Error on '{first_error['loc'][0]}': "
                f"{first_error['msg']}"
            )
            self._update_status(err_msg, "error")
            return

        # Fill template with validated data
        template = self.jinja_env.from_string(self.WIKI_TEMPLATE)
        self.generated_content = template.render(**model.model_dump())

        # Display generated code in output area
        with self.widgets.output_area:
            clear_output(wait=True)
            print(self.generated_content)

        # Update button states based on mode
        if mode == 'full':
            self.widgets.btn_download.disabled = False
            self.widgets.btn_copy.disabled = False
            self._update_status(
                "Code generated successfully. Ready to copy or download.",
                "success"
            )
        else:
            # Preview mode - enable copy but not download
            self.widgets.btn_copy.disabled = False
            self._update_status("Preview generated successfully.", "preview")

    def _safe_set_selection_value(self, widget, value):
        """
        Safely sets value of a selection widget.

        Args:
            widget: Widget to update
            value: Value to set
        """
        widget.value = value

    def _update_status(self, message, level='info'):
        """
        Updates the status message with color coding.

        Args:
            message (str): Status message to display
            level (str): 'info', 'success', 'warning', 'error', or 'preview'

        Notes:
            Different levels show different colors and prefixes.
        """
        # Color and prefix for each level
        color_map = {
            'error': ('red', '<b>Error:</b>'),
            'success': ('green', '<b>Success:</b>'),
            'warning': ('orange', '<b>Warning:</b>'),
            'preview': ('blue', '<i>Status:</i>'),
            'info': ('#555', '<i>Status:</i>')
        }
        color, prefix = color_map.get(level, ('#555', '<i>Status:</i>'))

        # Update status label
        self.widgets.status_label.value = (
            f"<div style='text-align:center; color:{color};'>"
            f"{prefix} {message}</div>"
        )

    def display_app(self):
        """
        Displays the application interface.

        Call this method to show the widget in Jupyter/Colab.
        """
        display(self.tabs)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Create and display the application when run directly
    app = NMSWikiRegionFormCreator()
    app.display_app()