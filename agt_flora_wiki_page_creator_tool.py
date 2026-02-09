"""
No Man's Sky Flora Wiki Generator

This module provides a complete interactive tool for generating wiki pages for flora
(plants) discovered in the game No Man's Sky. It includes functionality to:
- Convert portal glyphs to 3D coordinates in the game universe
- Generate procedural region names based on galactic coordinates
- Create properly formatted wiki markup with game data
- Provide an interactive web interface using Jupyter widgets
- Load game data from remote repositories for accuracy

The main classes are:
- NMSFloraWikiGenerator: Main application with interactive UI
- FloraDataModel: Data validation model for flora information
- RegionNameGenerator: Generates procedural region names from coordinates
- NMSGalaxyMap: Converts portal glyphs to 3D voxel coordinates
- Generator: Core algorithm for procedural name generation
- ByteUtils: Helper class for byte-level operations
- NMSData: Loads and manages game data from external sources
"""

import json
import operator
import re
import struct
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import arrow
import ipywidgets as widgets
import requests
from IPython.display import Javascript, clear_output, display
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Constants used in procedural name generation and coordinate calculations
# These values are game-specific and derived from No Man's Sky's internal algorithms

# Multiplier used in seed generation for procedural names
SEED_MULTIPLIER = [0x99, 0xF8, 0x76, 0x5A]

# Very small double value used as a scaling factor in probability calculations
TINY_DOUBLE_BYTES = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]

# Hash multipliers used in region name generation to create variation
HASH_MULTIPLIER_1 = [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64]
HASH_MULTIPLIER_2 = [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3]

# Bit masks for limiting values to specific byte sizes
BYTE_MASK = 0xFF          # Mask for single byte (0-255)
UINT16_MASK = 0xFFFF      # Mask for 16-bit unsigned integer
UINT16_MOD = 65536        # Modulus for 16-bit calculations (2^16)
INT16_OFFSET = 32768      # Offset to convert signed to unsigned 16-bit

# Voxel coordinate center points - used to adjust glyph coordinates to game's coordinate system
# These are the "zero points" in the game's 3D coordinate space
VOXEL_CENTER_XZ = 0x7FF   # Center for X and Z coordinates (horizontal plane)
VOXEL_CENTER_Y = 0x7F     # Center for Y coordinate (vertical axis)

# Limits for name generation to prevent overly long or infinite processing
MAX_NAME_LENGTH = 63              # Maximum characters in a generated name
MAX_BACKTRACK_ATTEMPTS = 50       # Maximum attempts to fix invalid name parts

# Constants for date formatting and default values
STARDATE_YEAR_OFFSET = 1716       # Year offset for AGT (Alliance of Galactic Travellers) stardate format
DEFAULT_CIVILIZATION = "Alliance of Galactic Travellers"  # Default civilization for discoveries
DEFAULT_RELEASE = "Breach"        # Default game release version
FALLBACK_REGION_NAME = "Unknown Region"  # Fallback when region name generation fails

# Special sentinel value indicating an empty result from name generation
EMPTY_SENTINEL = "__EMPTY__"

# Threshold for adding adornments (descriptive suffixes) to region names
# Lower values mean more regions get adornments
ADORNMENT_THRESHOLD = 0x50

# Character sets for vowel insertion rules in name generation
VOWELS = "aeiou"          # Standard vowels for basic vowel insertion
VOWELS_WITH_Y = "aeiouy"  # Vowels including 'y' for consecutive consonant checking

# UI-related constants
PLACEHOLDER_PREFIXES = ("Select ", "Type ")  # Text prefixes that indicate placeholder values in dropdowns

# Valid consonant pairs for name generation - determines which consonants can appear together
# These rules make generated names sound more natural and pronounceable
VALID_INITIAL_PAIRS = {'h': set('ctw'), 'l': set('bcfgps'), 'r': set('bcdfgkpt'), 'w': set('dgt'), 'y': set('hmr')}
VALID_FINAL_PAIRS = {'b': set('gn'), 'd': set('bdfghkmpst'), 'g': set('l'), 'p': set('bdhkt'), 'r': set('bfg'), 't': set('g')}


def sanitize_filename(name: str) -> str:
    """
    Converts a string to a safe filename by removing special characters.

    This function takes any string and converts it to a filename-safe version
    by replacing spaces with underscores and removing any characters that aren't
    alphanumeric, underscore, or hyphen. This prevents issues when saving files
    to disk.

    Args:
        name (str): The original string to convert to a filename.

    Returns:
        str: A sanitized version of the input string safe for use as a filename.

    Example:
        >>> sanitize_filename("My Flora: Test Subject")
        'My_Flora_Test_Subject'
    """
    # Replace spaces with underscores first, then remove any remaining unsafe characters
    return re.sub(r'[^a-zA-Z0-9_\-]', '', name.replace(' ', '_'))


class ByteUtils:
    """
    Utility class for byte-level operations used in procedural generation.

    This class provides static methods for manipulating lists of bytes (integers
    in range 0-255) that represent numbers in the game's algorithms. It handles
    operations like addition, subtraction, multiplication, and bitwise operations
    on variable-length byte arrays, simulating how the game engine processes data.

    The methods work on little-endian byte arrays (least significant byte first),
    which matches the game's internal data representation.
    """

    @staticmethod
    def parse(val: str, little_endian: bool = True) -> List[int]:
        """
        Converts a hexadecimal string to a list of byte values.

        Takes a string of hexadecimal characters (like "1A2B3C") and converts it
        to a list of integer byte values (0-255). If the string has an odd length,
        it pads with a leading zero. By default, returns bytes in little-endian
        order (least significant byte first).

        Args:
            val (str): Hexadecimal string to parse.
            little_endian (bool, optional): If True, reverse byte order (little-endian).
                Defaults to True.

        Returns:
            List[int]: List of byte values (0-255).

        Example:
            >>> ByteUtils.parse("1A2B", little_endian=True)
            [43, 26]
            >>> ByteUtils.parse("1A2B", little_endian=False)
            [26, 43]
        """
        # Pad with leading zero if odd number of characters
        if len(val) % 2 != 0:
            val = "0" + val

        # Convert each pair of hex characters to a byte value
        res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]

        # Reverse for little-endian (game's internal format)
        if little_endian:
            res.reverse()

        return res

    @staticmethod
    def format_short(op1: List[int]) -> List[int]:
        """
        Ensures a byte list has at least 2 bytes (16-bit).

        Pads a byte array with zeros if it's shorter than 2 bytes. This is used
        when the algorithm expects at least 2 bytes for 16-bit operations.

        Args:
            op1 (List[int]): Input byte array.

        Returns:
            List[int]: Byte array with at least 2 bytes.

        Example:
            >>> ByteUtils.format_short([0x01])
            [1, 0]
        """
        res = list(op1)
        # Add zero bytes until we have at least 2 bytes
        while len(res) < 2:
            res.append(0x00)
        return res

    @staticmethod
    def add(op1: List[int], op2: List[int]) -> List[int]:
        """
        Adds two byte arrays together, handling carry between bytes.

        Performs byte-by-byte addition with carry propagation. This simulates
        adding two multi-byte numbers where each element in the list represents
        one byte (8 bits) of the number.

        Args:
            op1 (List[int]): First byte array (added to op2).
            op2 (List[int]): Second byte array (base to add to).

        Returns:
            List[int]: Result byte array, may be longer than inputs if carry overflow.

        Example:
            >>> ByteUtils.add([0xFF, 0x01], [0x01, 0x00])
            [0, 2, 0]  # Because 0xFF + 0x01 = 0x100 with carry
        """
        # Start with copy of second operand
        result = list(op2)

        # Add each byte of first operand to corresponding position
        for i, val in enumerate(op1):
            result = ByteUtils._add_single(val, result, i)

        return result

    @staticmethod
    def _add_single(val: int, target_list: List[int], index: int) -> List[int]:
        """
        Helper method to add a single byte value at a specific position.

        Adds 'val' to the byte at position 'index' in 'target_list', handling
        carry to higher bytes if the sum exceeds 255 (one byte).

        Args:
            val (int): Byte value to add (0-255).
            target_list (List[int]): List of bytes to modify.
            index (int): Position in list where addition happens.

        Returns:
            List[int]: Modified byte list with carry handled.
        """
        # If position exists in list, add and handle carry
        if index < len(target_list):
            total = val + target_list[index]
            # Keep only lower 8 bits (0-255)
            target_list[index] = total & BYTE_MASK
            # Calculate carry (bits 8+)
            rem = (total >> 8) & BYTE_MASK

            # If there's carry, add it to next byte position
            if rem != 0:
                target_list = ByteUtils._add_single(rem, target_list, index + 1)
        else:
            # If position doesn't exist, append the value
            target_list.append(val)

        return target_list

    @staticmethod
    def sub(op1: List[int], op2: List[int]) -> List[int]:
        """
        Subtracts one byte array from another, handling borrow between bytes.

        Subtracts op1 from op2 (op2 - op1) with borrow propagation. This is
        the inverse of the add operation for multi-byte numbers.

        Args:
            op1 (List[int]): Byte array to subtract (subtractor).
            op2 (List[int]): Byte array to subtract from (minuend).

        Returns:
            List[int]: Result byte array.

        Example:
            >>> ByteUtils.sub([0x01], [0x02, 0x00])
            [1, 255]  # Because 0x200 - 0x01 = 0x1FF
        """
        result = list(op2)

        # Subtract each byte of first operand from corresponding position
        for i, val in enumerate(op1):
            result = ByteUtils._sub_single(val, result, i)

        return result

    @staticmethod
    def _sub_single(val: int, target_list: List[int], index: int) -> List[int]:
        """
        Helper method to subtract a single byte value at a specific position.

        Subtracts 'val' from the byte at position 'index' in 'target_list',
        handling borrow from higher bytes if needed.

        Args:
            val (int): Byte value to subtract (0-255).
            target_list (List[int]): List of bytes to modify.
            index (int): Position in list where subtraction happens.

        Returns:
            List[int]: Modified byte list with borrow handled.
        """
        if index < len(target_list):
            # Calculate difference (may be negative, causing borrow)
            diff = val - target_list[index]
            # Keep only lower 8 bits using two's complement
            target_list[index] = diff & BYTE_MASK
            # Calculate borrow
            rem = (diff >> 8) & BYTE_MASK

            # If there's borrow, propagate to next byte
            if rem != 0:
                target_list = ByteUtils._sub_single(rem, target_list, index + 1)
        else:
            # If position doesn't exist, append the value
            target_list.append(val)

        return target_list

    @staticmethod
    def multiply(op1: List[int], op2: List[int]) -> List[int]:
        """
        Multiplies two byte arrays together using grade-school multiplication.

        Performs multiplication of two multi-byte numbers by multiplying each
        byte of op1 with each byte of op2 and summing the results at appropriate
        positions. This is a standard long multiplication algorithm adapted for
        bytes with carry handling.

        Args:
            op1 (List[int]): First byte array (multiplicand).
            op2 (List[int]): Second byte array (multiplier).

        Returns:
            List[int]: Product as byte array.

        Note:
            This uses signed 16-bit intermediate calculations to match the
            game's algorithm exactly.
        """
        result = []

        # For each byte in first operand
        for i, v1 in enumerate(op1):
            rem = 0  # Carry from previous multiplication

            # Multiply with each byte in second operand
            for j, v2 in enumerate(op2):
                # Multiply bytes and add previous carry
                raw_prod = (v1 * v2) + rem

                # Convert to signed 16-bit for game's algorithm
                # The game uses signed arithmetic for some reason
                signed_prd = (raw_prod + INT16_OFFSET) % UINT16_MOD - INT16_OFFSET

                # Extract carry (upper 8 bits) and result (lower 8 bits)
                rem = (signed_prd >> 8) & BYTE_MASK
                res = signed_prd & BYTE_MASK

                # Position in result array = i + j (like decimal multiplication)
                idx = i + j

                # Add to existing result or append new byte
                if idx < len(result):
                    result = ByteUtils._add_single(res, result, idx)
                else:
                    result.append(res)

            # Handle any remaining carry after inner loop
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
        Shifts bytes left (toward more significant positions).

        Removes the first 'shift' bytes from the array, effectively shifting
        left by 8*shift bits. This is like bitwise left shift but on byte
        boundaries.

        Args:
            op1 (List[int]): Byte array to shift.
            shift (int): Number of bytes to remove from beginning.

        Returns:
            List[int]: Shifted byte array, or [0x00] if all bytes shifted out.

        Example:
            >>> ByteUtils.shl([1, 2, 3], 1)
            [2, 3]
        """
        # Remove first 'shift' bytes, return single zero if empty
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1: List[int], shift: int) -> List[int]:
        """
        Shifts bytes right (toward less significant positions).

        Removes the last 'shift' bytes from the array, effectively shifting
        right by 8*shift bits. This is like bitwise right shift but on byte
        boundaries.

        Args:
            op1 (List[int]): Byte array to shift.
            shift (int): Number of bytes to remove from end.

        Returns:
            List[int]: Shifted byte array, or [0x00] if all bytes shifted out.

        Example:
            >>> ByteUtils.shr([1, 2, 3], 1)
            [1, 2]
        """
        # Remove last 'shift' bytes, return single zero if empty
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1: List[int], roll: int) -> List[int]:
        """
        Rotates bytes left within the array.

        Moves bytes from the beginning to the end, preserving all bytes.
        This is like a circular shift on byte boundaries.

        Args:
            op1 (List[int]): Byte array to rotate.
            roll (int): Number of positions to rotate left.

        Returns:
            List[int]: Rotated byte array.

        Example:
            >>> ByteUtils.rol([1, 2, 3, 4], 1)
            [2, 3, 4, 1]
        """
        if not op1:
            return op1

        # Handle rotation greater than array length
        r = roll % len(op1)
        # Move first r bytes to end
        return op1[r:] + op1[:r]

    @staticmethod
    def zxd(op1: List[int], extend: int) -> List[int]:
        """
        Zero-extends a byte array to specified length.

        Adds zero bytes to the end of the array until it reaches 'extend' length.
        Used when algorithm needs fixed-size arrays but we have variable input.

        Args:
            op1 (List[int]): Byte array to extend.
            extend (int): Desired total length.

        Returns:
            List[int]: Extended array padded with zeros.

        Example:
            >>> ByteUtils.zxd([1, 2], 4)
            [1, 2, 0, 0]
        """
        # Start with copy, add zeros until desired length
        result = list(op1)
        return result + [0x00] * (extend - len(result))

    @staticmethod
    def sxd(op1: List[int], extend: int) -> List[int]:
        """
        Sign-extends a byte array to specified length.

        Adds bytes to the end based on the sign of the original number.
        If the most significant bit of the last byte is 1 (negative in two's
        complement), adds 0xFF bytes. Otherwise adds 0x00 bytes.

        Args:
            op1 (List[int]): Byte array to extend.
            extend (int): Desired total length.

        Returns:
            List[int]: Sign-extended array.

        Example:
            >>> ByteUtils.sxd([0xFF], 2)  # 0xFF is -1 in two's complement
            [255, 255]
        """
        result = list(op1)

        # Check sign bit of most significant byte (bit 7 = 128)
        if len(op1) > 0 and (op1[-1] >> 7) == 1:
            val = BYTE_MASK  # 0xFF for negative extension
        else:
            val = 0x00  # 0x00 for positive extension

        # Add sign-extension bytes
        for _ in range(extend - len(op1)):
            result.append(val)

        return result

    @staticmethod
    def logical_op(op1: List[int], op2: List[int], mode: int) -> List[int]:
        """
        Performs bitwise logical operations on two byte arrays.

        Applies AND, OR, or XOR operation byte-by-byte to two arrays.
        If arrays are different lengths, the shorter one is zero-extended.

        Args:
            op1 (List[int]): First byte array.
            op2 (List[int]): Second byte array.
            mode (int): Operation mode: 0=AND, 1=OR, 2=XOR.

        Returns:
            List[int]: Result of bitwise operation.

        Raises:
            KeyError: If mode is not 0, 1, or 2.
        """
        len1, len2 = len(op1), len(op2)

        # Make both arrays same length by zero-extending shorter one
        if len1 > len2:
            longer, shorter = list(op1), list(op2) + [0x00] * (len1 - len2)
        else:
            longer, shorter = list(op2), list(op1) + [0x00] * (len2 - len1)

        # Map mode numbers to Python bitwise operators
        ops = {0: operator.and_, 1: operator.or_, 2: operator.xor}
        op_func = ops[mode]

        # Apply operation to each byte pair
        res = []
        for i, val_l in enumerate(longer):
            res.append(op_func(val_l, shorter[i]))

        return res

    @staticmethod
    def xor(op1: List[int], op2: List[int]) -> List[int]:
        """
        Bitwise XOR of two byte arrays.

        Args:
            op1 (List[int]): First byte array.
            op2 (List[int]): Second byte array.

        Returns:
            List[int]: XOR result.
        """
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1: List[int], op2: List[int]) -> List[int]:
        """
        Bitwise AND of two byte arrays.

        Args:
            op1 (List[int]): First byte array.
            op2 (List[int]): Second byte array.

        Returns:
            List[int]: AND result.
        """
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1: List[int], op2: List[int]) -> List[int]:
        """
        Bitwise OR of two byte arrays.

        Args:
            op1 (List[int]): First byte array.
            op2 (List[int]): Second byte array.

        Returns:
            List[int]: OR result.
        """
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache: List[List[int]], move: int = 1) -> List[List[int]]:
        """
        Updates the procedural generation seed state.

        This is a core function in the name generation algorithm. It takes the
        current seed state (two-part cache) and advances it by 'move' steps,
        mixing the values using multiplication and addition. Each call produces
        a new pseudo-random state for the next part of name generation.

        Args:
            cache (List[List[int]]): Current seed state as [cache0, cache1].
            move (int, optional): Number of seed update steps to perform.
                Defaults to 1.

        Returns:
            List[List[int]]: Updated seed state.

        Note:
            This algorithm mimics the game's internal random number generation
            for procedural content.
        """
        for _ in range(move):
            # Step 1: Multiply cache0 with fixed multiplier
            step1 = ByteUtils.multiply(cache[0], SEED_MULTIPLIER)

            # Step 2: Add cache1 to the result
            result = ByteUtils.add(step1, cache[1])

            # Step 3: Update cache0 with left-shifted part
            cache[0] = ByteUtils.shl(result, 4)

            # Step 4: Update cache1 with right-shifted part
            cache[1] = ByteUtils.shr(result, 4)

        return cache

    @staticmethod
    def _pad(arr: List[int], length: int) -> List[int]:
        """
        Helper method to pad byte array with zeros to exact length.

        Args:
            arr (List[int]): Byte array to pad.
            length (int): Desired length.

        Returns:
            List[int]: Padded array.
        """
        result = list(arr)
        while len(result) < length:
            result.append(0)
        return result

    @staticmethod
    def to_uint32(arr: List[int], offset: int = 0) -> int:
        """
        Converts 4 bytes to an unsigned 32-bit integer.

        Interprets next 4 bytes starting at offset as little-endian unsigned int.

        Args:
            arr (List[int]): Byte array to convert.
            offset (int, optional): Starting position in array. Defaults to 0.

        Returns:
            int: Unsigned 32-bit integer value.
        """
        # Pad to 4 bytes if needed, then unpack as unsigned int
        return struct.unpack('<I', bytes(ByteUtils._pad(arr[offset:offset + 4], 4)))[0]

    @staticmethod
    def to_int32(arr: List[int], offset: int = 0) -> int:
        """
        Converts 4 bytes to a signed 32-bit integer.

        Interprets next 4 bytes starting at offset as little-endian signed int.

        Args:
            arr (List[int]): Byte array to convert.
            offset (int, optional): Starting position in array. Defaults to 0.

        Returns:
            int: Signed 32-bit integer value.
        """
        # Pad to 4 bytes if needed, then unpack as signed int
        return struct.unpack('<i', bytes(ByteUtils._pad(arr[offset:offset + 4], 4)))[0]

    @staticmethod
    def to_int16(arr: List[int], offset: int = 0) -> int:
        """
        Converts 2 bytes to a signed 16-bit integer.

        Interprets next 2 bytes starting at offset as little-endian signed short.

        Args:
            arr (List[int]): Byte array to convert.
            offset (int, optional): Starting position in array. Defaults to 0.

        Returns:
            int: Signed 16-bit integer value.
        """
        # Pad to 2 bytes if needed, then unpack as signed short
        return struct.unpack('<h', bytes(ByteUtils._pad(arr[offset:offset + 2], 2)))[0]

    @staticmethod
    def to_double(arr: List[int], offset: int = 0) -> float:
        """
        Converts 8 bytes to a double-precision floating point number.

        Interprets next 8 bytes starting at offset as little-endian double.

        Args:
            arr (List[int]): Byte array to convert.
            offset (int, optional): Starting position in array. Defaults to 0.

        Returns:
            float: Double-precision floating point value.
        """
        # Pad to 8 bytes if needed, then unpack as double
        return struct.unpack('<d', bytes(ByteUtils._pad(arr[offset:offset + 8], 8)))[0]

    @staticmethod
    def to_single(arr: List[int], offset: int = 0) -> float:
        """
        Converts 4 bytes to a single-precision floating point number.

        Interprets next 4 bytes starting at offset as little-endian float.

        Args:
            arr (List[int]): Byte array to convert.
            offset (int, optional): Starting position in array. Defaults to 0.

        Returns:
            float: Single-precision floating point value.
        """
        # Pad to 4 bytes if needed, then unpack as float
        return struct.unpack('<f', bytes(ByteUtils._pad(arr[offset:offset + 4], 4)))[0]

    @staticmethod
    def get_bytes_uint32(val: int) -> List[int]:
        """
        Converts unsigned 32-bit integer to 4-byte little-endian array.

        Args:
            val (int): Integer value to convert (0 to 4294967295).

        Returns:
            List[int]: 4-byte array in little-endian order.

        Example:
            >>> ByteUtils.get_bytes_uint32(0x12345678)
            [120, 86, 52, 18]
        """
        # Pack as little-endian unsigned int, convert to byte list
        return list(struct.pack('<I', val))


class StringExtensions:
    """
    String manipulation utilities for the procedural generation system.

    Provides methods for formatting hexadecimal values specifically for
    coordinate conversion and seed generation.
    """

    @staticmethod
    def short_to_formatted_hex(val: int, trunc: int) -> str:
        """
        Formats a 16-bit integer as hexadecimal with optional truncation.

        Takes a value up to 65535, converts to 4-character hex (like "01A3"),
        then returns only the last 'trunc' characters. Used for extracting
        specific parts of coordinates for seed generation.

        Args:
            val (int): Integer value to format (0-65535).
            trunc (int): Number of hex characters to keep from the end.

        Returns:
            str: Truncated hexadecimal string.

        Example:
            >>> StringExtensions.short_to_formatted_hex(0x01A3, 2)
            'A3'
        """
        # Mask to 16 bits and format as 4-character hex
        val = val & UINT16_MASK
        hex_str = f"{val:04X}"

        # Return last 'trunc' characters
        return hex_str[-trunc:]


class Generator:
    """
    Core procedural name generator for No Man's Sky.

    This class implements the complex algorithm that generates pronounceable,
    science-fiction-style names for regions, systems, and other game entities.
    It uses weighted character probabilities, vowel insertion rules, and
    backtracking to create names that sound natural but unique.

    The algorithm works by:
    1. Selecting initial characters from predefined alphabets
    2. Adding characters based on probability weights
    3. Applying linguistic rules for vowel/consonant patterns
    4. Adding adornments (descriptive suffixes) based on random chance
    """

    # Class constants shared by all instances
    MULTIPLIER = SEED_MULTIPLIER       # Seed update multiplier
    TINY_DOUBLE = TINY_DOUBLE_BYTES    # Tiny double for probability scaling

    @staticmethod
    def generate_name(cache0, cache1, letter_map, alphasets) -> str:
        """
        Generates a procedural name using the game's algorithm.

        This is the main entry point for name generation. It takes the current
        seed state and reference data (letter maps and alphabets) and produces
        a unique name. The process involves multiple steps of character selection,
        vowel insertion, and quality checks to ensure pronounceable results.

        Args:
            cache0: First part of seed state (list of byte lists).
            cache1: Second part of seed state with control values.
            letter_map: Dictionary mapping character sets to probability weights.
            alphasets: List of character sets for different name styles.

        Returns:
            str: Generated name, or empty string if generation failed.

        Raises:
            None: Returns empty string on failure instead of raising exceptions.
        """
        # Step 1: Get initial character triplet from alphabet set
        name = Generator.get_characters_from_alphaset(cache0, cache1, alphasets)

        # Check if we got the empty sentinel (no valid characters)
        if name == EMPTY_SENTINEL:
            return ""

        # Step 2: Advance seed for next operations
        ByteUtils.update_seed(cache0)

        # Step 3: Check if we should use alternate character selection mode
        # This mode uses different probability calculations
        alternate_check_bytes = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        is_alternate_char_mode = (ByteUtils.to_int16(alternate_check_bytes) != 0)

        # Step 4: Calculate how many additional characters to generate
        ByteUtils.update_seed(cache0)

        # Complex calculation for character count limit
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])

        # Convert to integer limit for the loop
        limit = ByteUtils.to_int16(ByteUtils.sxd(register0, 2))

        # Step 5: Generate additional characters if limit > 0
        if 0 < limit:
            i = 0
            safety = 0  # Counter to prevent infinite backtracking

            # Generate characters one by one until limit reached
            while i < limit:
                ByteUtils.update_seed(cache0)

                # Get last 3 characters for context-aware generation
                sub_str = name[i: i + 3]

                # Determine which alphabet set to use
                alphaset_idx = cache1[0][0] if cache1[0] else 0

                # Get probability weights for possible next characters
                char_weights = Generator.get_string_weights(sub_str, alphaset_idx, letter_map)

                # Generate random value for character selection
                val_u32 = ByteUtils.to_uint32(cache0[0])
                tiny_dbl = ByteUtils.to_double(Generator.TINY_DOUBLE)
                target = float(val_u32 * tiny_dbl)  # Value between 0 and 1

                # Step 6: Handle case where no valid characters found (backtrack)
                if char_weights is None:
                    i -= 1  # Go back one character
                    safety += 1

                    # Safety check to prevent infinite loops
                    if safety > MAX_BACKTRACK_ATTEMPTS:
                        break
                else:
                    safety = 0  # Reset safety counter on successful step
                    index = 0

                    # Step 7: Select character based on probability weights
                    if is_alternate_char_mode:
                        # Alternate mode: more random selection
                        target *= (len(char_weights) - 1)
                        b_tgt = list(struct.pack('<f', target))
                        op_and = ByteUtils.and_op(b_tgt, [0x00, 0x00, 0x00, 0x80])
                        op = ByteUtils.or_op(op_and, [0x00, 0x00, 0x00, 0x3F])
                        index = int(ByteUtils.to_single(op) + target)
                    else:
                        # Normal mode: weighted probability selection
                        weight = 0.0
                        j = 0
                        for cw in char_weights:
                            weight += cw[1]
                            if weight >= target:
                                break
                            j += 1
                        index = j

                    # Step 8: Add selected character to name
                    if index < len(char_weights):
                        name += char_weights[index][0]

                # Step 9: Enforce maximum name length
                if len(name) > MAX_NAME_LENGTH:
                    name = name[:MAX_NAME_LENGTH + 1]

                i += 1

        # Step 10: Return empty string if no name was generated
        if not name:
            return ""

        # Step 11: Apply vowel insertion rules for better pronunciation

        # Rule 1: Insert vowel if first two characters are both consonants
        # (unless they form a valid consonant pair)
        first = name[0]
        second = name[1] if len(name) > 1 else ''
        should_skip_vowel_insert = False

        if (first not in VOWELS) and (second not in VOWELS):
            # Special case: 's' followed by certain letters doesn't need vowel
            cond1 = first != 's' or second not in "hklmnprtwy"

            if cond1:
                # Check if this is a valid initial consonant pair
                if second in VALID_INITIAL_PAIRS and first in VALID_INITIAL_PAIRS[second]:
                    should_skip_vowel_insert = True

                # Insert vowel at position 1 if not a valid pair
                if not should_skip_vowel_insert:
                    name = Generator.insert_vowel(name, cache0, 1)

        # Rule 2: Check end of name for consonant issues
        last_char = name[-1]
        second_last_char = name[-2] if len(name) > 1 else ''

        # Don't check if second last is 'g' followed by vowel
        if len(name) > 1 and (second_last_char != 'g' or last_char in VOWELS):
            idx = len(name) - 1

            # Insert vowel before last character if:
            # 1. Last two chars form invalid final consonant pair, OR
            # 2. Last char is 'w' and second last isn't a vowel
            if (last_char in VALID_FINAL_PAIRS and second_last_char in VALID_FINAL_PAIRS[last_char]) or (last_char == 'w' and second_last_char not in VOWELS):
                name = Generator.insert_vowel(name, cache0, idx)

        # Rule 3: Check for too many consecutive consonants
        consecutive_count = Generator.get_consecutive_consonants(name)
        if consecutive_count != -1:
            ByteUtils.update_seed(cache0)

            # Calculate where to insert vowel to break consonant run
            mult = ByteUtils.multiply(cache0[0], [0x03])
            shr = ByteUtils.shr(mult, 4)
            add = ByteUtils.add(shr, [0x01])
            offset = ByteUtils.to_int32(ByteUtils.zxd(add, 4))

            # Insert vowel at calculated position
            name = Generator.insert_vowel(name, cache0, consecutive_count + offset)

        return name

    @staticmethod
    def get_characters_from_alphaset(cache0, cache1, alphasets) -> str:
        """
        Selects initial characters from an alphabet set.

        Chooses a 3-character starting sequence from one of the predefined
        alphabet sets based on the current seed. These sets contain common
        prefixes or syllables used in the game's naming system.

        Args:
            cache0: First part of seed state.
            cache1: Second part of seed state with alphabet index.
            alphasets: List of character set strings.

        Returns:
            str: 3-character starting sequence, or EMPTY_SENTINEL if failed.
        """
        ByteUtils.update_seed(cache0)

        # Get which alphabet set to use from cache1
        idx = cache1[0][0] if cache1[0] else 0

        # Safety check: ensure index is valid
        if idx >= len(alphasets):
            idx = 0

        alphaset_str = alphasets[idx]

        # Return sentinel if alphabet set is empty
        if not alphaset_str:
            return EMPTY_SENTINEL

        # Calculate start position in the alphabet string
        # Alphabet strings are concatenated 3-character groups
        length_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        register0 = ByteUtils.multiply(cache0[0], length_bytes)
        shr_reg = ByteUtils.shr(register0, 4)
        register1 = ByteUtils.format_short(ByteUtils.multiply(shr_reg, [0x03]))

        # Get start and end indices for 3-character slice
        start = ByteUtils.to_int16(register1)
        end = ByteUtils.to_int16(ByteUtils.add(register1, [0x03]))

        # Extract and return the 3-character sequence
        return alphaset_str[start:end]

    @staticmethod
    def get_string_weights(s, alphaset, letter_map):
        """
        Gets probability weights for possible next characters.

        Given a string prefix and alphabet set, returns a list of possible
        next characters with their probability weights. This enables
        context-aware character generation where some characters are more
        likely to follow certain prefixes.

        Args:
            s: Current string prefix (up to 3 chars).
            alphaset: Index of alphabet set to use.
            letter_map: Nested dictionary of character probabilities.

        Returns:
            List of (character, weight) tuples, or None if no matches.
        """
        # Check if we have data for this alphabet set
        if not letter_map or alphaset not in letter_map:
            return None

        subset = letter_map[alphaset]

        # Check if first character exists in the subset
        if not s or s[0] not in subset:
            return None

        # Recursively search for matching prefix
        return Generator.recursive_search(subset[s[0]], s)

    @staticmethod
    def recursive_search(arr, s):
        """
        Recursively searches for character probability data.

        Navigates the nested letter_map structure to find the probability
        weights for a given string prefix. The structure is organized as
        a tree where each level corresponds to character positions.

        Args:
            arr: Current level of the probability tree.
            s: String prefix to search for.

        Returns:
            List of (character, weight) tuples, or None if not found.
        """
        result = None

        # Search through array elements (tree nodes)
        for i, item in enumerate(arr):
            if result is not None:
                break

            # Check if this item has sub-structure
            if len(item) > 2:
                type_code, val = item[2], item[0]

                # "ja" type: jump if above (compare string values)
                if type_code == "ja":
                    # Convert strings to bytes for comparison
                    s_bytes = ByteUtils.zxd(list(s.encode('utf-8')), 4)
                    val_b = ByteUtils.zxd(list(str(val).encode('utf-8')), 4)

                    # If input string > comparison value, search deeper
                    if ByteUtils.to_int32(s_bytes) > ByteUtils.to_int32(val_b):
                        result = Generator.recursive_search(item[1], s)

                # "jz" type: jump if zero (exact match found)
                elif type_code == "jz" and str(val) == s:
                    # Extract weight data from this node
                    weights = [(w.get("Item1"), float(w.get("Item2", 0))) for w in item[1]]
                    return weights

        return result

    @staticmethod
    def insert_vowel(name, seed, index):
        """
        Inserts a vowel at the specified position in a name.

        Selects a random vowel based on the current seed and inserts it
        at the given position. Used to break up consonant clusters and
        improve name pronounceability.

        Args:
            name: Original name string.
            seed: Current seed state for random vowel selection.
            index: Position where vowel should be inserted.

        Returns:
            str: Name with vowel inserted, or original name if insertion failed.
        """
        ByteUtils.update_seed(seed)

        # Generate random value to select which vowel
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)

        # Check if we got a valid vowel index (0-4 for a-e-i-o-u)
        if calc and calc[0] < 5:
            # Ensure index is within bounds, then insert vowel
            if index <= len(name):
                return name[:index] + VOWELS[calc[0]] + name[index:]

        # Return original if insertion failed
        return name

    @staticmethod
    def get_consecutive_consonants(name):
        """
        Finds runs of consecutive consonants in a name.

        Scans the name to find where there are 4 or more consecutive
        consonants (excluding 'y'). Returns the position where such a
        run begins, or -1 if no problematic runs are found.

        Args:
            name: Name string to check.

        Returns:
            int: Position where 4+ consonant run begins, or -1 if none.
        """
        consecutive_count = 0

        for i, char in enumerate(name):
            # Track consecutive consonants (vowels reset counter)
            if consecutive_count < 3:
                if char not in VOWELS:
                    consecutive_count += 1
                else:
                    consecutive_count = 0
            else:
                # We have at least 3 consonants already
                # Check if this makes 4+ (including 'y' as consonant here)
                if char not in VOWELS_WITH_Y:
                    # Found 4+ consonant run, return start position
                    return i - 3
                else:
                    # 'y' or vowel breaks the run
                    consecutive_count = 0

        # No problematic consonant runs found
        return -1


class RegionNameGenerator:
    """
    Generates procedural region names for No Man's Sky.

    This class specializes the general name generator for region names,
    which have specific formatting rules and can include decorative
    adornments (like "Expanse" or "Nebula").

    Region names are generated from galactic coordinates (x, y, z) and
    galaxy index, creating unique but consistent names for each location.
    """

    # List of decorative suffixes that can be added to region names
    # The %NAME% placeholder gets replaced with the base region name
    PROC_ADORNMENTS = [
        "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
        "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
        "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
        "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux", "%NAME% Instability",
        "Sea of %NAME%", "The Arm of %NAME%", "%NAME% Spur", "%NAME% Shallows"
    ]

    @staticmethod
    def create_region_seed(x: int, y: int, z: int, galaxy: int) -> List[int]:
        """
        Creates a seed value from galactic coordinates.

        Combines galaxy index and 3D coordinates into a hexadecimal string,
        then converts to byte array for use in procedural generation.
        The format is: galaxy(2) + y(2) + z(3) + x(3) hex characters.

        Args:
            x (int): X coordinate in voxel space.
            y (int): Y coordinate in voxel space.
            z (int): Z coordinate in voxel space.
            galaxy (int): Galaxy index (0-255).

        Returns:
            List[int]: Byte array seed for name generation.

        Example:
            >>> RegionNameGenerator.create_region_seed(100, 50, -200, 0)
            [0x00, 0x32, 0x38, 0x64]  # Simplified example
        """
        # Convert each component to formatted hex strings
        s_gal = StringExtensions.short_to_formatted_hex(galaxy, 2)  # Galaxy: 2 chars
        s_y = StringExtensions.short_to_formatted_hex(y, 2)          # Y: 2 chars
        s_z = StringExtensions.short_to_formatted_hex(z, 3)          # Z: 3 chars
        s_x = StringExtensions.short_to_formatted_hex(x, 3)          # X: 3 chars

        # Combine and parse as byte array
        hex_str = s_gal + s_y + s_z + s_x
        return ByteUtils.parse(hex_str)

    @staticmethod
    def format_name(seed, letter_map, alphasets) -> str:
        """
        Generates a complete region name from a seed.

        Takes a coordinate seed and processes it through multiple hashing
        and transformation steps to create a unique region name. Has a
        chance to add decorative adornments based on random chance.

        Args:
            seed: Byte array seed from create_region_seed().
            letter_map: Character probability data.
            alphasets: Alphabet sets for name generation.

        Returns:
            str: Generated region name, or fallback on error.
        """
        # Initialize seed cache structures
        # cache0 holds the main random state, cache1 holds control values
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # Step 1: Initial transformation of input seed
        register0 = ByteUtils.shr(seed, 4)
        if register0:
            register0[0] //= 2  # Halve first byte

        # Step 2: XOR with original seed for initial mixing
        xor_res = ByteUtils.xor(register0, seed)

        # Step 3: First hash multiplication
        register0 = ByteUtils.multiply(xor_res, HASH_MULTIPLIER_1)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)

        # Step 4: Second hash multiplication
        register0 = ByteUtils.multiply(xor2, HASH_MULTIPLIER_2)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        register0 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)

        # Step 5: Final mixing for seed cache
        shl4 = ByteUtils.shl(register0, 4)
        xor_mid = ByteUtils.xor(ByteUtils.rol(shl4, 2), ByteUtils.shr(register0, 4))
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure cache0[0] is not zero (would break generation)
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])

        # Step 6: Calculate name length parameter
        ByteUtils.update_seed(cache0)
        calc_len = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4)
        cache1[2] = ByteUtils.add(calc_len, [0x06])

        # Step 7: Generate base name
        name = Generator.generate_name(cache0, cache1, letter_map, alphasets)

        # Return fallback if generation failed
        if not name:
            return FALLBACK_REGION_NAME

        # Skip processing if name contains brackets (special case)
        if "[" in name:
            return name

        # Capitalize first letter of name
        name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()

        # Step 8: Random chance to add decorative adornment
        ByteUtils.update_seed(cache0)
        mult_check = ByteUtils.multiply(cache0[0], [0x64])
        should_adorn = ByteUtils.shr(mult_check, 4)[0] < ADORNMENT_THRESHOLD

        if should_adorn:
            ByteUtils.update_seed(cache0)
            idx_cal = ByteUtils.multiply(cache0[0], [0x14])
            idx = ByteUtils.shr(idx_cal, 4)[0]

            # Safety check and apply adornment
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                adornment = RegionNameGenerator.PROC_ADORNMENTS[idx]
                name = adornment.replace("%NAME%", name)

        return name


class NMSGalaxyMap:
    """
    Converts portal glyphs to 3D coordinates in No Man's Sky universe.

    Portal glyphs are 12-character hexadecimal codes that represent
    specific locations in the game. This class decodes them into
    voxel coordinates (x, y, z) that can be used for region name
    generation and other calculations.

    The conversion accounts for the game's coordinate system where
    values wrap around at specific boundaries.
    """

    # Shift values for coordinate conversion
    # Coordinates are stored with an offset to handle negative values
    SHIFT_POS_XZ = 2049  # Offset for positive X/Z coordinates
    SHIFT_NEG_XZ = 2047  # Offset for negative X/Z coordinates
    SHIFT_POS_Y = 129    # Offset for positive Y coordinates
    SHIFT_NEG_Y = 127    # Offset for negative Y coordinates

    def glyphs_to_voxels(self, glyphs: str) -> Optional[Dict[str, int]]:
        """
        Converts 12-character portal glyphs to 3D voxel coordinates.

        Portal glyphs encode a location in the universe. The format is:
        - First 4 characters: Unknown/unused in this conversion
        - Characters 4-5: Y coordinate
        - Characters 6-8: Z coordinate
        - Characters 9-12: X coordinate

        Args:
            glyphs (str): 12-character hexadecimal portal code.

        Returns:
            Optional[Dict[str, int]]: Dictionary with 'x', 'y', 'z' keys,
                or None if input is invalid.

        Example:
            >>> map_logic = NMSGalaxyMap()
            >>> map_logic.glyphs_to_voxels("100104005006")
            {'x': 5, 'y': 4, 'z': 6}
        """
        def convert_coord(value, pos_shift, neg_shift):
            """
            Helper to convert encoded coordinate to signed integer.

            The game stores coordinates with an offset to avoid negative
            numbers. This function reverses that offset.
            """
            return value - pos_shift if value >= pos_shift else value + neg_shift

        # Clean and validate input
        g = glyphs.strip().upper()

        # Must be exactly 12 hex characters
        if len(g) != 12:
            return None

        try:
            # Parse coordinate components from glyph string
            # Positions based on No Man's Sky glyph encoding format
            y_hex = int(g[4:6], 16)    # Characters 5-6 (0-indexed 4:6)
            z_hex = int(g[6:9], 16)    # Characters 7-9
            x_hex = int(g[9:12], 16)   # Characters 10-12
        except ValueError:
            # Invalid hexadecimal characters
            return None

        # Convert each coordinate using appropriate offsets
        # X and Z use the same offsets (horizontal plane)
        # Y uses different offsets (vertical axis)
        cx = convert_coord(x_hex, self.SHIFT_POS_XZ, self.SHIFT_NEG_XZ)
        cz = convert_coord(z_hex, self.SHIFT_POS_XZ, self.SHIFT_NEG_XZ)
        cy = convert_coord(y_hex, self.SHIFT_POS_Y, self.SHIFT_NEG_Y)

        return {'x': cx, 'y': cy, 'z': cz}


class NMSData:
    """
    Loads and manages game data from external repositories.

    This class handles downloading and parsing the JSON data files
    needed for the application: flora information, galaxy lists,
    letter probability maps, and alphabet sets. It caches the data
    locally for use by the name generator and UI.

    Data is loaded from GitHub repositories maintained by the
    No Man's Sky community.
    """

    # Base URL for data files on GitHub
    BASE_URL = "https://raw.githubusercontent.com/2A03-Jikuu/nms-wiki-tool-py/refs/heads/main/datalist"

    # Specific data file URLs
    URL_FLORA = f"{BASE_URL}/flora_data.json"       # Flora biology data
    URL_GALAXIES = f"{BASE_URL}/galaxies.json"      # Galaxy names and indices
    URL_LETTER_MAP = f"{BASE_URL}/letter_map.json"  # Character probability data
    URL_ALPHASETS = f"{BASE_URL}/alphasets.json"    # Alphabet sets for names

    def __init__(self):
        """
        Initializes data structures for storing game data.

        Creates empty containers for all data types that will be
        loaded from external sources. Actual data loading happens
        in the load_remote_data() method.
        """
        self.GALAXIES = []           # List of galaxy names
        self.GALAXY_INDICES = {}     # Map galaxy name -> index
        self.BIOME_LIST = []         # Available biome types
        self.AGE_LIST = []           # Plant age descriptions
        self.ROOTS_LIST = []         # Root structure types
        self.NUTRIENTS_LIST = []     # Nutrient source types
        self.NOTES_LIST = []         # Analysis note types
        self.ELEMENT_LIST = []       # Harvestable elements
        self.LETTER_MAP = {}         # Character probability maps
        self.ALPHASETS = []          # Alphabet sets for name generation

    def load_remote_data(self) -> Tuple[bool, str]:
        """
        Downloads and parses all required game data from remote sources.

        Makes HTTP requests to GitHub to fetch JSON data files, then
        processes and stores them in the class attributes. Handles
        network errors and data parsing issues gracefully.

        Returns:
            Tuple[bool, str]: (success status, message description)
                - success: True if all data loaded, False otherwise
                - message: Description of result or error

        Raises:
            No exceptions raised externally; all errors caught and returned.
        """
        try:
            # Load flora biology data (biomes, ages, resources, etc.)
            r_flora = requests.get(self.URL_FLORA, timeout=15)
            r_flora.raise_for_status()  # Raise exception for HTTP errors
            flora = r_flora.json()

            # Extract and sort flora categories
            self.BIOME_LIST = sorted(flora.get("biomes", []))
            self.AGE_LIST = sorted(flora.get("ages", []))
            self.ROOTS_LIST = sorted(flora.get("roots", []))
            self.NUTRIENTS_LIST = sorted(flora.get("nutrients", []))
            self.NOTES_LIST = sorted(flora.get("notes", []))
            self.ELEMENT_LIST = flora.get("elements", [])

            # Load galaxy names and indices
            r_gal = requests.get(self.URL_GALAXIES, timeout=15)
            r_gal.raise_for_status()
            gal_json = r_gal.json()

            # Extract galaxy names and create name->index mapping
            self.GALAXIES = sorted([g["name"] for g in gal_json if "name" in g])
            self.GALAXY_INDICES = {g["name"].lower(): g["index"] for g in gal_json if "name" in g}

            # Load character probability data for name generation
            r_lmap = requests.get(self.URL_LETTER_MAP, timeout=15)
            r_lmap.raise_for_status()
            raw_map = r_lmap.json()

            # Convert string keys to integers (JSON stores all keys as strings)
            self.LETTER_MAP = {int(k): v for k, v in raw_map.items()}

            # Load alphabet sets for name generation
            r_alpha = requests.get(self.URL_ALPHASETS, timeout=15)
            r_alpha.raise_for_status()
            self.ALPHASETS = r_alpha.json()

            return True, "All data modules loaded successfully."

        except requests.RequestException as e:
            # Network-related errors (connection, timeout, HTTP errors)
            return False, f"Network Error: {e}"
        except json.JSONDecodeError as e:
            # Invalid JSON data received
            return False, f"JSON Parsing Error: {e}"
        except Exception as e:
            # Any other unexpected errors
            return False, f"Unexpected Error during data load: {type(e).__name__}: {e}"


@dataclass
class AppWidgets:
    """
    Container class for all UI widgets used in the application.

    This dataclass organizes all the Jupyter widgets in one place,
    making it easier to manage and reference them throughout the UI code.
    Each field is initialized with a default widget factory to ensure
    widgets are created when the dataclass is instantiated.

    Attributes:
        galaxy: Dropdown for selecting galaxy
        glyphs: Text input for portal glyphs
        region: Calculated region name (read-only)
        system: Text input for system name
        planet: Text input for planet name
        moon: Text input for moon name (optional)
        discoverer: Text input for discoverer name
        discoverer_link: Text input for wiki profile link
        discovery_date: Date picker for discovery date
        agt_stardate: Calculated stardate (read-only)
        civilized: Text input for civilization
        release: Text input for game version
        name: Text input for current flora name
        original_name: Text input for original/procedural name
        biome: Dropdown for biome selection
        age: Combobox for plant age
        roots: Combobox for root structure
        nutrients: Combobox for nutrient source
        notes: Combobox for analysis notes
        polymorphic: Integer input for variant count
        element_primary: Dropdown for primary resource
        element_secondary: Dropdown for secondary resource
        image: Text input for main image filename
        gallery: Text area for gallery image list
        btn_preview: Button to preview wiki code
        btn_copy: Button to copy code to clipboard
        btn_gen: Button to generate wiki code file
        btn_clear: Button to reset the form
        btn_example: Button to load example data
        btn_download: Button to download generated file
        output_area: Output widget for displaying generated code
        status_bar: HTML widget for status messages
    """
    # Location widgets
    galaxy: widgets.Combobox = field(default_factory=widgets.Combobox)
    glyphs: widgets.Text = field(default_factory=widgets.Text)
    region: widgets.Text = field(default_factory=widgets.Text)
    system: widgets.Text = field(default_factory=widgets.Text)
    planet: widgets.Text = field(default_factory=widgets.Text)
    moon: widgets.Text = field(default_factory=widgets.Text)

    # Discovery widgets
    discoverer: widgets.Text = field(default_factory=widgets.Text)
    discoverer_link: widgets.Text = field(default_factory=widgets.Text)
    discovery_date: widgets.DatePicker = field(default_factory=widgets.DatePicker)
    agt_stardate: widgets.Text = field(default_factory=widgets.Text)
    civilized: widgets.Text = field(default_factory=widgets.Text)
    release: widgets.Text = field(default_factory=widgets.Text)

    # Identity widgets
    name: widgets.Text = field(default_factory=widgets.Text)
    original_name: widgets.Text = field(default_factory=widgets.Text)

    # Biology widgets
    biome: widgets.Dropdown = field(default_factory=widgets.Dropdown)
    age: widgets.Combobox = field(default_factory=widgets.Combobox)
    roots: widgets.Combobox = field(default_factory=widgets.Combobox)
    nutrients: widgets.Combobox = field(default_factory=widgets.Combobox)
    notes: widgets.Combobox = field(default_factory=widgets.Combobox)
    polymorphic: widgets.IntText = field(default_factory=widgets.IntText)

    # Resource widgets
    element_primary: widgets.Dropdown = field(default_factory=widgets.Dropdown)
    element_secondary: widgets.Dropdown = field(default_factory=widgets.Dropdown)

    # Media widgets
    image: widgets.Text = field(default_factory=widgets.Text)
    gallery: widgets.Textarea = field(default_factory=widgets.Textarea)

    # Action buttons
    btn_preview: widgets.Button = field(default_factory=widgets.Button)
    btn_copy: widgets.Button = field(default_factory=widgets.Button)
    btn_gen: widgets.Button = field(default_factory=widgets.Button)
    btn_clear: widgets.Button = field(default_factory=widgets.Button)
    btn_example: widgets.Button = field(default_factory=widgets.Button)
    btn_download: widgets.Button = field(default_factory=widgets.Button)

    # Display widgets
    output_area: widgets.Output = field(default_factory=widgets.Output)
    status_bar: widgets.HTML = field(default_factory=widgets.HTML)


class FloraDataModel(BaseModel):
    """
    Pydantic model for validating flora data.

    This model defines the structure and validation rules for all
    flora data entered by the user. It ensures data consistency and
    prevents invalid data from being used in wiki page generation.

    Attributes:
        name: Current name of the flora (required)
        original_name: Original procedural name if renamed
        discoverer: Name of player who discovered the flora (required)
        discoverer_link: Wiki username of discoverer
        discovery_date: When the flora was discovered (required)
        civilized: Civilization the discoverer belongs to
        release: Game version when discovered
        galaxy: Which galaxy the flora is in (required)
        region: Region name (auto-calculated from glyphs)
        system: Star system name (required)
        planet: Planet name (required)
        moon: Moon name if on a moon
        glyphs: 12-character portal code (required)
        biome: Type of biome where flora grows
        age: Age description from analysis visor
        roots: Root structure description
        nutrients: Nutrient source description
        notes: Additional analysis notes
        polymorphic: Number of visual variants (1 or more)
        element_primary: Primary harvestable resource
        element_secondary: Secondary harvestable resource
        image: Main image filename for wiki
        gallery: List of gallery image filenames and captions
    """
    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Data fields with validation rules
    name: str = Field(..., min_length=1, description="Current flora name")
    original_name: Optional[str] = ""
    discoverer: str = Field(..., min_length=1)
    discoverer_link: Optional[str] = ""
    discovery_date: arrow.Arrow  # Special arrow type for dates
    civilized: Optional[str] = DEFAULT_CIVILIZATION
    release: Optional[str] = DEFAULT_RELEASE
    galaxy: str = Field(..., min_length=1)
    region: Optional[str] = ""
    system: str = Field(..., min_length=1)
    planet: str = Field(..., min_length=1)
    moon: Optional[str] = ""
    glyphs: str = Field(..., description="12-character portal code")
    biome: Optional[str] = ""
    age: Optional[str] = ""
    roots: Optional[str] = ""
    nutrients: Optional[str] = ""
    notes: Optional[str] = ""
    polymorphic: int = Field(1, ge=1, description="Number of variants")
    element_primary: Optional[str] = ""
    element_secondary: Optional[str] = ""
    image: Optional[str] = ""
    gallery: Optional[str] = ""

    @field_validator('glyphs')
    @classmethod
    def validate_glyphs(cls, v: str) -> str:
        """
        Validates portal glyph format.

        Ensures glyphs are exactly 12 hexadecimal characters (0-9, A-F).
        Converts to uppercase for consistency.

        Args:
            v: Glyph string to validate.

        Returns:
            str: Validated glyphs in uppercase.

        Raises:
            ValueError: If glyphs are empty or wrong format.
        """
        if not v:
            raise ValueError("Portal Hex cannot be empty.")

        # Must be exactly 12 hex characters
        if not re.match(r'^[0-9A-Fa-f]{12}$', v):
            raise ValueError("Portal Hex must be exactly 12 hex characters (0-9, A-F).")

        return v.upper()

    @field_validator('discovery_date')
    @classmethod
    def validate_discovery_date(cls, v):
        """
        Validates discovery date is provided.

        Args:
            v: Date value to check.

        Returns:
            Valid date.

        Raises:
            ValueError: If date is None.
        """
        if v is None:
            raise ValueError("Discovery Date is required.")
        return v


class NMSFloraWikiGenerator:
    """
    Main application class for the No Man's Sky Flora Wiki Generator.

    This class orchestrates the entire application:
    - Loads game data
    - Creates the interactive UI with tabs
    - Handles user interactions and events
    - Generates wiki markup from user input
    - Provides file download functionality

    The UI is built using Jupyter widgets and organized into logical tabs
    for different categories of information (location, identity, details, export).
    """

    # Wiki template using Jinja2 syntax with double curly braces escaped
    # The template contains placeholders that get filled with user data
    WIKI_TEMPLATE = """{{ '{{' }}Version|{{ release }}{{ '}}' }}
{{ '{{' }}AGT Notice{{ '}}' }}
{{ '{{' }}Flora infobox
| name = {{ name }}
| image = {{ image | replace('File:', '') | trim }}
| galaxy = {{ galaxy }}
| region = {{ region }}
| system = {{ system }}
| planet = {{ planet }}
| moon = {{ moon }}
| civilized = {{ civilized }}
| type =
| biome = {{ biome }}
| polymorphic = {{ polymorphic }}
| age = {{ age }}
| roots = {{ roots }}
| nut_source = {{ nutrients }}
| notes = {{ notes }}
| element_primary = {{ element_primary }}
| element_secondary = {{ element_secondary }}
| discovered = {{ discoverer }}
| discoveredlink = {{ discoverer_link }}
| discovered_on = {{ discovered_on }}
| mode = Normal
| researchteam = Alliance of Galactic Travellers
| release = {{ release }}
{{ '}}' }}
'''{{ name }}''' is a species of flora.

==Summary==
'''{{ name }}''' is a [[species]] of [[flora]].

==Alias Names==
{{ '{{' }}aliasc|text=Original|name={{ original_name if original_name else name }}{{ '}}' }}
{{ '{{' }}aliasc|text=Current|name={{ name }}{{ '}}' }}

==Location==
It can be found on the [[{{ loc_type }}]] [[{{ planet }}]]{% if moon %} [[{{ moon }}]]{% endif %} in the [[{{ system }}]] [[star system]].
{{ '{{' }}CoordGlyphConvert|{{ glyphs }}{{ '}}' }}

==Resources==
{{ res_txt }}

==Additional Information==
* Discovered {{ date_short }}. (AGT Stardate {{ stardate }})
* Research contributed by the Alliance of Galactic Travellers research team.

==Gallery==
<gallery>
{{ gallery }}
</gallery>

==AGT Galactic Archives==
{{ '{{' }}AGT Galactic Archive Sync{{ '}}' }}"""

    def __init__(self):
        """
        Initializes the wiki generator application.

        Sets up data loading, creates the UI, and displays the application.
        This is the main entry point that users call to start the tool.
        """
        # Initialize core components
        self.data = NMSData()                     # Game data loader
        self.widgets = AppWidgets()               # UI widgets container
        self.map_logic = NMSGalaxyMap()           # Coordinate converter
        self.jinja_template = Template(self.WIKI_TEMPLATE)  # Wiki template engine
        self.generated_content: str = ""          # Stores generated wiki code

        # Set up UI styles and layout
        self._define_styles_and_layouts()

        # Load game data and show initialization message
        print("Initializing System: Fetching NMS Data Repositories...")
        success, msg = self.data.load_remote_data()

        # Build the user interface
        self._setup_ui()
        self._connect_events()
        self._update_stardate_ui(None)  # Initialize stardate display

        # Clear initialization message and show final status
        clear_output()
        if not success:
            self._update_status(f"WARNING: Data Load Failed. Features missing. ({msg})", "error")
        else:
            self._update_status("System Ready. All data modules loaded.", "success")

        # Display the complete application
        display(self.app_container)

    def _define_styles_and_layouts(self):
        """
        Defines CSS styles and layout configurations for the UI.

        Creates consistent visual styling across all widgets and
        defines layout templates for different UI sections. This
        centralizes styling to make maintenance easier.
        """
        self.STYLE = {
            # Header styling for section titles
            'header': "font-weight:bold; font-size:16px; margin-top:20px; border-bottom:2px solid #00ACC1; padding-bottom:5px; color:#006064;",

            # Description text below headers
            'desc': "font-style:italic; font-size:12px; color:#555; margin-bottom:12px; line-height:1.4em; background-color:#E0F7FA; padding:8px; border-left:4px solid #00BCD4; border-radius:4px;",

            # Label styling for widget descriptions
            'label': {'description_width': '140px'},

            # Base styling for status messages
            'status_base': "padding:8px; margin-top:5px; border-radius:4px; font-weight:bold; text-align:center;",
        }

        # Status message variations with different colors
        self.STYLE['status_info'] = self.STYLE['status_base'] + "background-color:#E0F7FA; color:#006064;"
        self.STYLE['status_success'] = self.STYLE['status_base'] + "background-color:#E8F5E9; color:#2E7D32;"
        self.STYLE['status_error'] = self.STYLE['status_base'] + "background-color:#FFEBEE; color:#C62828;"

        # Layout configurations for different widget arrangements
        self.LAYOUT = {
            'widget': widgets.Layout(width='98%'),           # Standard widget width
            'gallery': widgets.Layout(width='100%', height='180px'),  # Gallery text area
            'col': widgets.Layout(width='50%'),              # Half-width column
            'full_row': widgets.Layout(width='100%', margin='5px 0'), # Full-width row
            'app': widgets.Layout(padding='10px', border='1px solid #B0BEC5', border_radius='5px')  # Main container
        }

    def _setup_ui(self):
        """
        Builds the complete user interface with tabs.

        Creates four main tabs for different data categories and
        assembles them into the main application container. Each
        tab is created by a dedicated method.
        """
        # Create individual tab contents
        tab1 = self._create_tab_location()   # Location & Discovery
        tab2 = self._create_tab_identity()   # Identity & Biology
        tab3 = self._create_tab_details()    # Resources & Media
        tab4 = self._create_tab_generate()   # Export

        # Create tab container with all tabs
        tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4])

        # Set tab titles
        headers = ['Location & Discovery', 'Identity & Biology', 'Resources & Media', 'Export']
        for i, h in enumerate(headers):
            tabs.set_title(i, h)

        # Create main application container
        self.app_container = widgets.VBox([tabs], layout=self.LAYOUT['app'])

    def _create_tab_location(self) -> widgets.VBox:
        """
        Creates the 'Location & Discovery' tab.

        Contains widgets for galactic location (galaxy, glyphs, region,
        system, planet, moon) and discovery information (discoverer,
        date, civilization, game version).

        Returns:
            widgets.VBox: Complete tab container.
        """
        return widgets.VBox([
            # Section 1: Galactic Location
            self._header('Galactic Location'),
            self._desc("Select the Galaxy and enter the Portal Glyphs found in your screenshot. The <b>Region Name</b> will be calculated automatically."),

            # Galaxy selection and glyph input
            self._two_col_row(
                self._create_widget('galaxy', widgets.Combobox, 'Galaxy',
                                   options=self.data.GALAXIES,
                                   placeholder='Type to search Galaxy...'),
                self._create_widget('glyphs', widgets.Text, 'Portal Hex',
                                   placeholder='e.g. 100104005006 (12 Hex Chars)')
            ),

            # Region and system names
            self._two_col_row(
                self._create_widget('region', widgets.Text, 'Region Name',
                                   placeholder='(Calculated automatically from Glyphs)',
                                   disabled=True),
                self._create_widget('system', widgets.Text, 'System Name',
                                   placeholder='e.g. Ocopadica')
            ),

            # Planet and moon names
            self._two_col_row(
                self._create_widget('planet', widgets.Text, 'Planet Name',
                                   placeholder='e.g. New Lennon'),
                self._create_widget('moon', widgets.Text, 'Moon Name',
                                   placeholder='(Optional) e.g. Moon of Lennon')
            ),

            # Section 2: Discovery Record
            self._header('Discovery Record'),
            self._desc("Enter your in-game details. The Stardate is automatically calculated based on the Discovery Date."),

            # Discoverer information
            self._two_col_row(
                self._create_widget('discoverer', widgets.Text, 'Discoverer Alias',
                                   placeholder='Your In-Game Username'),
                self._create_widget('discoverer_link', widgets.Text, 'Wiki Profile',
                                   placeholder='Your Wiki Username (optional)')
            ),

            # Date and stardate
            self._two_col_row(
                self._create_widget('discovery_date', widgets.DatePicker, 'Discovery Date',
                                   value=arrow.now().date()),
                self._create_widget('agt_stardate', widgets.Text, 'AGT Stardate',
                                   disabled=True,
                                   placeholder='(Auto-Calculated)')
            ),

            # Civilization and game version
            self._two_col_row(
                self._create_widget('civilized', widgets.Text, 'Civilization',
                                   value=DEFAULT_CIVILIZATION,
                                   placeholder='e.g. Galactic Hub'),
                self._create_widget('release', widgets.Text, 'Game Version',
                                   value=DEFAULT_RELEASE,
                                   placeholder='e.g. Worlds Part I')
            )
        ], layout=widgets.Layout(padding='20px'))

    def _create_tab_identity(self) -> widgets.VBox:
        """
        Creates the 'Identity & Biology' tab.

        Contains widgets for flora naming (current and original names)
        and biological characteristics (biome, age, roots, nutrients,
        notes, variant count).

        Returns:
            widgets.VBox: Complete tab container.
        """
        return widgets.VBox([
            # Section 1: Flora Identity
            self._header('Flora Identity'),
            self._desc("If you renamed the flora, put the new name in 'Current Name' and the procedural name in 'Original Name'."),

            # Name inputs
            self._two_col_row(
                self._create_widget('name', widgets.Text, 'Current Name',
                                   placeholder='e.g. F. Exemplaris'),
                self._create_widget('original_name', widgets.Text, 'Original Name',
                                   placeholder='Procedural Name (Only if renamed)')
            ),

            # Section 2: Biological Analysis
            self._header('Biological Analysis'),
            self._desc("Refer to the Analysis Visor (left panel). These fields support autocomplete typing."),

            # Biome and age
            self._two_col_row(
                self._create_widget('biome', widgets.Dropdown, 'Native Biome',
                                   options=self.data.BIOME_LIST,
                                   placeholder='Select Native Biome...'),
                self._create_widget('age', widgets.Combobox, 'Age',
                                   options=self.data.AGE_LIST,
                                   placeholder='Type or Select Age...')
            ),

            # Roots and nutrients
            self._two_col_row(
                self._create_widget('roots', widgets.Combobox, 'Root Structure',
                                   options=self.data.ROOTS_LIST,
                                   placeholder='Type or Select Root Structure...'),
                self._create_widget('nutrients', widgets.Combobox, 'Nutrient Source',
                                   options=self.data.NUTRIENTS_LIST,
                                   placeholder='Type or Select Nutrient Source...')
            ),

            # Notes and variant count (special layout)
            widgets.HBox([
                widgets.VBox([self._create_widget('notes', widgets.Combobox, 'Analysis Notes',
                                                 options=self.data.NOTES_LIST,
                                                 placeholder='Type or Select Analysis Notes...')],
                            layout=self.LAYOUT['col']),
                widgets.VBox([self._create_widget('polymorphic', widgets.IntText, 'Variant Count',
                                                 value=1)],
                            layout=self.LAYOUT['col'])
            ], layout=self.LAYOUT['full_row'])
        ], layout=widgets.Layout(padding='20px'))

    def _create_tab_details(self) -> widgets.VBox:
        """
        Creates the 'Resources & Media' tab.

        Contains widgets for harvestable resources (primary and secondary
        elements) and wiki media (main image and gallery images).

        Returns:
            widgets.VBox: Complete tab container.
        """
        return widgets.VBox([
            # Section 1: Harvestable Resources
            self._header('Harvestable Resources'),
            self._desc("What elements do you get when you mine this plant?"),

            # Resource elements
            self._two_col_row(
                self._create_widget('element_primary', widgets.Dropdown, 'Primary Element',
                                   options=self.data.ELEMENT_LIST,
                                   placeholder='Select Resource...'),
                self._create_widget('element_secondary', widgets.Dropdown, 'Secondary Element',
                                   options=self.data.ELEMENT_LIST,
                                   placeholder='Select Resource (Optional)...')
            ),

            # Section 2: Wiki Media
            self._header('Wiki Media'),
            self._desc("Enter filenames of images uploaded to the wiki. Do not include the 'https' link."),

            # Main image
            self._create_widget('image', widgets.Text, 'Main Infobox Image',
                               placeholder='e.g. File:MyFlora.jpg'),

            # Gallery images
            self._create_widget('gallery', widgets.Textarea, 'Gallery Images',
                               placeholder='e.g. File:Flora_Day.jpg|Daytime view\ne.g. File:Flora_Night.jpg|Glowing at night')
        ], layout=widgets.Layout(padding='20px'))

    def _create_tab_generate(self) -> widgets.VBox:
        """
        Creates the 'Export' tab with action buttons and output display.

        Contains buttons for generating, previewing, copying, and downloading
        wiki code, plus a status bar and output area for the generated content.

        Returns:
            widgets.VBox: Complete tab container.
        """
        # Configure button appearance and behavior
        self.widgets.btn_preview.description = 'Preview Code'
        self.widgets.btn_preview.button_style = 'info'
        self.widgets.btn_preview.icon = 'eye'

        self.widgets.btn_copy.description = 'Copy to Clipboard'
        self.widgets.btn_copy.button_style = 'primary'
        self.widgets.btn_copy.icon = 'copy'

        self.widgets.btn_gen.description = 'Generate File'
        self.widgets.btn_gen.button_style = 'success'
        self.widgets.btn_gen.icon = 'code'

        self.widgets.btn_download.description = 'Download .txt'
        self.widgets.btn_download.button_style = 'success'
        self.widgets.btn_download.icon = 'download'
        self.widgets.btn_download.disabled = True  # Enabled after generation

        self.widgets.btn_example.description = 'Load Example'
        self.widgets.btn_example.button_style = 'warning'
        self.widgets.btn_example.icon = 'upload'

        self.widgets.btn_clear.description = 'Reset Form'
        self.widgets.btn_clear.button_style = 'danger'
        self.widgets.btn_clear.icon = 'trash'

        return widgets.VBox([
            # Section 1: Action buttons
            self._header('Finalization'),
            self._desc("Review your data, then generate the Wiki Markup code. You can copy it or download a text file."),

            # Button row
            widgets.HBox([
                self.widgets.btn_preview,
                self.widgets.btn_copy,
                self.widgets.btn_gen,
                self.widgets.btn_download,
                self.widgets.btn_example,
                self.widgets.btn_clear
            ], layout=widgets.Layout(justify_content='center', margin='15px 0', flex_wrap='wrap')),

            # Status message display
            self.widgets.status_bar,

            # Section 2: Generated output
            self._header('Wikitext Output'),
            self.widgets.output_area

        ], layout=widgets.Layout(padding='20px'))

    def _create_widget(self, key, widget_class, description='', **kwargs):
        """
        Creates and configures a UI widget with consistent styling.

        Helper method that creates widgets with standard styling, handles
        placeholder text for different widget types, and stores reference
        in the AppWidgets container.

        Args:
            key (str): Attribute name in AppWidgets to store widget.
            widget_class: Widget class to instantiate (Text, Dropdown, etc.).
            description (str): Label text displayed next to widget.
            **kwargs: Additional widget-specific parameters.

        Returns:
            widget: Created and configured widget instance.
        """
        # Base parameters for all widgets
        params = {'description': description, 'style': self.STYLE['label']}
        params.update(kwargs)

        # Handle special layout requirements
        if widget_class == widgets.Textarea and key == 'gallery':
            params['layout'] = self.LAYOUT.get('gallery')
        elif widget_class == widgets.Textarea:
            params['layout'] = widgets.Layout(width='98%', height='100px')
        else:
            params['layout'] = self.LAYOUT.get('widget')

        # Handle placeholder text for different widget types
        if 'placeholder' in params:
            if widget_class == widgets.Dropdown:
                # Dropdowns: add placeholder as first option
                placeholder_txt = params.pop('placeholder')
                current_options = list(params.get('options', []))
                params['options'] = [placeholder_txt] + current_options
                params['value'] = placeholder_txt
            elif widget_class in [widgets.DatePicker, widgets.IntText]:
                # These don't support placeholders, remove parameter
                params.pop('placeholder')

        # Combobox-specific configuration
        if widget_class == widgets.Combobox:
            params['ensure_option'] = False  # Allow custom text input

        # Create widget and store reference
        w = widget_class(**params)
        setattr(self.widgets, key, w)
        return w

    def _header(self, text: str) -> widgets.HTML:
        """
        Creates a styled section header.

        Args:
            text (str): Header text content.

        Returns:
            widgets.HTML: HTML widget with header styling.
        """
        return widgets.HTML(f"<div style='{self.STYLE['header']}'>{text}</div>")

    def _desc(self, text: str) -> widgets.HTML:
        """
        Creates a styled description text block.

        Args:
            text (str): Description HTML content.

        Returns:
            widgets.HTML: HTML widget with description styling.
        """
        return widgets.HTML(f"<div style='{self.STYLE['desc']}'>{text}</div>")

    def _two_col_row(self, w1, w2=None) -> widgets.HBox:
        """
        Creates a two-column layout row.

        Places two widgets side-by-side in equal-width columns.
        If only one widget provided, it spans full width.

        Args:
            w1: First widget for left column.
            w2: Second widget for right column (optional).

        Returns:
            widgets.HBox: Horizontal box with two columns.
        """
        return widgets.HBox([
            widgets.VBox([w1], layout=self.LAYOUT['col']),
            widgets.VBox([w2] if w2 else [], layout=self.LAYOUT['col'])
        ], layout=self.LAYOUT['full_row'])

    def _update_status(self, message: str, level: str = 'info'):
        """
        Updates the status bar with a colored message.

        Args:
            message (str): Status message text (can contain HTML).
            level (str): Message type: 'info', 'success', or 'error'.
        """
        style = self.STYLE.get(f"status_{level}", self.STYLE['status_info'])
        self.widgets.status_bar.value = f"<div style='{style}'>{message}</div>"

    def _connect_events(self):
        """
        Connects event handlers to UI widgets.

        Sets up all the interactive behavior: button clicks, value changes,
        and automatic calculations when inputs change.
        """
        # Automatic calculations when values change
        self.widgets.discovery_date.observe(self._update_stardate_ui, names='value')
        self.widgets.glyphs.observe(self._calculate_region_ui, names='value')
        self.widgets.galaxy.observe(self._calculate_region_ui, names='value')

        # Button click handlers
        self.widgets.btn_preview.on_click(lambda b: self._generate_handler(mode='preview'))
        self.widgets.btn_gen.on_click(lambda b: self._generate_handler(mode='full'))
        self.widgets.btn_copy.on_click(self._copy_to_clipboard_handler)
        self.widgets.btn_clear.on_click(self._clear_form_handler)
        self.widgets.btn_example.on_click(self._load_example_handler)
        self.widgets.btn_download.on_click(self._download_handler)

    def _update_stardate_ui(self, change):
        """
        Updates the AGT stardate when discovery date changes.

        AGT stardate format: (Year+1716).Day.Month
        Example: January 15, 2024 becomes 3740.15.01

        Args:
            change: Widget change event (unused but required by observer).
        """
        dt = self.widgets.discovery_date.value
        if dt:
            # Convert to arrow date object for easy formatting
            arr = arrow.get(dt)

            # Calculate AGT stardate with year offset
            stardate = f"{arr.year + STARDATE_YEAR_OFFSET}.{arr.day}.{arr.month:02d}"
            self.widgets.agt_stardate.value = stardate
        else:
            self.widgets.agt_stardate.value = ""

    def _calculate_region_ui(self, change):
        """
        Calculates region name when glyphs or galaxy changes.

        Triggered automatically when user enters portal glyphs or selects
        a galaxy. Converts glyphs to coordinates, then generates procedural
        region name using the RegionNameGenerator.

        Args:
            change: Widget change event (unused but required by observer).
        """
        glyphs = self.widgets.glyphs.value.strip().upper()
        galaxy_name = self.widgets.galaxy.value.strip().lower()

        # Validate glyph format
        if not re.match(r'^[0-9A-F]{12}$', glyphs):
            return

        # Check if selected galaxy exists in our data
        if galaxy_name not in self.data.GALAXY_INDICES:
            return

        # Get galaxy index for coordinate calculation
        galaxy_index = self.data.GALAXY_INDICES[galaxy_name]

        # Convert glyphs to voxel coordinates
        voxels = self.map_logic.glyphs_to_voxels(glyphs)
        if not voxels:
            return

        try:
            # Calculate region-relative coordinates
            # Subtract center points to get position within region
            x = voxels['x'] - VOXEL_CENTER_XZ
            y = voxels['y'] - VOXEL_CENTER_Y
            z = voxels['z'] - VOXEL_CENTER_XZ

            # Generate region name from coordinates
            seed = RegionNameGenerator.create_region_seed(x, y, z, galaxy_index)
            name = RegionNameGenerator.format_name(seed, self.data.LETTER_MAP, self.data.ALPHASETS)

            # Update region field if value changed
            if self.widgets.region.value != name:
                self.widgets.region.value = name
        except Exception:
            # Clear region field on any error
            self.widgets.region.value = ""

    def _generate_handler(self, mode: str):
        """
        Handles wiki code generation (preview or full generation).

        Collects all form data, validates it, renders the wiki template,
        and either displays preview or saves to file based on mode.

        Args:
            mode (str): 'preview' to display only, 'full' to save file.
        """
        # Step 1: Collect raw data from all widgets
        raw_data = {}
        for field_name in FloraDataModel.model_fields:
            if hasattr(self.widgets, field_name):
                widget = getattr(self.widgets, field_name)
                val = getattr(widget, 'value', None)

                # Clean string values
                if isinstance(val, str):
                    # Remove placeholder selections
                    if isinstance(val, str) and val.startswith(PLACEHOLDER_PREFIXES):
                        val = ""
                    val = val.strip()

                raw_data[field_name] = val

        # Step 2: Convert discovery date to arrow object
        raw_date = raw_data.get('discovery_date')
        try:
            raw_data['discovery_date'] = arrow.get(raw_date) if raw_date else None
        except Exception:
            self._update_status("Invalid Discovery Date format.", "error")
            return

        # Step 3: Validate data using Pydantic model
        try:
            validated_data = FloraDataModel.model_validate(raw_data)
            model_dict = validated_data.model_dump()
        except ValidationError as e:
            # Extract first validation error for display
            error = e.errors()[0]
            msg = f"VALIDATION ERROR: {error['msg']} ('{error['loc'][0]}')"
            self._update_status(msg, 'error')
            return

        # Step 4: Prepare template context with additional calculated fields
        context = model_dict.copy()

        # Format dates for wiki
        dt = context['discovery_date']
        context['stardate'] = f"{dt.year + STARDATE_YEAR_OFFSET}.{dt.day}.{dt.month:02d}"
        context['discovered_on'] = dt.format('MMMM D, YYYY')
        context['date_short'] = dt.format('D-MMM-YYYY')

        # Determine location type (planet or moon)
        context['loc_type'] = "moon" if context['moon'] else "planet"

        # Generate resource description text
        r1 = context['element_primary']
        r2 = context['element_secondary']
        if r1 and r2:
            context['res_txt'] = f"This flora provides the resources [[{r1}]] and [[{r2}]] when harvested."
        elif r1:
            context['res_txt'] = f"This flora provides the resource [[{r1}]] when harvested."
        else:
            context['res_txt'] = "This flora provides no harvestable resources."

        # Step 5: Render wiki template
        self.generated_content = self.jinja_template.render(**context)

        # Step 6: Handle based on mode
        if mode == 'full':
            # Save to file with sanitized filename
            safe_name = sanitize_filename(context['name'])
            filename = f"{safe_name}_Flora.txt"

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.generated_content)
                self._update_status(f"SUCCESS: Saved as '{filename}'. Click Download.", 'success')
            except Exception as e:
                self._update_status(f"ERROR: Could not save file locally. ({e})", 'error')
        else:
            # Preview mode - just show success message
            self._update_status("Preview generated.", 'info')

        # Enable download button since we have content
        self.widgets.btn_download.disabled = False

        # Display generated code in output area
        with self.widgets.output_area:
            clear_output(wait=True)
            print(self.generated_content)

    def _copy_to_clipboard_handler(self, b):
        """
        Copies generated wiki code to clipboard using JavaScript.

        Note: Only works in Jupyter environments with JavaScript support.

        Args:
            b: Button click event (unused).
        """
        if not self.generated_content:
            self._update_status("No content to copy. Generate a preview first.", "error")
            return

        # Escape special characters for JavaScript string
        safe_str = self.generated_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        # Use JavaScript to copy to clipboard
        display(Javascript(f'navigator.clipboard.writeText(`{safe_str}`);'))
        self._update_status("SUCCESS: Copied to clipboard!", 'success')

    def _clear_form_handler(self, b):
        """
        Resets all form fields to default values.

        Args:
            b: Button click event (unused).
        """
        # Reset each widget based on its type
        for f_obj in fields(self.widgets):
            w = getattr(self.widgets, f_obj.name)

            if hasattr(w, 'value'):
                if isinstance(w, (widgets.Text, widgets.Textarea, widgets.Combobox)):
                    w.value = ""
                elif isinstance(w, widgets.Dropdown) and w.options:
                    # Reset to first option (usually placeholder)
                    w.value = w.options[0] if w.options else None
                elif isinstance(w, widgets.IntText):
                    w.value = 1

        # Set specific defaults
        self.widgets.discovery_date.value = arrow.now().date()
        self.widgets.civilized.value = DEFAULT_CIVILIZATION
        self.widgets.release.value = DEFAULT_RELEASE
        self.widgets.btn_download.disabled = True

        # Clear generated content and output area
        self.generated_content = ""
        with self.widgets.output_area:
            clear_output()

        self._update_status("Form reset.", 'info')

    def _load_example_handler(self, b):
        """
        Loads example data into the form for demonstration.

        Args:
            b: Button click event (unused).
        """
        # Clear form first
        self._clear_form_handler(None)

        # Example data for demonstration
        example = {
            'name': "F. TestSubjectia",
            'original_name': "Spikus Gekinus",
            'discoverer': "Traveller117",
            'galaxy': "Euclid",
            'glyphs': "11FE00800801",
            'system': "Starfall",
            'planet': "New Lennon",
            'biome': "Lush",
            'age': "Ancient",
            'element_primary': "Carbon",
            'element_secondary': "Oxygen",
            'image': "File:FloraExample.jpg"
        }

        # Populate widgets with example data
        for k, v in example.items():
            if hasattr(self.widgets, k):
                widget = getattr(self.widgets, k)

                # Handle dropdowns specially (must match exact option)
                if k in ['biome', 'element_primary', 'element_secondary']:
                    if v in widget.options:
                        widget.value = v
                else:
                    widget.value = v

        self._update_status("Example loaded. Region calculation triggered.", 'info')

    def _download_handler(self, b):
        """
        Triggers download of generated wiki file (Google Colab only).

        Note: This only works in Google Colab environment where the
        files.download() function is available.

        Args:
            b: Button click event (unused).
        """
        if not self.generated_content:
            self._update_status("No content to download. Generate code first.", "error")
            return

        try:
            # Google Colab-specific download functionality
            from google.colab import files

            # Create safe filename from flora name
            safe_name = sanitize_filename(self.widgets.name.value.strip() or "Unnamed")
            filename = f"{safe_name}_Flora.txt"

            # Write file and trigger download
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.generated_content)
            files.download(filename)

        except ImportError:
            # Not in Google Colab
            self._update_status("Download only supported in Colab.", 'error')
        except Exception as e:
            # Any other download error
            self._update_status(f"Download failed: {e}", "error")


# Application entry point
if __name__ == '__main__':
    # Create and run the application when script is executed directly
    app = NMSFloraWikiGenerator()