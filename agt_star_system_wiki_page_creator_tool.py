"""
No Man's Sky Star System Wiki Creator

This module provides a complete application for generating wiki markup code for
documenting star systems in the video game No Man's Sky. It allows users to
input system details through an interactive form, automatically calculates
missing data like coordinates and region names using the game's procedural
generation algorithms, and outputs formatted text ready for wiki publishing.

The application consists of several main components:
1. Procedural generation utilities that mimic the game's name generation
2. Data management for fetching and storing game information
3. User interface widgets for data input
4. Validation models to ensure data correctness
5. The main application controller that ties everything together

Key features:
- Converts portal glyphs to 3D coordinates
- Generates procedural region names
- Validates user input
- Creates properly formatted wiki code
- Provides an interactive form with auto-calculation

Example usage:
    After running this script, an interactive form will appear where you can
    enter star system details. Fill in the required fields and click "Generate"
    to create wiki markup code.
"""

import math
import re
import struct
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

import requests
import arrow
from jinja2 import Template
from pydantic import BaseModel, ValidationError, field_validator, ConfigDict

import ipywidgets as widgets
from ipywidgets import (
    Layout, HBox, VBox, Tab, Text, Textarea, Dropdown, Button,
    IntText, FloatText, DatePicker, Checkbox, GridBox, Combobox,
    Accordion, HTML, Output
)
from IPython.display import display, clear_output, Javascript, FileLink


# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================

# Coordinate System Constants
# These values are used to convert between the game's glyph system and 3D coordinates
# The game stores coordinates with specific offsets that we need to account for
SHIFT_POS_XZ = 2049  # Offset value for positive X and Z coordinates
SHIFT_NEG_XZ = 2047  # Offset value for negative X and Z coordinates
SHIFT_POS_Y = 129    # Offset value for positive Y coordinates (vertical axis)
SHIFT_NEG_Y = 127    # Offset value for negative Y coordinates
CENTER_X, CENTER_Y, CENTER_Z = 2047, 127, 2047  # Center point of the galaxy (coordinates origin)
LY_SCALE = 400       # Scale factor: 1 coordinate unit equals 400 Light Years

# Procedural Generation "Magic" Constants
# These are fixed byte sequences from the game's code that ensure we generate
# the same names the game would generate for given inputs
MAGIC_MULTIPLIER = [0x99, 0xF8, 0x76, 0x5A]  # Used in random number generation
TINY_DOUBLE = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]  # Small double value for probability calculations
REGION_MIXER_1 = [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64]  # Region name generation constant 1
REGION_MIXER_2 = [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3]  # Region name generation constant 2
SUFFIX_PROBABILITY_THRESHOLD = 0x50  # 80 out of 255 chance to add a suffix to region names
NAME_GEN_SAFETY_LIMIT = 50           # Maximum attempts to generate a valid name before giving up
MAX_NAME_LENGTH = 64                 # Maximum characters allowed in generated names

# Regex Patterns for input validation
GLYPH_REGEX = r"^[0-9A-F]{12}$"  # Exactly 12 hexadecimal characters for portal addresses
HEX_REGEX = r"^[0-9A-F]*$"       # Any valid hexadecimal string

# UI Styling Constants
# CSS styles for headers and descriptions in the user interface
HEADER_STYLE = (
    "font-weight:bold; font-size:16px; margin-top:20px; "
    "border-bottom:2px solid #00ACC1; padding-bottom:5px; "
    "color:#006064;"
)
DESC_STYLE = (
    "font-style:italic; font-size:13px; color:#444; "
    "margin-bottom:15px; line-height:1.5em; background-color:#E0F7FA; "
    "padding:10px; border-left:5px solid #00BCD4; border-radius:4px;"
)

# Flavor text suffixes for region names (e.g., "Haud Void", "Raisu Boundary")
# These make generated region names sound more interesting and varied
PROC_ADORNMENTS = [
    "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
    "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
    "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
    "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux",
    "%NAME% Instability", "Sea of %NAME%", "The Arm of %NAME%",
    "%NAME% Spur", "%NAME% Shallows"
]


# =============================================================================
# 2. PROCEDURAL GENERATION UTILITIES
# =============================================================================

class ByteUtils:
    """
    A collection of low-level byte manipulation utilities.

    This class mimics how the game's C++ code performs arithmetic operations
    on bytes, which is necessary to generate the exact same names and values
    that the game would generate. It handles operations like addition with
    carry, multiplication with overflow, and bitwise operations at the byte level.

    Attributes:
        None (all methods are static)
    """

    @staticmethod
    def hex_string_to_bytes(val: str, little_endian: bool = True) -> List[int]:
        """Convert a hexadecimal string into a list of integer bytes.

        This is used to convert portal glyph codes and other hex values into
        the byte format needed for procedural generation calculations.

        Args:
            val: The hexadecimal string to convert (e.g., "0A1F").
            little_endian: If True, reverses byte order (standard for x86 CPUs).
                          If False, keeps bytes in the order they appear.

        Returns:
            A list of integers where each integer is a byte value (0-255).

        Example:
            >>> ByteUtils.hex_string_to_bytes("0A1F")
            [31, 10]  # With little_endian=True (default)
            >>> ByteUtils.hex_string_to_bytes("0A1F", little_endian=False)
            [10, 31]
        """
        # Ensure the string has even length by adding a leading zero if needed
        # This is necessary because hex values come in pairs of characters
        if len(val) % 2 != 0:
            val = "0" + val

        # Convert each pair of hex characters to an integer byte value
        # Example: "0A" becomes 10, "1F" becomes 31
        res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]

        # Reverse the byte order if little-endian format is requested
        # This matches how most computer systems store multi-byte values
        if little_endian:
            res.reverse()
        return res

    @staticmethod
    def pad_to_word(op1: List[int]) -> List[int]:
        """Ensure a byte list is at least 2 bytes long by padding with zeros.

        Some game algorithms expect values to be at least 16 bits (2 bytes),
        so this method adds zero bytes if the list is too short.

        Args:
            op1: The byte list to pad.

        Returns:
            A new list with at least 2 bytes, padded with zeros if needed.

        Example:
            >>> ByteUtils.pad_to_word([0x01])
            [0x01, 0x00]
        """
        res = op1.copy()
        while len(res) < 2:
            res.append(0x00)
        return res

    @staticmethod
    def add(op1: List[int], op2: List[int]) -> List[int]:
        """Add two byte lists together, handling carry between bytes.

        This mimics how a CPU adds multi-byte values, where overflow from
        one byte is carried over to the next byte.

        Args:
            op1: First byte list to add.
            op2: Second byte list to add.

        Returns:
            A new byte list containing the sum.

        Example:
            >>> ByteUtils.add([0xFF, 0x01], [0x01, 0x00])
            [0x00, 0x02]  # 0xFF + 0x01 = 0x100, carry 1 to next byte
        """
        result = op2.copy()
        for i in range(len(op1)):
            result = ByteUtils._add_single(op1[i], result, i)
        return result

    @staticmethod
    def _add_single(val: int, target_list: List[int], index: int) -> List[int]:
        """Helper method to add a single byte to a specific position in a byte list.

        This handles the carry operation when adding bytes. If the sum of two
        bytes exceeds 255 (0xFF), the overflow is carried to the next byte.

        Args:
            val: The byte value to add (0-255).
            target_list: The byte list to add to.
            index: The position in the list to add the byte.

        Returns:
            The updated byte list with carry handled.

        Note:
            This method calls itself recursively if there's carry to handle.
        """
        if index < len(target_list):
            # Add the byte values
            total = val + target_list[index]
            # Keep only the lowest 8 bits (0-255)
            target_list[index] = total & 0xFF
            # Extract the carry (the 9th bit and beyond)
            rem = (total >> 8) & 0xFF
            # If there's carry, add it to the next byte position
            if rem:
                target_list = ByteUtils._add_single(rem, target_list, index + 1)
        else:
            # If we're past the end of the list, append the byte
            target_list.append(val)
        return target_list

    @staticmethod
    def sub(op1: List[int], op2: List[int]) -> List[int]:
        """Subtract op1 from op2, handling borrow between bytes.

        This mimics how a CPU subtracts multi-byte values, where if a byte
        needs to borrow from the next byte when subtracting.

        Args:
            op1: The byte list to subtract.
            op2: The byte list to subtract from.

        Returns:
            A new byte list containing the difference (op2 - op1).

        Example:
            >>> ByteUtils.sub([0x01, 0x00], [0x00, 0x01])
            [0xFF, 0x00]  # Borrow needed from second byte
        """
        result = op2.copy()
        for i in range(len(op1)):
            result = ByteUtils._sub_single(op1[i], result, i)
        return result

    @staticmethod
    def _sub_single(val: int, target_list: List[int], index: int) -> List[int]:
        """Helper method to subtract a single byte from a specific position.

        This handles the borrow operation when subtracting bytes. If a byte
        is too small to subtract from, it borrows from the next byte.

        Args:
            val: The byte value to subtract (0-255).
            target_list: The byte list to subtract from.
            index: The position in the list to subtract from.

        Returns:
            The updated byte list with borrow handled.

        Note:
            This method calls itself recursively if there's borrow to handle.
        """
        if index < len(target_list):
            # Subtract the byte values
            diff = val - target_list[index]
            # Keep only the lowest 8 bits (0-255)
            target_list[index] = diff & 0xFF
            # Extract the borrow (the 9th bit and beyond)
            rem = (diff >> 8) & 0xFF
            # If there's borrow, subtract it from the next byte position
            if rem:
                target_list = ByteUtils._sub_single(rem, target_list, index + 1)
        else:
            # If we're past the end of the list, append the byte
            target_list.append(val)
        return target_list

    @staticmethod
    def multiply(op1: List[int], op2: List[int]) -> List[int]:
        """Multiply two byte lists together, handling overflow between bytes.

        This mimics how a CPU multiplies multi-byte values, similar to
        doing long multiplication by hand.

        Args:
            op1: First byte list to multiply.
            op2: Second byte list to multiply.

        Returns:
            A new byte list containing the product.

        Example:
            >>> ByteUtils.multiply([0x02], [0x80])
            [0x00, 0x01]  # 2 * 128 = 256 = 0x0100
        """
        result: List[int] = []
        for i in range(len(op1)):
            rem = 0
            for j in range(len(op2)):
                # Multiply bytes and add any remainder from previous multiplication
                raw_prod = (op1[i] * op2[j]) + rem
                # This logic simulates signed 16-bit math overflow
                # (how the game's C++ code handles multiplication)
                signed_prd = (raw_prod + 32768) % 65536 - 32768
                rem = (signed_prd >> 8) & 0xFF
                res = signed_prd & 0xFF

                # Add result to appropriate position in the result
                idx = i + j
                if idx < len(result):
                    result = ByteUtils._add_single(res, result, idx)
                else:
                    result.append(res)

            # Handle remaining carry after multiplying all bytes
            if rem > 0:
                idx = i + len(op2)
                if idx < len(result):
                    result = ByteUtils._add_single(rem, result, idx)
                else:
                    result.append(rem)
        # Return at least one zero byte if result is empty
        return result or [0x00]

    @staticmethod
    def slice_lower_bytes(op1: List[int], shift: int) -> List[int]:
        """Return the first 'shift' number of bytes from the list.

        This is like taking the lower part of a multi-byte value.
        Used to extract specific portions of calculation results.

        Args:
            op1: The byte list to slice.
            shift: How many bytes to take from the beginning.

        Returns:
            The first 'shift' bytes, or zeros if the list is too short.

        Example:
            >>> ByteUtils.slice_lower_bytes([0x01, 0x02, 0x03, 0x04], 2)
            [0x01, 0x02]
        """
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def slice_upper_bytes(op1: List[int], shift: int) -> List[int]:
        """Return the bytes starting from position 'shift' to the end.

        This is like taking the upper part of a multi-byte value.
        Used to extract specific portions of calculation results.

        Args:
            op1: The byte list to slice.
            shift: The starting position for slicing.

        Returns:
            Bytes from position 'shift' to the end, or zeros if beyond length.

        Example:
            >>> ByteUtils.slice_upper_bytes([0x01, 0x02, 0x03, 0x04], 2)
            [0x03, 0x04]
        """
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1: List[int], roll: int) -> List[int]:
        """Rotate the list left: move items from start to end.

        This is like a circular shift operation on bytes.
        Used in region name generation to mix up byte patterns.

        Args:
            op1: The byte list to rotate.
            roll: How many positions to rotate.

        Returns:
            The rotated byte list.

        Example:
            >>> ByteUtils.rol([0x01, 0x02, 0x03, 0x04], 1)
            [0x02, 0x03, 0x04, 0x01]
        """
        if not op1:
            return [0x00]
        # Use modulo to handle roll values larger than list length
        r = roll % len(op1)
        return op1[r:] + op1[:r]

    @staticmethod
    def zxd(op1: List[int], extend: int) -> List[int]:
        """Zero-extend: pad the list with zeros until it reaches 'extend' length.

        This ensures a byte list has a specific length by adding zeros.
        Used when algorithms expect values of a certain size.

        Args:
            op1: The byte list to extend.
            extend: The desired total length.

        Returns:
            The extended byte list padded with zeros.

        Example:
            >>> ByteUtils.zxd([0x01, 0x02], 4)
            [0x01, 0x02, 0x00, 0x00]
        """
        return op1.copy() + [0x00] * (extend - len(op1))

    @staticmethod
    def sxd(op1: List[int], extend: int) -> List[int]:
        """Sign-extend: pad the list, copying the sign bit (positive/negative flag).

        This preserves the sign of a signed number when extending it.
        Used when dealing with signed integer values.

        Args:
            op1: The byte list to extend.
            extend: The desired total length.

        Returns:
            The extended byte list with sign preserved.

        Example:
            For a negative number (most significant bit = 1):
            >>> ByteUtils.sxd([0x80], 2)  # 0x80 = -128 in signed 8-bit
            [0x80, 0xFF]  # Extended with 0xFF to preserve negative sign
        """
        result = op1.copy()
        # Check the most significant bit of the last byte to determine sign
        # If the bit is 1, the number is negative; extend with 0xFF
        # If the bit is 0, the number is positive; extend with 0x00
        val = 0xFF if (op1 and (op1[-1] >> 7) == 1) else 0x00
        for _ in range(extend - len(op1)):
            result.append(val)
        return result

    @staticmethod
    def logical_op(op1: List[int], op2: List[int], mode: int) -> List[int]:
        """Perform bitwise logic (AND, OR, XOR) on two byte lists.

        This applies the operation byte-by-byte between two lists.
        Used in various mixing operations during name generation.

        Args:
            op1: First byte list.
            op2: Second byte list.
            mode: 0 for AND, 1 for OR, 2 for XOR.

        Returns:
            A new byte list with the logical operation applied.

        Example:
            >>> ByteUtils.logical_op([0x0F], [0xF0], 0)  # AND
            [0x00]
            >>> ByteUtils.logical_op([0x0F], [0xF0], 1)  # OR
            [0xFF]
            >>> ByteUtils.logical_op([0x0F], [0xF0], 2)  # XOR
            [0xFF]
        """
        # Pad the shorter list with zeros to match lengths
        # This ensures we can operate on all bytes
        l1, l2 = len(op1), len(op2)
        if l1 > l2:
            longer, shorter = op1.copy(), op2.copy() + [0x00] * (l1 - l2)
        else:
            longer, shorter = op2.copy(), op1.copy() + [0x00] * (l2 - l1)

        res = []
        for i in range(len(longer)):
            if mode == 0:
                res.append(longer[i] & shorter[i])  # Bitwise AND
            elif mode == 1:
                res.append(longer[i] | shorter[i])  # Bitwise OR
            else:
                res.append(longer[i] ^ shorter[i])  # Bitwise XOR
        return res

    @staticmethod
    def xor(op1: List[int], op2: List[int]) -> List[int]:
        """Bitwise XOR operation on two byte lists.

        XOR is commonly used in random number generation and mixing algorithms.

        Args:
            op1: First byte list.
            op2: Second byte list.

        Returns:
            A new byte list containing op1 XOR op2.

        Example:
            >>> ByteUtils.xor([0x0F], [0xF0])
            [0xFF]
        """
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1: List[int], op2: List[int]) -> List[int]:
        """Bitwise AND operation on two byte lists.

        Args:
            op1: First byte list.
            op2: Second byte list.

        Returns:
            A new byte list containing op1 AND op2.

        Example:
            >>> ByteUtils.and_op([0x0F], [0xF0])
            [0x00]
        """
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1: List[int], op2: List[int]) -> List[int]:
        """Bitwise OR operation on two byte lists.

        Args:
            op1: First byte list.
            op2: Second byte list.

        Returns:
            A new byte list containing op1 OR op2.

        Example:
            >>> ByteUtils.or_op([0x0F], [0xF0])
            [0xFF]
        """
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache: List[List[int]], move: int = 1) -> List[List[int]]:
        """Advance the random number generator state.

        This implements the game's random number generation algorithm which
        is used to generate names and other procedural content. The same
        input will always produce the same output, making it deterministic.

        Args:
            cache: The current seed state (split into two parts).
            move: How many steps to advance the generator.

        Returns:
            The updated seed state.

        Example:
            >>> seed = [[0x01, 0x02], [0x03, 0x04]]
            >>> ByteUtils.update_seed(seed)
            [[...], [...]]  # New seed values
        """
        for _ in range(move):
            # Multiply the first part by a magic constant
            step1 = ByteUtils.multiply(cache[0], MAGIC_MULTIPLIER)
            # Add the second part
            result = ByteUtils.add(step1, cache[1])
            # Update the two parts of the seed
            cache[0] = ByteUtils.slice_lower_bytes(result, 4)
            cache[1] = ByteUtils.slice_upper_bytes(result, 4)
        return cache

    # Standard Python struct packing helpers
    # These methods convert between byte lists and standard Python data types

    @staticmethod
    def to_uint32(arr: List[int], offset: int = 0) -> int:
        """Convert 4 bytes to an unsigned 32-bit integer.

        Args:
            arr: The byte list to convert from.
            offset: Starting position in the list (default 0).

        Returns:
            An unsigned 32-bit integer.
        """
        # Take 4 bytes starting at offset, pad with zeros if needed
        chunk = (arr[offset:offset + 4] + [0] * 4)[:4]
        # '<I' means little-endian unsigned int
        return struct.unpack('<I', bytes(chunk))[0]

    @staticmethod
    def to_int32(arr: List[int], offset: int = 0) -> int:
        """Convert 4 bytes to a signed 32-bit integer.

        Args:
            arr: The byte list to convert from.
            offset: Starting position in the list (default 0).

        Returns:
            A signed 32-bit integer.
        """
        chunk = (arr[offset:offset + 4] + [0] * 4)[:4]
        # '<i' means little-endian signed int
        return struct.unpack('<i', bytes(chunk))[0]

    @staticmethod
    def to_int16(arr: List[int], offset: int = 0) -> int:
        """Convert 2 bytes to a signed 16-bit integer.

        Args:
            arr: The byte list to convert from.
            offset: Starting position in the list (default 0).

        Returns:
            A signed 16-bit integer.
        """
        chunk = (arr[offset:offset + 2] + [0] * 2)[:2]
        # '<h' means little-endian signed short (16-bit)
        return struct.unpack('<h', bytes(chunk))[0]

    @staticmethod
    def to_double(arr: List[int], offset: int = 0) -> float:
        """Convert 8 bytes to a double-precision floating point number.

        Args:
            arr: The byte list to convert from.
            offset: Starting position in the list (default 0).

        Returns:
            A double-precision floating point number.
        """
        chunk = (arr[offset:offset + 8] + [0] * 8)[:8]
        # '<d' means little-endian double
        return struct.unpack('<d', bytes(chunk))[0]

    @staticmethod
    def to_single(arr: List[int], offset: int = 0) -> float:
        """Convert 4 bytes to a single-precision floating point number.

        Args:
            arr: The byte list to convert from.
            offset: Starting position in the list (default 0).

        Returns:
            A single-precision floating point number.
        """
        chunk = (arr[offset:offset + 4] + [0] * 4)[:4]
        # '<f' means little-endian float
        return struct.unpack('<f', bytes(chunk))[0]

    @staticmethod
    def get_bytes_uint32(val: int) -> List[int]:
        """Convert an unsigned 32-bit integer to a list of 4 bytes.

        Args:
            val: The integer to convert.

        Returns:
            A list of 4 bytes in little-endian order.

        Example:
            >>> ByteUtils.get_bytes_uint32(0x01020304)
            [0x04, 0x03, 0x02, 0x01]  # Little-endian order
        """
        return list(struct.pack('<I', val))


class StringExtensions:
    """Helper methods for formatting strings into hexadecimal codes.

    This class provides utilities for converting numerical values to hex
    strings with specific formatting requirements used in the game's
    procedural generation algorithms.
    """

    @staticmethod
    def short_to_formatted_hex(val: int, trunc: int) -> str:
        """Convert an integer to a hex string and keep the last N characters.

        This is used to extract specific parts of coordinate values for
        region name generation.

        Args:
            val: The integer value to convert.
            trunc: How many hex characters to keep from the end.

        Returns:
            A hexadecimal string with the specified length.

        Example:
            >>> StringExtensions.short_to_formatted_hex(0x1234, 2)
            "34"  # Last 2 characters
        """
        # Mask to 16 bits to ensure we only work with the lower 2 bytes
        val = val & 0xFFFF
        # Convert to hex with exactly 4 characters (padding with zeros)
        hex_str = f"{val:04X}"
        # Return the last 'trunc' characters
        return hex_str[-trunc:]


class Generator:
    """
    The Procedural Name Generator.

    This class uses a Markov-chain-like approach where the probability of
    the next letter depends on the previous letters. It mimics the game's
    native naming algorithm to generate names for star systems, regions,
    and other procedurally generated content.

    The generator works by:
    1. Starting with a seed value (from coordinates or other input)
    2. Using the seed to pick initial characters from predefined sets
    3. Building the name letter by letter based on probability tables
    4. Applying phonetic rules to make names pronounceable

    Attributes:
        None (all methods are static)
    """

    @staticmethod
    def generate_name(cache0: List[List[int]], cache1: List[List[int]]) -> str:
        """Generate a procedural name using the game's algorithm.

        This is the main entry point for name generation. It takes seed data
        and configuration to produce a name that matches what the game would
        generate for the same input.

        Args:
            cache0: Random number generator state (seed).
            cache1: Configuration data (alphabet sets, name lengths).

        Returns:
            A generated name string, or empty string if generation fails.

        Example:
            >>> seed = [[0x01, 0x02, 0x03], [0x04, 0x05, 0x06]]
            >>> config = [[0x00], [0x06], [0x08]]
            >>> Generator.generate_name(seed, config)
            "Leksha"  # Example generated name
        """
        # Step 1: Pick starting characters based on the seed
        name = Generator.get_characters_from_alphaset(cache0, cache1)
        if name == "__EMPTY__":
            return ""

        # Step 2: Decide if we use the 'alternate' generation method
        # This is based on a random check of the seed
        ByteUtils.update_seed(cache0)
        check_op = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        alternate = (ByteUtils.to_int16(check_op) != 0)
        ByteUtils.update_seed(cache0)

        # Step 3: Calculate how long the name should be
        limit = Generator._calculate_length(cache0, cache1)

        # Step 4: Construct the name letter by letter
        name = Generator._build_string(name, limit, alternate, cache0, cache1)

        if not name:
            return ""

        # Step 5: Apply rules to make sure it's pronounceable
        return Generator._apply_phonetic_rules(name, cache0)

    @staticmethod
    def _calculate_length(cache0: List[List[int]], cache1: List[List[int]]) -> int:
        """Calculate the target length of the name based on the seed.

        The game determines name length using a specific formula that
        combines the seed with configuration data.

        Args:
            cache0: Random number generator state.
            cache1: Configuration data.

        Returns:
            The target length for the name (number of characters).
        """
        # This series of operations replicates the game's length calculation
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.slice_upper_bytes(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])
        return ByteUtils.to_int16(ByteUtils.sxd(register0, 2))

    @staticmethod
    def _build_string(name: str, limit: int, alternate: bool,
                      cache0: List[List[int]], cache1: List[List[int]]) -> str:
        """Add characters to the name until the limit is reached.

        This is the core of the name generation algorithm. It builds the
        name character by character using probability tables to determine
        which letter should come next based on previous letters.

        Args:
            name: The starting characters for the name.
            limit: How many more characters to add.
            alternate: Whether to use alternate generation method.
            cache0: Random number generator state.
            cache1: Configuration data.

        Returns:
            The generated name string.

        Raises:
            No explicit exceptions, but will stop if generation fails too many times.
        """
        if limit <= 0:
            return name

        i, safety = 0, 0
        while i < limit:
            # Advance the random number generator for this character
            ByteUtils.update_seed(cache0)
            # Look at the last 3 characters to determine next letter probabilities
            sub_str = name[i: i + 3]

            # Get the list of possible next letters and their weights
            alphaset_idx = cache1[0][0] if cache1 and cache1[0] else 0
            char_weights = Generator.get_string_weights(sub_str, alphaset_idx)

            # Convert RNG output to a float between 0 and 1 for probability selection
            val_u32 = ByteUtils.to_uint32(cache0[0])
            target = float(val_u32 * ByteUtils.to_double(TINY_DOUBLE))

            if char_weights is None:
                # If we don't know what comes next, step back and try again
                i -= 1
                safety += 1
                # Safety check to prevent infinite loops
                if safety > NAME_GEN_SAFETY_LIMIT:
                    break
            else:
                safety = 0
                index = 0

                if alternate:
                    # Alternative calculation method (less common path)
                    target *= (len(char_weights) - 1)
                    b_tgt = list(struct.pack('<f', target))
                    op = ByteUtils.or_op(
                        ByteUtils.and_op(b_tgt, [0, 0, 0, 0x80]),
                        [0, 0, 0, 0x3F]
                    )
                    index = int(ByteUtils.to_single(op) + target)
                else:
                    # Standard method: sum weights until we exceed our random target
                    weight = 0.0
                    for j, cw in enumerate(char_weights):
                        weight += cw[1]
                        if weight >= target:
                            index = j
                            break

                # Append the chosen letter to the name
                if index < len(char_weights):
                    name += char_weights[index][0]

            # Safety check to prevent extremely long names
            if len(name) > MAX_NAME_LENGTH:
                name = name[:MAX_NAME_LENGTH + 1]
            i += 1
        return name

    @staticmethod
    def _apply_phonetic_rules(name: str, cache0: List[List[int]]) -> str:
        """Fix common pronunciation issues in the raw generated string.

        The game applies rules to make generated names more pronounceable
        and avoid awkward consonant clusters.

        Args:
            name: The raw generated name.
            cache0: Random number generator state (for inserting vowels).

        Returns:
            The name with phonetic improvements applied.
        """
        # Rule 1: Insert a vowel if we start with two consonants
        # This prevents names like "Qtarn" that are hard to pronounce
        if len(name) > 1 and name[0] not in "aeiou" and name[1] not in "aeiou":
            # Special exception for "sh" and similar combinations
            cond = (name[0] != 's' or name[1] not in "hklmnprtwy")
            if cond and (name[1] in "ctw" if name[0] == 'h' else True):
                name = Generator.insert_vowel(name, cache0, 1)

        # Rule 2: Insert a vowel before awkward ending consonants
        # This prevents names ending with hard-to-pronounce combinations
        if len(name) > 1 and (name[-2] != 'g' or name[-1] in "aeiou"):
            ult, penult = name[-1], name[-2]
            if ((ult == 'b' and penult in "gn") or
                (ult == 'd' and penult in "bdfghkmpst") or
                (ult == 'p' and penult in "bdhkt")):
                name = Generator.insert_vowel(name, cache0, len(name) - 1)

        # Rule 3: Break up long sequences of consonants
        # Find where 3 or more consonants appear in a row
        cons = Generator.get_consecutive_consonants(name)
        if cons != -1:
            ByteUtils.update_seed(cache0)
            # Calculate where to insert a vowel
            mult = ByteUtils.multiply(cache0[0], [0x03])
            shr = ByteUtils.slice_upper_bytes(mult, 4)
            add = ByteUtils.add(shr, [0x01])
            offset = ByteUtils.to_int32(ByteUtils.zxd(add, 4))
            name = Generator.insert_vowel(name, cache0, cons + offset)
        return name

    @staticmethod
    def get_characters_from_alphaset(cache0: List[List[int]], cache1: List[List[int]]) -> str:
        """Select the initial 3 characters from a predefined alphabet set.

        The game uses different "alphasets" (alphabet sets) for different
        types of names. This method picks which set to use and selects
        the starting characters from it.

        Args:
            cache0: Random number generator state.
            cache1: Configuration data.

        Returns:
            The starting 3-character string, or "__EMPTY__" if no set is found.
        """
        ByteUtils.update_seed(cache0)
        idx = cache1[0][0] if cache1[0] else 0

        # Safety check: ensure the index is valid
        if idx >= len(NMSData.ALPHASETS):
            idx = 0
        alphaset_str = NMSData.ALPHASETS[idx]
        if not alphaset_str:
            return "__EMPTY__"

        # Calculate random start position within the alphabet string
        # The game stores alphabet data in groups of 3 characters
        length_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        register0 = ByteUtils.multiply(cache0[0], length_bytes)
        register1 = ByteUtils.pad_to_word(
            ByteUtils.multiply(ByteUtils.slice_upper_bytes(register0, 4), [0x03])
        )
        start = ByteUtils.to_int16(register1)
        end = ByteUtils.to_int16(ByteUtils.add(register1, [0x03]))
        return alphaset_str[start:end]

    @staticmethod
    def get_string_weights(s: str, alphaset: int) -> Optional[List[Tuple[str, float]]]:
        """Look up the probability table for what letter follows string 's'.

        The game uses probability tables (Markov chains) to determine
        which letters are likely to follow given sequences of letters.
        This method looks up those tables.

        Args:
            s: The string to look up (typically 1-3 characters).
            alphaset: Which alphabet set to use.

        Returns:
            A list of (character, weight) tuples, or None if not found.
        """
        if alphaset not in NMSData.LETTER_MAP:
            return None
        subset = NMSData.LETTER_MAP[alphaset]
        if not s or s[0] not in subset:
            return None
        return Generator.recursive_search(subset[s[0]], s)

    @staticmethod
    def recursive_search(arr: List[Any], s: str) -> Optional[List[Tuple[str, float]]]:
        """Search the nested probability data structure for a matching string.

        The probability data is stored in a complex nested structure that
        needs to be searched recursively to find the right entry.

        Args:
            arr: The nested data structure to search.
            s: The string to search for.

        Returns:
            A list of (character, weight) tuples if found, None otherwise.
        """
        for item in arr:
            if len(item) > 2:
                type_code, val = item[2], item[0]
                if type_code == "ja":
                    # Compare encoded strings to find the matching sequence
                    s_enc = list(s.encode())
                    val_enc = list(str(val).encode())
                    if ByteUtils.to_int32(ByteUtils.zxd(s_enc, 4)) > ByteUtils.to_int32(ByteUtils.zxd(val_enc, 4)):
                        res = Generator.recursive_search(item[1], s)
                        if res:
                            return res
                elif type_code == "jz" and str(val) == s:
                    # Found the match: return the list of (character, weight) tuples
                    return [
                        (w.get("Item1"), float(w.get("Item2", 0)))
                        for w in item[1]
                    ]
        return None

    @staticmethod
    def insert_vowel(name: str, seed: List[List[int]], index: int) -> str:
        """Insert a random vowel (a, e, i, o, u) at the specified index.

        This is used by the phonetic rules to break up consonant clusters
        and make names more pronounceable.

        Args:
            name: The name to modify.
            seed: Random number generator state for selecting which vowel.
            index: Where to insert the vowel.

        Returns:
            The modified name with vowel inserted.
        """
        ByteUtils.update_seed(seed)
        calc = ByteUtils.slice_upper_bytes(ByteUtils.multiply(seed[0], [0x05]), 4)
        if calc and calc[0] < 5 and index <= len(name):
            return name[:index] + "aeiou"[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name: str) -> int:
        """Find the index where 3 consonants appear in a row.

        This helps identify where names might be hard to pronounce.

        Args:
            name: The name to check.

        Returns:
            The index where 3 consonants start, or -1 if not found.
        """
        cons = 0
        for i, c in enumerate(name):
            if c not in "aeiou":
                cons += 1
                # Check for 3 consonants in a row (excluding 'y' which is sometimes vowel-like)
                if cons >= 3 and c not in "aeiouy":
                    return i - 3
            else:
                cons = 0
        return -1


class RegionNameGenerator:
    """Generates region names (e.g., "Haud Void") based on spatial coordinates.

    This class takes X, Y, Z coordinates and a galaxy index and generates
    the procedural region name that the game would assign to that location.

    Example:
        >>> RegionNameGenerator.create_region_seed(100, 50, -200, 0)
        >>> RegionNameGenerator.format_name(seed)
        "Haud Void"
    """

    @staticmethod
    def create_region_seed(x: int, y: int, z: int, galaxy: int) -> List[int]:
        """Create a seed number from the X, Y, Z coordinates and galaxy index.

        The game combines coordinates in a specific way to create a unique
        seed for each region. This seed is then used to generate the name.

        Args:
            x: X coordinate in game units.
            y: Y coordinate in game units.
            z: Z coordinate in game units.
            galaxy: Galaxy index (0 for Euclid, 1 for Hilbert, etc.).

        Returns:
            A list of bytes representing the region seed.
        """
        # Convert each component to hex with specific formatting
        s_gal = StringExtensions.short_to_formatted_hex(galaxy, 2)
        s_y = StringExtensions.short_to_formatted_hex(y, 2)
        s_z = StringExtensions.short_to_formatted_hex(z, 3)
        s_x = StringExtensions.short_to_formatted_hex(x, 3)
        # Combine and convert to bytes
        return ByteUtils.hex_string_to_bytes(s_gal + s_y + s_z + s_x)

    @staticmethod
    def format_name(seed: List[int]) -> str:
        """Generate the full region name including suffixes.

        This is the main method for generating region names. It takes a
        seed created from coordinates and produces a complete region name
        like "Haud Void" or "Raisu Boundary".

        Args:
            seed: The region seed generated from coordinates.

        Returns:
            The generated region name.

        Example:
            >>> seed = RegionNameGenerator.create_region_seed(100, 50, -200, 0)
            >>> RegionNameGenerator.format_name(seed)
            "Haud Void"
        """
        # Initialize caches for the name generation algorithm
        cache0, cache1 = [[], []], [[0x00], [0x06], []]
        # Scramble the seed to create the starting state
        cache0 = RegionNameGenerator._scramble_seed(seed, cache0)

        # Generate the main part of the name (e.g., "Haud")
        name = RegionNameGenerator._generate_base_name(cache0, cache1)

        if not name or "[" in name:
            return name if name else "Unknown Region"

        # Capitalize the first letter and add suffix (e.g., "Void", "Expanse")
        name = name[0].upper() + name[1:]
        name = RegionNameGenerator._apply_adornment(name, cache0)
        return name

    @staticmethod
    def _scramble_seed(seed: List[int], cache0: List[List[int]]) -> List[List[int]]:
        """Mix the seed bytes with magic numbers to randomize it.

        The game applies several mixing operations to the seed before
        using it for name generation. This ensures names vary even with
        similar coordinates.

        Args:
            seed: The original region seed.
            cache0: The cache to initialize.

        Returns:
            The initialized cache with scrambled seed values.
        """
        # First mixing operation
        register0 = ByteUtils.slice_upper_bytes(seed, 4)
        if register0:
            register0[0] //= 2
        xor_res = ByteUtils.xor(register0, seed)

        # Second mixing with first magic constant
        reg0 = ByteUtils.multiply(xor_res, REGION_MIXER_1)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.slice_upper_bytes(reg0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), reg0)

        # Third mixing with second magic constant
        reg0 = ByteUtils.multiply(xor2, REGION_MIXER_2)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.slice_upper_bytes(reg0, 4)) // 2
        reg0 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), reg0)

        # Final mixing operations
        shl4 = ByteUtils.slice_lower_bytes(reg0, 4)
        xor_mid = ByteUtils.xor(ByteUtils.rol(shl4, 2), ByteUtils.slice_upper_bytes(reg0, 4))
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure we don't have a zero seed (which would cause problems)
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])
        return cache0

    @staticmethod
    def _generate_base_name(cache0: List[List[int]], cache1: List[List[int]]) -> str:
        """Reuse the standard Generator class to make the name part.

        Region names use the same core generation algorithm as other
        procedural names, just with different configuration.

        Args:
            cache0: Random number generator state.
            cache1: Configuration data.

        Returns:
            The base name without suffix (e.g., "Haud").
        """
        ByteUtils.update_seed(cache0)
        # Calculate name length
        calc_len = ByteUtils.slice_upper_bytes(ByteUtils.multiply(cache0[0], [0x04]), 4)
        cache1[2] = ByteUtils.add(calc_len, [0x06])
        # Generate the name using the main Generator class
        return Generator.generate_name(cache0, cache1)

    @staticmethod
    def _apply_adornment(name: str, cache0: List[List[int]]) -> str:
        """Randomly add a suffix like 'Nebula' or 'Sector'.

        Region names have an 80/255 chance of getting a suffix that makes
        them sound more interesting.

        Args:
            name: The base name (e.g., "Haud").
            cache0: Random number generator state.

        Returns:
            The name with suffix if applicable (e.g., "Haud Void").
        """
        ByteUtils.update_seed(cache0)
        mult_check = ByteUtils.multiply(cache0[0], [0x64])

        # Check against probability threshold (80/255 ≈ 31% chance)
        if ByteUtils.slice_upper_bytes(mult_check, 4)[0] < SUFFIX_PROBABILITY_THRESHOLD:
            ByteUtils.update_seed(cache0)
            # Pick which suffix to use
            idx_cal = ByteUtils.multiply(cache0[0], [0x14])
            idx = ByteUtils.slice_upper_bytes(idx_cal, 4)[0]

            if idx < len(PROC_ADORNMENTS):
                # Replace %NAME% placeholder with the actual name
                name = PROC_ADORNMENTS[idx].replace("%NAME%", name)
        return name


# =============================================================================
# 3. DATA TIER
# =============================================================================

class NMSData:
    """
    Handles fetching and storing static game data from external sources.

    This class downloads JSON files from GitHub containing lists of
    factions, economy types, procedural generation letter weights, and
    other game data. It acts as a central database for the application.

    All data is loaded once when the application starts and cached
    for use throughout the session.

    Example:
        >>> NMSData.initialize()
        >>> print(NMSData.GALAXIES[:3])
        ['Euclid', 'Hilbert Dimension', 'Calypso']
    """

    # Base URL for fetching game data from GitHub
    _BASE_URL = "https://raw.githubusercontent.com/2A03-Jikuu/nms-wiki-tool-py/refs/heads/main/datalist"
    TIMEOUT = 10  # Seconds to wait for network requests
    MAX_RETRIES = 3  # How many times to retry failed downloads

    # Data Containers - These will be filled when initialize() is called
    CLASS_TO_COLOR: Dict[str, str] = {}  # Maps star class letters to colors (e.g., 'M' -> 'Red')
    GALAXIES: List[str] = []  # List of galaxy names
    GALAXY_MAP: Dict[int, str] = {}  # Maps galaxy indices to names
    FACTIONS: List[str] = []  # List of possible factions (Gek, Vy'keen, Korvax, etc.)
    ECONOMY_LIST: List[str] = []  # List of economy types (Manufacturing, Mining, etc.)
    WEALTH_LIST: List[str] = []  # List of wealth levels (Destitute, Poor, etc.)
    CONFLICT_LIST: List[str] = []  # List of conflict levels (Tranquil, Low, etc.)
    COMMODITIES: List[str] = []  # List of trade commodities
    UPGRADES_MT: List[str] = []  # Multi-tool upgrade modules
    UPGRADES_SS: List[str] = []  # Starship upgrade modules
    UPGRADES_EC: List[str] = []  # Exocraft upgrade modules
    UPGRADES_ES: List[str] = []  # Exosuit upgrade modules
    PLATFORMS: List[str] = []  # Game platforms (PC, PS4, Xbox, etc.)
    MODES: List[str] = []  # Game modes (Normal, Survival, Creative, etc.)

    # Procedural generation data
    LETTER_MAP: Dict[int, Any] = {}  # Probability tables for name generation
    ALPHASETS: List[str] = []  # Alphabet sets for different name types

    @classmethod
    def _fetch_with_retry(cls, url: str, description: str):
        """Download JSON data from the web, retrying if it fails.

        This handles network errors by retrying up to MAX_RETRIES times
        with a 1-second delay between attempts.

        Args:
            url: The URL to fetch data from.
            description: Human-readable description for error messages.

        Returns:
            The parsed JSON data.

        Raises:
            requests.RequestException: If all retries fail.
        """
        for attempt in range(cls.MAX_RETRIES):
            try:
                response = requests.get(url, timeout=cls.TIMEOUT)
                # Raise an exception for HTTP errors (404, 500, etc.)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                # If this was the last attempt, raise the exception
                if attempt == cls.MAX_RETRIES - 1:
                    print(f"❌ Failed to fetch {description}: {e}")
                    raise
                # Otherwise wait and retry
                time.sleep(1)

    @classmethod
    def initialize(cls):
        """Load all necessary data from the remote repository.

        This method must be called before using any other parts of the
        application. It fetches multiple JSON files and populates the
        class attributes with game data.

        Raises:
            Exception: If data loading fails, some features may not work.
        """
        try:
            print("⏳ Fetching System Data...")
            # Fetch main system data (factions, economies, etc.)
            data = cls._fetch_with_retry(f"{cls._BASE_URL}/system_data.json", "System Data")

            # Populate data containers from the fetched JSON
            cls.CLASS_TO_COLOR = data.get('CLASS_TO_COLOR', {})
            cls.FACTIONS = data.get('FACTIONS', [])
            cls.ECONOMY_LIST = sorted(data.get('ECONOMY_LIST', []))
            cls.WEALTH_LIST = sorted(data.get('WEALTH_LIST', []))
            cls.CONFLICT_LIST = sorted(data.get('CONFLICT_LIST', []))
            cls.COMMODITIES = sorted(data.get('COMMODITIES', []))
            cls.UPGRADES_MT = sorted(data.get('UPGRADES_MT', []))
            cls.UPGRADES_SS = sorted(data.get('UPGRADES_SS', []))
            cls.UPGRADES_EC = sorted(data.get('UPGRADES_EC', []))
            cls.UPGRADES_ES = sorted(data.get('UPGRADES_ES', []))
            cls.PLATFORMS = data.get('PLATFORMS', [])
            cls.MODES = data.get('MODES', [])

            print("⏳ Fetching Galaxy List...")
            # Fetch galaxy data
            gal_data = cls._fetch_with_retry(f"{cls._BASE_URL}/galaxies.json", "Galaxy List")
            cls.GALAXIES = [g['name'] for g in gal_data]
            cls.GALAXY_MAP = {g['index']: g['name'] for g in gal_data}

            print("⏳ Fetching Procedural Tables...")
            # Fetch procedural generation data
            cls.ALPHASETS = cls._fetch_with_retry(f"{cls._BASE_URL}/alphasets.json", "Alphasets")
            letter_raw = cls._fetch_with_retry(f"{cls._BASE_URL}/letter_map.json", "Letter Map")
            cls.LETTER_MAP = {int(k): v for k, v in letter_raw.items()}

            print("✅ Data initialization complete.")
        except Exception as e:
            # If data loading fails, provide default data so the app still runs
            print(f"❌ Critical Error initializing data. Some features may not work. {e}")
            cls.GALAXIES = ['Euclid', 'Hilbert Dimension']


# =============================================================================
# 4. WIDGET TIER
# =============================================================================

@dataclass
class AppWidgets:
    """
    A container to hold all the UI elements (buttons, text boxes, dropdowns).

    Using a dataclass like this keeps variables organized instead of having
    hundreds of global variables floating around. Each widget corresponds to
    a field in the star system data form.

    Attributes:
        name (Text): Input for the system's wiki name.
        original_name (Text): Input for the system's original (procedural) name.
        galaxy (Combobox): Dropdown for selecting which galaxy the system is in.
        region (Text): Display field for the calculated region name.
        spectral_class (Text): Input for the star's spectral class (e.g., M7p).
        star_color (Dropdown): Dropdown for selecting the star's color.
        multiple_stars (Dropdown): Dropdown for selecting how many visible stars.
        planet_count (IntText): Input for number of planets in the system.
        moon_count (IntText): Input for number of moons in the system.
        celestial_names (Textarea): Multi-line input for planet and moon names.
        glyphs (Text): Input for the 12-character portal glyph code.
        coordinates (Text): Display field for calculated 3D coordinates.
        system_id (IntText): Display field for the system's index number.
        distance (Text): Display field for distance to galaxy center in light years.
        image (Text): Input for the main infobox image filename.
        nav_image (Text): Input for the navigation map image filename.
        gallery_images (Textarea): Multi-line input for gallery image filenames.
        faction (Dropdown): Dropdown for selecting the dominant faction.
        civilized (Text): Input for civilization name (usually "Alliance of Galactic Travellers").
        economy_desc (Combobox): Dropdown for selecting the economy type.
        wealth (Combobox): Dropdown for selecting the wealth level.
        economy_buy (FloatText): Input for buy price modifier percentage.
        economy_sell (FloatText): Input for sell price modifier percentage.
        conflict (Combobox): Dropdown for selecting conflict level.
        age (Text): Input for the system's age in billions of years.
        water (Checkbox): Checkbox for whether water is present.
        dissonant (Checkbox): Checkbox for whether the system is dissonant.
        discoverer (Text): Input for discoverer's in-game name.
        discoverer_link (Text): Input for discoverer's wiki username.
        date (DatePicker): Date picker for discovery date.
        stardate (Text): Display field for calculated AGT stardate.
        platform (Dropdown): Dropdown for game platform (PC, PS4, etc.).
        mode (Dropdown): Dropdown for game mode (Normal, Survival, etc.).
        release (Text): Input for game version when discovered.
        commodity_checks (Dict[str, Checkbox]): Dictionary of checkboxes for trade commodities.
        upgrade_merchants (Dict[str, Dropdown]): Dictionary of dropdowns for upgrade merchants.
    """

    # System Identity
    name: Text = field(init=False)
    original_name: Text = field(init=False)
    galaxy: Combobox = field(init=False)
    region: Text = field(init=False)
    spectral_class: Text = field(init=False)
    star_color: Dropdown = field(init=False)
    multiple_stars: Dropdown = field(init=False)

    # Celestial Bodies
    planet_count: IntText = field(init=False)
    moon_count: IntText = field(init=False)
    celestial_names: Textarea = field(init=False)

    # Location
    glyphs: Text = field(init=False)
    coordinates: Text = field(init=False)
    system_id: IntText = field(init=False)
    distance: Text = field(init=False)

    # Media
    image: Text = field(init=False)
    nav_image: Text = field(init=False)
    gallery_images: Textarea = field(init=False)

    # Faction & Economy
    faction: Dropdown = field(init=False)
    civilized: Text = field(init=False)
    economy_desc: Combobox = field(init=False)
    wealth: Combobox = field(init=False)
    economy_buy: FloatText = field(init=False)
    economy_sell: FloatText = field(init=False)
    conflict: Combobox = field(init=False)

    # Characteristics
    age: Text = field(init=False)
    water: Checkbox = field(init=False)
    dissonant: Checkbox = field(init=False)

    # Discovery
    discoverer: Text = field(init=False)
    discoverer_link: Text = field(init=False)
    date: DatePicker = field(init=False)
    stardate: Text = field(init=False)
    platform: Dropdown = field(init=False)
    mode: Dropdown = field(init=False)
    release: Text = field(init=False)

    # Collections
    commodity_checks: Dict[str, Checkbox] = field(default_factory=dict, init=False)
    upgrade_merchants: Dict[str, Dropdown] = field(default_factory=dict, init=False)


# =============================================================================
# 5. VALIDATION TIER
# =============================================================================

class SystemDataModel(BaseModel):
    """
    Defines the rules and structure for valid Star System data using Pydantic.

    This model ensures that when the user types data into the form, it actually
    makes sense before we try to generate the wiki code. Pydantic automatically
    validates data types, formats, and custom rules.

    Example:
        >>> data = SystemDataModel(name="Test System", system_id=123)
        >>> print(data.name)
        "Test System"
    """

    # Pydantic configuration: automatically strip whitespace from strings
    model_config = ConfigDict(str_strip_whitespace=True)

    # Basic Info
    name: str = "Unnamed System"  # The system's name (required, has default)
    original_name: Optional[str] = ""  # Original procedural name (optional)
    image: Optional[str] = ""  # Main infobox image filename
    nav_image: Optional[str] = ""  # Navigation map image filename
    gallery_images: List[str] = []  # List of additional gallery images

    # Location
    region: Optional[str] = ""  # Region name (calculated from coordinates)
    galaxy: Optional[str] = ""  # Galaxy name
    multiple_stars: Optional[str] = ""  # Number of visible stars
    distance: Optional[str] = ""  # Distance to galaxy center
    star_color: Optional[str] = ""  # Star color
    spectral_class: Optional[str] = ""  # Spectral class (e.g., M7p)
    system_id: int = 0  # System index number
    coordinates: Optional[str] = ""  # 3D coordinates in hex format
    glyphs: Optional[str] = ""  # 12-character portal glyph code

    # Bodies
    planet_count: int = 0  # Number of planets
    moon_count: int = 0  # Number of moons
    water: bool = False  # Whether water is present
    dissonant: bool = False  # Whether system is dissonant

    # Economy
    gateway: Optional[str] = ""  # Gateway system status
    faction: Optional[str] = ""  # Dominant faction
    economy_desc: Optional[str] = ""  # Economy type
    economy_buy: Optional[float] = None  # Buy price modifier
    economy_sell: Optional[float] = None  # Sell price modifier
    wealth: Optional[str] = ""  # Wealth level
    conflict: Optional[str] = ""  # Conflict level

    # Meta
    platform: Optional[str] = ""  # Game platform
    mode: Optional[str] = ""  # Game mode
    civilized: Optional[str] = ""  # Civilization name
    discoverer: Optional[str] = ""  # Discoverer's name
    discoverer_link: Optional[str] = ""  # Discoverer's wiki username
    release: Optional[str] = ""  # Game version

    researchteam: str = "Alliance of Galactic Travellers"  # Fixed value
    misc: Optional[str] = ""  # Miscellaneous notes
    age: Optional[str] = ""  # System age in billions of years
    stardate: Optional[str] = ""  # AGT stardate format
    discovery_date: Optional[Any] = None  # Discovery date object
    celestial_names: List[str] = []  # List of planet/moon names
    commodities: List[str] = []  # List of available trade commodities
    ss_merchants: Dict[str, List[str]] = {}  # Upgrade merchants by type

    @field_validator('glyphs', mode='before')
    def validate_glyphs(cls, v):
        """Ensure the glyph code is exactly 12 hex characters.

        Portal glyphs must be 12 hexadecimal characters (0-9, A-F).
        This validator checks the format and converts to uppercase.

        Args:
            v: The glyph string to validate.

        Returns:
            The validated glyph string in uppercase.

        Raises:
            ValueError: If glyphs are not exactly 12 hex characters.

        Example:
            >>> validate_glyphs("0123456789ab")
            "0123456789AB"
        """
        if v and (len(v) != 12 or not re.match(HEX_REGEX, v, re.I)):
            raise ValueError("Portal Glyphs must be a 12-character hex string.")
        return v.upper() if v else ""

    @field_validator('economy_buy', 'economy_sell', mode='before')
    def empty_numeric_to_none(cls, v):
        """Convert empty strings in numeric fields to None.

        This allows users to leave price modifier fields blank without
        causing validation errors. Blank fields become None instead of 0.0.

        Args:
            v: The value to check (could be string, float, or None).

        Returns:
            None if the value is empty or zero, otherwise the original value.
        """
        if v == 0.0 or v == "" or v is None:
            return None
        return v

    @field_validator('age')
    def validate_age(cls, v):
        """Ensure age is a valid number.

        The age field should contain a decimal number (like "3.7" for
        3.7 billion years). This checks the format but doesn't enforce
        a specific range.

        Args:
            v: The age string to validate.

        Returns:
            The validated age string.

        Raises:
            ValueError: If age is not a valid number.
        """
        if v and not re.match(r"^\d*\.?\d+$", v):
             raise ValueError("Age must be a valid number.")
        return v

    @field_validator('spectral_class')
    def validate_spectral(cls, v):
        """Check if the star class starts with a real star type letter.

        Valid star types in No Man's Sky are O, B, A, F, G, K, M, E, L, T, Y.
        This ensures users don't enter completely invalid spectral classes.

        Args:
            v: The spectral class string to validate.

        Returns:
            The validated spectral class string.

        Raises:
            ValueError: If spectral class doesn't start with a valid letter.
        """
        if v and v[0].upper() not in "OBAFGKMELTY":
            raise ValueError("Spectral Class must start with a valid stellar type (O,B,A,F,G,K,M,E,L,T,Y).")
        return v


# =============================================================================
# 6. PRESENTATION TIER: Main Application
# =============================================================================

class NMSStarSystemWikiCreator:
    """
    The main controller for the application.

    This class sets up the User Interface, connects buttons to functions,
    handles the math for coordinate conversion, and generates the final
    wiki text using a template. It's the central class that ties all the
    other components together.

    Example:
        >>> app = NMSStarSystemWikiCreator()
        # An interactive form will appear in Jupyter Notebook
    """

    # This is the Jinja2 template for the final Wiki output.
    # {{ variable }} gets replaced with data from the form.
    # The template creates properly formatted wiki markup code.
    WIKI_TEMPLATE = Template("""{{ '{{Version|' + release + '}}' }}
{{ '{{AGT Notice}}' }}
{{ '{{System infobox' }}
| name= {{ name|e }}
| image= {{ image|e }}
| region= {{ region|e }}
| galaxy= {{ galaxy|e }}
| multiplestars= {{ multiple_stars }}
| distance= {{ distance }}
| color= {{ star_color }}
| class= {{ spectral_class|e }}
| systemid= {{ system_id }}
| coordinates= {{ coordinates }}
| portalglyphs= {{ glyphs }}
| planet= {{ planet_count }}
| moon= {{ moon_count }}
| water= {{ 'Yes' if water else 'No' }}
| dissonant= {{ 'Yes' if dissonant else 'No' }}
| gateway= {{ gateway }}
| faction= {{ faction }}
| economy= {{ economy_desc }}
| economybuy= {{ economy_buy if economy_buy is not none else '' }}
| economysell= {{ economy_sell if economy_sell is not none else '' }}
| wealth= {{ wealth }}
| conflict= {{ conflict }}
| platform= {{ platform }}
| mode= {{ mode }}
| civilized= {{ civilized|e }}
| discovered= {{ discoverer|e }}
| discoveredlink= {{ discoverer_link|e }}
| release= {{ release|e }}
| researchteam= {{ researchteam }}
| misc= {{ misc }}
}}

'''{{ name|e }}''' is a star system.

==Summary==
'''{{ name|e }}''' is a [[System colours|{{ star_color.lower() if star_color else 'unknown' }}]] [[star system]] in the [[{{ region|e }}]] [[region]] of the [[{{ galaxy|e }}]] galaxy.

{% if 'Uncharted' in faction %}The system is [[uncharted]] so no [[faction]] inhabits this system. The system [[economy]] is non-existent.
{% elif 'Abandoned' in faction %}The [[{{ faction.replace(" Abandoned", "") }}]] [[faction]] abandoned this system. The system [[economy]] is non-existent.
{% else %}The [[{{ faction }}]] [[faction]] inhabits this system. The system [[economy]] is primarily {{ economy_desc|lower }}. The economic conditions are {{ wealth|lower }}.
{% endif %}

The star system is estimated to be {{ age }} billion years old.

==Alias Names==
{{ '{{aliasc|text=Original|name=' + (original_name|e or name|e) + '}}' }}
{{ '{{aliasc|text=' + release + '|name=' + name|e + '}}' }}

==Planets & Moons==
{{ '{{PM|' + planet_count|string + '|' + moon_count|string + '}}' }}
{% for name in celestial_names %}
* {{ name|e }}{% endfor %}

==Location Information==
It is located in the {{ region|e }} region and is approximately {{ distance }} light years from the [[Galaxy Centre|galactic centre]].

{{ '{{CoordGlyphConvert|' + coordinates + '}}' }}

===Navigation Images===
{% if nav_image %}[[File:{{ nav_image|e }}|400px]]{% else %}<!-- No Nav Image -->{% endif %}

==Space Station==
{% if 'Uncharted' in faction %}As this system is uncharted, there is no [[Space station]].
{% elif 'Abandoned' in faction %}This [[space station]] is abandoned. The following [[Trade Commodities]] may still be found:
{% else %}The following [[Trade Commodities]] are found at the [[Galactic Trade Terminal]]:
{% endif %}{% if commodities %}{% for c in commodities %}
* {{ '{{Resource2icon|' + c + '}}' }} [[{{ c }}]]{% endfor %}{% else %}
No notable commodities.{% endif %}
{% if has_merchants %}
{{ '{{SSMerchants' }}
{% for p, items in ss_merchants.items() %}{% for i in range(1, 6) %}{% if items[i-1] %}|{{p}}{{i}}={{ items[i-1]|e }}
{% endif %}{% endfor %}{% endfor %}}}
{% endif %}

==Additional Information==
* Discovered on {{ discovery_date.format('DD-MMM-YYYY') if discovery_date else '' }} (AGT Stardate {{ stardate }}).
* Discovery documented by [[Alliance of Galactic Travellers]] explorer ''{{ discoverer|e }}''.
* Survey research contributed by the AGT research team.

==Gallery==
<gallery>
{% for img in gallery_images %}File:{{ img|e }}
{% endfor %}</gallery>

{{ '{{AGT Galactic Archive Sync}}' }}
""")

    def __init__(self):
        """Initialize the UI, Data, and Event Handlers.

        This constructor sets up the entire application:
        1. Initializes data references
        2. Creates the widget container
        3. Sets up styling
        4. Builds the user interface
        5. Connects event handlers
        6. Displays the application
        """
        self.data = NMSData  # Reference to the data class
        self.widgets = AppWidgets()  # Container for all UI widgets
        self.generated_content = ""  # Stores the last generated wiki code
        self._reset_confirm_active = False  # Flag for reset confirmation

        # Styling definitions for consistent UI appearance
        self.LABEL_STYLE = {'description_width': '140px'}  # Width for widget labels
        self.WIDGET_LAYOUT = Layout(width='98%')  # Standard widget width
        self.TEXTAREA_LAYOUT = Layout(width='98%', height='120px')  # Text area size
        self.COL_LAYOUT = Layout(width='50%')  # Two-column layout width
        self.FULL_ROW = Layout(width='100%', margin='5px 0')  # Full-width row layout

        # Build the interface
        self._create_ui_components()  # Create all widgets
        self._layout_tabs()  # Organize widgets into tabs
        self._connect_events()  # Connect buttons and inputs to functions

    def _create_ui_components(self):
        """Create all the input boxes, buttons, and other UI widgets.

        This method creates every widget needed for the form and stores
        them in the self.widgets dataclass. Widgets are organized by
        category (location, star info, demographics, etc.).
        """
        # 1. Location & Identification widgets
        self._create_widget(Combobox, 'galaxy', 'Galaxy *', options=self.data.GALAXIES, placeholder='e.g. Euclid')
        self._create_widget(Text, 'glyphs', 'Portal Glyphs (Hex) *', placeholder='12 Hex Chars')
        self._create_widget(Text, 'coordinates', 'Coordinates', disabled=True)
        self._create_widget(Text, 'distance', 'Dist. to Core (LY)', disabled=True)
        self._create_widget(IntText, 'system_id', 'System Index', disabled=True)
        self._create_widget(Text, 'region', 'Region Name', disabled=True)
        self._create_widget(Text, 'name', 'System Name *')
        self._create_widget(Text, 'original_name', 'Original Name')

        # 2. Star Information widgets
        self._create_widget(Text, 'spectral_class', 'Spectral Class')
        self._create_widget(Dropdown, 'star_color', 'Star Color', options=['Yellow', 'Red', 'Green', 'Blue', 'Purple', 'Black', 'Unknown'])
        self._create_widget(Dropdown, 'multiple_stars', 'Visible Stars', options=['1', '2', '3'])

        # 3. Planetary System widgets
        self._create_widget(IntText, 'planet_count', 'Planet Count', value=0)
        self._create_widget(IntText, 'moon_count', 'Moon Count', value=0)
        self._create_widget(Textarea, 'celestial_names', 'Body Names List')
        # Counter display for planets/moons parsed from the text area
        self.celestial_counter = HTML(value="<span style='color:#777; font-size:10px; margin-left:145px'>0 planets, 0 moons parsed</span>")

        # 4. Media widgets
        self._create_widget(Text, 'image', 'Infobox Image')
        self._create_widget(Text, 'nav_image', 'Navigation Map Image')
        self._create_widget(Textarea, 'gallery_images', 'Gallery Images')

        # 5. Demographics widgets
        self._create_widget(Dropdown, 'faction', 'Dominant Faction', options=self.data.FACTIONS)
        self._create_widget(Text, 'civilized', 'Civilization', value='Alliance of Galactic Travellers')
        self._create_widget(Combobox, 'economy_desc', 'Economy Type', options=self.data.ECONOMY_LIST)
        self._create_widget(Combobox, 'wealth', 'Wealth Level', options=self.data.WEALTH_LIST)
        self._create_widget(FloatText, 'economy_buy', 'Buy Modifier %', value=0.0)
        self._create_widget(FloatText, 'economy_sell', 'Sell Modifier %', value=0.0)
        self._create_widget(Combobox, 'conflict', 'Conflict Level', options=self.data.CONFLICT_LIST)
        self._create_widget(Text, 'age', 'Age (Billions)')
        self._create_widget(Checkbox, 'water', 'Water Present', layout=Layout(width='auto', margin='0 20px 0 0'))
        self._create_widget(Checkbox, 'dissonant', 'Dissonant System', layout=Layout(width='auto'))

        # 6. Discovery widgets
        self._create_widget(Text, 'discoverer', 'Discoverer Alias *')
        self._create_widget(Text, 'discoverer_link', 'Wiki Username')
        self._create_widget(DatePicker, 'date', 'Discovery Date', value=arrow.now().date())
        self._create_widget(Text, 'stardate', 'AGT Stardate', disabled=True)
        self._create_widget(Dropdown, 'platform', 'Platform', options=self.data.PLATFORMS)
        self._create_widget(Dropdown, 'mode', 'Game Mode', options=self.data.MODES, value='Normal')
        self._create_widget(Text, 'release', 'Game Version', value='Breach')

        # Checkboxes for trade commodities (dynamically created from data)
        self.widgets.commodity_checks = {
            c: Checkbox(description=c, layout=Layout(width='auto'))
            for c in self.data.COMMODITIES
        }

        # Action Buttons
        self.btn_preview = Button(description='Preview Code', button_style='info', icon='eye')
        self.btn_gen = Button(description='Generate & Save', button_style='success', icon='code')
        self.btn_copy = Button(description='Copy to Clipboard', button_style='primary', icon='copy', disabled=True)
        self.btn_download = Button(description='Download File', button_style='primary', icon='download', disabled=True)
        self.btn_clear = Button(description='Reset Form', button_style='danger', icon='trash')
        self.btn_example = Button(description='Load Example', button_style='warning', icon='upload')

        # Output Display areas
        self.output = Output(layout={'border': '1px solid #ccc', 'height': '400px', 'overflow_y': 'scroll', 'padding': '10px', 'font_family': 'monospace'})
        self.status_output = Output()

    def _create_widget(self, widget_class, key, description, store_in_dict=False, **kwargs):
        """Helper to create a widget with consistent styling.

        This method ensures all widgets have the same look and feel,
        and are properly stored in the widgets dataclass.

        Args:
            widget_class: The type of widget to create (Text, Dropdown, etc.).
            key: The attribute name to store the widget under.
            description: The label text for the widget.
            store_in_dict: If True, store in a dictionary instead of as an attribute.
            **kwargs: Additional parameters to pass to the widget constructor.

        Returns:
            The created widget instance.
        """
        # Choose layout based on widget type
        layout = self.TEXTAREA_LAYOUT if widget_class in [Textarea] else self.WIDGET_LAYOUT
        # Combine all parameters
        params = {'description': description, 'style': self.LABEL_STYLE, 'layout': layout, **kwargs}

        # Remove any unsupported parameters
        if 'ensure_option' in params:
             del params['ensure_option']

        # Create the widget instance
        widget_instance = widget_class(**params)

        # Store the widget in the appropriate place
        if store_in_dict:
            self.widgets.upgrade_merchants[key] = widget_instance
        else:
            setattr(self.widgets, key, widget_instance)
        return widget_instance

    # UI Layout Helpers
    def _header(self, text):
        """Create a styled header HTML element.

        Args:
            text: The header text.

        Returns:
            An HTML widget with header styling.
        """
        return HTML(f"<div style='{HEADER_STYLE}'>{text}</div>")

    def _desc(self, text):
        """Create a styled description HTML element.

        Args:
            text: The description text.

        Returns:
            An HTML widget with description styling.
        """
        return HTML(f"<div style='{DESC_STYLE}'>{text}</div>")

    def _two_col_row(self, w1, w2=None):
        """Create a two-column layout row.

        Args:
            w1: Widget for the left column.
            w2: Widget for the right column (optional).

        Returns:
            An HBox containing the widgets in two columns.
        """
        return HBox([VBox([w1], layout=self.COL_LAYOUT), VBox([w2] if w2 else [], layout=self.COL_LAYOUT)], layout=self.FULL_ROW)

    def _layout_tabs(self):
        """Arrange widgets into Tabbed pages for organized display.

        This method groups related widgets into tabs:
        1. Location & ID
        2. Star Info
        3. Media
        4. Demographics
        5. Discovery
        6. Space Station
        7. Generate

        Each tab contains a vertical box (VBox) with headers, descriptions,
        and widgets arranged in rows.
        """
        w = self.widgets  # Short alias for cleaner code

        # Tab 1: Location & Identification
        t1 = VBox([
            self._header('Location Source'),
            self._desc("Select Galaxy and enter 12-char Hex Glyphs. Coordinates/Region are calculated automatically."),
            self._two_col_row(w.galaxy, w.glyphs),
            self._header('Calculated Location Data'),
            self._desc("Read-only fields updated from Glyphs."),
            self._two_col_row(w.coordinates, w.distance),
            self._two_col_row(w.region, w.system_id),
            self._header('System Identity'),
            self._desc("Enter Wiki Name. If renamed from procedural, enter original name too."),
            self._two_col_row(w.name, w.original_name)
        ], layout=Layout(padding='20px'))

        # Tab 2: Star Information
        t2 = VBox([
            self._header('Star Characteristics'),
            self._desc("Check Galaxy Map for Spectral Class. Color auto-selects based on class."),
            self._two_col_row(w.spectral_class, w.star_color),
            self._two_col_row(w.multiple_stars),
            self._header('Planetary System'),
            self._desc("Enter counts. List names one per line. Prefix moons with '-'."),
            self._two_col_row(w.planet_count, w.moon_count),
            w.celestial_names,
            self.celestial_counter
        ], layout=Layout(padding='20px'))

        # Tab 3: Media
        t3 = VBox([
            self._header('Wiki Media'),
            self._desc("Filenames only. No 'File:' prefix."),
            self._two_col_row(w.image, w.nav_image),
            self._desc("Additional gallery images (one per line)."),
            w.gallery_images
        ], layout=Layout(padding='20px'))

        # Tab 4: Demographics
        t4 = VBox([
            self._header('Faction & Civilization'),
            self._two_col_row(w.faction, w.civilized),
            self._header('Economy'),
            self._desc("Data from Galaxy Map."),
            self._two_col_row(w.economy_desc, w.wealth),
            self._two_col_row(w.economy_buy, w.economy_sell),
            self._header('Environment'),
            self._two_col_row(w.conflict, w.age),
            HBox([w.water, w.dissonant], layout=Layout(margin='10px 0'))
        ], layout=Layout(padding='20px'))

        # Tab 5: Discovery
        t5 = VBox([
            self._header('Discovery Record'),
            self._desc("In-game name. Wiki account if different."),
            self._two_col_row(w.discoverer, w.discoverer_link),
            self._two_col_row(w.date, w.stardate),
            self._header('Platform & Version'),
            self._two_col_row(w.platform, w.mode),
            self._two_col_row(w.release)
        ], layout=Layout(padding='20px'))

        # Tab 6: Space Station
        # Create a grid of commodity checkboxes
        comm_box = GridBox(
            children=list(self.widgets.commodity_checks.values()),
            layout=Layout(grid_template_columns="repeat(auto-fill, minmax(220px, 1fr))", height='250px', overflow_y='scroll', border='1px solid #ccc', padding='8px')
        )

        # Create accordion for upgrade merchants (collapsible sections)
        acc_children = []
        # Define the four merchant types and their data
        for prefix, items in [('MT', self.data.UPGRADES_MT), ('SS', self.data.UPGRADES_SS), ('EC', self.data.UPGRADES_EC), ('ES', self.data.UPGRADES_ES)]:
             # Create 5 dropdowns for each merchant type (slots 1-5)
             acc_children.append(VBox([self._create_widget(Dropdown, f"upgrade_{prefix}{i}", f"Slot {i}", options=[''] + items, store_in_dict=True) for i in range(1, 6)]))
        acc = Accordion(children=acc_children)
        # Set titles for each accordion section
        for i, t in enumerate(['Multi-Tool', 'Starship', 'Exocraft', 'Exosuit']): acc.set_title(i, f'{t} Upgrade Merchant')
        acc.selected_index = 0  # Open first section by default

        t6 = VBox([
            self._header('Trade Commodities'),
            self._desc("Select all available at the station terminal."),
            comm_box,
            self._header('S-Class Upgrades'),
            self._desc("S-Class tech sold by merchants."),
            acc
        ], layout=Layout(padding='20px'))

        # Tab 7: Generate
        t7 = VBox([
            self._header('Finalization'),
            self._desc("Review data. Click Generate to create file."),
            HBox([self.btn_preview, self.btn_gen, self.btn_copy, self.btn_download, HTML("&nbsp;&nbsp;|&nbsp;&nbsp;"), self.btn_example, self.btn_clear], layout=Layout(justify_content='center', margin='15px 0')),
            self.status_output, self._header('Code Output'), self.output
        ], layout=Layout(padding='20px'))

        # Create the tab container with all 7 tabs
        self.tabs = Tab(children=[t1, t2, t3, t4, t5, t6, t7])
        # Set tab titles
        for i, t in enumerate(['Location & ID', 'Star Info', 'Media', 'Demographics', 'Discovery', 'Space Station', 'Generate']): self.tabs.set_title(i, t)
        # Display the tabs in the notebook
        display(self.tabs)

    def _connect_events(self):
        """Connect buttons and inputs to their logic functions.

        This method sets up event handlers so that when users interact
        with the form (type text, click buttons, select options), the
        appropriate functions are called.
        """
        # Connect input field observers (trigger on value change)
        self.widgets.glyphs.observe(self._update_coord_and_region, names='value')
        self.widgets.galaxy.observe(self._update_coord_and_region, names='value')
        self.widgets.spectral_class.observe(self._update_color_ui, names='value')
        self.widgets.date.observe(self._update_stardate_ui, names='value')
        self.widgets.celestial_names.observe(self._update_celestial_counter, names='value')

        # Initialize some UI elements
        self._update_stardate_ui(None)

        # Connect button click handlers
        self.btn_preview.on_click(lambda b: self._generate('preview'))
        self.btn_gen.on_click(lambda b: self._generate('full'))
        self.btn_copy.on_click(self._copy_to_clipboard)
        self.btn_download.on_click(self._download_file)
        self.btn_clear.on_click(self._clear_form)
        self.btn_example.on_click(self._load_example)

        # Disable export buttons when form changes (to prevent using outdated code)
        for w in [self.widgets.name, self.widgets.discoverer, self.widgets.platform]:
            w.observe(self._disable_export_buttons, names='value')

    def _disable_export_buttons(self, change):
        """Disable copy/download buttons if the form is modified after generation.

        This prevents users from accidentally copying or downloading
        outdated wiki code after they've changed the form data.

        Args:
            change: The widget change event (not used, required by observer pattern).
        """
        if self.btn_download.disabled is False:
            self.btn_download.disabled = True
            self.btn_copy.disabled = True
            with self.status_output:
                print("⚠️ Form changed. Regenerate code to enable export.")

    def _update_celestial_counter(self, change):
        """Update the text showing how many planets/moons were parsed from the text area.

        This provides immediate feedback to users about how many bodies
        they've listed in the celestial names text area.

        Args:
            change: The widget change event (not used, required by observer pattern).
        """
        # Split text into lines and filter out empty lines
        lines = [l for l in self.widgets.celestial_names.value.split('\n') if l.strip()]
        # Count planets (lines not starting with '-') and moons (lines starting with '-')
        p_count = sum(1 for l in lines if not l.strip().startswith('-'))
        m_count = len(lines) - p_count
        # Update the counter display
        self.celestial_counter.value = f"<span style='color:#777; font-size:10px; margin-left:145px'>{p_count} planets, {m_count} moons parsed</span>"

    def _update_coord_and_region(self, change):
        """Calculate X, Y, Z, Distance, and Region Name from Glyphs.

        This is the core coordinate conversion function. When users enter
        portal glyphs, this method:
        1. Validates the glyph format
        2. Converts hex to coordinates
        3. Calculates distance from galaxy center
        4. Generates the region name
        5. Updates all related fields

        Args:
            change: The widget change event (contains new glyph value).
        """
        # Get and clean the glyph input
        g_raw = str(getattr(self.widgets.glyphs, 'value', '')).strip().upper()
        # Ensure the UI shows the uppercase version
        if self.widgets.glyphs.value != g_raw:
             self.widgets.glyphs.value = g_raw
             return

        # Get the selected galaxy name
        g_name = str(getattr(self.widgets.galaxy, 'value', '')).strip()

        # Validate glyph format (must be 12 hex chars)
        if len(g_raw) != 12 or not re.match(HEX_REGEX, g_raw):
            self._reset_location_fields()
            return

        try:
            # Parse Hex values from the glyph string
            # Each part of the glyph represents different coordinate components
            s_idx = int(g_raw[1:4], 16)  # System index (positions 1-3)
            y, z, x = int(g_raw[4:6], 16), int(g_raw[6:9], 16), int(g_raw[9:12], 16)  # Y, Z, X coordinates

            # Convert portal code to actual coordinates using game's formula
            # The game stores coordinates with offsets that we need to reverse
            cx = (x - SHIFT_POS_XZ) if x >= SHIFT_POS_XZ else (x + SHIFT_NEG_XZ)
            cz = (z - SHIFT_POS_XZ) if z >= SHIFT_POS_XZ else (z + SHIFT_NEG_XZ)
            cy = (y - SHIFT_POS_Y) if y >= SHIFT_POS_Y else (y + SHIFT_NEG_Y)

            # Calculate distance from galaxy center using Pythagorean theorem in 3D
            # Then multiply by the light-year scale factor (400 LY per coordinate unit)
            dist = int(math.sqrt((cx - CENTER_X)**2 + (cy - CENTER_Y)**2 + (cz - CENTER_Z)**2) * LY_SCALE)

            # Update UI fields with calculated values
            self.widgets.coordinates.value = f"{cx:04X}:{cy:04X}:{cz:04X}:{s_idx:04X}"
            self.widgets.system_id.value = s_idx
            self.widgets.distance.value = f"{dist:,}"  # Format with commas for thousands

            # Generate Region Name from coordinates
            # First find the galaxy index from the galaxy name
            gal_idx = next((k for k, v in self.data.GALAXY_MAP.items() if v == g_name), 0)
            # Create region seed from adjusted coordinates
            reg_seed = RegionNameGenerator.create_region_seed(cx - 0x7FF, cy - 0x7F, cz - 0x7FF, gal_idx)
            # Generate and display the region name
            self.widgets.region.value = RegionNameGenerator.format_name(reg_seed)
            # Change glyph text color to green to indicate valid input
            self.widgets.glyphs.style.text_color = 'green'

        except (ValueError, IndexError, KeyError) as e:
            # Handle any errors in calculation (invalid hex, missing galaxy, etc.)
            self._reset_location_fields(error=str(e))

    def _reset_location_fields(self, error=None):
        """Clear calculated fields if input is invalid.

        When users enter invalid glyphs or clear the field, this method
        resets all the calculated fields to empty or error states.

        Args:
            error: Optional error message to display.
        """
        self.widgets.coordinates.value = "Error" if error else ""
        self.widgets.system_id.value = 0
        self.widgets.distance.value = ""
        self.widgets.region.value = f"Error: {error}" if error else ""
        self.widgets.glyphs.style.text_color = 'black'

    def _update_stardate_ui(self, change):
        """Calculate the AGT Stardate format (Year.Day.Month).

        The Alliance of Galactic Travellers uses a custom stardate format
        based on the discovery date. This converts the selected date to
        that format.

        Args:
            change: The widget change event (not used, required by observer pattern).
        """
        if d := self.widgets.date.value:
            # AGT stardate format: (Year + 1716).Day.Month
            self.widgets.stardate.value = f"{d.year + 1716}.{d.day}.{d.month:02d}"

    def _update_color_ui(self, change):
        """Auto-select star color based on the first letter of Spectral Class.

        The game has standard colors for different star types:
        - O, B: Blue
        - A, F: White/Green
        - G: Yellow
        - K, M: Red/Orange
        - E, L, T, Y: Brown/Black

        Args:
            change: The widget change event (contains new spectral class).
        """
        c = str(getattr(self.widgets.spectral_class, 'value', '')).strip().upper()
        if c and c[0] in self.data.CLASS_TO_COLOR:
            self.widgets.star_color.value = self.data.CLASS_TO_COLOR[c[0]]

    def _clean_value(self, value):
        """Helper to clean empty or default-looking strings.

        Converts placeholder values, "Select..." options, and None to
        empty strings for cleaner data processing.

        Args:
            value: The value to clean.

        Returns:
            Cleaned string or empty string.
        """
        if value is None or any(p in str(value) for p in ["Select ", "..."]):
            return ""
        return str(value).strip()

    def _gather_data_and_validate(self):
        """Collect data from widgets and validate it using the Pydantic model.

        This method:
        1. Extracts values from all widgets
        2. Cleans and formats the data
        3. Passes it through the Pydantic model for validation
        4. Returns either valid data or None (with error messages)

        Returns:
            A validated SystemDataModel instance, or None if validation fails.
        """
        w = self.widgets  # Short alias

        # Extract and clean data from all widgets
        raw_data = {
            'name': w.name.value.strip(),
            'original_name': w.original_name.value.strip(),
            'galaxy': self._clean_value(w.galaxy.value),
            'region': w.region.value.strip(),
            'spectral_class': self._clean_value(w.spectral_class.value),
            'star_color': self._clean_value(w.star_color.value),
            'multiple_stars': self._clean_value(w.multiple_stars.value),
            'planet_count': w.planet_count.value,
            'moon_count': w.moon_count.value,
            'celestial_names': [ln.strip() for ln in w.celestial_names.value.split('\n') if ln.strip()],
            'glyphs': w.glyphs.value.strip(),
            'coordinates': w.coordinates.value,
            'system_id': w.system_id.value,
            'distance': w.distance.value,
            'image': w.image.value.strip(),
            'nav_image': w.nav_image.value.strip(),
            'gallery_images': [ln.strip() for ln in w.gallery_images.value.split('\n') if ln.strip()],
            'faction': self._clean_value(w.faction.value),
            'civilized': w.civilized.value,
            'economy_desc': self._clean_value(w.economy_desc.value),
            'wealth': self._clean_value(w.wealth.value),
            'economy_buy': w.economy_buy.value,
            'economy_sell': w.economy_sell.value,
            'conflict': self._clean_value(w.conflict.value),
            'age': w.age.value.strip(),
            'water': w.water.value,
            'dissonant': w.dissonant.value,
            'discoverer': w.discoverer.value.strip(),
            'discoverer_link': w.discoverer_link.value.strip(),
            'discovery_date': arrow.get(w.date.value) if w.date.value else None,
            'stardate': w.stardate.value,
            'platform': self._clean_value(w.platform.value),
            'mode': self._clean_value(w.mode.value),
            'release': w.release.value,
            'commodities': [c for c, chk in w.commodity_checks.items() if chk.value],
            'ss_merchants': {p: [self._clean_value(w.upgrade_merchants[f'upgrade_{p}{i}'].value) for i in range(1, 6)] for p in ['MT', 'SS', 'EC', 'ES']}
        }

        try:
            # Validate the data using Pydantic
            return SystemDataModel(**raw_data)
        except ValidationError as e:
            # Display validation errors to the user
            with self.status_output:
                clear_output(wait=True)
                print("❌ Validation Errors:")
                for err in e.errors():
                    print(f" - {err['loc'][0]}: {err['msg']}")
            return None

    def _generate(self, mode):
        """Generate the wiki code from the form data.

        This is the main generation function that:
        1. Validates the form data
        2. Prepares the template context
        3. Renders the wiki template
        4. Displays the result
        5. Enables export buttons if in 'full' mode

        Args:
            mode: Either 'preview' (just show) or 'full' (enable export).
        """
        # Step 1: Validate the form data
        validated_data = self._gather_data_and_validate()
        if not validated_data:
            return  # Validation failed, errors already displayed

        # Step 2: Prepare template context
        context = validated_data.model_dump()  # Convert to dictionary

        # Step 3: Check if there are any merchants to display
        # This determines whether to include the merchant section in the output
        context['has_merchants'] = any(
            item for sublist in context['ss_merchants'].values() for item in sublist
        )

        # Step 4: Render the template
        self.generated_content = self.WIKI_TEMPLATE.render(context)

        # Step 5: Display the generated code
        with self.output:
            clear_output(wait=True)
            print(self.generated_content)

        # Step 6: Update status and enable/disable buttons
        with self.status_output:
            clear_output(wait=True)
            if mode == 'full':
                # Enable export buttons for full generation
                self.btn_download.disabled = False
                self.btn_copy.disabled = False
                print("✅ Wiki code generated and ready to export.")
            else:
                # Keep export disabled for preview
                self.btn_download.disabled = True
                self.btn_copy.disabled = True
                print("🔵 Preview generated (Buttons disabled until full generation).")

    def _clear_form(self, b):
        """Reset the form to default empty state.

        This method clears all form fields. It uses a two-step confirmation
        to prevent accidental resets: first click shows "Confirm Reset?",
        second click actually resets.

        Args:
            b: The button click event (not used).
        """
        # Two-step confirmation to prevent accidental resets
        if not self._reset_confirm_active:
            # First click: show confirmation message
            self.btn_clear.description = "Confirm Reset?"
            self._reset_confirm_active = True
            time.sleep(0.1)  # Small delay for visual feedback
            return

        # Second click: actually reset the form
        # Reset all widget values to defaults
        for widget in self.widgets.__dict__.values():
            if not hasattr(widget, 'value'):
                continue
            if isinstance(widget, (Text, Textarea, Combobox)):
                widget.value = ''
            elif isinstance(widget, Dropdown):
                widget.value = None
            elif isinstance(widget, (IntText, FloatText)):
                widget.value = 0
            elif isinstance(widget, DatePicker):
                widget.value = arrow.now().date()
            elif isinstance(widget, Checkbox):
                widget.value = False

        # Reset commodity checkboxes
        for w in self.widgets.commodity_checks.values():
            w.value = False
        # Reset upgrade merchant dropdowns
        for w in self.widgets.upgrade_merchants.values():
            w.value = None

        # Set some fields to sensible defaults
        self.widgets.civilized.value = "Alliance of Galactic Travellers"
        self.widgets.release.value = "Breach"

        # Clear output areas
        self.output.clear_output()
        self.generated_content = ""
        self.btn_download.disabled = True
        self.btn_copy.disabled = True

        # Reset button text and confirmation flag
        self.btn_clear.description = "Reset Form"
        self._reset_confirm_active = False

        # Show confirmation message
        with self.status_output:
            clear_output()
            print("⚪ Form reset.")

    def _load_example(self, b):
        """Fill the form with dummy data for testing and demonstration.

        This provides users with sample data so they can see how the
        form works without having to look up real system information.

        Args:
            b: The button click event (not used).
        """
        # First reset the form
        self._reset_confirm_active = True
        self._clear_form(None)

        w = self.widgets  # Short alias
        # Fill form with example data
        w.name.value = 'AGT Ontohan-Leksha ACP'
        w.original_name.value = 'Ontohan-Leksha'
        w.galaxy.value = 'Ontiniangp'
        w.glyphs.value = '015AF3545C3E'
        w.spectral_class.value = 'M7p'
        w.image.value = 'ontohandisc.png'
        w.planet_count.value = 3
        w.celestial_names.value = 'Lusworl Minor\nEciu XI\nWolynox Gamma'
        w.faction.value = 'Gek'
        w.economy_desc.value = 'Manufacturing'
        w.wealth.value = 'Destitute'
        w.discoverer.value = 'AnimalCrackr'

        # Show status message
        with self.status_output:
            clear_output()
            print("🟡 Example data loaded.")

    def _copy_to_clipboard(self, b):
        """Use Javascript to copy the generated text to the clipboard.

        This only works in Jupyter Notebook environments where JavaScript
        can be executed in the browser.

        Args:
            b: The button click event (not used).
        """
        if not self.generated_content:
            return  # Nothing to copy
        # Create JavaScript code to copy text to clipboard
        js_code = f"const e=document.createElement('textarea');e.value=`{self.generated_content.replace('`', '\\`')}`;document.body.appendChild(e);e.select();document.execCommand('copy');document.body.removeChild(e);"
        display(Javascript(js_code))
        # Show confirmation
        with self.status_output:
            print("📋 Wiki code copied to clipboard!")

    def _download_file(self, b):
        """Save the generated code as a .txt file.

        In Google Colab, this triggers a download. In regular Jupyter,
        it creates a download link.

        Args:
            b: The button click event (not used).
        """
        if not self.generated_content:
            return  # Nothing to download

        # Create a safe filename from the system name
        filename = re.sub(r'[^a-zA-Z0-9_\\-]', '', self.widgets.name.value.replace(' ', '_')) + "_System.txt"

        try:
            # Try Google Colab download method
            from google.colab import files
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.generated_content)
            files.download(filename)
            msg = f"💾 Downloading '{filename}' (Colab)..."
        except ImportError:
            # Regular Jupyter: create a file link
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.generated_content)
            display(FileLink(filename))
            msg = f"💾 File created: '{filename}' (Click link above to download)"

        # Show status message
        with self.status_output:
            clear_output()
            print(msg)


# =============================================================================
# 7. INITIALIZATION
# =============================================================================

# Display initial loading message
display(HTML("<div><span style='color:blue'><b>Please Wait:</b></span> Initializing App and Fetching Data...</div>"))

# Initialize game data (download from GitHub)
NMSData.initialize()

# Clear the loading message
clear_output()

# Create and display the main application
app = NMSStarSystemWikiCreator()