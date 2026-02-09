"""
No Man's Sky Starship Wiki Form Generator

This module provides a complete interactive form for generating wiki page code
for starships in the game No Man's Sky. It includes:
- Coordinate and region name generation from portal glyphs
- Form validation with Pydantic models
- Interactive Jupyter widget interface
- Wiki template rendering with Jinja2

Key Classes:
    ByteUtils: Handles byte-level operations used in procedural generation
    Generator: Generates procedural region names
    RegionNameGenerator: Creates region names from coordinates
    NMSGalaxyMap: Converts portal glyphs to galactic coordinates
    NMSData: Loads and manages game data from external sources
    NMSWikiStarshipFormCreator: Main application with interactive form

The form allows players to input ship discovery details and automatically
generates properly formatted wiki code for the No Man's Sky Fandom wiki.
"""

import json
import re
import struct
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Optional

import arrow
import ipywidgets as widgets
import jinja2
import requests
from IPython.display import Javascript, clear_output, display
from ipywidgets import (
    Button, Combobox, DatePicker, Dropdown, FloatText, HBox, HTML, IntText,
    Layout, Output, Tab, Text, Textarea, VBox
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Maximum iterations to prevent infinite loops in name generation
MAX_SAFETY_ITERATIONS = 50
# Maximum allowed length for generated names
MAX_NAME_LENGTH = 64
# Total maximum iterations for name generation process
MAX_TOTAL_ITERATIONS = 500
# Year offset for converting real dates to in-game stardates
STARDATE_YEAR_OFFSET = 1716


class ByteUtils:
    """
    Utility class for byte-level operations used in procedural generation.

    This class provides static methods for manipulating lists of bytes (integers
    0-255) including parsing, arithmetic, bitwise operations, and conversions.
    These operations mimic how the game handles byte arrays for procedural
    generation algorithms.

    Methods:
        parse: Converts hex string to byte list
        format_short: Ensures byte list is at least 2 bytes
        add/sub/multiply: Arithmetic operations on byte lists
        shl/shr: Bit shift operations
        rol: Rotate left operation
        logical_op: Bitwise AND/OR/XOR operations
        update_seed: Updates random number generator seed
        to_*/get_bytes_*: Convert between byte lists and numeric types
    """

    @staticmethod
    def parse(val: str, little_endian: bool = True) -> list[int]:
        """
        Converts a hexadecimal string into a list of byte values.

        This method takes a hex string (like "1A2B") and converts it to a list
        of integers where each integer represents one byte (0-255). If the
        string has an odd length, it adds a leading zero. The bytes can be
        returned in reverse order to handle little-endian format.

        Args:
            val (str): Hexadecimal string to parse
            little_endian (bool): If True, returns bytes in reverse order
                                  (least significant byte first)

        Returns:
            list[int]: List of byte values (0-255) parsed from hex string

        Example:
            >>> ByteUtils.parse("1A2B")
            [43, 26]  # With little_endian=True (default)
            >>> ByteUtils.parse("1A2B", little_endian=False)
            [26, 43]  # With little_endian=False
        """
        # Add leading zero if hex string has odd length
        if len(val) % 2 != 0:
            val = "0" + val

        # Convert each 2-character hex pair to integer
        res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]

        # Reverse for little-endian (game uses little-endian)
        if little_endian:
            res.reverse()

        return res

    @staticmethod
    def format_short(op1: list[int]) -> list[int]:
        """
        Ensures a byte list is at least 2 bytes long by padding with zeros.

        This is used when the game expects a 16-bit (2-byte) value but the
        input might be shorter. For example, when working with coordinates
        that should be 2 bytes but might be represented with fewer.

        Args:
            op1 (list[int]): Input byte list

        Returns:
            list[int]: Same list padded with zeros to length 2 if needed

        Example:
            >>> ByteUtils.format_short([0x01])
            [1, 0]  # Padded to 2 bytes
        """
        res = list(op1)
        while len(res) < 2:
            res.append(0x00)  # Add zero byte
        return res

    @staticmethod
    def add(op1: list[int], op2: list[int]) -> list[int]:
        """
        Adds two byte lists together as multi-byte numbers.

        This performs addition similar to how processors add multi-byte
        integers, handling carry between bytes. Both lists are treated as
        little-endian numbers (least significant byte first).

        Args:
            op1 (list[int]): First byte list (addend)
            op2 (list[int]): Second byte list (addend)

        Returns:
            list[int]: Result of addition as byte list

        Raises:
            None: Uses internal error handling

        Example:
            >>> ByteUtils.add([0xFF], [0x01])
            [0, 1]  # 255 + 1 = 256 (0x0100 in little-endian)
        """
        result = list(op2)
        # Add each byte from op1 to corresponding position in result
        for i in range(len(op1)):
            result = ByteUtils._add_single(op1[i], result, i)
        return result

    @staticmethod
    def _add_single(val, target_list, index):
        """
        Internal helper to add a single byte with carry handling.

        This method adds one byte to a specific position in the target list,
        handling overflow (carry) to the next byte position if the sum exceeds
        255. It's called recursively if there's carry to the next byte.

        Args:
            val (int): Byte value (0-255) to add
            target_list (list[int]): Byte list to add to
            index (int): Position in target_list to add to

        Returns:
            list[int]: Updated byte list after addition
        """
        if index < len(target_list):
            # Add values and keep only lower 8 bits (0-255)
            total = val + target_list[index]
            target_list[index] = total & 0xFF
            # Calculate carry (bits 8-15 shifted down)
            rem = (total >> 8) & 0xFF
            # If there's carry, add it to next byte position
            if rem != 0:
                target_list = ByteUtils._add_single(rem, target_list, index + 1)
        else:
            # If index beyond current length, append the value
            target_list.append(val)
        return target_list

    @staticmethod
    def sub(op1: list[int], op2: list[int]) -> list[int]:
        """
        Subtracts one byte list from another as multi-byte numbers.

        Similar to add() but performs subtraction, handling borrowing between
        bytes when needed. Both lists are treated as little-endian numbers.

        Args:
            op1 (list[int]): Byte list to subtract from (minuend)
            op2 (list[int]): Byte list to subtract (subtrahend)

        Returns:
            list[int]: Result of subtraction as byte list

        Example:
            >>> ByteUtils.sub([0x00, 0x01], [0x01])
            [255, 0]  # 256 - 1 = 255 (0xFF00 in little-endian)
        """
        result = list(op2)
        # Subtract each byte from op1 from corresponding position in result
        for i in range(len(op1)):
            result = ByteUtils._sub_single(op1[i], result, i)
        return result

    @staticmethod
    def _sub_single(val, target_list, index):
        """
        Internal helper to subtract a single byte with borrow handling.

        This method subtracts one byte from a specific position in the target
        list, handling underflow (borrow) from the next byte if needed.

        Args:
            val (int): Byte value (0-255) to subtract
            target_list (list[int]): Byte list to subtract from
            index (int): Position in target_list to subtract from

        Returns:
            list[int]: Updated byte list after subtraction
        """
        if index < len(target_list):
            # Subtract and keep only lower 8 bits
            diff = val - target_list[index]
            target_list[index] = diff & 0xFF
            # Calculate borrow (bits 8-15 shifted down, negative means borrow)
            rem = (diff >> 8) & 0xFF
            if rem != 0:
                target_list = ByteUtils._sub_single(rem, target_list, index + 1)
        else:
            # If index beyond current length, append the value
            target_list.append(val)
        return target_list

    @staticmethod
    def multiply(op1: list[int], op2: list[int]) -> list[int]:
        """
        Multiplies two byte lists as multi-byte numbers.

        This performs multiplication similar to how processors multiply
        multi-byte integers, handling partial products and carries.
        It treats both lists as little-endian numbers.

        Args:
            op1 (list[int]): First byte list (multiplicand)
            op2 (list[int]): Second byte list (multiplier)

        Returns:
            list[int]: Product as byte list

        Example:
            >>> ByteUtils.multiply([0x02], [0x80])
            [0, 1]  # 2 * 128 = 256 (0x0100 in little-endian)
        """
        result = []
        # Multiply each byte of op1 with each byte of op2
        for i in range(len(op1)):
            rem = 0  # Carry from previous multiplication
            for j in range(len(op2)):
                # Calculate product of two bytes plus any carry
                raw_prod = (op1[i] * op2[j]) + rem
                # Convert to signed 16-bit (game's signed arithmetic)
                signed_prd = (raw_prod + 32768) % 65536 - 32768
                # Get upper byte (carry) and lower byte (result)
                rem = (signed_prd >> 8) & 0xFF
                res = signed_prd & 0xFF
                idx = i + j  # Position in result for this partial product
                # Add this partial product to the result
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
    def shl(op1, shift):
        """
        Shift bytes left (remove bytes from beginning).

        This removes the first 'shift' bytes from the list, effectively
        shifting left. If shifting beyond list length, returns single zero byte.

        Args:
            op1 (list[int]): Byte list to shift
            shift (int): Number of bytes to shift left

        Returns:
            list[int]: Shifted byte list

        Example:
            >>> ByteUtils.shl([1, 2, 3], 1)
            [2, 3]  # First byte removed
        """
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1, shift):
        """
        Shift bytes right (remove bytes from end).

        This removes the last 'shift' bytes from the list, effectively
        shifting right. If shifting beyond list length, returns single zero byte.

        Args:
            op1 (list[int]): Byte list to shift
            shift (int): Number of bytes to shift right

        Returns:
            list[int]: Shifted byte list

        Example:
            >>> ByteUtils.shr([1, 2, 3], 1)
            [1, 2]  # Last byte removed
        """
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1, roll):
        """
        Rotate byte list left (circular shift).

        Bytes moved from beginning to end, preserving all data.
        Rotation wraps around if roll exceeds list length.

        Args:
            op1 (list[int]): Byte list to rotate
            roll (int): Number of positions to rotate left

        Returns:
            list[int]: Rotated byte list

        Example:
            >>> ByteUtils.rol([1, 2, 3], 1)
            [2, 3, 1]  # First element moved to end
        """
        if not op1:
            return op1
        # Handle roll larger than list length using modulo
        r = roll % len(op1)
        return op1[r:] + op1[:r]

    @staticmethod
    def zxd(op1, extend):
        """
        Zero-extend byte list to specified length.

        Adds zero bytes to the end of the list until it reaches 'extend' length.
        Used when a number needs more bytes but should be unsigned.

        Args:
            op1 (list[int]): Byte list to extend
            extend (int): Desired total length

        Returns:
            list[int]: Extended byte list with zeros

        Example:
            >>> ByteUtils.zxd([1, 2], 4)
            [1, 2, 0, 0]
        """
        return list(op1) + [0x00] * (extend - len(op1))

    @staticmethod
    def sxd(op1, extend):
        """
        Sign-extend byte list to specified length.

        Adds bytes to preserve the sign (positive or negative) when extending.
        If the most significant bit of last byte is 1 (negative), adds 0xFF bytes.
        Otherwise adds 0x00 bytes.

        Args:
            op1 (list[int]): Byte list to extend
            extend (int): Desired total length

        Returns:
            list[int]: Sign-extended byte list

        Example:
            >>> ByteUtils.sxd([0xFF], 2)
            [255, 255]  # Negative number extended with 0xFF
        """
        result = list(op1)
        # Check sign bit of most significant byte (bit 7)
        val = 0xFF if (len(op1) > 0 and (op1[-1] >> 7) == 1) else 0x00
        # Append sign-preserving bytes
        for _ in range(extend - len(op1)):
            result.append(val)
        return result

    @staticmethod
    def logical_op(op1, op2, mode):
        """
        Perform bitwise logical operation on two byte lists.

        Handles AND (mode 0), OR (mode 1), or XOR (mode 2) operations.
        Lists are aligned by padding shorter list with zeros, then operation
        is applied byte-by-byte.

        Args:
            op1 (list[int]): First byte list
            op2 (list[int]): Second byte list
            mode (int): 0=AND, 1=OR, 2=XOR

        Returns:
            list[int]: Result of logical operation

        Raises:
            None: Mode is validated internally

        Example:
            >>> ByteUtils.logical_op([0xF0], [0x0F], 0)
            [0]  # 0xF0 AND 0x0F = 0x00
        """
        len1, len2 = len(op1), len(op2)
        # Use longer list as base, pad shorter list with zeros
        longer = list(op1) if len1 > len2 else list(op2)
        shorter = (list(op2) if len1 > len2 else list(op1)) + [0x00] * abs(len1 - len2)
        res = []
        # Apply operation to each byte pair
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
        Bitwise XOR (exclusive OR) of two byte lists.

        Each bit is 1 only if bits differ between inputs.
        Wrapper for logical_op with mode 2.

        Args:
            op1 (list[int]): First byte list
            op2 (list[int]): Second byte list

        Returns:
            list[int]: XOR result as byte list

        Example:
            >>> ByteUtils.xor([0xF0], [0xFF])
            [15]  # 0xF0 XOR 0xFF = 0x0F
        """
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1, op2):
        """
        Bitwise AND of two byte lists.

        Each bit is 1 only if both input bits are 1.
        Wrapper for logical_op with mode 0.

        Args:
            op1 (list[int]): First byte list
            op2 (list[int]): Second byte list

        Returns:
            list[int]: AND result as byte list

        Example:
            >>> ByteUtils.and_op([0xF0], [0x0F])
            [0]  # 0xF0 AND 0x0F = 0x00
        """
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1, op2):
        """
        Bitwise OR of two byte lists.

        Each bit is 1 if either input bit is 1.
        Wrapper for logical_op with mode 1.

        Args:
            op1 (list[int]): First byte list
            op2 (list[int]): Second byte list

        Returns:
            list[int]: OR result as byte list

        Example:
            >>> ByteUtils.or_op([0xF0], [0x0F])
            [255]  # 0xF0 OR 0x0F = 0xFF
        """
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache, move=1):
        """
        Update random number generator seed using game's algorithm.

        The game uses a linear congruential generator (LCG) with specific
        multiplier. This updates the seed state for procedural generation.

        Args:
            cache (list): Two-element list containing seed state [part1, part2]
            move (int): Number of times to update the seed

        Returns:
            list: Updated seed cache

        Example:
            >>> seed = [[1], [2]]
            >>> ByteUtils.update_seed(seed)
            [[...], [...]]  # Updated seed values
        """
        multiplier = [0x99, 0xF8, 0x76, 0x5A]  # Game's LCG multiplier
        for _ in range(move):
            # Multiply, add, then split result into two parts
            step1 = ByteUtils.multiply(cache[0], multiplier)
            result = ByteUtils.add(step1, cache[1])
            cache[0] = ByteUtils.shl(result, 4)   # First 4 bytes become part1
            cache[1] = ByteUtils.shr(result, 4)   # Remaining bytes become part2
        return cache

    @staticmethod
    def to_uint32(arr):
        """
        Convert byte list to unsigned 32-bit integer.

        Interprets first 4 bytes as little-endian unsigned integer.
        Pads with zeros if shorter than 4 bytes.

        Args:
            arr (list[int]): Byte list (0-4 bytes)

        Returns:
            int: Unsigned 32-bit integer value

        Example:
            >>> ByteUtils.to_uint32([1, 0, 0, 0])
            1  # 0x01000000 in little-endian = 1
        """
        return struct.unpack('<I', bytes((arr + [0]*4)[:4]))[0]

    @staticmethod
    def to_int32(arr):
        """
        Convert byte list to signed 32-bit integer.

        Interprets first 4 bytes as little-endian signed integer.
        Pads with zeros if shorter than 4 bytes.

        Args:
            arr (list[int]): Byte list (0-4 bytes)

        Returns:
            int: Signed 32-bit integer value

        Example:
            >>> ByteUtils.to_int32([255, 255, 255, 255])
            -1  # All bits 1 = -1 in two's complement
        """
        return struct.unpack('<i', bytes((arr + [0]*4)[:4]))[0]

    @staticmethod
    def to_int16(arr):
        """
        Convert byte list to signed 16-bit integer.

        Interprets first 2 bytes as little-endian signed integer.
        Pads with zeros if shorter than 2 bytes.

        Args:
            arr (list[int]): Byte list (0-2 bytes)

        Returns:
            int: Signed 16-bit integer value

        Example:
            >>> ByteUtils.to_int16([255, 255])
            -1  # 0xFFFF = -1 in two's complement
        """
        return struct.unpack('<h', bytes((arr + [0]*2)[:2]))[0]

    @staticmethod
    def to_double(arr):
        """
        Convert byte list to 64-bit floating point (double).

        Interprets first 8 bytes as little-endian double precision float.
        Pads with zeros if shorter than 8 bytes.

        Args:
            arr (list[int]): Byte list (0-8 bytes)

        Returns:
            float: Double precision floating point value

        Example:
            >>> ByteUtils.to_double([0, 0, 0, 0, 0, 0, 240, 63])
            1.0  # Double representation of 1.0
        """
        return struct.unpack('<d', bytes((arr + [0]*8)[:8]))[0]

    @staticmethod
    def to_single(arr):
        """
        Convert byte list to 32-bit floating point (single).

        Interprets first 4 bytes as little-endian single precision float.
        Pads with zeros if shorter than 4 bytes.

        Args:
            arr (list[int]): Byte list (0-4 bytes)

        Returns:
            float: Single precision floating point value

        Example:
            >>> ByteUtils.to_single([0, 0, 128, 63])
            1.0  # Float representation of 1.0
        """
        return struct.unpack('<f', bytes((arr + [0]*4)[:4]))[0]

    @staticmethod
    def get_bytes_uint32(val):
        """
        Convert unsigned 32-bit integer to byte list.

        Creates 4-byte little-endian representation of the integer.

        Args:
            val (int): Integer value (0 to 2^32-1)

        Returns:
            list[int]: 4-byte little-endian representation

        Example:
            >>> ByteUtils.get_bytes_uint32(256)
            [0, 1, 0, 0]  # 0x00000100 in little-endian
        """
        return list(struct.pack('<I', val))


class Generator:
    """
    Generates procedural names using game's algorithm.

    This class implements the complex name generation algorithm used by
    No Man's Sky for creating region and system names. It uses byte-level
    operations and probability weights to generate pronounceable names.

    Class Attributes:
        TINY_DOUBLE (list[int]): Special constant used in probability calculation

    Methods:
        generate_name: Main name generation method
        post_process: Applies linguistic rules to improve name quality
        get_from_alphaset: Selects base name fragment from alphabet sets
        get_weights: Gets character probability weights for continuation
        recursive_search: Searches letter map tree for weight data
        insert_vowel: Inserts vowel at specified position
        get_consecutive_consonants: Finds long consonant sequences
    """

    # Small double value (2^-13) used as multiplier in probability calculation
    TINY_DOUBLE = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]

    @staticmethod
    def generate_name(cache0, cache1, alphasets, letter_map):
        """
        Generate a procedural name using game's algorithm.

        This is the main name generation function that:
        1. Gets a base 3-character fragment from alphabet sets
        2. Determines if using alternate character selection mode
        3. Calculates how many additional characters to add
        4. Selects each character based on probability weights
        5. Applies post-processing rules

        Args:
            cache0 (list): First part of random seed cache
            cache1 (list): Second part of random seed cache with metadata
            alphasets (list): List of alphabet set strings (groups of 3 chars)
            letter_map (dict): Nested dictionary of character probability weights

        Returns:
            str: Generated name, or empty string if generation fails

        Raises:
            None: Uses safety limits to prevent infinite loops

        Example:
            >>> seed_cache = [[...], [...]]
            >>> metadata_cache = [[...], [...], [...]]
            >>> Generator.generate_name(seed_cache, metadata_cache, ["abc", "def"], {})
            "Abcdef"  # Example generated name
        """
        # Step 1: Get initial 3-character fragment from alphabet sets
        name = Generator.get_from_alphaset(cache0, cache1, alphasets)
        if name == "__EMPTY__":
            return ""  # Empty alphabet set means no name

        # Step 2: Update seed and check for alternate character selection mode
        ByteUtils.update_seed(cache0)
        check_op = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        is_alternate_char_mode = (ByteUtils.to_int16(check_op) != 0)
        ByteUtils.update_seed(cache0)

        # Step 3: Calculate how many additional characters to generate
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        reg0 = ByteUtils.sub(step5, [0x03])
        limit = ByteUtils.to_int16(ByteUtils.sxd(reg0, 2))

        # Step 4: Generate additional characters up to the calculated limit
        if 0 < limit:
            i, safety, total_iterations = 0, 0, 0
            while i < limit:
                ByteUtils.update_seed(cache0)
                sub_str = name[i: i + 3]  # Last 3 chars for context
                idx = cache1[0][0] if cache1[0] else 0
                weights = Generator.get_weights(sub_str, idx, letter_map)

                # Calculate random target value for character selection
                val_u32 = ByteUtils.to_uint32(cache0[0])
                tiny_dbl = ByteUtils.to_double(Generator.TINY_DOUBLE)
                target = float(val_u32 * tiny_dbl)

                if weights is None:
                    # No weights found - backtrack and try again
                    i -= 1
                    safety += 1
                    if safety > MAX_SAFETY_ITERATIONS:
                        break  # Safety break to prevent infinite loop
                else:
                    safety = 0
                    index = 0

                    # Select character based on mode
                    if is_alternate_char_mode:
                        # Alternate mode: use floating point calculation
                        target *= (len(weights) - 1)
                        op_and = ByteUtils.and_op(list(struct.pack('<f', target)),
                                                  [0x00, 0x00, 0x00, 0x80])
                        op = ByteUtils.or_op(op_and, [0x00, 0x00, 0x00, 0x3F])
                        index = int(ByteUtils.to_single(op) + target)
                    else:
                        # Normal mode: use cumulative probability weights
                        weight = 0.0
                        for k, cw in enumerate(weights):
                            weight += cw[1]  # Add weight
                            if weight >= target:
                                index = k
                                break
                            index = k + 1

                    # Add selected character to name
                    if index < len(weights):
                        name += weights[index][0]

                # Safety checks to prevent excessive generation
                total_iterations += 1
                if total_iterations > MAX_TOTAL_ITERATIONS:
                    break
                if len(name) >= MAX_NAME_LENGTH:
                    name = name[:MAX_NAME_LENGTH]
                    break
                i += 1

        # Step 5: Apply linguistic rules to improve name quality
        return Generator.post_process(name, cache0)

    @staticmethod
    def post_process(name, cache0):
        """
        Apply linguistic rules to improve generated name quality.

        This function:
        1. Inserts vowels between difficult consonant clusters at start
        2. Inserts vowels between difficult consonant clusters at end
        3. Breaks up long sequences of consecutive consonants

        Args:
            name (str): Raw generated name
            cache0 (list): Random seed cache for vowel selection

        Returns:
            str: Processed name with better linguistic properties

        Example:
            >>> Generator.post_process("bctr", [[...], [...]])
            "bacter"  # Vowel inserted between 'b' and 'c'
        """
        if not name:
            return ""

        # Check start of name for difficult consonant clusters
        first, second = name[0], name[1] if len(name) > 1 else ''
        should_skip_vowel_insert = False

        # Rule: Insert vowel if first two chars are consonants (not a,e,i,o,u)
        if (first not in "aeiou") and (second not in "aeiou"):
            cond1 = first != 's' or second not in "hklmnprtwy"
            # Check for specific difficult consonant pairs
            if cond1 and any([(second == 'h' and first in "ctw"),
                              (second == 'l' and first in "bcfgps"),
                              (second == 'r' and first in "bcdfgkpt"),
                              (second == 'w' and first in "dgt"),
                              (second == 'y' and first in "hmr")]):
                should_skip_vowel_insert = True

            # Insert vowel if not skipping
            if not should_skip_vowel_insert:
                name = Generator.insert_vowel(name, cache0, 1)

        # Check end of name for difficult consonant clusters
        ult, penult = name[-1], name[-2] if len(name) > 1 else ''
        if len(name) > 1 and (penult != 'g' or ult in "aeiou"):
            # Check for specific difficult ending pairs
            if any([(ult == 'b' and penult in "gn"),
                    (ult == 'd' and penult in "bdfghkmpst"),
                    (ult == 'g' and penult == 'l'),
                    (ult == 'p' and penult in "bdhkt"),
                    (ult == 'r' and penult in "bfg"),
                    (ult == 't' and penult == 'g'),
                    (ult == 'w' and penult not in "aeiou")]):
                name = Generator.insert_vowel(name, cache0, len(name) - 1)

        # Break up long sequences of consecutive consonants
        consonance = Generator.get_consecutive_consonants(name)
        if consonance != -1:
            ByteUtils.update_seed(cache0)
            mult = ByteUtils.multiply(cache0[0], [0x03])
            offset = ByteUtils.to_int32(ByteUtils.zxd(
                ByteUtils.add(ByteUtils.shr(mult, 4), [0x01]), 4))
            name = Generator.insert_vowel(name, cache0, consonance + offset)

        return name

    @staticmethod
    def get_from_alphaset(cache0, cache1, alphasets):
        """
        Select initial 3-character fragment from alphabet sets.

        Uses random seed to select which alphabet set and which 3-character
        fragment within that set to use as name base.

        Args:
            cache0 (list): Random seed cache
            cache1 (list): Metadata cache containing alphabet set index
            alphasets (list): List of alphabet set strings

        Returns:
            str: 3-character fragment, or "__EMPTY__" if no valid set

        Example:
            >>> alphasets = ["abcdef", "ghijkl"]
            >>> Generator.get_from_alphaset(seed, metadata, alphasets)
            "abc"  # First 3 chars from first set
        """
        ByteUtils.update_seed(cache0)
        # Get alphabet set index from metadata
        idx = cache1[0][0] if cache1[0] else 0
        if idx >= len(alphasets):
            idx = 0  # Default to first set if index out of range

        alphaset_str = alphasets[idx]
        if not alphaset_str:
            return "__EMPTY__"

        # Calculate which 3-character segment to use
        len_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        reg0 = ByteUtils.multiply(cache0[0], len_bytes)
        reg1 = ByteUtils.format_short(ByteUtils.multiply(
            ByteUtils.shr(reg0, 4), [0x03]))
        start = ByteUtils.to_int16(reg1)
        end = ByteUtils.to_int16(ByteUtils.add(reg1, [0x03]))

        return alphaset_str[start:end]

    @staticmethod
    def get_weights(s, alphaset_idx, letter_map):
        """
        Get probability weights for next character based on context.

        Looks up in letter_map what characters can follow the given 3-character
        context string, along with their probability weights.

        Args:
            s (str): 3-character context string
            alphaset_idx (int): Which alphabet set is being used
            letter_map (dict): Nested dictionary of character weights

        Returns:
            list or None: List of (character, weight) tuples, or None if not found

        Example:
            >>> letter_map = {"0": {"a": [["b", 0.5], ["c", 0.5]]}}
            >>> Generator.get_weights("ab", 0, letter_map)
            [("b", 0.5), ("c", 0.5)]  # Possible next chars after "ab"
        """
        key = str(alphaset_idx)
        if not letter_map or key not in letter_map:
            return None

        subset = letter_map[key]
        if not s or s[0] not in subset:
            return None

        return Generator.recursive_search(subset[s[0]], s)

    @staticmethod
    def recursive_search(arr, s):
        """
        Recursively search letter map tree for weight data.

        The letter map is a tree structure where each node can have:
        - A "ja" (jump if above) node: go to child if string > value
        - A "jz" (jump if zero) node: return weights if string == value
        - A leaf node: return the weights list

        Args:
            arr (list): Current level of the letter map tree
            s (str): Context string being searched for

        Returns:
            list or None: List of (character, weight) tuples, or None

        Example:
            >>> tree = [["ja", 100, [...]], ["jz", "abc", weights_list]]
            >>> Generator.recursive_search(tree, "abc")
            weights_list  # Found exact match
        """
        for item in arr:
            if len(item) > 2:
                type_code, val = item[2], item[0]
                if type_code == "ja":
                    # Jump to child if string value > node value
                    s_b = ByteUtils.zxd(list(s.encode()), 4)
                    val_b = ByteUtils.zxd(list(str(val).encode()), 4)
                    if ByteUtils.to_int32(s_b) > ByteUtils.to_int32(val_b):
                        res = Generator.recursive_search(item[1], s)
                        if res:
                            return res
                elif type_code == "jz" and str(val) == s:
                    # Exact match found - return weights
                    return [(w.get("Item1"), float(w.get("Item2", 0)))
                            for w in item[1]]
        return None

    @staticmethod
    def insert_vowel(name, seed, index):
        """
        Insert a vowel at specified position in name.

        Randomly selects which vowel (a,e,i,o,u) to insert based on seed.

        Args:
            name (str): Original name
            seed (list): Random seed cache for vowel selection
            index (int): Position to insert vowel (0 = before first char)

        Returns:
            str: Name with vowel inserted

        Example:
            >>> Generator.insert_vowel("bctr", seed, 1)
            "bacter"  # 'a' inserted at position 1
        """
        ByteUtils.update_seed(seed)
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)
        # Check if we have a valid vowel index (0-4)
        if calc and calc[0] < 5 and index <= len(name):
            return name[:index] + "aeiou"[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name):
        """
        Find position where too many consecutive consonants occur.

        Scans name for sequences of 4+ consonants (not including 'y' as break).
        Returns position where 4th consonant in sequence occurs, or -1 if none.

        Args:
            name (str): Name to check

        Returns:
            int: Index where 4th consonant starts, or -1 if no long sequence

        Example:
            >>> Generator.get_consecutive_consonants("bcdfg")
            3  # Position where 4th consonant ('f') starts
        """
        consonance = 0
        for i, char in enumerate(name):
            if consonance < 3:
                # Count consecutive consonants, reset on vowel
                consonance = 0 if char in "aeiou" else consonance + 1
            else:
                # Already have 3 consonants, check 4th
                if char not in "aeiouy":
                    return i - 3  # Return start of 4-consonant sequence
                else:
                    consonance = 0  # 'y' breaks the sequence
        return -1


class RegionNameGenerator:
    """
    Generates region names from galactic coordinates.

    This class implements the game's algorithm for generating region names
    based on galactic coordinates (x, y, z) and galaxy index. It produces
    the procedural region names seen in the galactic map.

    Class Attributes:
        PROC_ADORNMENTS (list[str]): Suffixes and prefixes that can be
            added to region names (e.g., "Void", "Expanse", "Sea of")

    Methods:
        create_region_seed: Creates seed from coordinates for name generation
        format_name: Generates full region name with possible adornment
    """

    # List of decorative suffixes/prefixes that can be added to region names
    PROC_ADORNMENTS = [
        "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
        "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
        "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
        "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux", "%NAME% Instability",
        "Sea of %NAME%", "The Arm of %NAME%", "%NAME% Spur", "%NAME% Shallows"
    ]

    @staticmethod
    def create_region_seed(x, y, z, galaxy_idx):
        """
        Create seed value from coordinates for name generation.

        Combines galaxy index and coordinates into a hex string, then
        converts to byte list. Format: galaxy(2) + y(2) + z(3) + x(3) hex digits.

        Args:
            x (int): X coordinate in region space
            y (int): Y coordinate in region space
            z (int): Z coordinate in region space
            galaxy_idx (int): Index of galaxy (0 for Euclid)

        Returns:
            list[int]: Byte list seed for name generation

        Example:
            >>> RegionNameGenerator.create_region_seed(100, 50, -100, 0)
            [0, 0, 50, 0, 156, 255, 100, 0]  # Example byte list
        """
        def fmt(val, trunc):
            """Format integer as hex, truncate to specified digits."""
            return f"{val & 0xFFFF:04X}"[-trunc:]

        # Build 10-character hex string: GGY YZZ ZXX (2-2-3-3 format)
        hex_str = fmt(galaxy_idx, 2) + fmt(y, 2) + fmt(z, 3) + fmt(x, 3)
        return ByteUtils.parse(hex_str)

    @staticmethod
    def format_name(seed, alphasets, letter_map):
        """
        Generate full region name from seed.

        This is the main method that:
        1. Processes seed through several mixing operations
        2. Sets up random number generator state
        3. Generates base name using Generator class
        4. Possibly adds decorative adornment (50% chance)

        Args:
            seed (list[int]): Byte list seed from create_region_seed
            alphasets (list): Alphabet sets for name generation
            letter_map (dict): Character probability weights

        Returns:
            str: Generated region name, or "Unknown Region" on error

        Raises:
            None: Catches all exceptions and returns default name

        Example:
            >>> seed = [0, 0, 50, 0, 156, 255, 100, 0]
            >>> RegionNameGenerator.format_name(seed, alphasets, letter_map)
            "Sea of Abcdef"  # Example region name with adornment
        """
        # Initialize random number generator caches
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # Step 1: Process seed through mixing operations (game's algorithm)
        reg0 = ByteUtils.shr(seed, 4)
        if reg0:
            reg0[0] //= 2

        xor_res = ByteUtils.xor(reg0, seed)
        reg0 = ByteUtils.multiply(xor_res,
            [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64])[:8]

        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(reg0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), reg0)
        reg0 = ByteUtils.multiply(xor2,
            [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3])[:8]

        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(reg0, 4)) // 2
        reg0 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), reg0)

        # Step 2: Set up random number generator state
        shl4 = ByteUtils.shl(reg0, 4)
        xor_mid = ByteUtils.xor(ByteUtils.rol(shl4, 2), ByteUtils.shr(reg0, 4))
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure cache0[0] is not zero (would break generator)
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])

        ByteUtils.update_seed(cache0)
        cache1[2] = ByteUtils.add(
            ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4), [0x06])

        # Step 3: Generate base name
        name = Generator.generate_name(cache0, cache1, alphasets, letter_map)
        if not name or "[" in name:
            return "Unknown Region"

        # Capitalize first letter
        name = name[0].upper() + name[1:]

        # Step 4: 50% chance to add decorative adornment
        ByteUtils.update_seed(cache0)
        if ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x64]), 4)[0] < 0x50:
            ByteUtils.update_seed(cache0)
            idx = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x14]), 4)[0]
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                name = RegionNameGenerator.PROC_ADORNMENTS[idx].replace(
                    "%NAME%", name)

        return name


class NMSGalaxyMap:
    """
    Handles conversions between portal glyphs and galactic coordinates.

    Portal glyphs are 12-character hex codes that encode location in the
    No Man's Sky universe. This class decodes them into x, y, z coordinates
    and galaxy index.

    Methods:
        glyphs_to_coords: Converts portal glyph string to coordinate dictionary
    """

    @staticmethod
    def glyphs_to_coords(glyphs: str):
        """
        Convert portal glyph string to galactic coordinates.

        Portal glyph format: 12 hex characters encoding:
        - Characters 1-4: Galaxy index and Y coordinate
        - Characters 5-9: Z coordinate
        - Characters 10-12: X coordinate

        The coordinates use a signed representation centered at 0x800/0x80.

        Args:
            glyphs (str): 12-character hex string (portal address)

        Returns:
            dict or None: Dictionary with keys 'x','y','z','s' (galaxy index),
                         or None if glyphs are invalid

        Raises:
            None: Returns None on any parsing error

        Example:
            >>> NMSGalaxyMap.glyphs_to_coords("1205D058AC1D")
            {'x': 1234, 'y': 56, 'z': -789, 's': 291}
        """
        try:
            # Parse hex segments from glyph string
            y_hex = int(glyphs[4:6], 16)   # Characters 5-6: Y coordinate
            z_hex = int(glyphs[6:9], 16)   # Characters 7-9: Z coordinate
            x_hex = int(glyphs[9:12], 16)  # Characters 10-12: X coordinate
            s_hex = int(glyphs[1:4], 16)   # Characters 2-4: Galaxy index
        except ValueError:
            return None  # Invalid hex characters

        def convert(val, is_y=False):
            """
            Convert hex value to signed coordinate.

            Coordinates are stored as offset from center:
            - X and Z: Center at 0x800 (2048), range ±2047
            - Y: Center at 0x80 (128), range ±127
            """
            limit = 0x80 if is_y else 0x800
            shift_pos = 0x81 if is_y else 0x801  # For values above center
            shift_neg = 0x7F if is_y else 0x7FF  # For values below center

            if val > limit:
                return val - shift_pos   # Positive offset from center
            else:
                return val + shift_neg   # Negative offset from center

        return {
            'x': convert(x_hex),          # X coordinate
            'y': convert(y_hex, is_y=True),  # Y coordinate (special range)
            'z': convert(z_hex),          # Z coordinate
            's': s_hex                    # Galaxy index
        }


class NMSData:
    """
    Loads and manages game data from external JSON files.

    This class downloads ship data, galaxy lists, and procedural generation
    data from GitHub repositories. It provides properties for easy access
    to sorted and organized game data.

    Attributes:
        URL_BASE (str): Base URL for data files on GitHub

    Properties:
        GALAXY_OPTIONS: List of galaxy names
        GALAXY_INDEX_MAP: Dictionary mapping galaxy names to indices
        LETTER_MAP: Character probability weights for name generation
        ALPHASETS: Alphabet sets for name generation
        SHIP_TYPES/COLORS/ECONOMY_LIST/etc.: Various ship-related data lists
    """

    # Base URL for data files in GitHub repository
    URL_BASE = "https://raw.githubusercontent.com/2A03-Jikuu/nms-wiki-tool-py/refs/heads/main/datalist"

    def __init__(self):
        """
        Initialize NMSData and load all game data.

        Loads ship data, galaxy information, letter maps, and alphabet sets
        from external JSON files. Initializes empty structures if loading fails.
        """
        self._ship_data = {}          # All ship-related data
        self._galaxy_list = []        # List of galaxy names
        self._galaxy_map = {}         # Map galaxy name -> index
        self._letter_map = {}         # Character probability weights
        self._alphasets = []          # Alphabet sets for name generation

        self._load_data()  # Load all data on initialization

    def _load_data(self):
        """
        Load all game data files from remote URLs.

        Attempts to load 4 JSON files:
        1. ship_data.json: Ship types, colors, parts, etc.
        2. galaxies.json: Galaxy names and indices
        3. letter_map.json: Character probability weights
        4. alphasets.json: Alphabet sets for name generation

        Prints warnings if any file fails to load.
        """
        # Load ship data (types, colors, parts, etc.)
        try:
            resp = requests.get(f"{self.URL_BASE}/ship_data.json")
            resp.raise_for_status()  # Raise exception for HTTP errors
            self._ship_data = resp.json()
            self._sort_ship_data()  # Sort lists for better UI display
        except requests.RequestException:
            self._ship_data = {}
            print("⚠️ Failed to load ship_data.json")

        # Load galaxy list and index mapping
        try:
            resp = requests.get(f"{self.URL_BASE}/galaxies.json")
            resp.raise_for_status()
            g_data = resp.json()
            # Sort galaxies by index for consistent display
            s_gal = sorted(g_data, key=lambda x: x.get('index', 0))
            self._galaxy_list = [g['name'] for g in s_gal if 'name' in g]
            self._galaxy_map = {g['name']: g['index']
                               for g in s_gal if 'name' in g and 'index' in g}
        except requests.RequestException:
            self._galaxy_list = []
            self._galaxy_map = {}
            print("⚠️ Failed to load galaxies.json")

        # Load letter map for name generation
        try:
            resp = requests.get(f"{self.URL_BASE}/letter_map.json")
            resp.raise_for_status()
            self._letter_map = resp.json()
        except requests.RequestException:
            self._letter_map = {}
            print("⚠️ Failed to load letter_map.json")

        # Load alphabet sets for name generation
        try:
            resp = requests.get(f"{self.URL_BASE}/alphasets.json")
            resp.raise_for_status()
            self._alphasets = resp.json()
        except requests.RequestException:
            self._alphasets = []
            print("⚠️ Failed to load alphasets.json")

    def _sort_ship_data(self):
        """
        Sort lists in ship data for better UI display.

        Sorts all list-type values in ship_data dictionary alphabetically.
        Also sorts lists inside nested dictionaries for consistent ordering.
        """
        if not self._ship_data:
            return

        # Sort top-level lists
        for field_name in ['ship_types', 'ship_colors', 'upgrade_modules',
                          'living_ship_upgrades', 'location_options', 'economy_list']:
            if field_name in self._ship_data and isinstance(self._ship_data[field_name], list):
                self._ship_data[field_name].sort()

        # Sort lists inside nested dictionaries
        for field_name in ['ship_subtypes', 'wings', 'thrusters',
                          'hull_accessories', 'other_accessories']:
            if field_name in self._ship_data and isinstance(self._ship_data[field_name], dict):
                for k in self._ship_data[field_name]:
                    if isinstance(self._ship_data[field_name][k], list):
                        self._ship_data[field_name][k].sort()

    # Property getters for easy access to sorted data
    @property
    def GALAXY_OPTIONS(self):
        """List of galaxy names sorted by index."""
        return self._galaxy_list

    @property
    def GALAXY_INDEX_MAP(self):
        """Dictionary mapping galaxy name to index."""
        return self._galaxy_map

    @property
    def LETTER_MAP(self):
        """Character probability weights for name generation."""
        return self._letter_map

    @property
    def ALPHASETS(self):
        """Alphabet sets for name generation."""
        return self._alphasets

    @property
    def SHIP_TYPES(self):
        """List of ship type names (Fighter, Explorer, etc.)."""
        return self._ship_data.get('ship_types', [])

    @property
    def SHIP_COLORS(self):
        """List of possible ship color names."""
        return self._ship_data.get('ship_colors', [])

    @property
    def ECONOMY_LIST(self):
        """List of economy types (★, ★★, ★★★, 💀)."""
        return self._ship_data.get('economy_list', [])

    @property
    def LOCATION_OPTIONS(self):
        """List of location types (Space Station, Trading Post, etc.)."""
        return self._ship_data.get('location_options', [])

    @property
    def UPGRADE_MODULES(self):
        """List of upgrade module names."""
        return self._ship_data.get('upgrade_modules', [])

    @property
    def LIVING_SHIP_UPGRADES(self):
        """List of living ship specific upgrades."""
        return self._ship_data.get('living_ship_upgrades', [])

    @property
    def SHIP_SUBTYPES(self):
        """Dictionary mapping ship types to their subtypes."""
        return self._ship_data.get('ship_subtypes', {})

    @property
    def WINGS(self):
        """Dictionary mapping ship types to available wing options."""
        return self._ship_data.get('wings', {})

    @property
    def THRUSTERS(self):
        """Dictionary mapping ship types to available thruster options."""
        return self._ship_data.get('thrusters', {})

    @property
    def HULL_ACCESSORIES(self):
        """Dictionary mapping ship types to available hull accessories."""
        return self._ship_data.get('hull_accessories', {})

    @property
    def OTHER_ACCESSORIES(self):
        """Dictionary mapping ship types to available other accessories."""
        return self._ship_data.get('other_accessories', {})

    @property
    def EXAMPLE_DATA(self):
        """Dictionary of example ship data for each ship type."""
        return self._ship_data.get('example_data', {})


@dataclass
class AppWidgets:
    """
    Data class holding all UI widget references.

    This class organizes all the Jupyter widgets used in the form
    into logical groups. Using a dataclass makes it easy to access
    widgets by name and keeps the UI setup code organized.

    Fields are organized by section:
    - Location & Coordinates widgets
    - Ship Identity widgets
    - Discovery Info widgets
    - Colors & Parts widgets
    - Stats & Media widgets
    - Action buttons and output widgets
    """
    # Location & Coordinates section widgets
    galaxy: Combobox = field(default_factory=Combobox)
    portalglyphs: Text = field(default_factory=Text)
    region: Text = field(default_factory=Text)
    coordinates: Text = field(default_factory=Text)
    system: Text = field(default_factory=Text)
    economy: Dropdown = field(default_factory=Dropdown)
    planet: Text = field(default_factory=Text)
    moon: Text = field(default_factory=Text)
    location: Dropdown = field(default_factory=Dropdown)
    axes_text: Text = field(default_factory=Text)

    # Ship Identity section widgets
    name: Text = field(default_factory=Text)
    class_: Dropdown = field(default_factory=Dropdown)
    type: Dropdown = field(default_factory=Dropdown)
    subtype: Dropdown = field(default_factory=Dropdown)
    techslots: IntText = field(default_factory=IntText)
    cargoslots: IntText = field(default_factory=IntText)
    inventory: Text = field(default_factory=Text)
    cost: Text = field(default_factory=Text)

    # Discovery Info section widgets
    discovered_by_alias: Text = field(default_factory=Text)
    discovered_by_link: Text = field(default_factory=Text)
    discovery_date: DatePicker = field(default_factory=DatePicker)
    agt_stardate: Text = field(default_factory=Text)
    platform: Dropdown = field(default_factory=Dropdown)
    mode: Dropdown = field(default_factory=Dropdown)
    civilized: Text = field(default_factory=Text)
    release: Text = field(default_factory=Text)

    # Colors & Parts section widgets
    primarycolor: Dropdown = field(default_factory=Dropdown)
    secondarycolor: Dropdown = field(default_factory=Dropdown)
    accent: Dropdown = field(default_factory=Dropdown)
    wings: Dropdown = field(default_factory=Dropdown)
    engines: Dropdown = field(default_factory=Dropdown)
    hullacc: Dropdown = field(default_factory=Dropdown)
    otheracc: Dropdown = field(default_factory=Dropdown)

    # Stats & Media section widgets
    damageB: FloatText = field(default_factory=FloatText)
    shieldB: FloatText = field(default_factory=FloatText)
    warpB: FloatText = field(default_factory=FloatText)
    maneuverB: FloatText = field(default_factory=FloatText)
    upgrade1: Dropdown = field(default_factory=Dropdown)
    upgrade2: Dropdown = field(default_factory=Dropdown)
    upgrade3: Dropdown = field(default_factory=Dropdown)
    upgrade4: Dropdown = field(default_factory=Dropdown)
    image: Text = field(default_factory=Text)
    gallery_images: Textarea = field(default_factory=Textarea)

    # Action buttons and output widgets
    filename: Text = field(default_factory=Text)
    btn_preview: Button = field(default_factory=Button)
    btn_gen: Button = field(default_factory=Button)
    btn_copy: Button = field(default_factory=Button)
    btn_download: Button = field(default_factory=Button)
    btn_clear: Button = field(default_factory=Button)
    status_label: HTML = field(default_factory=HTML)
    output: Output = field(default_factory=Output)


class ShipModel(BaseModel):
    """
    Pydantic model for validating ship form data.

    This model defines all fields that can be entered in the form,
    their data types, validation rules, and cleaning logic. Using
    Pydantic ensures all data is properly validated before generating
    wiki code.

    Attributes:
        model_config: Pydantic configuration allowing custom types
        name through gallery_images: All form fields with validation

    Validators:
        clean_inputs: Cleans string inputs and converts empty to None
        convert_date_to_arrow: Converts date strings to arrow objects
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    # Ship Identity fields
    name: str = Field(..., min_length=1)  # Required, non-empty
    class_: Optional[str] = None
    type: Optional[str] = None
    subtype: Optional[str] = None
    techslots: Optional[int] = Field(default=0, ge=0)  # Must be >= 0
    cargoslots: Optional[int] = Field(default=0, ge=0)  # Must be >= 0
    inventory: Optional[str] = None
    cost: Optional[str] = None

    # Discovery Info fields
    discovered_by_alias: Optional[str] = None
    discovered_by_link: Optional[str] = None
    discovery_date: Optional[arrow.Arrow] = None
    platform: Optional[str] = None
    mode: Optional[str] = None
    civilized: Optional[str] = None
    release: Optional[str] = None

    # Location fields
    galaxy: Optional[str] = None
    region: Optional[str] = None
    system: Optional[str] = None
    economy: Optional[str] = None
    planet: Optional[str] = None
    moon: Optional[str] = None
    location: Optional[str] = None
    axes_text: Optional[str] = None
    portalglyphs: Optional[str] = Field(
        default=None, pattern=r"^[0-9A-F]{12}$")  # Must be 12 hex chars

    # Colors & Parts fields
    primarycolor: Optional[str] = None
    secondarycolor: Optional[str] = None
    accent: Optional[str] = None
    wings: Optional[str] = None
    engines: Optional[str] = None
    hullacc: Optional[str] = None
    otheracc: Optional[str] = None

    # Stats & Media fields
    damageB: Optional[float] = Field(default=0.0, ge=0)  # Must be >= 0
    shieldB: Optional[float] = Field(default=0.0, ge=0)  # Must be >= 0
    warpB: Optional[float] = Field(default=0.0, ge=0)  # Must be >= 0
    maneuverB: Optional[float] = Field(default=0.0, ge=0)  # Must be >= 0
    upgrade1: Optional[str] = None
    upgrade2: Optional[str] = None
    upgrade3: Optional[str] = None
    upgrade4: Optional[str] = None
    image: Optional[str] = None
    gallery_images: Optional[str] = None

    @field_validator('*', mode='before')
    @classmethod
    def clean_inputs(cls, v):
        """
        Clean all input values before validation.

        For strings: strip whitespace, convert empty strings to None
        For numbers: convert values <= 0 to None
        For dropdowns: convert placeholder values to None

        Args:
            v: Input value of any type

        Returns:
            Cleaned value or None
        """
        if isinstance(v, str):
            v = v.strip()
            if v == "" or v.startswith("- Select"):
                return None  # Empty or placeholder becomes None
        if isinstance(v, (int, float)) and v <= 0:
            return None  # Zero or negative numbers become None
        return v

    @field_validator('discovery_date', mode='before')
    @classmethod
    def convert_date_to_arrow(cls, v):
        """
        Convert discovery date to arrow object for consistent handling.

        Args:
            v: Date string, date object, or arrow object

        Returns:
            arrow.Arrow object or None
        """
        return arrow.get(v) if v else None


class NMSWikiStarshipFormCreator:
    """
    Main application class for the starship wiki form generator.

    This class creates the complete Jupyter notebook interface with:
    - Tabbed form sections for different data categories
    - Real-time coordinate calculation from portal glyphs
    - Procedural region name generation
    - Form validation with Pydantic
    - Wiki template rendering with Jinja2
    - Preview, copy, and download functionality

    Class Attributes:
        WIKI_TEMPLATE: Jinja2 template for wiki page code
        DEFAULT_*: Default values for various fields
        ECONOMY_SPAWN_CHANCE_MAP: Maps economy levels to spawn chances
        INVENTORY_SIZE_THRESHOLDS: Defines ship size based on slot counts
    """

    # Jinja2 template for generating wiki page code
    # Uses double curly braces that won't conflict with wiki template syntax
    WIKI_TEMPLATE = jinja2.Template("""{{ '{{' }}PAGEStarship
| name = {{ name | default('') }}
| civstub = {{ '{{' }}AGT Notice}}
| image = {{ image | default('') }}
| galaxy = {{ galaxy | default('') }}
| region = {{ region | default('') }}
| system = {{ system | default('') }}
| planet = {{ planet | default('') }}
| moon = {{ moon | default('') }}
| location = {{ location | default('') }}
| axes = {{ axes_text | default('') }}
| coordinates = {{ coordinates | default('') }}
| portalglyphs = {{ portalglyphs | default('') }}
| economy = {{ economy | default('') }}
| type = {{ type | default('') }}
| subtype = {{ subtype | default('') }}
| exotic = {{ exotic | default('') }}
| class = {{ class_ | default('') }}
| inventory = {{ inventory | default('') }}
| slots = {{ cargoslots | default('') }}
| techslots = {{ techslots | default('') }}
| cost = {{ cost | default('') }}
| civilized = {{ civilized | default('') }}
| discovered = {{ discovered_by_alias | default('') }}
| discoveredlink = {{ discovered_by_link | default('') }}
| mode = {{ mode | default('') }}
| platform = {{ platform | default('') }}
| release = {{ release | default('') }}
| damageB = {{ damageB | default('') }}
| shieldB = {{ shieldB | default('') }}
| warpB = {{ warpB | default('') }}
| maneuverB = {{ maneuverB | default('') }}
| researchteam = AGT Bureau of Starship Registration
| discoverydate = {{ discovery_date | default('') }}
| AGTstardate = {{ agt_stardate | default('') }}
| primarycolor = {{ primarycolor | default('') }}
| secondarycolor = {{ secondarycolor | default('') }}
| accent = {{ accent | default('') }}
| schance = {{ schance | default('') }}
| wings = {{ wings | default('') }}
| engines = {{ engines | default('') }}
| hullacc = {{ hullacc | default('') }}
| otheracc = {{ otheracc | default('') }}
| upgrade1 = {{ upgrade1 | default('') }}
| upgrade2 = {{ upgrade2 | default('') }}
| upgrade3 = {{ upgrade3 | default('') }}
| upgrade4 = {{ upgrade4 | default('') }}
{{ '}}' }}
==Gallery==
<gallery>
{{ gallery_images | default('') }}
</gallery>

==AGT Galactic Archives==
{{ '{{' }}AGT Galactic Archive Sync}}""")

    # Default values and placeholders
    DEFAULT_PLACEHOLDER = "- Select -"
    DEFAULT_CIVILIZATION = "Alliance of Galactic Travellers"
    DEFAULT_RELEASE = "Breach"

    # Map economy levels (★ symbols) to spawn chance percentages
    ECONOMY_SPAWN_CHANCE_MAP = {'★':'0%','★★':'1%','★★★':'2%','💀':'5%'}

    # Define ship size categories based on cargo and tech slot counts
    # Format: (min_cargo_slots, min_tech_slots, size_label)
    INVENTORY_SIZE_THRESHOLDS = {
        'Fighter': [(30,19,'Large'),(24,19,'Medium'),(24,14,'Small')],
        'Explorer': [(30,24,'Large'),(24,19,'Medium'),(24,14,'Small')],
        'Hauler': [(30,12,'Small')],
        'Shuttle': [(24,12,'Small')],
        'Solar': [(24,13,'Small')],
        'Exotic': [(24,20,'Small')],
        'Interceptor': [(32,22,'Large')]
    }

    def __init__(self):
        """
        Initialize the form creator application.

        Sets up data loading, defines UI styles, creates all widgets,
        connects event handlers, and displays the interface.
        """
        # Load game data
        self.data = NMSData()
        # Initialize coordinate converter
        self.map_logic = NMSGalaxyMap()
        # Initialize widget container (populated in _setup_ui)
        self.widgets = AppWidgets()
        # Store generated wiki content
        self.generated_content = ""

        # Setup UI components
        self._define_styles()      # Define CSS-like styles
        self._setup_ui()           # Create all widgets and layout
        self._connect_events()     # Connect widget event handlers

        # Initialize UI state
        self._update_stardate_ui(None)  # Set initial stardate
        self.on_ship_type_change(None)  # Initialize ship type dropdowns
        self.on_location_change(None)   # Initialize location field state

    def _define_styles(self):
        """Define CSS-like styles for UI elements."""
        # Header style (section titles)
        self.H_STYLE = "font-weight:bold; font-size:16px; margin-top:20px; " \
                      "border-bottom:2px solid #2196F3; padding-bottom:5px; " \
                      "color:#0D47A1;"
        # Description style (section explanations)
        self.D_STYLE = "font-style:italic; font-size:12px; color:#333; " \
                      "margin-bottom:12px; line-height:1.4em; " \
                      "background-color:#E3F2FD; padding:8px; " \
                      "border-left:4px solid #42A5F5; border-radius:4px;"
        # Label style for form fields
        self.L_STYLE = {'description_width': '140px'}
        # Layouts for different widget types
        self.W_LAYOUT = Layout(width='98%')           # Standard width
        self.TXT_LAYOUT = Layout(width='98%', height='180px')  # Textarea
        self.COL_LAYOUT = Layout(width='50%')         # Half-width column
        self.FULL_ROW = Layout(width='100%', margin='5px 0')   # Full width row

    def _setup_ui(self):
        """
        Create all UI widgets and arrange them in tabbed layout.

        This method:
        1. Creates individual widgets for each form field
        2. Groups widgets into logical sections
        3. Creates tabs for different categories
        4. Displays the complete interface
        """
        # Create Location & Coordinates widgets
        w_galaxy_cb = self._make(Combobox, 'Galaxy',
                                 options=self.data.GALAXY_OPTIONS,
                                 placeholder="Select Galaxy (e.g. Euclid)")
        w_glyphs_txt = self._make(Text, 'Portal Glyphs',
                                 placeholder='e.g., 1205D058AC1D')
        w_reg = self._make(Text, 'Region',
                          placeholder="Auto-calculated from Glyphs",
                          disabled=True)
        w_coo = self._make(Text, 'Coordinates', disabled=True,
                          placeholder="Auto-calculated")
        w_sys = self._make(Text, 'Star System',
                          placeholder="Enter System Name")
        w_eco = self._make(Dropdown, 'Economy',
                          options=self.data.ECONOMY_LIST,
                          placeholder='Select Economy Level')
        w_pla = self._make(Text, 'Planet',
                          placeholder='Enter Planet Name')
        w_moo = self._make(Text, 'Moon',
                          placeholder='Enter Moon Name (Optional)')
        w_loc = self._make(Dropdown, 'Specific Location',
                          options=self.data.LOCATION_OPTIONS,
                          placeholder='Select Location Type')
        w_axs = self._make(Text, 'Planetary Axes',
                          placeholder='e.g. +40.23, -112.90')

        # Create Ship Identity widgets
        w_nam = self._make(Text, 'Ship Name',
                          placeholder='e.g. The Radiant Pillar')
        w_cla = self._make(Dropdown, 'Class',
                          options=['S', 'A', 'B', 'C'],
                          placeholder='Select Ship Class')
        w_typ = self._make(Dropdown, 'Type',
                          options=self.data.SHIP_TYPES,
                          placeholder='Select Ship Type')
        w_sub = self._make(Dropdown, 'Subtype',
                          placeholder='(Available after Type selection)')
        w_tsc = self._make(IntText, 'Tech Slots')
        w_csc = self._make(IntText, 'Cargo Slots')
        w_inv = self._make(Text, 'Inventory Size', disabled=True,
                          placeholder='Auto-calculated')
        w_cos = self._make(Text, 'Cost',
                          placeholder='e.g. 15,000,000 Units')

        # Create Discovery Info widgets
        w_dis = self._make(Text, 'Discoverer Alias',
                          placeholder="Enter In-Game Name")
        w_lnk = self._make(Text, 'Wiki User Link',
                          placeholder="e.g. User:YourName")
        w_dat = self._make(DatePicker, 'Discovery Date',
                          value=arrow.now().date())
        w_sta = self._make(Text, 'AGT Stardate', disabled=True)
        w_pla_p = self._make(Dropdown, 'Platform',
                            options=['PC', 'PS4', 'PS5', 'Xbox', 'Switch', 'Mac'],
                            placeholder='Select Platform')
        w_mod = self._make(Dropdown, 'Game Mode',
                          options=['Normal', 'Survival', 'Permadeath', 'Creative', 'Relaxed'],
                          placeholder='Select Mode')
        w_civ = self._make(Text, 'Civilization',
                          value=self.DEFAULT_CIVILIZATION,
                          placeholder="Enter Civ Name")
        w_rel = self._make(Text, 'Release Version',
                          value=self.DEFAULT_RELEASE,
                          placeholder="e.g. Worlds Part I")

        # Create Colors & Parts widgets
        w_c1 = self._make(Dropdown, 'Primary Color',
                         options=self.data.SHIP_COLORS,
                         placeholder='Select Primary')
        w_c2 = self._make(Dropdown, 'Secondary Color',
                         options=self.data.SHIP_COLORS,
                         placeholder='Select Secondary')
        w_acc = self._make(Dropdown, 'Accent Color',
                          options=self.data.SHIP_COLORS,
                          placeholder='Select Accent')
        w_win = self._make(Dropdown, 'Wings',
                          placeholder='Select Ship Type First')
        w_eng = self._make(Dropdown, 'Thrusters',
                          placeholder='Select Ship Type First')
        w_hul = self._make(Dropdown, 'Hull Acc',
                          placeholder='(Shuttle Only)')
        w_oth = self._make(Dropdown, 'Other Acc',
                          placeholder='(Shuttle Only)')

        # Create Stats & Media widgets
        w_dmg = self._make(FloatText, 'Damage Pot.')
        w_shd = self._make(FloatText, 'Shield Str.')
        w_wrp = self._make(FloatText, 'Hyperdrive')
        w_man = self._make(FloatText, 'Maneuver.')
        w_u1 = self._make(Dropdown, 'Upgrade 1',
                         placeholder='Select Module')
        w_u2 = self._make(Dropdown, 'Upgrade 2',
                         placeholder='Select Module')
        w_u3 = self._make(Dropdown, 'Upgrade 3',
                         placeholder='Select Module')
        w_u4 = self._make(Dropdown, 'Upgrade 4',
                         placeholder='Select Module')
        w_img = self._make(Text, 'Main Image',
                          placeholder='e.g. File:ShipName.png')
        w_gallery_txt = self._make(Textarea, 'Gallery',
                                  placeholder='Format: File:Image.jpg|Description (One per line)')

        # Create action buttons and output widgets
        w_fil = self._make(Text, 'Filename',
                          placeholder='Auto-generated from Ship Name')
        w_bp = Button(description='Preview', button_style='info', icon='eye',
                     layout=Layout(flex='1 1 auto', margin='0 5px'))
        w_bg = Button(description='Generate', button_style='success', icon='code',
                     layout=Layout(flex='1 1 auto', margin='0 5px'))
        w_bc = Button(description='Copy', button_style='primary', icon='clipboard',
                     disabled=True, layout=Layout(flex='1 1 auto', margin='0 5px'))
        w_bd = Button(description='Download', button_style='primary', icon='download',
                     disabled=True, layout=Layout(flex='1 1 auto', margin='0 5px'))
        w_br = Button(description='Reset', button_style='danger', icon='trash',
                     layout=Layout(flex='1 1 auto', margin='0 5px'))
        w_stt = HTML(value="<i>Ready.</i>")
        w_out = Output(layout={'border': '1px solid #ccc', 'height': '400px',
                              'overflow_y': 'scroll', 'padding': '10px'})

        # Store all widgets in AppWidgets dataclass
        self.widgets = AppWidgets(
            galaxy=w_galaxy_cb, portalglyphs=w_glyphs_txt, region=w_reg,
            coordinates=w_coo, system=w_sys, economy=w_eco, planet=w_pla,
            moon=w_moo, location=w_loc, axes_text=w_axs,
            name=w_nam, class_=w_cla, type=w_typ, subtype=w_sub,
            techslots=w_tsc, cargoslots=w_csc, inventory=w_inv, cost=w_cos,
            discovered_by_alias=w_dis, discovered_by_link=w_lnk,
            discovery_date=w_dat, agt_stardate=w_sta, platform=w_pla_p,
            mode=w_mod, civilized=w_civ, release=w_rel,
            primarycolor=w_c1, secondarycolor=w_c2, accent=w_acc,
            wings=w_win, engines=w_eng, hullacc=w_hul, otheracc=w_oth,
            damageB=w_dmg, shieldB=w_shd, warpB=w_wrp, maneuverB=w_man,
            upgrade1=w_u1, upgrade2=w_u2, upgrade3=w_u3, upgrade4=w_u4,
            image=w_img, gallery_images=w_gallery_txt,
            filename=w_fil, btn_preview=w_bp, btn_gen=w_bg, btn_copy=w_bc,
            btn_download=w_bd, btn_clear=w_br, status_label=w_stt, output=w_out
        )

        # Create tab 1: Location & Coordinates
        t_loc = [
            self._h('Location & Coordinates'),
            self._d("Enter the Galaxy Name and Portal Glyphs first. The Region and Galactic Coordinates will be calculated automatically."),
            self._row(self.widgets.galaxy, self.widgets.portalglyphs),
            self._row(self.widgets.region, self.widgets.coordinates),
            self._row(self.widgets.system, self.widgets.economy),
            self._h('Planet Details'),
            self._d("Specify the planetary body and precise location of the find."),
            self._row(self.widgets.planet, self.widgets.moon),
            self._row(self.widgets.location, self.widgets.axes_text)
        ]

        # Create tab 2: Ship Identity
        t_shp = [
            self._h('Ship Identity'),
            self._d("Enter the core identity details of the starship."),
            self._row(self.widgets.name, self.widgets.class_),
            self._row(self.widgets.type, self.widgets.subtype),
            self._h('Inventory & Cost'),
            self._d("Enter the slot counts to auto-calculate size. Enter the unit cost."),
            self._row(self.widgets.techslots, self.widgets.cargoslots),
            self._row(self.widgets.inventory, self.widgets.cost)
        ]

        # Create tab 3: Discovery Info
        t_dsc = [
            self._h('Discovery'),
            self._d("Credit the discoverer and date. Stardate is auto-calculated."),
            self._row(self.widgets.discovered_by_alias, self.widgets.discovered_by_link),
            self._row(self.widgets.discovery_date, self.widgets.agt_stardate),
            self._row(self.widgets.platform, self.widgets.mode),
            self._row(self.widgets.civilized, self.widgets.release)
        ]

        # Create tab 4: Colors & Parts
        t_cfg = [
            self._h('Colors'),
            self._d("Select the color palette used by the ship."),
            self._row(self.widgets.primarycolor, self.widgets.secondarycolor),
            self._row(self.widgets.accent),
            self._h('Parts'),
            self._d("Select the procedural parts. Dropdowns update based on 'Ship Type'."),
            self._row(self.widgets.wings, self.widgets.engines),
            self._row(self.widgets.hullacc, self.widgets.otheracc)
        ]

        # Create tab 5: Stats & Media
        t_sts = [
            self._h('Stats'),
            self._d("Enter the base statistics found in the scanner view."),
            self._row(self.widgets.damageB, self.widgets.shieldB),
            self._row(self.widgets.warpB, self.widgets.maneuverB),
            self._h('Upgrades'),
            self._d("List any notable pre-installed technology modules."),
            self._row(self.widgets.upgrade1, self.widgets.upgrade2),
            self._row(self.widgets.upgrade3, self.widgets.upgrade4),
            self._h('Media'),
            self._d("Enter the main infobox filename and a list of gallery images."),
            self._row(self.widgets.image),
            self.widgets.gallery_images
        ]

        # Create tab 6: Generate & Tools
        # Create example load buttons for each ship type
        ex_btns = [Button(description=f'Load {ship_type}', button_style='warning',
                         layout=Layout(width='auto', margin='2px'))
                  for ship_type in self.data.SHIP_TYPES]

        # Connect each button to load example data
        for btn in ex_btns:
            btn.on_click(partial(self.on_load_example_click,
                                ship_type=btn.description.replace('Load ', '')))

        t_gen = [
            self._h('Tools'),
            self._d("Pre-load example data or clear the form."),
            widgets.GridBox(ex_btns,
                           layout=widgets.Layout(grid_template_columns="repeat(4, 1fr)",
                                                grid_gap='5px')),
            self._h('Output'),
            self._d("Generate the code, then Copy to Clipboard or Download the file."),
            HBox([self.widgets.btn_preview, self.widgets.btn_gen, self.widgets.btn_copy,
                  self.widgets.btn_download, self.widgets.btn_clear],
                 layout=Layout(justify_content='space-between', margin='15px 0')),
            self.widgets.status_label,
            self._row(self.widgets.filename),
            self.widgets.output
        ]

        # Create tabbed interface with padding
        pad = Layout(padding='20px')
        self.tabs = Tab(children=[VBox(i, layout=pad) for i in
                                 [t_loc, t_shp, t_dsc, t_cfg, t_sts, t_gen]])

        # Set tab titles
        for i, title in enumerate(['Location', 'Ship Info', 'Discovery',
                                   'Parts/Color', 'Stats/Media', 'Generate']):
            self.tabs.set_title(i, title)

        # Display the complete interface
        display(self.tabs)

    def _make(self, widget_cls, desc, **kwargs):
        """
        Helper method to create standardized widgets.

        Applies consistent styling, layout, and placeholder handling
        for all widget types.

        Args:
            widget_cls: Widget class (Text, Dropdown, etc.)
            desc (str): Description label for the widget
            **kwargs: Additional widget-specific parameters

        Returns:
            Widget instance with applied styling
        """
        # Start with basic layout
        layout = self.TXT_LAYOUT if widget_cls == Textarea else self.W_LAYOUT
        params = {'description': desc, 'style': self.L_STYLE, 'layout': layout}

        # Handle placeholder specially for different widget types
        if 'placeholder' in kwargs:
            params['placeholder'] = kwargs.pop('placeholder')

        # Special handling for Dropdown widgets
        if 'options' in kwargs:
            opts = kwargs.pop('options')
            if widget_cls == Dropdown:
                # Add placeholder as first option
                params['options'] = [f"- {params.get('placeholder', 'Select')} -"] + opts
                params['value'] = params['options'][0]  # Select placeholder by default
            else:
                params['options'] = opts

        # Set default values for numeric inputs
        if widget_cls in (IntText, FloatText):
            params['value'] = 0

        # Apply any additional parameters
        params.update(kwargs)

        # Clean up parameters for specific widget types
        if widget_cls == Dropdown and 'placeholder' in params:
            del params['placeholder']  # Dropdown uses first option as placeholder

        if widget_cls in (IntText, FloatText, DatePicker) and 'placeholder' in params:
            del params['placeholder']  # Numeric widgets don't use placeholders

        if widget_cls == Combobox and 'placeholder' not in params:
            params['placeholder'] = ''  # Ensure Combobox has placeholder

        return widget_cls(**params)

    def _h(self, title):
        """Create HTML header element with predefined style."""
        return HTML(f"<div style='{self.H_STYLE}'>{title}</div>")

    def _d(self, description):
        """Create HTML description element with predefined style."""
        return HTML(f"<div style='{self.D_STYLE}'>{description}</div>")

    def _row(self, widget_left, widget_right=None):
        """
        Create a horizontal row with one or two widgets.

        Args:
            widget_left: Left widget (or only widget)
            widget_right: Optional right widget

        Returns:
            HBox containing the widget(s) in two columns
        """
        return HBox([VBox([widget_left], layout=self.COL_LAYOUT),
                    VBox([widget_right] if widget_right else [],
                         layout=self.COL_LAYOUT)],
                   layout=self.FULL_ROW)

    def _connect_events(self):
        """
        Connect all widget event handlers.

        Sets up observers for value changes and click handlers
        for buttons to make the form interactive.
        """
        # Connect value change handlers
        self.widgets.type.observe(self.on_ship_type_change, names='value')
        self.widgets.location.observe(self.on_location_change, names='value')
        self.widgets.portalglyphs.observe(self.on_glyphs_change, names='value')
        self.widgets.galaxy.observe(self.on_glyphs_change, names='value')
        self.widgets.discovery_date.observe(self._update_stardate_ui, names='value')
        self.widgets.cargoslots.observe(self.on_stats_change, names='value')
        self.widgets.techslots.observe(self.on_stats_change, names='value')

        # Connect button click handlers
        self.widgets.btn_preview.on_click(lambda _: self._generate_content("preview"))
        self.widgets.btn_gen.on_click(lambda _: self._generate_content("full"))
        self.widgets.btn_copy.on_click(self._copy_to_clipboard)
        self.widgets.btn_download.on_click(self._download_file)
        self.widgets.btn_clear.on_click(self._clear_form)

    def on_glyphs_change(self, change):
        """
        Handle portal glyphs input change.

        When glyphs are entered:
        1. Validate glyph format (12 hex characters)
        2. Convert to coordinates if valid
        3. Generate region name if galaxy is selected

        Args:
            change: Widget change event (unused but required by observer)
        """
        glyphs = (self.widgets.portalglyphs.value or "").strip().upper()

        # Validate glyph format (12 hex characters)
        if not re.match(r"^[0-9A-F]{12}$", glyphs):
            self.widgets.coordinates.value = ""
            self.widgets.region.value = ""
            return

        # Convert glyphs to coordinates
        coords = self.map_logic.glyphs_to_coords(glyphs)
        if coords:
            # Format coordinates as hex string
            self.widgets.coordinates.value = \
                f"{coords['x']:04X}:{coords['y']:04X}:{coords['z']:04X}:{coords['s']:04X}"

            # Generate region name if galaxy is selected
            gal_idx = self.data.GALAXY_INDEX_MAP.get(self.widgets.galaxy.value)
            if gal_idx is not None and self.data.ALPHASETS:
                # Convert coordinates to region space (centered)
                center_xz = 0x7FF  # Center for X and Z coordinates
                center_y = 0x7F    # Center for Y coordinate
                reg_x = coords['x'] - center_xz
                reg_y = coords['y'] - center_y
                reg_z = coords['z'] - center_xz

                try:
                    # Create seed and generate region name
                    seed = RegionNameGenerator.create_region_seed(
                        reg_x, reg_y, reg_z, gal_idx)
                    self.widgets.region.value = RegionNameGenerator.format_name(
                        seed, self.data.ALPHASETS, self.data.LETTER_MAP)
                except (IndexError, ValueError, KeyError, struct.error) as e:
                    self.widgets.region.value = f"Error: {e}"

    def on_ship_type_change(self, change):
        """
        Handle ship type selection change.

        When ship type changes:
        1. Disable class dropdown for Exotic ships (always S-class)
        2. Update secondary color options for Exotic ships
        3. Update all part dropdowns (wings, thrusters, etc.)
        4. Update upgrade module lists
        5. Recalculate inventory size

        Args:
            change: Widget change event (unused but required by observer)
        """
        ship_type = self.widgets.type.value
        is_exo = ship_type == 'Exotic'

        # Exotic ships are always S-class and have limited color options
        self.widgets.class_.disabled = is_exo
        self.widgets.accent.disabled = is_exo
        if is_exo:
            self.widgets.class_.value = 'S'  # Exotic ships are always S-class
            self.widgets.secondarycolor.options = [self.DEFAULT_PLACEHOLDER,
                                                   'Gold', 'Silver']
        else:
            self.widgets.secondarycolor.options = [self.DEFAULT_PLACEHOLDER] + \
                                                  self.data.SHIP_COLORS

        def update_part_dropdown(widget, data_dict, placeholder_prefix):
            """
            Helper to update part dropdown based on ship type.

            Args:
                widget: Dropdown widget to update
                data_dict: Dictionary mapping ship types to part lists
                placeholder_prefix: Text for placeholder option
            """
            opts = data_dict.get(ship_type, [])
            widget.options = [f"- {placeholder_prefix} -"] + opts
            widget.disabled = not bool(opts)  # Disable if no options

        # Update all part dropdowns
        update_part_dropdown(self.widgets.subtype, self.data.SHIP_SUBTYPES,
                            'Select Subtype')
        update_part_dropdown(self.widgets.wings, self.data.WINGS, 'Select Wings')
        update_part_dropdown(self.widgets.engines, self.data.THRUSTERS,
                            'Select Thrusters')
        update_part_dropdown(self.widgets.hullacc, self.data.HULL_ACCESSORIES,
                            'Select Hull Acc')
        update_part_dropdown(self.widgets.otheracc, self.data.OTHER_ACCESSORIES,
                            'Select Other Acc')

        # Update upgrade module lists (include living ship upgrades if needed)
        mods = sorted(self.data.UPGRADE_MODULES +
                     (self.data.LIVING_SHIP_UPGRADES if ship_type == 'Living Ship' else []))
        for i in range(1, 5):
            getattr(self.widgets, f'upgrade{i}').options = [self.DEFAULT_PLACEHOLDER] + mods

        # Recalculate inventory size based on new ship type
        self.on_stats_change(None)

    def on_stats_change(self, change):
        """
        Calculate inventory size based on slot counts.

        Compares cargo and tech slot counts against thresholds for the
        current ship type to determine size category (Small, Medium, Large).

        Args:
            change: Widget change event (unused but required by observer)
        """
        ship_type = self.widgets.type.value
        cargo = self.widgets.cargoslots.value
        tech = self.widgets.techslots.value

        # Find first matching size threshold for this ship type
        thresholds = self.INVENTORY_SIZE_THRESHOLDS.get(ship_type, [])
        self.widgets.inventory.value = next(
            (size_label for cargo_threshold, tech_threshold, size_label in thresholds
             if cargo >= cargo_threshold and tech >= tech_threshold),
            ""  # Empty string if no threshold matches
        )

    def on_location_change(self, change):
        """
        Handle location type change.

        Disables planetary axes field for space locations (no coordinates
        on planets/moons).

        Args:
            change: Widget change event (unused but required by observer)
        """
        is_space_location = self.widgets.location.value in ['Space Station',
                                                           'Outlaw Station']
        self.widgets.axes_text.disabled = is_space_location
        if is_space_location:
            self.widgets.axes_text.value = ''  # Clear axes for space locations

    def _update_stardate_ui(self, change):
        """
        Update AGT stardate display based on discovery date.

        Converts real date to in-game stardate format: YYYY.DD.MM
        with year offset for game timeline.

        Args:
            change: Widget change event (unused but required by observer)
        """
        date_arrow = arrow.get(self.widgets.discovery_date.value) \
            if self.widgets.discovery_date.value else None
        if date_arrow:
            # Format: (year + offset).day.month
            self.widgets.agt_stardate.value = \
                f"{date_arrow.year+STARDATE_YEAR_OFFSET}.{date_arrow.day}.{date_arrow.month:02d}"
        else:
            self.widgets.agt_stardate.value = ""

    def _generate_content(self, mode):
        """
        Generate wiki content from form data.

        Args:
            mode (str): "preview" for preview mode, "full" for generation mode

        Steps:
        1. Collect all widget values
        2. Validate with ShipModel
        3. Prepare template context
        4. Render wiki template
        5. Display results and update UI state
        """
        self.widgets.status_label.value = "<b>Processing...</b>"

        # Step 1: Collect all widget values
        raw = {}
        for widget_field in fields(self.widgets):
            widget = getattr(self.widgets, widget_field.name)
            if (hasattr(widget, 'value') and
                not isinstance(widget, (Button, Output, HTML))):
                raw[widget_field.name] = widget.value

        # Step 2: Validate data with Pydantic model
        try:
            model = ShipModel(**raw)
        except ValidationError as e:
            # Show first validation error
            self.widgets.status_label.value = \
                f"<b style='color:red'>Error: {e.errors()[0]['msg']}</b>"
            return

        # Step 3: Prepare template context
        template_context = {k: str(v) for k, v in
                           model.model_dump(exclude_none=True).items()}

        # Special handling for exotic ships (subtype becomes exotic field)
        if template_context.get('type') == 'Exotic':
            template_context['exotic'] = template_context.pop('subtype', '')
        else:
            template_context['exotic'] = ''

        # Calculate spawn chance based on economy
        economy_value = template_context.get('economy', '')
        if template_context.get('type') in ['Exotic', 'Living Ship']:
            template_context['schance'] = '100%'  # Always spawn at space stations
        else:
            template_context['schance'] = \
                self.ECONOMY_SPAWN_CHANCE_MAP.get(economy_value[:2], '')

        # Format discovery date
        if model.discovery_date:
            template_context['discovery_date'] = \
                model.discovery_date.format('DD-MMM-YYYY')

        # Format gallery images (one per line)
        template_context['gallery_images'] = "\n".join(
            [line.strip() for line in
             raw.get('gallery_images', '').split('\n') if line.strip()])

        # Step 4: Render wiki template
        try:
            self.generated_content = self.WIKI_TEMPLATE.render(template_context)
        except jinja2.TemplateError as e:
            self.widgets.status_label.value = \
                f"<b style='color:red'>Template Error: {e}</b>"
            return

        # Step 5: Display results
        with self.widgets.output:
            clear_output(wait=True)
            print(self.generated_content)

        # Update UI based on mode
        if mode == 'preview':
            self.widgets.status_label.value = \
                "<b style='color:green'>Preview generated.</b>"
        else:
            self.widgets.status_label.value = \
                "<b style='color:green'>Wiki code generated — ready to copy or download.</b>"
            self.widgets.btn_download.disabled = False
            self.widgets.btn_copy.disabled = False

            # Auto-generate filename if not set
            if not self.widgets.filename.value:
                safe = re.sub(r'[^a-zA-Z0-9_-]', '',
                             model.name.replace(' ', '_'))
                self.widgets.filename.value = f"{safe}_Wiki.txt"

    def _clear_form(self, _):
        """
        Reset all form fields to default values.

        Args:
            _: Button click event (unused)
        """
        for widget_field in fields(self.widgets):
            widget = getattr(self.widgets, widget_field.name)
            if not hasattr(widget, 'value'):
                continue

            # Reset based on widget type
            if widget_field.name == 'discovery_date':
                widget.value = arrow.now().date()  # Today's date
            elif widget_field.name == 'civilized':
                widget.value = self.DEFAULT_CIVILIZATION
            elif widget_field.name == 'release':
                widget.value = self.DEFAULT_RELEASE
            elif isinstance(widget.value, (int, float)):
                widget.value = 0  # Reset numeric fields
            elif isinstance(widget, Dropdown):
                widget.value = widget.options[0]  # First (placeholder) option
            else:
                widget.value = ""  # Clear text fields

        # Clear output and reset state
        self.widgets.output.clear_output()
        self.generated_content = ""
        self.widgets.btn_copy.disabled = True
        self.widgets.btn_download.disabled = True
        self.widgets.status_label.value = "<i>Form cleared.</i>"

        # Re-initialize dependent fields
        self.on_ship_type_change(None)

    def on_load_example_click(self, _, ship_type):
        """
        Load example data for specified ship type.

        Args:
            _: Button click event (unused)
            ship_type (str): Type of ship to load example for
        """
        # Clear form first
        self._clear_form(None)

        # Load example data for this ship type
        data = self.data.EXAMPLE_DATA.get(ship_type)
        if data:
            # Set ship type first (triggers other updates)
            if 'type' in data:
                self.widgets.type.value = data['type']

            # Set all other fields from example data
            for k, v in data.items():
                if hasattr(self.widgets, k):
                    widget = getattr(self.widgets, k)
                    if isinstance(widget, Dropdown):
                        # Only set if value exists in options
                        widget.value = v if v in widget.options else widget.options[0]
                    else:
                        widget.value = v

            self.widgets.status_label.value = f"<b>Loaded {ship_type}</b>"
        else:
            self.widgets.status_label.value = \
                f"<b style='color:orange'>No example data for {ship_type}</b>"

    def _copy_to_clipboard(self, _):
        """
        Copy generated wiki code to clipboard.

        Uses JavaScript clipboard API (works in Jupyter).

        Args:
            _: Button click event (unused)
        """
        if self.generated_content:
            display(Javascript(
                f"navigator.clipboard.writeText({json.dumps(self.generated_content)});"))
            self.widgets.status_label.value = "<b>Copied!</b>"

    def _download_file(self, _):
        """
        Download generated wiki code as text file.

        Only works in Google Colab environment.

        Args:
            _: Button click event (unused)
        """
        if self.generated_content:
            try:
                from google.colab import files
                filename = self.widgets.filename.value or "ship.txt"
                # Sanitize filename
                filename = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)
                filename = filename.lstrip('.')  # Remove leading dots
                if not filename:
                    filename = "ship.txt"

                # Write file and trigger download
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.generated_content)
                files.download(filename)
            except ImportError:
                # Not in Colab - show error
                self.widgets.status_label.value = \
                    "<b style='color:red'>Colab only</b>"


# Main execution block
if __name__ == '__main__':
    try:
        # Check if running in Jupyter/IPython
        get_ipython()
        # Create and display the application
        app = NMSWikiStarshipFormCreator()
    except NameError:
        # Not in Jupyter - print instructions
        print("Run in Jupyter.")