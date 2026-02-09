"""
No Man's Sky Base Wiki Generator for Jupyter/Colab Notebooks

This module provides a complete graphical interface for creating No Man's Sky base
wiki pages. It includes tools for converting portal glyphs to galactic coordinates,
generating procedurally-correct region names, and formatting all base information
into the official wiki markup format.

Key Components:
1. NMSData - Loads game data from external sources
2. ByteUtils & Generator - Implements procedural name generation algorithms
3. NMSGalaxyMap - Converts portal glyphs to coordinates
4. WikiDataModel - Validates user input
5. NmsWikiGenerator - Main application with tabbed interface
"""

# Import standard library modules for data handling and byte manipulation
import struct
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Dict, List, Optional

# Import third-party libraries for UI, templates, and data validation
import arrow
import jinja2
import requests
from pydantic import BaseModel, Field, ValidationError, field_validator, ValidationInfo
import ipywidgets as widgets
from ipywidgets import (
    Button, Checkbox, Combobox, DatePicker, Dropdown, GridBox,
    HBox, HTML, Layout, Output, Tab, Text, Textarea, VBox
)
from IPython.display import Javascript, clear_output, display


class ByteUtilsConstants:
    """
    Stores shared mathematical constants used by the procedural generation engine.

    These values are hardcoded in the No Man's Sky game code and must match exactly
    for the region name generation to be accurate. They control byte operations,
    scrambling algorithms, and generation limits.
    """
    # Multipliers and scrambling arrays from the game's procedural generation code
    MULTIPLIER_ARRAY = [0x99, 0xF8, 0x76, 0x5A]
    SCRAMBLE_MULT_1 = [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64]
    SCRAMBLE_MULT_2 = [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3]

    # Operation mode identifiers for bitwise logic
    AND_MODE = 0
    OR_MODE = 1
    XOR_MODE = 2

    # Game coordinate system center points (for voxel space conversion)
    VOXEL_CENTER_XZ = 0x7FF    # 2047 in decimal - center of X and Z axes
    VOXEL_CENTER_Y = 0x7F      # 127 in decimal - center of Y axis

    # Threshold for adding adornments to region names (50% chance)
    ADORN_THRESHOLD = 0x50     # 80 in decimal - 80/100 = 80% threshold

    # Safety limit to prevent infinite loops in name generation
    MAX_SAFETY_ITERATIONS = 50

    # Maximum length for generated region names
    MAX_NAME_LENGTH = 64

    # Sentinel value indicating an empty result
    EMPTY_SENTINEL = "__EMPTY__"

    # Bytes representing a floating point scale factor (0.0000152587890625)
    FLOAT_SCALE_BYTES = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]


class NMSGalaxyMapConstants:
    """
    Stores constants for galaxy coordinate calculations.

    These values handle the conversion between portal glyphs (12-character hex codes)
    and galactic coordinates (X, Y, Z positions in the game universe).
    """
    # Coordinate shift values for handling negative positions in hex representation
    SHIFT_POS_XZ = 2049  # Add this to positive coordinates
    SHIFT_NEG_XZ = 2047  # Add this to negative coordinates
    SHIFT_POS_Y = 129    # Add this to positive Y coordinates
    SHIFT_NEG_Y = 127    # Add this to negative Y coordinates

    # Fixed length of portal glyph strings (always 12 hexadecimal characters)
    GLYPH_LENGTH = 12

    # Valid characters for portal glyph input
    HEX_CHARS = '0123456789ABCDEF'


class NMSData:
    """
    Loads and manages external game data from GitHub repositories.

    This class fetches dynamic game data (galaxies, base parts, features) instead of
    hardcoding it, making the tool automatically update when game content changes.
    All data is loaded from JSON files hosted online.

    Attributes:
        galaxies (List[str]): Names of all galaxies in No Man's Sky
        galaxy_to_index (Dict[str, int]): Maps galaxy names to their numeric IDs
        base_types (Dict[str, str]): Base type names and their descriptions
        power_map (Dict[str, str]): Power condition names and descriptions
        terrain_conditions (List[str]): Available terrain condition options
        features_list (List[str]): Available base feature options
        nearby_list (List[str]): Available nearby point-of-interest options
        load_status (str): Status of data loading operation
        LETTER_MAP (Dict[int, str]): Character mapping for name generation
        ALPHASETS (List[str]): Character sets for different name styles
    """

    # Base URL for all external data files
    BASE_URL = "https://raw.githubusercontent.com/2A03-Jikuu/nms-wiki-tool-py/refs/heads/main/datalist"

    # Individual data file URLs
    URL_BASE_DATA = f"{BASE_URL}/base_data.json"
    URL_GALAXIES = f"{BASE_URL}/galaxies.json"
    URL_LETTER_MAP = f"{BASE_URL}/letter_map.json"
    URL_ALPHASETS = f"{BASE_URL}/alphasets.json"

    # Static storage for procedural generation data (shared across all instances)
    LETTER_MAP = {}
    ALPHASETS = []

    def __init__(self):
        """Initialize empty data containers that will be populated from external sources."""
        self.galaxies = []                     # List of galaxy names
        self.galaxy_to_index = {}              # Maps names to numeric indices
        self.base_types = {}                   # Base type descriptions
        self.power_map = {}                    # Power condition descriptions
        self.terrain_conditions = []           # Available terrain conditions
        self.features_list = []                # Available base features
        self.nearby_list = []                  # Available nearby points of interest
        self.load_status = "Not Loaded"        # Current loading status

    def fetch_remote_data(self):
        """
        Download all required data files from GitHub and populate local data structures.

        This method performs multiple HTTP requests to fetch JSON data files,
        then processes and stores them in appropriate formats. If loading fails,
        it provides fallback data to prevent application crashes.

        Raises:
            requests.exceptions.RequestException: If network connection fails
            ValueError: If JSON parsing fails
            KeyError: If expected data fields are missing from JSON

        Example:
            >>> data_loader = NMSData()
            >>> data_loader.fetch_remote_data()
            Loaded 256 galaxies and configuration data.
        """
        try:
            # --- Load Galaxy Data ---
            # Fetch list of galaxies and their numeric indices
            r_gal = requests.get(self.URL_GALAXIES, timeout=15)
            r_gal.raise_for_status()  # Raise exception for HTTP errors
            gal_data = r_gal.json()

            # Extract galaxy names and create name-to-index mapping
            self.galaxies = [item['name'] for item in gal_data]
            self.galaxy_to_index = {
                item['name']: item['index'] for item in gal_data
            }

            # --- Load Base Configuration Data ---
            # Fetch base types, power conditions, terrain conditions, etc.
            r_base = requests.get(self.URL_BASE_DATA, timeout=15)
            r_base.raise_for_status()
            base_data = r_base.json()

            # Populate all base-related data structures
            self.base_types = base_data.get('base_types', {})
            self.power_map = base_data.get('power_map', {})
            self.terrain_conditions = sorted(
                base_data.get('terrain_conditions', [])
            )
            self.features_list = sorted(base_data.get('features_list', []))
            self.nearby_list = sorted(base_data.get('nearby_list', []))

            # --- Load Procedural Generation Data ---
            # These are stored as class variables since they're used by the Generator class

            # Load letter mapping for name generation (JSON keys are strings, convert to int)
            r_let = requests.get(self.URL_LETTER_MAP, timeout=15)
            r_let.raise_for_status()
            raw_letters = r_let.json()
            NMSData.LETTER_MAP = {int(k): v for k, v in raw_letters.items()}

            # Load character sets for different name styles
            r_alpha = requests.get(self.URL_ALPHASETS, timeout=15)
            r_alpha.raise_for_status()
            NMSData.ALPHASETS = r_alpha.json()

            # Update status and confirm successful loading
            self.load_status = "Success"
            print(
                f"Loaded {len(self.galaxies)} galaxies and configuration data."
            )

        except Exception as e:
            # Handle any loading errors gracefully with fallback data
            self.load_status = f"Error: {str(e)}"
            print(f"FAILED TO LOAD EXTERNAL DATA: {e}")

            # Provide minimal fallback data to prevent crashes
            self.galaxies = ['Euclid', 'Hilbert Dimension']
            self.galaxy_to_index = {'Euclid': 0, 'Hilbert Dimension': 1}


class ByteUtils:
    """
    Provides low-level byte manipulation operations that mimic C++ behavior.

    No Man's Sky is written in C++ where integer overflow wraps around
    (e.g., 255 + 1 = 0). Python doesn't do this automatically, so this class
    implements byte-by-byte arithmetic with wrapping to match the game's behavior.
    All operations work on lists of bytes (integers 0-255).

    Note: This is essential for accurate procedural generation because the
    game's random number generator depends on exact C++ integer behavior.
    """

    @staticmethod
    def parse(value, little_endian=True):
        """
        Convert a hexadecimal string into a list of byte values.

        Args:
            value (str): Hexadecimal string (e.g., "015A" or "FF")
            little_endian (bool): If True, reverse byte order (least significant byte first)

        Returns:
            List[int]: List of byte values (integers 0-255)

        Example:
            >>> ByteUtils.parse("015A")
            [90, 1]  # With little_endian=True (default)
            >>> ByteUtils.parse("015A", little_endian=False)
            [1, 90]  # Without byte reversal
        """
        # Pad with leading zero if string has odd length
        if len(value) % 2 != 0:
            value = "0" + value

        # Convert each 2-character hex pair to an integer
        result = [int(value[i:i + 2], 16) for i in range(0, len(value), 2)]

        # Reverse for little-endian (game's internal format)
        if little_endian:
            result.reverse()
        return result

    @staticmethod
    def format_short(op1):
        """
        Ensure a byte list is at least 2 bytes long by padding with zeros.

        Args:
            op1 (List[int]): Input byte list

        Returns:
            List[int]: Padded byte list with minimum 2 bytes

        Example:
            >>> ByteUtils.format_short([0x01])
            [0x01, 0x00]
        """
        result = list(op1)
        while len(result) < 2:
            result.append(0x00)
        return result

    @staticmethod
    def add(op1, op2):
        """
        Add two byte lists together with proper carry handling.

        This mimics C++ addition where each byte (0-255) overflows to the next byte.
        The operation is performed byte-by-byte starting from the least significant byte.

        Args:
            op1 (List[int]): First operand (byte list)
            op2 (List[int]): Second operand (byte list)

        Returns:
            List[int]: Result of addition as byte list

        Example:
            # Adding 255 + 1 = 256, which becomes [0, 1] in little-endian bytes
            >>> ByteUtils.add([0xFF], [0x01])
            [0x00, 0x01]
        """
        result = list(op2)
        for i in range(len(op1)):
            result = ByteUtils._add_single(op1[i], result, i)
        return result

    @staticmethod
    def _add_single(val, target_list, index):
        """
        Helper method for single-byte addition with recursive carry.

        Args:
            val (int): Byte value to add (0-255)
            target_list (List[int]): Current result being built
            index (int): Current byte position

        Returns:
            List[int]: Updated result with carry handled
        """
        if index < len(target_list):
            # Add the byte values
            total = val + target_list[index]

            # Keep only the lower 8 bits (0-255)
            target_list[index] = total & 0xFF

            # Calculate carry for next byte
            rem = (total >> 8) & 0xFF

            # Recursively add carry to next byte position
            if rem != 0:
                target_list = ByteUtils._add_single(
                    rem, target_list, index + 1
                )
        else:
            # If we're past the end, append the value
            target_list.append(val)
        return target_list

    @staticmethod
    def sub(op1, op2):
        """
        Subtract op1 from op2 with proper borrow handling.

        This mimics C++ subtraction where each byte (0-255) underflows to borrow
        from the next byte. Works from least to most significant byte.

        Args:
            op1 (List[int]): Value to subtract
            op2 (List[int]): Starting value

        Returns:
            List[int]: Result of subtraction as byte list

        Example:
            # Subtracting 1 from 0 = -1, which wraps to 255
            >>> ByteUtils.sub([0x01], [0x00])
            [0xFF]
        """
        result = list(op2)
        for i in range(len(op1)):
            result = ByteUtils._sub_single(op1[i], result, i)
        return result

    @staticmethod
    def _sub_single(val, target_list, index):
        """
        Helper method for single-byte subtraction with recursive borrow.

        Args:
            val (int): Byte value to subtract
            target_list (List[int]): Current result being built
            index (int): Current byte position

        Returns:
            List[int]: Updated result with borrow handled
        """
        if index < len(target_list):
            # Subtract the byte values
            diff = val - target_list[index]

            # Keep only the lower 8 bits (0-255, wraps on underflow)
            target_list[index] = diff & 0xFF

            # Calculate borrow for next byte (negative means we borrowed)
            rem = (diff >> 8) & 0xFF

            # Recursively handle borrow in next byte position
            if rem != 0:
                target_list = ByteUtils._sub_single(
                    rem, target_list, index + 1
                )
        else:
            # If we're past the end, append the value (with sign extension)
            target_list.append(val)
        return target_list

    @staticmethod
    def multiply(op1, op2):
        """
        Multiply two byte lists with signed 16-bit wrapping (mimics C++).

        This is crucial for the seed scrambling algorithm. The game uses
        signed 16-bit multiplication that wraps around when exceeding 32767
        or going below -32768.

        Args:
            op1 (List[int]): First operand
            op2 (List[int]): Second operand

        Returns:
            List[int]: Result of multiplication as byte list

        Example:
            # Multiplying near the 16-bit limit
            >>> result = ByteUtils.multiply([0xFF, 0x7F], [0x02])
        """
        result = []

        # Perform multiplication byte by byte (like manual long multiplication)
        for i in range(len(op1)):
            rem = 0  # Carry from previous multiplication
            for j in range(len(op2)):
                # Multiply bytes and add any carry from previous step
                raw_prod = (op1[i] * op2[j]) + rem

                # Convert to signed 16-bit and wrap (C++ behavior)
                signed_prd = (raw_prod + 32768) % 65536 - 32768

                # Extract high and low bytes
                rem = (signed_prd >> 8) & 0xFF  # Carry for next position
                res = signed_prd & 0xFF         # Result for this position

                # Add to appropriate position in result
                idx = i + j
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

    # --- Bitwise Shift Operations ---
    # These simulate C++ shift operators on byte arrays

    @staticmethod
    def shl(op1, shift):
        """
        Shift bytes left (toward more significant positions).

        Args:
            op1 (List[int]): Byte array to shift
            shift (int): Number of positions to shift

        Returns:
            List[int]: Shifted byte array

        Example:
            >>> ByteUtils.shl([0x01, 0x02], 1)
            [0x02]  # First byte shifted out
        """
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1, shift):
        """
        Shift bytes right (toward less significant positions).

        Args:
            op1 (List[int]): Byte array to shift
            shift (int): Number of positions to shift

        Returns:
            List[int]: Shifted byte array

        Example:
            >>> ByteUtils.shr([0x01, 0x02], 1)
            [0x01]  # Last byte shifted out
        """
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rol(op1, roll):
        """
        Rotate bytes left (circular shift).

        Args:
            op1 (List[int]): Byte array to rotate
            roll (int): Number of positions to rotate

        Returns:
            List[int]: Rotated byte array

        Example:
            >>> ByteUtils.rol([0x01, 0x02, 0x03], 1)
            [0x02, 0x03, 0x01]
        """
        return (
            op1[roll % len(op1):] + op1[:roll % len(op1)] if op1 else op1
        )

    # --- Byte Array Padding Operations ---

    @staticmethod
    def zxd(op1, extend):
        """
        Zero-extend a byte array to specified length.

        Adds zeros to the end (most significant side in little-endian).

        Args:
            op1 (List[int]): Byte array to extend
            extend (int): Desired final length

        Returns:
            List[int]: Extended array padded with zeros

        Example:
            >>> ByteUtils.zxd([0x01], 3)
            [0x01, 0x00, 0x00]
        """
        return list(op1) + [0x00] * (extend - len(op1))

    @staticmethod
    def sxd(op1, extend):
        """
        Sign-extend a byte array to specified length.

        Replicates the sign bit (most significant bit of last byte)
        to preserve two's complement representation of signed numbers.

        Args:
            op1 (List[int]): Byte array to extend
            extend (int): Desired final length

        Returns:
            List[int]: Sign-extended array

        Example:
            # Negative number (0x80 has MSB set = negative)
            >>> ByteUtils.sxd([0x80], 2)
            [0x80, 0xFF]
        """
        result = list(op1)

        # Check sign bit of most significant byte
        val = 0xFF if (len(op1) > 0 and (op1[-1] >> 7) == 1) else 0x00

        # Extend with sign value
        for _ in range(extend - len(op1)):
            result.append(val)
        return result

    # --- Bitwise Logic Operations ---

    @staticmethod
    def logical_op(op1, op2, mode):
        """
        Perform AND, OR, or XOR operation on two byte arrays.

        Args:
            op1 (List[int]): First operand
            op2 (List[int]): Second operand
            mode (int): Operation type (AND_MODE, OR_MODE, or XOR_MODE)

        Returns:
            List[int]: Result of bitwise operation

        Raises:
            ValueError: If mode is not one of the recognized constants
        """
        # Make arrays equal length by padding with zeros
        l1, l2 = len(op1), len(op2)
        if l1 > l2:
            longer = list(op1)
            shorter = list(op2) + [0x00] * (l1 - l2)
        else:
            longer = list(op2)
            shorter = list(op1) + [0x00] * (l2 - l1)

        # Perform operation byte by byte
        result = []
        for i in range(len(longer)):
            if mode == ByteUtilsConstants.AND_MODE:
                result.append(longer[i] & shorter[i])
            elif mode == ByteUtilsConstants.OR_MODE:
                result.append(longer[i] | shorter[i])
            else:  # XOR_MODE
                result.append(longer[i] ^ shorter[i])
        return result

    @staticmethod
    def xor(op1, op2):
        """Exclusive OR of two byte arrays."""
        return ByteUtils.logical_op(op1, op2, ByteUtilsConstants.XOR_MODE)

    @staticmethod
    def and_op(op1, op2):
        """Bitwise AND of two byte arrays."""
        return ByteUtils.logical_op(op1, op2, ByteUtilsConstants.AND_MODE)

    @staticmethod
    def or_op(op1, op2):
        """Bitwise OR of two byte arrays."""
        return ByteUtils.logical_op(op1, op2, ByteUtilsConstants.OR_MODE)

    @staticmethod
    def update_seed(cache, move=1):
        """
        Advance the pseudo-random number generator state.

        This implements the game's specific random number generation algorithm
        that's used for procedural content. The algorithm mixes two seed values
        using multiplication and bit shifting.

        Args:
            cache (List[List[int]]): Current seed state [seed0, seed1]
            move (int): Number of times to advance the generator

        Returns:
            List[List[int]]: Updated seed state

        Example:
            >>> seed_state = [[0x01, 0x02], [0x03, 0x04]]
            >>> ByteUtils.update_seed(seed_state)
            [[...], [...]]  # New mixed state
        """
        for _ in range(move):
            # Multiply seed0 by constant array
            step1 = ByteUtils.multiply(cache[0], ByteUtilsConstants.MULTIPLIER_ARRAY)

            # Add seed1 to result
            result = ByteUtils.add(step1, cache[1])

            # Update seeds: first 4 bits become new seed0, rest become seed1
            cache[0] = ByteUtils.shl(result, 4)
            cache[1] = ByteUtils.shr(result, 4)
        return cache

    # --- Binary Conversion Methods ---
    # Convert byte arrays to standard Python numeric types

    @staticmethod
    def to_uint32(arr, offset=0):
        """
        Convert 4 bytes to unsigned 32-bit integer.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            int: Unsigned 32-bit integer value

        Example:
            >>> ByteUtils.to_uint32([0x01, 0x00, 0x00, 0x00])
            1  # Little-endian: 0x00000001
        """
        chunk = arr[offset:offset + 4]
        while len(chunk) < 4:
            chunk.append(0)
        return struct.unpack('<I', bytes(chunk))[0]

    @staticmethod
    def to_int32(arr, offset=0):
        """
        Convert 4 bytes to signed 32-bit integer.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            int: Signed 32-bit integer value

        Example:
            >>> ByteUtils.to_int32([0xFF, 0xFF, 0xFF, 0xFF])
            -1  # Two's complement representation
        """
        chunk = arr[offset:offset + 4]
        while len(chunk) < 4:
            chunk.append(0)
        return struct.unpack('<i', bytes(chunk))[0]

    @staticmethod
    def to_int16(arr, offset=0):
        """
        Convert 2 bytes to signed 16-bit integer.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            int: Signed 16-bit integer value
        """
        chunk = arr[offset:offset + 2]
        while len(chunk) < 2:
            chunk.append(0)
        return struct.unpack('<h', bytes(chunk))[0]

    @staticmethod
    def to_double(arr, offset=0):
        """
        Convert 8 bytes to double-precision floating point.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            float: Double-precision floating point value
        """
        chunk = arr[offset:offset + 8]
        while len(chunk) < 8:
            chunk.append(0)
        return struct.unpack('<d', bytes(chunk))[0]

    @staticmethod
    def to_single(arr, offset=0):
        """
        Convert 4 bytes to single-precision floating point.

        Args:
            arr (List[int]): Byte array
            offset (int): Starting position in array

        Returns:
            float: Single-precision floating point value
        """
        chunk = arr[offset:offset + 4]
        while len(chunk) < 4:
            chunk.append(0)
        return struct.unpack('<f', bytes(chunk))[0]

    @staticmethod
    def get_bytes_uint32(val):
        """
        Convert unsigned 32-bit integer to 4-byte array.

        Args:
            val (int): Integer value (0 to 4,294,967,295)

        Returns:
            List[int]: 4-byte array in little-endian order
        """
        return list(struct.pack('<I', val))


class StringExtensions:
    """
    Provides string formatting utilities for hexadecimal conversion.

    These methods handle the conversion of coordinate values to the specific
    hex string formats required by the procedural generation algorithms.
    """

    @staticmethod
    def short_to_formatted_hex(value, truncate_length):
        """
        Convert a 16-bit integer to hexadecimal string with truncation.

        Handles negative numbers by converting to two's complement
        representation before formatting.

        Args:
            value (int): Integer value to convert (-32768 to 65535)
            truncate_length (int): Number of hex characters to keep

        Returns:
            str: Truncated hexadecimal string

        Example:
            >>> StringExtensions.short_to_formatted_hex(255, 2)
            "FF"
            >>> StringExtensions.short_to_formatted_hex(-1, 2)
            "FF"  # Two's complement: 0xFFFF truncated to 2 chars
        """
        # Mask to 16-bit to handle negative numbers (two's complement)
        value = value & 0xFFFF

        # Convert to 4-character hex string
        hex_str = f"{value:04X}"

        # Return only the required characters (from the end for little-endian)
        return hex_str[-truncate_length:]


class Generator:
    """
    Generates procedural region names using game-accurate algorithms.

    This class implements the complex name generation logic from No Man's Sky,
    which uses weighted character tables (alphasets) and linguistic rules
    to create pronounceable but alien-sounding region names.

    The generation process:
    1. Starts with an initial 3-character seed from an alphaset
    2. Adds characters based on statistical weights and randomness
    3. Applies linguistic fixes to avoid unpronounceable combinations
    4. Possibly adds adornments (e.g., "The Arm of [Name]")
    """

    @staticmethod
    def generate_name(cache0, cache1):
        """
        Generate a region name from seed values.

        This is the main name generation loop that builds a name character
        by character using statistical weights from the game's letter tables.

        Args:
            cache0 (List[List[int]]): Primary seed state for randomness
            cache1 (List[List[int]]): Secondary seed state for parameters

        Returns:
            str: Generated region name, or empty string if generation fails

        Example:
            >>> seed0 = [[0x01, 0x02, 0x03], [0x04, 0x05, 0x06]]
            >>> seed1 = [[0x00], [0x06], []]
            >>> Generator.generate_name(seed0, seed1)
            "Ebyr"
        """
        # Step 1: Get initial 3-character seed from alphaset
        name = Generator.get_characters_from_alphaset(cache0, cache1)

        # If alphaset returned empty sentinel, return empty name
        if name == ByteUtilsConstants.EMPTY_SENTINEL:
            return ""

        # Step 2: Advance random seed and determine generation mode
        ByteUtils.update_seed(cache0)

        # Check if we should use alternate character selection method
        check_op = ByteUtils.zxd(ByteUtils.and_op(cache0[0], [0x01]), 2)
        alternate_char_getter = (ByteUtils.to_int16(check_op) != 0)
        ByteUtils.update_seed(cache0)

        # Step 3: Calculate how many additional characters to generate
        # This formula determines name length based on seed values
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])
        limit = ByteUtils.to_int16(ByteUtils.sxd(register0, 2))

        # Step 4: Generate additional characters
        if 0 < limit:
            i, safety = 0, 0
            while i < limit:
                ByteUtils.update_seed(cache0)

                # Get last 3 characters for context-based weighting
                sub_str = name[i: i + 3]
                alphaset_idx = cache1[0][0] if cache1[0] else 0

                # Get statistical weights for possible next characters
                char_weights = Generator.get_string_weights(
                    sub_str, alphaset_idx
                )

                # Generate random target value for character selection
                val_u32 = ByteUtils.to_uint32(cache0[0])
                tiny_dbl = ByteUtils.to_double(ByteUtilsConstants.FLOAT_SCALE_BYTES)
                target = float(val_u32 * tiny_dbl)

                # If no weights available, backtrack and try again
                if char_weights is None:
                    i = max(0, i - 1)  # Backtrack one character
                    safety += 1

                    # Safety break to prevent infinite loops
                    if safety > ByteUtilsConstants.MAX_SAFETY_ITERATIONS:
                        break
                else:
                    safety = 0
                    index = 0

                    # Select character based on weights and random target
                    if alternate_char_getter:
                        # Alternate method: scale target and add offset
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
                        # Standard method: cumulative probability selection
                        weight, j = 0.0, 0
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
                if len(name) > (ByteUtilsConstants.MAX_NAME_LENGTH - 1):
                    name = name[:ByteUtilsConstants.MAX_NAME_LENGTH]
                i += 1

        # Step 5: Apply linguistic fixes and return result
        return Generator.linguistic_fix(name, cache0) if name else ""

    @staticmethod
    def linguistic_fix(name, cache0):
        """
        Apply linguistic rules to improve name pronounceability.

        The game tries to avoid unpronounceable consonant clusters and
        awkward letter combinations. This method inserts vowels where needed.

        Args:
            name (str): Raw generated name
            cache0 (List[List[int]]): Seed state for random vowel selection

        Returns:
            str: Name with linguistic improvements applied
        """
        # Rule 1: Fix awkward starts (e.g., "Xq...")
        first, second = name[0], name[1] if len(name) > 1 else ''
        goto_finalise = False

        # Check if first two characters are both consonants (not a, e, i, o, u)
        if (first not in "aeiou") and (second not in "aeiou"):
            # Special cases that are allowed (like "sh", "ch", etc.)
            if first != 's' or second not in "hklmnprtwy":
                conds = [
                    (second == 'h' and first in "ctw"),  # "ch", "th", "wh"
                    (second == 'l' and first in "bcfgps"),  # "bl", "cl", etc.
                    (second == 'r' and first in "bcdfgkpt"),  # "br", "cr", etc.
                    (second == 'w' and first in "dgt"),  # "dw", "gw", "tw"
                    (second == 'y' and first in "hmr")   # "hy", "my", "ry"
                ]
                if any(conds):
                    goto_finalise = True
                if not goto_finalise:
                    # Insert vowel between two consonants
                    name = Generator.insert_vowel(name, cache0, 1)

        # Rule 2: Fix awkward endings (e.g., "...bg")
        ult, penult = name[-1], name[-2] if len(name) > 1 else ''
        if len(name) > 1 and (penult != 'g' or ult in "aeiou"):
            conds = [
                (ult == 'b' and penult in "gn"),    # "gb", "nb"
                (ult == 'd' and penult in "bdfghkmpst"),  # "bd", "dd", etc.
                (ult == 'g' and penult == 'l'),     # "lg"
                (ult == 'p' and penult in "bdhkt"), # "bp", "dp", etc.
                (ult == 'r' and penult in "bfg"),   # "br", "fr", "gr"
                (ult == 't' and penult == 'g'),     # "gt"
                (ult == 'w' and penult not in "aeiou")  # consonant + "w"
            ]
            if any(conds):
                # Insert vowel before awkward ending
                name = Generator.insert_vowel(name, cache0, len(name) - 1)

        # Rule 3: Fix consecutive consonants (more than 3 in a row)
        consonance = Generator.get_consecutive_consonants(name)
        if consonance != -1:
            ByteUtils.update_seed(cache0)

            # Calculate random position to insert vowel
            mult = ByteUtils.multiply(cache0[0], [0x03])
            shr = ByteUtils.shr(mult, 4)
            add = ByteUtils.add(shr, [0x01])
            offset = ByteUtils.to_int32(ByteUtils.zxd(add, 4))

            # Insert vowel at calculated position
            name = Generator.insert_vowel(name, cache0, consonance + offset)
        return name

    @staticmethod
    def get_characters_from_alphaset(cache0, cache1):
        """
        Get initial 3-character seed from the selected alphaset.

        Alphasets are character tables that define the style of names
        (different sets produce different "sounding" names).

        Args:
            cache0 (List[List[int]]): Seed for random selection
            cache1 (List[List[int]]): Contains alphaset index

        Returns:
            str: 3-character seed, or empty sentinel if alphaset is empty
        """
        # Advance random seed
        ByteUtils.update_seed(cache0)

        # Get alphaset index from cache
        idx = cache1[0][0] if cache1[0] else 0

        # Safety check: ensure index is within bounds
        if idx >= len(NMSData.ALPHASETS):
            idx = 0

        # Get the alphaset string (characters grouped in threes)
        alphaset_str = NMSData.ALPHASETS[idx]

        # Return empty sentinel if alphaset is empty
        if not alphaset_str:
            return ByteUtilsConstants.EMPTY_SENTINEL

        # Calculate random starting position within alphaset
        length_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        register0 = ByteUtils.multiply(cache0[0], length_bytes)
        shr_reg = ByteUtils.shr(register0, 4)
        register1 = ByteUtils.format_short(
            ByteUtils.multiply(shr_reg, [0x03])
        )

        # Extract 3-character substring from calculated position
        start = ByteUtils.to_int16(register1)
        end = ByteUtils.to_int16(ByteUtils.add(register1, [0x03]))
        return alphaset_str[start:end]

    @staticmethod
    def get_string_weights(s, alphaset):
        """
        Get statistical weights for possible next characters.

        The game uses a Markov-like model where the probability of the
        next character depends on the previous characters.

        Args:
            s (str): Current character context (1-3 characters)
            alphaset (int): Index of current alphaset

        Returns:
            Optional[List[Tuple[str, float]]]: List of (character, weight) pairs,
            or None if no matching data found
        """
        # Check if letter map is loaded and alphaset exists
        if not NMSData.LETTER_MAP or alphaset not in NMSData.LETTER_MAP:
            return None

        # Get subset for this alphaset
        subset = NMSData.LETTER_MAP[alphaset]

        # Check if first character exists in subset
        if not s or s[0] not in subset:
            return None

        # Recursively search for matching context
        return Generator.recursive_search(subset[s[0]], s)

    @staticmethod
    def recursive_search(arr, s):
        """
        Recursively search through letter map structure for character weights.

        The letter map is a complex nested structure that encodes character
        transition probabilities based on context.

        Args:
            arr (List): Nested structure from letter map
            s (str): Character context to search for

        Returns:
            Optional[List[Tuple[str, float]]]: Weights for next characters
        """
        result, i = None, 0

        # Iterate through array until found or exhausted
        while result is None and i < len(arr):
            item = arr[i]
            if len(item) > 2:
                type_code, val = item[2], item[0]

                # Type "ja": Jump if above (binary search in tree)
                if type_code == "ja":
                    # Compare strings as 32-bit integers
                    s_bytes = ByteUtils.zxd(list(s.encode('utf-8')), 4)
                    val_b = ByteUtils.zxd(list(str(val).encode('utf-8')), 4)

                    # If search string > node value, search right subtree
                    if ByteUtils.to_int32(s_bytes) > ByteUtils.to_int32(val_b):
                        result = Generator.recursive_search(item[1], s)

                # Type "jz": Jump if zero (exact match found)
                elif type_code == "jz" and str(val) == s:
                    # Extract weights from matched node
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
        Insert a vowel at specified position in the name.

        Randomly selects which vowel (a, e, i, o, u) to insert
        based on the current seed state.

        Args:
            name (str): Current name
            seed (List[List[int]]): Random seed state
            index (int): Position to insert vowel

        Returns:
            str: Name with vowel inserted
        """
        ByteUtils.update_seed(seed)

        # Generate random number 0-4 to select vowel
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)

        # Insert vowel if calculation succeeded
        if calc and calc[0] < 5:
            if index <= len(name):
                vowels = "aeiou"
                return name[:index] + vowels[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name):
        """
        Find position where too many consecutive consonants occur.

        Looks for sequences of 4 or more consonants (not counting 'y'
        as a consonant in some positions).

        Args:
            name (str): Name to check

        Returns:
            int: Position where 4+ consonants start, or -1 if none found
        """
        consonance = 0  # Count of consecutive consonants

        for i in range(len(name)):
            # Reset count when vowel found
            if consonance < 3:
                if name[i] not in "aeiou":
                    consonance += 1
                else:
                    consonance = 0
            else:
                # Found 3 consonants, check next character
                if name[i] not in "aeiouy":  # 'y' can sometimes act as vowel
                    return i - 3  # Return start of 4-consonant sequence
                else:
                    consonance = 0
        return -1  # No problematic sequence found


class RegionNameGenerator:
    """
    Generates procedurally-correct region names from coordinates.

    This class combines galaxy index and voxel coordinates to create a seed,
    then uses the Generator class to produce a name, and optionally adds
    adornments (like "The Arm of [Name]").

    The process:
    1. Convert coordinates to hex seed string
    2. Scramble seed using game's algorithm
    3. Generate base name
    4. Possibly add adornment with 50% chance
    """

    # Predefined adornments that can be added to region names
    PROC_ADORNMENTS = [
        "%NAME% Adjunct", "%NAME% Void", "%NAME% Expanse", "%NAME% Terminus",
        "%NAME% Boundary", "%NAME% Fringe", "%NAME% Cluster", "%NAME% Mass",
        "%NAME% Band", "%NAME% Cloud", "%NAME% Nebula", "%NAME% Quadrant",
        "%NAME% Sector", "%NAME% Anomaly", "%NAME% Conflux",
        "%NAME% Instability", "Sea of %NAME%", "The Arm of %NAME%",
        "%NAME% Spur", "%NAME% Shallows"
    ]

    @staticmethod
    def create_region_seed(x, y, z, galaxy):
        """
        Convert coordinates and galaxy index to hex seed string.

        The format is: Galaxy(2) + Y(2) + Z(3) + X(3) = 10 hex characters

        Args:
            x (int): Voxel X coordinate (after centering)
            y (int): Voxel Y coordinate (after centering)
            z (int): Voxel Z coordinate (after centering)
            galaxy (int): Galaxy index (0 for Euclid, 1 for Hilbert, etc.)

        Returns:
            List[int]: Byte array representation of the seed

        Example:
            >>> RegionNameGenerator.create_region_seed(100, 50, -200, 0)
            [0x00, 0x32, 0x38, ...]  # Hex bytes
        """
        # Convert each component to hex with specific lengths
        s_gal = StringExtensions.short_to_formatted_hex(galaxy, 2)  # Galaxy: 2 chars
        s_y = StringExtensions.short_to_formatted_hex(y, 2)         # Y: 2 chars
        s_z = StringExtensions.short_to_formatted_hex(z, 3)         # Z: 3 chars
        s_x = StringExtensions.short_to_formatted_hex(x, 3)         # X: 3 chars

        # Concatenate and parse to byte array
        hex_str = s_gal + s_y + s_z + s_x
        return ByteUtils.parse(hex_str)

    @staticmethod
    def format_name(seed):
        """
        Generate region name from seed bytes.

        This is the main entry point for region name generation:
        1. Scramble the seed using game's algorithm
        2. Generate base name
        3. Optionally add adornment
        4. Capitalize first letter

        Args:
            seed (List[int]): Byte array seed from coordinates

        Returns:
            str: Generated region name, or "Unknown Region" on failure

        Example:
            >>> seed = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]
            >>> RegionNameGenerator.format_name(seed)
            "The Arm of Ebyr"
        """
        # Initialize seed caches for the generation algorithm
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # --- Seed Scrambling Phase ---
        # This mimics the game's seed mixing algorithm exactly

        # First mixing step
        register0 = ByteUtils.shr(seed, 4)
        if register0:
            register0[0] //= 2
        xor_res = ByteUtils.xor(register0, seed)

        # Second mixing step with first scramble constant
        register0 = ByteUtils.multiply(xor_res, ByteUtilsConstants.SCRAMBLE_MULT_1)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)

        # Third mixing step with second scramble constant
        register0 = ByteUtils.multiply(xor2, ByteUtilsConstants.SCRAMBLE_MULT_2)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        register0 = ByteUtils.xor(
            ByteUtils.get_bytes_uint32(val_u32), register0
        )

        # Final mixing and cache population
        shl4 = ByteUtils.shl(register0, 4)
        xor_mid = ByteUtils.xor(
            ByteUtils.rol(shl4, 2), ByteUtils.shr(register0, 4)
        )
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure cache0[0] is not zero (would break generation)
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])

        # Advance seed and calculate name length parameter
        ByteUtils.update_seed(cache0)
        calc_len = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4)
        cache1[2] = ByteUtils.add(calc_len, [0x06])

        # --- Name Generation ---
        name = Generator.generate_name(cache0, cache1)

        # Handle generation failures
        if not name or "[" in name:  # "[" indicates error in some alphasets
            return "Unknown Region"

        # Capitalize first letter
        name = name[0].upper() + name[1:]

        # --- Adornment Logic (50% Chance) ---
        # Roll random number 0-99 and check against threshold (80)
        ByteUtils.update_seed(cache0)
        mult_check = ByteUtils.multiply(cache0[0], [0x64])  # 0x64 = 100 decimal
        should_adorn = ByteUtils.shr(mult_check, 4)[0] < ByteUtilsConstants.ADORN_THRESHOLD

        if should_adorn:
            # Select random adornment from list
            ByteUtils.update_seed(cache0)
            idx_cal = ByteUtils.multiply(cache0[0], [0x14])  # 0x14 = 20 decimal (list length)
            idx = ByteUtils.shr(idx_cal, 4)[0]

            # Apply adornment if index is valid
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                adornment = RegionNameGenerator.PROC_ADORNMENTS[idx]
                name = adornment.replace("%NAME%", name)

        return name


class NMSGalaxyMap:
    """
    Converts portal glyphs to galactic coordinates.

    Portal glyphs are 12-character hexadecimal codes that encode a location
    in the No Man's Sky universe. This class decodes them into X, Y, Z
    coordinates and region data.

    The glyph format: [P][SSS][YY][ZZZ][XXX]
    P = Planet index within system (0-F)
    SSS = Solar System index
    YY = Y coordinate
    ZZZ = Z coordinate
    XXX = X coordinate
    """

    # Use constants from NMSGalaxyMapConstants
    SHIFT_POS_XZ = NMSGalaxyMapConstants.SHIFT_POS_XZ
    SHIFT_NEG_XZ = NMSGalaxyMapConstants.SHIFT_NEG_XZ
    SHIFT_POS_Y = NMSGalaxyMapConstants.SHIFT_POS_Y
    SHIFT_NEG_Y = NMSGalaxyMapConstants.SHIFT_NEG_Y

    def __init__(self):
        """Initialize the galaxy map converter (no setup required)."""
        pass

    def glyphs_to_region_data(self, glyphs: str):
        """
        Convert portal glyph string to galactic coordinates and region data.

        Args:
            glyphs (str): 12-character hexadecimal portal code

        Returns:
            Optional[Dict]: Dictionary with coordinate data, or None if invalid
            Format: {
                "coords_full": "XXXX:YYYY:ZZZZ:SSSS",
                "raw_values": {'x': int, 'y': int, 'z': int}
            }

        Raises:
            ValueError: If glyphs contain non-hex characters

        Example:
            >>> map = NMSGalaxyMap()
            >>> map.glyphs_to_region_data("0123456789AB")
            {
                "coords_full": "07FF:0080:07FF:0123",
                "raw_values": {'x': 2047, 'y': 128, 'z': 2047}
            }
        """
        # Clean and validate input
        glyph_string = glyphs.strip().upper()

        # Check length requirement
        if len(glyph_string) != NMSGalaxyMapConstants.GLYPH_LENGTH:
            return None

        try:
            # Parse glyph components according to format
            # [1:4] = Characters 1-3: Solar System index
            # [4:6] = Characters 4-5: Y coordinate
            # [6:9] = Characters 6-8: Z coordinate
            # [9:12] = Characters 9-11: X coordinate
            s_hex = int(glyph_string[1:4], 16)
            y_hex = int(glyph_string[4:6], 16)
            z_hex = int(glyph_string[6:9], 16)
            x_hex = int(glyph_string[9:12], 16)
        except ValueError:
            # Invalid hex characters
            return None

        # --- Convert hex to signed coordinates ---
        # The game stores coordinates as offset from center (2047 for X/Z, 127 for Y)
        # Values >= shift threshold represent negative coordinates

        # X coordinate conversion
        if x_hex >= self.SHIFT_POS_XZ:
            cx = x_hex - self.SHIFT_POS_XZ  # Positive coordinate
        else:
            cx = x_hex + self.SHIFT_NEG_XZ  # Negative coordinate (wrapped)

        # Z coordinate conversion (same logic as X)
        if z_hex >= self.SHIFT_POS_XZ:
            cz = z_hex - self.SHIFT_POS_XZ
        else:
            cz = z_hex + self.SHIFT_NEG_XZ

        # Y coordinate conversion (different threshold)
        if y_hex >= self.SHIFT_POS_Y:
            cy = y_hex - self.SHIFT_POS_Y
        else:
            cy = y_hex + self.SHIFT_NEG_Y

        return {
            # Format: XXXX:YYYY:ZZZZ:SSSS (16-bit hex values)
            "coords_full": f"{cx:04X}:{cy:04X}:{cz:04X}:{s_hex:04X}",

            # Raw integer values for further processing
            "raw_values": {'x': cx, 'y': cy, 'z': cz}
        }


@dataclass
class AppWidgets:
    """
    Container class for all UI widget objects.

    This dataclass stores references to every widget in the application,
    making them easily accessible throughout the code. Each field corresponds
    to a specific input field in the user interface.

    Attributes:
        name (Text): Base name input
        image (Text): Main image filename input
        builder (Text): Builder's name input
        builderlink (Text): Builder's wiki profile link
        civilized (Text): Civilization name input
        platform (Dropdown): Platform selection (PC, PS4, etc.)
        mode (Dropdown): Game mode selection (Normal, Survival, etc.)
        release (Text): Game release version input
        portalglyphs (Text): Portal glyph code input
        coordinates (Text): Calculated coordinates display
        axes_text (Text): Latitude/Longitude input
        galaxy (Combobox): Galaxy selection with autocomplete
        region (Text): Calculated region name display
        system (Text): Star system name input
        planet (Text): Planet name input
        moon (Text): Moon name input (optional)
        type (Dropdown): Base type selection
        layout (Textarea): Base layout description
        pwr_cond (Dropdown): Power condition selection
        ter_cond (Dropdown): Terrain condition selection
        is_farm (Checkbox): Farming facility checkbox
        is_geobay (Checkbox): Exocraft geobay checkbox
        is_terminal (Checkbox): Trade terminal checkbox
        is_landingpad (Checkbox): Landing pad checkbox
        is_arena (Checkbox): Arena facility checkbox
        is_racetrack (Checkbox): Racetrack checkbox
        start_date (DatePicker): Construction start date
        start_agt (Text): Calculated AGT stardate for start
        finish_date (DatePicker): Construction finish date
        finish_agt (Text): Calculated AGT stardate for finish
        survey_date (DatePicker): Survey date
        survey_agt (Text): Calculated AGT stardate for survey
        gallery_images (Textarea): Gallery image list
        video (Textarea): Video embed codes
        external_links (Textarea): External link list
        feature_checks (Dict[str, Checkbox]): Feature checkbox dictionary
        nearby_checks (Dict[str, Checkbox]): Nearby POI checkbox dictionary
    """

    # Location tab widgets
    name: Text = field(default=None)
    image: Text = field(default=None)
    builder: Text = field(default=None)
    builderlink: Text = field(default=None)
    civilized: Text = field(default=None)
    platform: Dropdown = field(default=None)
    mode: Dropdown = field(default=None)
    release: Text = field(default=None)
    portalglyphs: Text = field(default=None)
    coordinates: Text = field(default=None)
    axes_text: Text = field(default=None)
    galaxy: Combobox = field(default=None)
    region: Text = field(default=None)
    system: Text = field(default=None)
    planet: Text = field(default=None)
    moon: Text = field(default=None)

    # Details tab widgets
    type: Dropdown = field(default=None)
    layout: Textarea = field(default=None)
    pwr_cond: Dropdown = field(default=None)
    ter_cond: Dropdown = field(default=None)
    is_farm: Checkbox = field(default=None)
    is_geobay: Checkbox = field(default=None)
    is_terminal: Checkbox = field(default=None)
    is_landingpad: Checkbox = field(default=None)
    is_arena: Checkbox = field(default=None)
    is_racetrack: Checkbox = field(default=None)

    # Media tab widgets
    start_date: DatePicker = field(default=None)
    start_agt: Text = field(default=None)
    finish_date: DatePicker = field(default=None)
    finish_agt: Text = field(default=None)
    survey_date: DatePicker = field(default=None)
    survey_agt: Text = field(default=None)
    gallery_images: Textarea = field(default=None)
    video: Textarea = field(default=None)
    external_links: Textarea = field(default=None)

    # Feature tab widget collections (populated dynamically)
    feature_checks: dict = field(default_factory=dict)
    nearby_checks: dict = field(default_factory=dict)


class WikiDataModel(BaseModel):
    """
    Pydantic model for validating base data before wiki generation.

    This model ensures all required fields are present and valid before
    attempting to generate wiki markup. It includes custom validators for
    game-specific data formats and requirements.

    Attributes:
        name (str): Base name (required, min 1 character)
        image (str): Image filename (required, min 1 character)
        builder (str): Builder name (required, min 1 character)
        builderlink (Optional[str]): Wiki user page link
        civilized (Optional[str]): Civilization name
        platform (str): Platform (PC, PS4, etc.)
        mode (str): Game mode (Normal, Survival, etc.)
        release (str): Game version
        portalglyphs (str): 12-character hex code
        coordinates (str): Galactic coordinates
        axes_text (str): Latitude/Longitude
        galaxy (str): Galaxy name
        region (str): Region name
        system (str): Star system name
        planet (str): Planet name
        moon (Optional[str]): Moon name (if applicable)
        type (str): Base type
        layout (Optional[str]): Layout description
        pwr_cond (str): Power condition
        ter_cond (str): Terrain condition
        is_farm (bool): Has farming
        is_geobay (bool): Has exocraft geobay
        is_terminal (bool): Has trade terminal
        is_landingpad (bool): Has landing pad
        is_arena (bool): Has arena
        is_racetrack (bool): Has racetrack
        feature_list (List[str]): Selected features
        nearby_list (List[str]): Selected nearby POIs
        start_date (Any): Construction start date
        finish_date (Any): Construction finish date
        survey_date (Any): Survey date
        start_agt (str): AGT stardate for start
        finish_agt (str): AGT stardate for finish
        survey_agt (str): AGT stardate for survey
        gallery_images (Optional[str]): Gallery image list
        video (Optional[str]): Video embed codes
        external_links (Optional[str]): External link list
    """

    # Identity fields
    name: str = Field(min_length=1)
    image: str = Field(min_length=1)
    builder: str = Field(min_length=1)
    builderlink: Optional[str] = ""
    civilized: Optional[str] = "Alliance of Galactic Travellers"

    # Game details
    platform: str
    mode: str
    release: str

    # Location fields
    portalglyphs: str = Field(pattern=r"^[0-9A-F]{12}$")
    coordinates: str
    axes_text: str
    galaxy: str
    region: str
    system: str
    planet: str
    moon: Optional[str] = ""

    # Base details
    type: str
    layout: Optional[str] = ""
    pwr_cond: str
    ter_cond: str

    # Facility flags
    is_farm: bool
    is_geobay: bool
    is_terminal: bool
    is_landingpad: bool
    is_arena: bool
    is_racetrack: bool

    # Feature lists
    feature_list: List[str] = []
    nearby_list: List[str] = []

    # Date fields
    start_date: Any
    finish_date: Any
    survey_date: Any
    start_agt: str
    finish_agt: str
    survey_agt: str

    # Media fields
    gallery_images: Optional[str] = ""
    video: Optional[str] = ""
    external_links: Optional[str] = ""

    @field_validator(
        'platform', 'mode', 'type', 'pwr_cond', 'ter_cond', 'galaxy',
        'region', 'system', 'planet', 'axes_text'
    )
    @classmethod
    def check_not_empty(cls, v: str, info: ValidationInfo) -> str:
        """
        Validate that required fields are not empty or placeholder values.

        Args:
            v (str): Field value to validate
            info (ValidationInfo): Validation context with field name

        Returns:
            str: Validated value

        Raises:
            ValueError: If field is empty or contains placeholder text
        """
        # Check for empty values or dropdown placeholders
        if not v or "- Select" in v:
            raise ValueError(f"{info.field_name} is required.")
        return v


class NmsWikiGenerator:
    """
    Main application class for the No Man's Sky Wiki Generator.

    This class orchestrates the entire application:
    1. Loads external game data
    2. Creates the tabbed user interface
    3. Handles user interactions and calculations
    4. Generates final wiki markup from validated data

    The application runs entirely within Jupyter/Colab notebooks and provides
    a complete GUI for creating wiki pages without manual formatting.

    Attributes:
        data (NMSData): Loaded game data
        widgets (AppWidgets): All UI widget references
        map_logic (NMSGalaxyMap): Glyph to coordinate converter
        generated_content (str): Last generated wiki markup
        jinja_env (jinja2.Environment): Template rendering environment
        jinja_template (jinja2.Template): Wiki template
        tabs (Tab): Main tabbed interface container
        HEADER_STYLE (str): CSS for section headers
        DESC_STYLE (str): CSS for description text
        LABEL_STYLE (dict): Widget label styling
        WIDGET_LAYOUT (Layout): Default widget layout
        TALL_TEXT_LAYOUT (Layout): Layout for textareas
        COL_LAYOUT (Layout): Two-column layout
        FULL_ROW (Layout): Full-width row layout
    """

    # Jinja2 template for wiki markup generation
    # Uses double curly braces escaped as {{ '{{' }} and {{ '}}' }}
    WIKI_TEMPLATE = """{{ '{{' }}Version|{{ release }}{{ '}}' }}
{{ '{{' }}AGT Notice{{ '}}' }}
{{ '{{' }}Base infobox
| name = {{ name }}
| image = {{ image }}
| civilized = {{ civilized }}
| builder = {{ builder }}
| builderlink = {{ builderlink }}
| galaxy = {{ galaxy }}
| region = {{ region }}
| system = {{ system }}
| planet = {{ planet }}
| moon = {{ moon }}
| axes = {{ axes_text }}
| coordinates = {{ coordinates }}
| portalglyphs = {{ '{{' }}Gl/Small|{{ portalglyphs }}{{ '}}' }}
| mode = {{ mode }}
| platform = {{ platform }}
| release = {{ release }}
| farm = {{ 'Y' if is_farm else 'N' }}
| geobay = {{ 'Y' if is_geobay else 'N' }}
| arena = {{ 'Y' if is_arena else 'N' }}
| landingpad = {{ 'Y' if is_landingpad else 'N' }}
| racetrack = {{ 'Y' if is_racetrack else 'N' }}
| terminal = {{ 'Y' if is_terminal else 'N' }}
| type = {{ type }}
{{ '}}' }}
'''{{ name }}''' is a player base.

==Summary==
'''{{ name }}''' is a [[Habitable Base|player base]], located on the [[planet]] [[{{ planet }}]] in the [[{{ system }}]] system.
{% if civilized == 'Alliance of Galactic Travellers' -%}
This base is located in an [[Alliance of Galactic Travellers]] [[star system]].
{% endif -%}
The base is located at {{ latlong_code }}.

{{ base_type_description }}

==Construction Builder==
Constructed by ''{{ builder }}''{% if civilized %} of [[{{ civilized }}]]{% endif %}.

==Layout==
{{ layout }}

==Features==
{% if feature_list -%}
{% for item in feature_list -%}
* [[{{ item }}]]
{% endfor -%}
{% else -%}
None
{%- endif %}

==Nearby Interest==
{% if nearby_list -%}
{% for item in nearby_list -%}
* [[{{ item }}]]
{% endfor -%}
{% else -%}
None
{%- endif %}

==Additional Information==
* Site Construction started approximately [[AGT Stardate]] {{ start_agt }} ({{ start_date_str }}).
* Site Construction finished approximately [[AGT Stardate]] {{ finish_agt }} ({{ finish_date_str }}).
* {{ power_condition_description }}
* The accessibility/terrain regrowth situation is classified as: {{ ter_cond }}
* Documentation based on a site survey conducted on [[AGT Stardate]] {{ survey_agt }} ({{ survey_date_str }}).
* Site Surveyed by ''{{ builder }}''.

==Gallery==
<gallery>
{{ gallery_content }}
</gallery>
{%- if video %}

==Video==
{{ video }}
{%- endif -%}
{%- if external_links %}

==External Links==
{{ external_links }}
{%- endif %}

==AGT Galactic Archives==
{{ '{{' }}AGT Galactic Archive Sync{{ '}}' }}"""

    def __init__(self):
        """
        Initialize the wiki generator application.

        Sets up data loading, UI creation, event handlers, and initial calculations.
        Automatically displays the interface when instantiated in a notebook.
        """
        # Load external game data
        self.data = NMSData()
        self.data.fetch_remote_data()

        # Initialize component containers
        self.widgets = AppWidgets()                    # UI widgets
        self.map_logic = NMSGalaxyMap()               # Coordinate calculator
        self.generated_content = ""                   # Last generated output

        # Set up template engine for wiki markup generation
        self.jinja_env = jinja2.Environment(
            loader=jinja2.BaseLoader(), lstrip_blocks=True
        )
        self.jinja_template = self.jinja_env.from_string(self.WIKI_TEMPLATE)

        # Define UI styling and layouts
        self._define_styles()

        # Build the user interface
        self._setup_ui()

        # Connect event handlers
        self._connect_events()

        # Perform initial calculations (glyph conversion)
        self._on_glyph_change(None)

        # Initialize date fields with current AGT stardates
        for prefix in ['start', 'finish', 'survey']:
            self._on_date_change(
                {'new': getattr(self.widgets, f'{prefix}_date').value}, prefix=prefix
            )

    def _define_styles(self):
        """Define CSS styles and layout configurations for the UI."""

        # Header style for section titles
        self.HEADER_STYLE = (
            "font-weight:bold; font-size:16px; margin-top:20px; "
            "border-bottom:2px solid #00ACC1; padding-bottom:5px; "
            "color:#006064;"
        )

        # Description style for explanatory text below headers
        self.DESC_STYLE = (
            "font-style:italic; font-size:12px; color:#555; "
            "margin-bottom:12px; background-color:#E0F7FA; padding:8px; "
            "border-left:4px solid #00BCD4;"
        )

        # Label styling for widget descriptions
        self.LABEL_STYLE = {'description_width': '140px'}

        # Layout configurations for different widget types
        self.WIDGET_LAYOUT = Layout(width='98%')
        self.TALL_TEXT_LAYOUT = Layout(width='98%', height='140px')
        self.COL_LAYOUT = Layout(width='50%')
        self.FULL_ROW = Layout(width='100%', margin='5px 0')

    def _setup_ui(self):
        """
        Create the main tabbed interface with all six tabs.

        The interface is organized into logical tabs:
        1. Location & Galaxy: Coordinates and system details
        2. Base Identity: Name, builder, and game info
        3. Base Details: Type, layout, and facilities
        4. Features: Base parts and nearby points of interest
        5. Media: Dates, gallery, videos, and links
        6. Generate: Preview and output controls
        """
        # Create tab containers
        self.tabs = Tab([
            self._tab_location(),      # Tab 1: Location data
            self._tab_identity(),      # Tab 2: Base identity
            self._tab_details(),       # Tab 3: Base details
            self._tab_features(),      # Tab 4: Features and POIs
            self._tab_media(),         # Tab 5: Media and dates
            self._tab_generate()       # Tab 6: Generation controls
        ])

        # Set tab titles
        titles = [
            'Location & Galaxy', 'Base Identity', 'Base Details',
            'Features', 'Media', 'Generate'
        ]
        for i, t in enumerate(titles):
            self.tabs.set_title(i, t)

        # Display the interface
        display(self.tabs)

    def _tab_location(self):
        """
        Create the Location & Galaxy tab.

        Contains widgets for:
        - Galaxy selection
        - Portal glyph input (auto-calculates coordinates and region)
        - Star system and planet details
        - Latitude/Longitude

        Returns:
            VBox: Complete location tab layout
        """
        return VBox([
            # Galactic Coordinates Section
            self._header('Galactic Coordinates'),
            self._desc(
                "Enter Galaxy and Glyphs. Coordinates and Region Name "
                "will auto-calculate."
            ),
            self._create_row(
                self._create_widget(
                    Combobox, 'galaxy', 'Galaxy:',
                    options=self.data.galaxies, placeholder="Start typing..."
                ),
                self._create_widget(
                    Text, 'portalglyphs', 'Portal Glyphs:',
                    placeholder='e.g., 207AF89D6D66'
                )
            ),
            self._create_row(
                self._create_widget(
                    Text, 'coordinates', 'Coordinates:', disabled=True
                ),
                self._create_widget(
                    Text, 'region', 'Region Name:',
                    placeholder='(Auto-calculated)', disabled=True
                )
            ),

            # System Details Section
            self._header('System Details'),
            self._desc(
                "Enter Star System and Planet/Moon names from discovery menu."
            ),
            self._create_row(
                self._create_widget(
                    Text, 'system', 'Star System:',
                    placeholder='e.g., Ogtialabi-Vez'
                ),
                self._create_widget(
                    Text, 'planet', 'Planet Name:',
                    placeholder='e.g., New Lennon'
                )
            ),
            self._create_row(
                self._create_widget(
                    Text, 'moon', 'Moon Name:',
                    placeholder='(Leave empty if on planet)'
                ),
                self._create_widget(
                    Text, 'axes_text', 'Lat / Long:',
                    placeholder='e.g., +14.33 / -122.50'
                )
            )
        ], layout=Layout(padding='20px'))

    def _tab_identity(self):
        """
        Create the Base Identity tab.

        Contains widgets for:
        - Base name and main image
        - Builder information and civilization
        - Platform, game mode, and version

        Returns:
            VBox: Complete identity tab layout
        """
        return VBox([
            # Base Identity Section
            self._header('Base Identity'),
            self._desc(
                "Base Name becomes the Wiki Page Title. Image must match an "
                "uploaded filename."
            ),
            self._create_row(
                self._create_widget(
                    Text, 'name', 'Base Name:',
                    placeholder='e.g., AGT Starfall Outpost'
                ),
                self._create_widget(
                    Text, 'image', 'Main Image File:',
                    placeholder='File:Starfall_Base_v1.png'
                )
            ),

            # Builder & Civilization Section
            self._header('Builder & Civilization'),
            self._desc(
                "Provide your in-game username and any specific "
                "Wiki username if different."
            ),
            self._create_row(
                self._create_widget(
                    Text, 'builder', 'Builder Name:',
                    placeholder='In-Game Username'
                ),
                self._create_widget(
                    Text, 'builderlink', 'Wiki User Page:',
                    placeholder='Wiki Username (Optional)'
                )
            ),
            self._create_widget(
                Text, 'civilized', 'Civilization:',
                value='Alliance of Galactic Travellers'
            ),

            # Game Details Section
            self._header('Game Details'),
            self._desc(
                "Select platform, game mode, and the current update version."
            ),
            self._create_row(
                self._create_widget(
                    Dropdown, 'platform', 'Platform:',
                    options=['PC', 'PS4', 'PS5', 'Xbox', 'Switch', 'Mac'],
                    placeholder=True
                ),
                self._create_widget(
                    Dropdown, 'mode', 'Game Mode:',
                    options=[
                        'Normal', 'Survival', 'Permadeath', 'Creative'
                    ],
                    placeholder=True
                )
            ),
            self._create_widget(
                Text, 'release', 'Release Version:', value='Breach'
            )
        ], layout=Layout(padding='20px'))

    def _tab_details(self):
        """
        Create the Base Details tab.

        Contains widgets for:
        - Base type and layout description
        - Power and terrain conditions
        - Facility checkboxes (farm, geobay, etc.)

        Returns:
            VBox: Complete details tab layout
        """
        # Create facility checkbox widgets
        keys = [
            'is_farm', 'is_geobay', 'is_terminal', 'is_landingpad',
            'is_arena', 'is_racetrack'
        ]
        labels = [
            'Farming', 'Geobay', 'Trade Term', 'Landing Pad',
            'Arena', 'Racetrack'
        ]
        facility_checkboxes = [
            self._create_widget(
                Checkbox, key, label, indent=False, layout=Layout(width='auto')
            )
            for key, label in zip(keys, labels)
        ]

        return VBox([
            # Classification Section
            self._header('Classification'),
            self._desc("Select type and describe layout."),
            self._create_widget(
                Dropdown, 'type', 'Base Type:',
                options=sorted(self.data.base_types.keys()),
                placeholder=True
            ),
            self._create_widget(
                Textarea, 'layout', 'Description:',
                placeholder='e.g., Built into a mountain...'
            ),

            # Status Section
            self._header('Status'),
            self._desc("Current power and terrain conditions."),
            self._create_row(
                self._create_widget(
                    Dropdown, 'pwr_cond', 'Power:',
                    options=sorted(self.data.power_map.keys()),
                    placeholder=True
                ),
                self._create_widget(
                    Dropdown, 'ter_cond', 'Terrain:',
                    options=self.data.terrain_conditions,
                    placeholder=True
                )
            ),

            # Facilities Section
            self._header('Facilities'),
            self._desc("Major amenities available."),
            GridBox(
                facility_checkboxes,
                layout=Layout(grid_template_columns="repeat(3, 200px)")
            )
        ], layout=Layout(padding='20px'))

    def _tab_features(self):
        """
        Create the Features tab.

        Contains dynamic checkboxes for:
        - Base features (parts and constructions)
        - Nearby points of interest

        Returns:
            VBox: Complete features tab layout
        """
        # Create feature checkboxes from loaded data
        feature_checkboxes = [
            Checkbox(description=c, layout=Layout(width='auto'), indent=False)
            for c in self.data.features_list
        ]

        # Store references for later access
        self.widgets.feature_checks = {
            c: w for c, w in zip(self.data.features_list, feature_checkboxes)
        }

        # Create nearby POI checkboxes from loaded data
        nearby_checkboxes = [
            Checkbox(description=c, layout=Layout(width='auto'), indent=False)
            for c in self.data.nearby_list
        ]

        # Store references for later access
        self.widgets.nearby_checks = {
            c: w for c, w in zip(self.data.nearby_list, nearby_checkboxes)
        }

        return VBox([
            # Base Features Section
            self._header('Base Features'),
            self._desc("Specific base parts used."),
            GridBox(
                feature_checkboxes,
                layout=Layout(
                    grid_template_columns="repeat(3, 1fr)",
                    height='250px', overflow_y='scroll',
                    border='1px solid #ccc'
                )
            ),

            # Nearby POIs Section
            self._header('Nearby POIs'),
            self._desc("Points of interest within view."),
            GridBox(
                nearby_checkboxes,
                layout=Layout(
                    grid_template_columns="repeat(3, 1fr)",
                    height='250px', overflow_y='scroll',
                    border='1px solid #ccc'
                )
            )
        ], layout=Layout(padding='20px'))

    def _tab_media(self):
        """
        Create the Media tab.

        Contains widgets for:
        - Construction and survey dates (auto-calculates AGT stardates)
        - Gallery images (with caption support)
        - Video embeds and external links

        Returns:
            VBox: Complete media tab layout
        """
        # Default to today's date for date pickers
        today = arrow.utcnow().to('local').date()

        return VBox([
            # Dates Section
            self._header('Dates'),
            self._desc("AGT Stardates calculated automatically."),
            self._create_row(
                self._create_widget(
                    DatePicker, 'start_date', 'Start:', value=today
                ),
                self._create_widget(
                    Text, 'start_agt', 'AGT:', disabled=True
                )
            ),
            self._create_row(
                self._create_widget(
                    DatePicker, 'finish_date', 'Finish:', value=today
                ),
                self._create_widget(
                    Text, 'finish_agt', 'AGT:', disabled=True
                )
            ),
            self._create_row(
                self._create_widget(
                    DatePicker, 'survey_date', 'Survey:', value=today
                ),
                self._create_widget(
                    Text, 'survey_agt', 'AGT:', disabled=True
                )
            ),

            # Media Section
            self._header('Media'),
            self._desc(
                "Gallery Format: File:ImageName.ext|Caption. One per line."
            ),
            self._create_widget(
                Textarea, 'gallery_images', 'Gallery:',
                placeholder='File:Img.png|Caption'
            ),
            self._create_widget(
                Textarea, 'video', 'Videos:',
                layout=Layout(width='98%', height='80px')
            ),
            self._create_widget(
                Textarea, 'external_links', 'Links:',
                layout=Layout(width='98%', height='80px')
            )
        ], layout=Layout(padding='20px'))

    def _tab_generate(self):
        """
        Create the Generate tab.

        Contains:
        - Action buttons (Preview, Copy, Download, Example, Reset)
        - Status display area
        - Output preview area

        Returns:
            VBox: Complete generation tab layout
        """
        # Create action buttons with icons and styles
        self.btn_prev = Button(
            description='Preview', button_style='info', icon='eye'
        )
        self.btn_copy = Button(
            description='Copy Code', button_style='primary', icon='copy'
        )
        self.btn_dl = Button(
            description='Download', button_style='success',
            icon='download', disabled=True  # Disabled until content generated
        )
        self.btn_ex = Button(
            description='Example', button_style='warning', icon='upload'
        )
        self.btn_clr = Button(
            description='Reset', button_style='danger', icon='trash'
        )

        # Output area for previewing generated wiki code
        self.out = Output(
            layout={
                'border': '1px solid #ccc', 'height': '400px',
                'overflow_y': 'scroll', 'padding': '10px'
            }
        )

        # Status display for validation messages
        self.status = HTML()

        return VBox([
            # Controls Section
            self._header('Finalization'),
            self._desc("Validate, Preview, and Copy the final Wiki Code."),
            HBox(
                [
                    self.btn_prev, self.btn_copy, self.btn_dl,
                    self.btn_ex, self.btn_clr
                ],
                layout=Layout(justify_content='center')
            ),
            self.status,

            # Output Section
            self._header('Output'),
            self.out
        ], layout=Layout(padding='20px'))

    def _create_widget(self, cls, key, desc, **kwargs):
        """
        Create a widget and store it in the AppWidgets container.

        This factory method handles widget creation with consistent styling
        and automatically stores references for later access.

        Args:
            cls (Widget class): Type of widget to create (Text, Dropdown, etc.)
            key (str): Attribute name in AppWidgets
            desc (str): Description label for the widget
            **kwargs: Additional widget-specific parameters

        Returns:
            Widget: Created widget instance
        """
        # Determine layout based on widget type
        is_tall = (
            cls == Textarea and key in ['layout', 'gallery_images']
        )
        widget_layout = self.TALL_TEXT_LAYOUT if is_tall else self.WIDGET_LAYOUT

        # Base widget parameters
        widget_params = {
            'description': desc,
            'style': self.LABEL_STYLE,
            'layout': widget_layout
        }

        # Special handling for Dropdown widgets with placeholders
        if 'placeholder' in kwargs and cls == Dropdown:
            # Add placeholder option as first choice
            widget_params['options'] = (
                [f"- Select {desc.replace(':', '')} -"] +
                kwargs.pop('options', [])
            )
            widget_params['value'] = widget_params['options'][0]

        # Apply any additional parameters
        widget_params.update(kwargs)

        # Special handling for Combobox widgets
        if cls == Combobox:
            widget_params['ensure_option'] = False

        # Create widget and store reference
        widget = cls(**widget_params)
        setattr(self.widgets, key, widget)
        return widget

    def _create_row(self, left_widget, right_widget):
        """
        Create a two-column row layout.

        Args:
            left_widget (Widget): Widget for left column
            right_widget (Widget): Widget for right column

        Returns:
            HBox: Horizontal box with two equal-width columns
        """
        return HBox(
            [
                VBox([left_widget], layout=self.COL_LAYOUT),
                VBox([right_widget], layout=self.COL_LAYOUT)
            ],
            layout=self.FULL_ROW
        )

    def _header(self, title):
        """
        Create a styled section header.

        Args:
            title (str): Header text

        Returns:
            HTML: Styled header widget
        """
        return HTML(f"<div style='{self.HEADER_STYLE}'>{title}</div>")

    def _desc(self, text):
        """
        Create a styled description text block.

        Args:
            text (str): Description text

        Returns:
            HTML: Styled description widget
        """
        return HTML(f"<div style='{self.DESC_STYLE}'>{text}</div>")

    def _connect_events(self):
        """Connect event handlers to all interactive widgets."""

        # Connect glyph change handler (triggers coordinate calculation)
        self.widgets.portalglyphs.observe(
            self._on_glyph_change, names='value'
        )

        # Connect galaxy change handler (affects region calculation)
        self.widgets.galaxy.observe(self._on_glyph_change, names='value')

        # Connect date change handlers for all date fields
        for prefix in ['start', 'finish', 'survey']:
            getattr(self.widgets, f'{prefix}_date').observe(
                partial(self._on_date_change, prefix=prefix), names='value'
            )

        # Connect button click handlers
        self.btn_prev.on_click(lambda _button: self._process(False))  # Preview
        self.btn_copy.on_click(lambda _button: self._process(True))   # Copy
        self.btn_clr.on_click(self._clear)                            # Reset
        self.btn_ex.on_click(self._example)                           # Example
        self.btn_dl.on_click(self._download)                          # Download

    def _on_glyph_change(self, change):
        """
        Handle changes to portal glyph input.

        When glyphs are entered or changed, this method:
        1. Validates the glyph format
        2. Calculates galactic coordinates
        3. Generates region name using procedural algorithm

        Args:
            change (dict): Widget change event (contains new value)
        """
        # Get current glyph value and clean it
        glyphs = self.widgets.portalglyphs.value.strip().upper()

        # Validate glyph format (12 hex characters)
        if len(glyphs) != NMSGalaxyMapConstants.GLYPH_LENGTH or not all(x in NMSGalaxyMapConstants.HEX_CHARS for x in glyphs):
            self.widgets.coordinates.value = "Invalid Glyphs" if glyphs else ""
            self.widgets.region.value = ""
            return

        # Convert glyphs to region data
        region_data = self.map_logic.glyphs_to_region_data(glyphs)

        if region_data:
            # Display calculated coordinates
            self.widgets.coordinates.value = region_data['coords_full']

            # Generate and display region name
            self._calc_region(region_data['raw_values'])
        else:
            # Display error for invalid glyphs
            self.widgets.coordinates.value = "Error"
            self.widgets.region.value = ""

    def _calc_region(self, raw):
        """
        Calculate region name from coordinates and galaxy.

        Args:
            raw (dict): Raw coordinate values {'x': int, 'y': int, 'z': int}
        """
        # Get selected galaxy name
        galaxy_name = self.widgets.galaxy.value

        # Check if galaxy exists in data and alphasets are loaded
        if galaxy_name not in self.data.galaxy_to_index or not NMSData.ALPHASETS:
            self.widgets.region.placeholder = "Data Error"
            return

        # Convert coordinates to voxel space (relative to galaxy center)
        # This matches the game's internal coordinate system for name generation
        x = raw['x'] - ByteUtilsConstants.VOXEL_CENTER_XZ
        y = raw['y'] - ByteUtilsConstants.VOXEL_CENTER_Y
        z = raw['z'] - ByteUtilsConstants.VOXEL_CENTER_XZ

        try:
            # Create seed from coordinates and galaxy index
            seed = RegionNameGenerator.create_region_seed(
                x, y, z, self.data.galaxy_to_index[galaxy_name]
            )

            # Generate and display region name
            self.widgets.region.value = RegionNameGenerator.format_name(seed)

        except (IndexError, KeyError, ValueError, TypeError, struct.error):
            # Handle any generation errors
            self.widgets.region.value = "Gen Error"

    def _on_date_change(self, change, prefix):
        """
        Handle date picker changes and calculate AGT stardates.

        AGT (Alliance of Galactic Travellers) stardate format:
        Year + 1716 . Day . Month (YYYY.DD.MM)

        Args:
            change (dict): Widget change event with 'new' date value
            prefix (str): Date field prefix ('start', 'finish', or 'survey')
        """
        if change['new']:
            try:
                # Parse selected date
                arrow_date = arrow.get(change['new'])

                # Calculate AGT stardate: add 1716 to year
                agt_year = arrow_date.year + 1716
                agt_date_str = f"{agt_year:04d}.{arrow_date.day:02d}.{arrow_date.month:02d}"

                # Update corresponding AGT field
                getattr(self.widgets, f'{prefix}_agt').value = agt_date_str
            except Exception:
                # Handle date parsing errors
                getattr(self.widgets, f'{prefix}_agt').value = ""
        else:
            # Clear AGT field if date is cleared
            getattr(self.widgets, f'{prefix}_agt').value = ""

    def _process(self, should_copy):
        """
        Main processing function: validates data and generates wiki markup.

        Args:
            should_copy (bool): If True, copy result to clipboard after generation
        """
        # Update status to show validation in progress
        self.status.value = "Validating..."

        # --- Collect Data from Widgets ---
        widget_data = {}

        # Collect values from all simple widgets (with .value attribute)
        for f in fields(AppWidgets):
            widget = getattr(self.widgets, f.name, None)
            if widget is not None and hasattr(widget, 'value'):
                widget_data[f.name] = widget.value

        # Collect values from feature checkboxes
        widget_data['feature_list'] = sorted([
            k for k, checkbox in self.widgets.feature_checks.items() if checkbox.value
        ])

        # Collect values from nearby POI checkboxes
        widget_data['nearby_list'] = sorted([
            k for k, checkbox in self.widgets.nearby_checks.items() if checkbox.value
        ])

        # Clean glyphs input (uppercase, no spaces)
        widget_data['portalglyphs'] = widget_data['portalglyphs'].strip().upper()

        try:
            # --- Validate Data Using Pydantic Model ---
            validated_model = WikiDataModel.model_validate(widget_data)

            # --- Prepare Template Context ---
            context = validated_model.model_dump()

            # Clean image filename (remove 'File:' prefix if present)
            context['image'] = context['image'].replace('File:', '').strip()

            # Format dates for display (DD-MMM-YYYY format)
            for key in ['start', 'finish', 'survey']:
                date_val = context[f'{key}_date']
                try:
                    context[f'{key}_date_str'] = (
                        arrow.get(date_val).format('DD-MMM-YYYY')
                        if date_val else '?'
                    )
                except Exception:
                    context[f'{key}_date_str'] = '?'

            # Set gallery content (use main image if no gallery specified)
            context['gallery_content'] = (
                context['gallery_images']
                if context['gallery_images'] else f"{context['image']}|{context['name']}"
            )

            # Set template variables from loaded data
            context['latlong_code'] = context['axes_text']
            context['base_type_description'] = self.data.base_types.get(
                context['type'], ''
            )
            context['power_condition_description'] = self.data.power_map.get(
                context['pwr_cond'], ''
            )

            # --- Generate Wiki Markup ---
            self.generated_content = self.jinja_template.render(context)

            # --- Display Results ---
            with self.out:
                clear_output()
                print(self.generated_content)

            # Enable download button now that we have content
            self.btn_dl.disabled = False

            if should_copy:
                # Copy to clipboard using JavaScript (works in Jupyter/Colab)
                import json
                js_content = json.dumps(self.generated_content)
                js = f'navigator.clipboard.writeText({js_content});'
                display(Javascript(js))
                self.status.value = "<b style='color:green'>Copied to clipboard!</b>"
            else:
                self.status.value = "<b style='color:green'>Preview generated.</b>"

        except ValidationError as e:
            # Handle validation errors by displaying them to user
            msg = '<br>'.join(err['msg'] for err in e.errors())
            self.status.value = f"<b style='color:red'>Validation Error: {msg}</b>"

    def _clear(self, _button=None):
        """
        Reset all widgets to their default/empty state.

        Args:
            _button: Button instance (ignored, required by click handler)
        """
        # Clear all widget values
        for field_info in fields(AppWidgets):
            widget = getattr(self.widgets, field_info.name, None)
            if widget and hasattr(widget, 'value'):
                if isinstance(widget, (Text, Textarea, Combobox)):
                    widget.value = ""
                elif isinstance(widget, Dropdown):
                    widget.value = widget.options[0]  # Reset to placeholder
                elif isinstance(widget, Checkbox):
                    widget.value = False
                elif isinstance(widget, DatePicker):
                    # Reset to today's date
                    widget.value = arrow.utcnow().to('local').date()

        # Clear feature checkboxes
        for checkbox in self.widgets.feature_checks.values():
            checkbox.value = False

        # Clear nearby POI checkboxes
        for checkbox in self.widgets.nearby_checks.values():
            checkbox.value = False

        # Clear output area and status
        self.out.clear_output()
        self.status.value = ""
        self.generated_content = ""
        self.btn_dl.disabled = True  # Disable download until new content

    def _example(self, _button=None):
        """
        Fill widgets with example data for demonstration/testing.

        Args:
            _button: Button instance (ignored, required by click handler)
        """
        # Clear existing data first
        self._clear()

        # Fill with example values
        self.widgets.name.value = "AGT Deep Sea"
        self.widgets.image.value = "File:Test.png"
        self.widgets.builder.value = "Traveller"
        self.widgets.galaxy.value = "Euclid"
        self.widgets.portalglyphs.value = "1205D058AC1D"
        self.widgets.system.value = "Alpha"
        self.widgets.planet.value = "Beta"
        self.widgets.axes_text.value = "+10/-10"

        # Set example dropdown values if options exist
        if "Industrial" in self.widgets.type.options:
            self.widgets.type.value = "Industrial"
        if "Sufficiently Powered" in self.widgets.pwr_cond.options:
            self.widgets.pwr_cond.value = "Sufficiently Powered"
        if "No obstruction" in self.widgets.ter_cond.options:
            self.widgets.ter_cond.value = "No obstruction"
        if "Normal" in self.widgets.mode.options:
            self.widgets.mode.value = "Normal"
        if "PC" in self.widgets.platform.options:
            self.widgets.platform.value = "PC"

        self.widgets.release.value = "Breach"

    def _download(self, _button=None):
        """
        Download generated wiki markup as a text file.

        In Google Colab, this triggers browser download.
        In local Jupyter, it saves to local filesystem.

        Args:
            _button: Button instance (ignored, required by click handler)
        """
        try:
            # Try Colab download first
            from google.colab import files
            with open('BaseWiki.txt', 'w') as f:
                f.write(self.generated_content)
            files.download('BaseWiki.txt')

        except ImportError:
            # Fall back to local file save (for Jupyter)
            try:
                with open('BaseWiki.txt', 'w') as f:
                    f.write(self.generated_content)
                self.status.value = "<b style='color:green'>File saved as BaseWiki.txt</b>"
            except IOError as e:
                self.status.value = f"<b style='color:red'>File write error: {e}</b>"


# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    """
    Main execution block for standalone testing.

    When this file is run directly (not imported), it creates and displays
    the wiki generator application. In Jupyter/Colab notebooks, this happens
    automatically when the cell containing NmsWikiGenerator() is executed.
    """
    app = NmsWikiGenerator()
