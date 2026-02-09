"""
No Man's Sky (NMS) Wiki Fauna Page Generator.

This module provides a complete interactive application for generating wiki
pages for fauna (creatures) in the game No Man's Sky. It creates properly
formatted wiki markup by collecting creature data through an intuitive form
interface, performing automatic calculations for coordinates and region names,
and generating the final wiki code that can be copied directly to the NMS wiki.

The application includes:
- Interactive form with multiple tabs for different data categories
- Automatic data fetching from external repositories
- Coordinate calculations from portal glyphs
- Procedural region name generation
- Wiki template rendering with validation
- Example data loading and form reset functionality

Key Classes:
- NMSData: Fetches and manages external game data
- ByteUtils: Handles low-level byte operations for calculations
- ProceduralNameGenerator: Generates procedural names for regions
- RegionNameGenerator: Creates region names from coordinates
- NMSWikiFaunaGenerator: Main application with UI and generation logic
"""

import re
import struct
import os
from dataclasses import dataclass, field, fields

import arrow
import requests
from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, ValidationError, Field
import ipywidgets as widgets
from ipywidgets import (
    Layout, HBox, VBox, Tab, Text, Textarea, Dropdown, Button,
    FloatText, DatePicker, Combobox, Checkbox, HTML, Output
)
from IPython.display import display, clear_output, Javascript


# Regular expression to validate portal glyph format (12 hexadecimal characters)
GLYPH_PATTERN = r"^[0-9A-F]{12}$"

# Constants for voxel coordinate calculations (used in converting glyphs to coordinates)
VOXEL_CENTER_XZ = 0x7FF  # Center point for X and Z axes in the coordinate system
VOXEL_CENTER_Y = 0x7F    # Center point for Y axis (vertical)

# Thresholds and offsets for converting raw glyph values to signed coordinates
X_THRESHOLD = 2049  # Values at or above this become negative coordinates
X_OFFSET = 2047     # Offset to add for negative X/Z coordinates
Y_THRESHOLD = 129   # Values at or above this become negative Y coordinates
Y_OFFSET = 127      # Offset to add for negative Y coordinates

# Default values for form fields
DEFAULT_IMAGE = "nmsMisc_NotAvailable.png"
DEFAULT_CIVILIZATION = "Alliance of Galactic Travellers"
DEFAULT_RELEASE = "Breach"

# Safety limit to prevent infinite loops in name generation
SAFETY_LIMIT = 50

# Maximum length for generated names to prevent excessively long names
NAME_MAX_LENGTH = 64

# Maximum number of consecutive consonants allowed before inserting a vowel
CONSECUTIVE_CONSONANT_LIMIT = 3

# Year offset for converting real dates to in-game AGT stardates
STARDATE_YEAR_OFFSET = 1716

# Threshold value for deciding whether to add adornments to region names
ADORN_THRESHOLD = 0x50

# List of placeholder text values that should be treated as empty selections
PLACEHOLDER_SENTINELS = frozenset([
    'Select Genus Classification...',
    'Select Rarity...',
    'Select Activity Period...',
    'Select Gender...',
    'Select Gender (Optional)...',
    'Select Ecosystem...',
    'Select (Optional)...',
    'Select Platform...',
    'Start typing...',
    'Start typing to search...'
])


class NMSData:
    """
    Manages external game data fetched from remote repositories.

    This class loads various data files needed for the application, including
    creature classifications, diets, behaviors, galaxies, and letter mappings
    for procedural name generation. All data is fetched from GitHub repositories
    to ensure it stays up-to-date with the game.

    Attributes:
        URL_BASE (str): Base URL for all data files
        GENUS_DB (dict): Database of creature genus classifications
        GENUS_OPTIONS (list): Sorted list of genus display names for dropdowns
        DIET_DB (dict): Database of diet types and their classifications
        DIET_OPTIONS (list): Sorted list of diet names for combobox
        GALAXIES (list): Sorted list of galaxy names
        GALAXY_MAP (dict): Mapping from galaxy names to their numeric indices
        GENDER_STANDARD (list): Standard gender options for biological creatures
        GENDER_ROBOT (list): Gender options for robotic/mechanical creatures
        BEHAVIORS (list): List of creature behavior patterns
        NOTES_LIST (list): List of possible scan notes/flavor text
        LETTER_MAP (dict): Mapping for procedural name generation weights
        ALPHASETS (list): Character sets used in procedural name generation
    """

    # Base URL for all data files (stored in a GitHub repository)
    URL_BASE = (
        "https://raw.githubusercontent.com/2A03-Jikuu/"
        "nms-wiki-tool-py/refs/heads/main/datalist"
    )

    # Individual data file URLs
    URL_GENUS = f"{URL_BASE}/genus.json"
    URL_DIETS = f"{URL_BASE}/diets.json"
    URL_GENDERS = f"{URL_BASE}/genders.json"
    URL_BEHAVIORS = f"{URL_BASE}/behaviors.json"
    URL_NOTES = f"{URL_BASE}/notes.json"
    URL_GALAXIES = f"{URL_BASE}/galaxies.json"
    URL_LETTER_MAP = f"{URL_BASE}/letter_map.json"
    URL_ALPHASETS = f"{URL_BASE}/alphasets.json"

    def __init__(self):
        """
        Initializes the data manager and loads all external data.

        All data attributes start as empty and are populated by calling
        _load_external_data(), which fetches JSON files from the remote URLs.
        """
        self.GENUS_DB = {}
        self.GENUS_OPTIONS = []
        self.DIET_DB = {}
        self.DIET_OPTIONS = []
        self.GALAXIES = []
        self.GALAXY_MAP = {}
        self.GENDER_STANDARD = []
        self.GENDER_ROBOT = []
        self.BEHAVIORS = []
        self.NOTES_LIST = []
        self.LETTER_MAP = {}
        self.ALPHASETS = []
        self._load_external_data()

    def _fetch_json(self, url, description):
        """
        Fetches JSON data from a URL with error handling.

        Args:
            url (str): The URL to fetch JSON data from
            description (str): Human-readable description of the data for error messages

        Returns:
            dict or None: The parsed JSON data, or None if fetching or parsing failed

        Raises:
            Prints error messages but doesn't raise exceptions to allow the
            application to continue with partial data
        """
        try:
            # Send HTTP GET request to the URL
            response = requests.get(url)
            # Raise an exception for HTTP errors (4xx or 5xx responses)
            response.raise_for_status()
            try:
                # Parse the response content as JSON
                return response.json()
            except ValueError as e:
                # Handle JSON parsing errors (malformed JSON)
                print(f"Error parsing JSON from {description}: {e}")
                return None
        except Exception as e:
            # Handle network errors, timeouts, etc.
            print(f"Error fetching {description}: {e}")
            return None

    def _load_external_data(self):
        """
        Loads all external data files and populates the class attributes.

        This method sequentially fetches each data file, processes it,
        and stores it in the appropriate attribute. If any file fails to load,
        that specific dataset will remain empty but the application continues.
        """
        # Load genus classification data
        genus_data = self._fetch_json(self.URL_GENUS, "Genus DB")
        if genus_data:
            self.GENUS_DB = genus_data
            # Create sorted list of (display_name, key) tuples for dropdown options
            self.GENUS_OPTIONS = sorted(
                [(v.get('display', k), k) for k, v in self.GENUS_DB.items()],
                key=lambda x: x[0]
            )

        # Load diet data
        diet_data = self._fetch_json(self.URL_DIETS, "Diet DB")
        if diet_data:
            self.DIET_DB = diet_data
            self.DIET_OPTIONS = sorted(self.DIET_DB.keys())

        # Load gender data (separate lists for standard and robotic creatures)
        gender_data = self._fetch_json(self.URL_GENDERS, "Gender DB")
        if gender_data:
            self.GENDER_STANDARD = sorted(gender_data.get('standard', []))
            self.GENDER_ROBOT = sorted(gender_data.get('robot', []))

        # Load behavior patterns
        behaviors_data = self._fetch_json(self.URL_BEHAVIORS, "Behaviors")
        if behaviors_data:
            self.BEHAVIORS = sorted(behaviors_data)

        # Load scan notes/flavor text options
        notes_data = self._fetch_json(self.URL_NOTES, "Notes")
        if notes_data:
            self.NOTES_LIST = sorted(notes_data)

        # Load galaxy data
        galaxy_data = self._fetch_json(self.URL_GALAXIES, "Galaxies")
        if galaxy_data:
            # Extract and sort galaxy names
            self.GALAXIES = sorted([
                g['name'] for g in galaxy_data if g.get('name')
            ])
            # Create mapping from galaxy name to its index (for region name generation)
            self.GALAXY_MAP = {
                g['name']: g['index'] for g in galaxy_data if g.get('name')
            }

        # Load letter mapping for procedural name generation
        letter_map_data = self._fetch_json(self.URL_LETTER_MAP, "Letter Map")
        if letter_map_data:
            # Convert string keys to integers (JSON keys are always strings)
            self.LETTER_MAP = {int(k): v for k, v in letter_map_data.items()}

        # Load alphasets (character sets for name generation)
        alphasets_data = self._fetch_json(self.URL_ALPHASETS, "Alphasets")
        if alphasets_data:
            self.ALPHASETS = alphasets_data
        else:
            # Default to empty alphasets if loading fails
            self.ALPHASETS = [""] * 8


class ByteUtils:
    """
    Provides utility methods for byte-level operations and numeric conversions.

    This class handles the low-level byte manipulation needed for coordinate
    calculations and procedural name generation. It mimics the byte operations
    performed by the game's engine when generating names and processing coordinates.

    Class Attributes:
        SEED_MULTIPLIER (list): Constant multiplier used in seed generation
    """

    # Constant multiplier used in the seed update algorithm
    SEED_MULTIPLIER = [0x99, 0xF8, 0x76, 0x5A]

    @staticmethod
    def parse(val, little_endian=True):
        """
        Converts a hexadecimal string to a list of byte values.

        Args:
            val (str): Hexadecimal string (may have odd length)
            little_endian (bool): If True, reverse byte order (least significant first)

        Returns:
            list: List of integer byte values (0-255)

        Example:
            >>> ByteUtils.parse("1A2B3C", True)
            [60, 43, 26]
        """
        # Ensure the string has even length by adding leading zero if needed
        if len(val) % 2 != 0:
            val = "0" + val
        # Convert each 2-character hex pair to an integer
        res = [int(val[i:i + 2], 16) for i in range(0, len(val), 2)]
        # Reverse for little-endian format (game uses little-endian)
        if little_endian:
            res.reverse()
        return res

    @staticmethod
    def format_short(op1):
        """
        Ensures a byte list has at least 2 bytes (16-bit).

        Args:
            op1 (list): Input byte list

        Returns:
            list: Byte list padded with zeros to length 2 if needed
        """
        res = list(op1)
        while len(res) < 2:
            res.append(0x00)
        return res

    @staticmethod
    def add(op1, op2):
        """
        Adds two byte lists together, handling carry between bytes.

        Args:
            op1 (list): First operand (byte list)
            op2 (list): Second operand (byte list), result is added to this

        Returns:
            list: Result of addition as byte list
        """
        result = list(op2)
        # Add each byte from op1 to the corresponding position in result
        for i in range(len(op1)):
            result = ByteUtils._add_single(op1[i], result, i)
        return result

    @staticmethod
    def _add_single(val, target_list, index):
        """
        Helper method to add a single value at a specific position.

        Args:
            val (int): Value to add (0-255)
            target_list (list): List to add value to
            index (int): Position in list where value should be added

        Returns:
            list: Updated target list with carry propagated if needed
        """
        if index < len(target_list):
            # Add value to existing byte at this position
            total = val + target_list[index]
            # Keep only the lower 8 bits (0-255)
            target_list[index] = total & 0xFF
            # Calculate carry (bits that overflowed beyond 8 bits)
            rem = (total >> 8) & 0xFF
            if rem != 0:
                # Propagate carry to next byte position
                target_list = ByteUtils._add_single(rem, target_list, index + 1)
        else:
            # If position doesn't exist, append the value
            target_list.append(val)
        return target_list

    @staticmethod
    def sub(op1, op2):
        """
        Subtracts op1 from op2 (op2 - op1), handling borrow between bytes.

        Args:
            op1 (list): Value to subtract (byte list)
            op2 (list): Value to subtract from (byte list)

        Returns:
            list: Result of subtraction as byte list
        """
        result = list(op2)
        # Subtract each byte from op1 from the corresponding position in result
        for i in range(len(op1)):
            result = ByteUtils._sub_single(op1[i], result, i)
        return result

    @staticmethod
    def _sub_single(val, target_list, index):
        """
        Helper method to subtract a single value at a specific position.

        Args:
            val (int): Value to subtract (0-255)
            target_list (list): List to subtract from
            index (int): Position in list where subtraction occurs

        Returns:
            list: Updated target list with borrow propagated if needed
        """
        if index < len(target_list):
            # Subtract value from existing byte at this position
            diff = val - target_list[index]
            # Keep only the lower 8 bits (0-255), using two's complement for negative
            target_list[index] = diff & 0xFF
            # Calculate borrow (bits that underflowed)
            rem = (diff >> 8) & 0xFF
            if rem != 0:
                # Propagate borrow to next byte position
                target_list = ByteUtils._sub_single(rem, target_list, index + 1)
        else:
            # If position doesn't exist, append the value (negative case)
            target_list.append(val)
        return target_list

    @staticmethod
    def multiply(op1, op2):
        """
        Multiplies two byte lists, handling overflow between bytes.

        This implements a cross-multiplication algorithm similar to how
        processors multiply multi-byte values.

        Args:
            op1 (list): First factor (byte list)
            op2 (list): Second factor (byte list)

        Returns:
            list: Product as byte list
        """
        result = []
        # Multiply each byte of op1 with each byte of op2
        for i in range(len(op1)):
            rem = 0  # Carry from previous multiplication
            for j in range(len(op2)):
                # Multiply bytes and add any carry from previous step
                raw_prod = (op1[i] * op2[j]) + rem
                # Convert to signed 16-bit (game uses signed arithmetic)
                signed_prd = (raw_prod + 32768) % 65536 - 32768
                # Extract carry (upper 8 bits) and result (lower 8 bits)
                rem = (signed_prd >> 8) & 0xFF
                res = signed_prd & 0xFF
                # Position in result is sum of indices (like polynomial multiplication)
                idx = i + j
                if idx < len(result):
                    # Add to existing value at this position
                    result = ByteUtils._add_single(res, result, idx)
                else:
                    # Create new position
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
        Shift bytes left (like bitwise left shift for multi-byte values).

        Args:
            op1 (list): Byte list to shift
            shift (int): Number of bytes to shift

        Returns:
            list: Shifted byte list, or [0x00] if shift exceeds length
        """
        # Shift left by removing bytes from the beginning
        return op1[:shift] if len(op1) > shift else [0x00]

    @staticmethod
    def shr(op1, shift):
        """
        Shift bytes right (like bitwise right shift for multi-byte values).

        Args:
            op1 (list): Byte list to shift
            shift (int): Number of bytes to shift

        Returns:
            list: Shifted byte list, or empty list if shift exceeds length
        """
        # Shift right by removing bytes from the beginning
        return op1[shift:] if len(op1) > shift else [0x00]

    @staticmethod
    def rotate_left(op1, roll):
        """
        Rotates bytes left (circular shift).

        Args:
            op1 (list): Byte list to rotate
            roll (int): Number of positions to rotate

        Returns:
            list: Rotated byte list
        """
        if not op1:
            return op1
        # Calculate effective rotation (handles rotations larger than list size)
        r = roll % len(op1)
        # Move first 'r' bytes to the end
        return op1[r:] + op1[:r]

    @staticmethod
    def zero_extend(op1, extend):
        """
        Extends a byte list with zeros to specified length.

        Args:
            op1 (list): Byte list to extend
            extend (int): Desired total length

        Returns:
            list: Extended list padded with zeros at the end
        """
        return list(op1) + [0x00] * (extend - len(op1))

    @staticmethod
    def sign_extend(op1, extend):
        """
        Extends a byte list with sign-preserving values.

        For signed numbers, extends with 0xFF if the number is negative
        (highest bit of last byte is 1), otherwise with 0x00.

        Args:
            op1 (list): Byte list to extend
            extend (int): Desired total length

        Returns:
            list: Sign-extended byte list
        """
        result = list(op1)
        # Check if the original value is negative (highest bit of last byte is 1)
        val = 0xFF if (len(op1) > 0 and (op1[-1] >> 7) == 1) else 0x00
        # Pad with the sign-preserving value
        for _ in range(extend - len(op1)):
            result.append(val)
        return result

    @staticmethod
    def logical_op(op1, op2, mode):
        """
        Performs bitwise logical operations on two byte lists.

        Args:
            op1 (list): First operand (byte list)
            op2 (list): Second operand (byte list)
            mode (int): 0 for AND, 1 for OR, 2 for XOR

        Returns:
            list: Result of logical operation as byte list
        """
        # Make both lists the same length by zero-padding the shorter one
        l1, l2 = len(op1), len(op2)
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
            op1 (list): First operand
            op2 (list): Second operand

        Returns:
            list: XOR result as byte list
        """
        return ByteUtils.logical_op(op1, op2, 2)

    @staticmethod
    def and_op(op1, op2):
        """
        Bitwise AND of two byte lists.

        Args:
            op1 (list): First operand
            op2 (list): Second operand

        Returns:
            list: AND result as byte list
        """
        return ByteUtils.logical_op(op1, op2, 0)

    @staticmethod
    def or_op(op1, op2):
        """
        Bitwise OR of two byte lists.

        Args:
            op1 (list): First operand
            op2 (list): Second operand

        Returns:
            list: OR result as byte list
        """
        return ByteUtils.logical_op(op1, op2, 1)

    @staticmethod
    def update_seed(cache, move=1):
        """
        Updates the seed cache using a specific algorithm.

        This mimics the game's procedural generation algorithm for updating
        random seeds used in name generation.

        Args:
            cache (list): Seed cache with two elements [cache0, cache1]
            move (int): Number of times to apply the update

        Returns:
            list: Updated seed cache
        """
        for _ in range(move):
            # Multiply cache[0] by the constant multiplier
            step1 = ByteUtils.multiply(cache[0], ByteUtils.SEED_MULTIPLIER)
            # Add cache[1] to the result
            result = ByteUtils.add(step1, cache[1])
            # Update cache[0] with left-shifted part
            cache[0] = ByteUtils.shl(result, 4)
            # Update cache[1] with right-shifted part
            cache[1] = ByteUtils.shr(result, 4)
        return cache

    @staticmethod
    def to_uint32(arr, offset=0):
        """
        Converts 4 bytes to an unsigned 32-bit integer.

        Args:
            arr (list): Byte list
            offset (int): Starting position in the list

        Returns:
            int: Unsigned 32-bit integer value
        """
        # Extract 4 bytes (pad with zeros if needed)
        chunk = arr[offset:offset + 4]
        while len(chunk) < 4:
            chunk.append(0)
        # Interpret as little-endian unsigned integer
        return struct.unpack('<I', bytes(chunk))[0]

    @staticmethod
    def to_int32(arr, offset=0):
        """
        Converts 4 bytes to a signed 32-bit integer.

        Args:
            arr (list): Byte list
            offset (int): Starting position in the list

        Returns:
            int: Signed 32-bit integer value
        """
        chunk = arr[offset:offset + 4]
        while len(chunk) < 4:
            chunk.append(0)
        # Interpret as little-endian signed integer
        return struct.unpack('<i', bytes(chunk))[0]

    @staticmethod
    def to_int16(arr, offset=0):
        """
        Converts 2 bytes to a signed 16-bit integer.

        Args:
            arr (list): Byte list
            offset (int): Starting position in the list

        Returns:
            int: Signed 16-bit integer value
        """
        chunk = arr[offset:offset + 2]
        while len(chunk) < 2:
            chunk.append(0)
        # Interpret as little-endian signed short
        return struct.unpack('<h', bytes(chunk))[0]

    @staticmethod
    def to_double(arr, offset=0):
        """
        Converts 8 bytes to a double-precision floating point number.

        Args:
            arr (list): Byte list
            offset (int): Starting position in the list

        Returns:
            float: Double-precision floating point value
        """
        chunk = arr[offset:offset + 8]
        while len(chunk) < 8:
            chunk.append(0)
        # Interpret as little-endian double
        return struct.unpack('<d', bytes(chunk))[0]

    @staticmethod
    def to_single(arr, offset=0):
        """
        Converts 4 bytes to a single-precision floating point number.

        Args:
            arr (list): Byte list
            offset (int): Starting position in the list

        Returns:
            float: Single-precision floating point value
        """
        chunk = arr[offset:offset + 4]
        while len(chunk) < 4:
            chunk.append(0)
        # Interpret as little-endian float
        return struct.unpack('<f', bytes(chunk))[0]

    @staticmethod
    def get_bytes_uint32(val):
        """
        Converts an unsigned 32-bit integer to a 4-byte list.

        Args:
            val (int): Unsigned 32-bit integer

        Returns:
            list: 4-byte list in little-endian order
        """
        # Pack integer as little-endian unsigned 32-bit and convert to list
        return list(struct.pack('<I', val))


class HexFormatter:
    """
    Provides formatting utilities for hexadecimal values.
    """

    @staticmethod
    def short_to_formatted_hex(val, trunc):
        """
        Formats a value as hexadecimal with specified truncation.

        Args:
            val (int): Integer value to format
            trunc (int): Number of hex digits to keep (truncates from right)

        Returns:
            str: Formatted hexadecimal string

        Example:
            >>> HexFormatter.short_to_formatted_hex(0x1234, 3)
            "234"
        """
        # Mask to ensure 16-bit value
        val = val & 0xFFFF
        # Format as 4-digit hex and keep only the last 'trunc' digits
        return f"{val:04X}"[-trunc:]


class ProceduralNameGenerator:
    """
    Generates procedural names using the game's algorithm.

    This class implements the exact name generation algorithm used by No Man's Sky
    for generating system and region names. It uses seeded random generation with
    specific rules for vowel insertion and consonant clustering.

    Class Attributes:
        _TINY_DOUBLE_BYTES (list): Constant used in floating point calculations
        VOWELS (frozenset): Set of vowels used in basic vowel checks
        VOWELS_Y (frozenset): Set of vowels including 'y' for consonant checks
    """

    # Constant bytes representing a very small double value (approximately 1e-9)
    _TINY_DOUBLE_BYTES = [0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xF0, 0x3D]

    # Vowel sets for different checks in the name generation algorithm
    VOWELS = frozenset("aeiou")
    VOWELS_Y = frozenset("aeiouy")

    @staticmethod
    def generate_name(cache0, cache1, data_source):
        """
        Generates a procedural name using the game's algorithm.

        Args:
            cache0 (list): Primary seed cache for random generation
            cache1 (list): Secondary cache with generation parameters
            data_source (NMSData): Data source for alphasets and letter mappings

        Returns:
            str: Generated name, or empty string if generation fails

        Note:
            This is a complex algorithm that mimics the game's exact name
            generation process, including vowel insertion rules and consonant
            cluster breaking.
        """
        # Step 1: Get initial character triplet from alphaset
        name = ProceduralNameGenerator.get_characters_from_alphaset(cache0, cache1, data_source)
        if name == "__EMPTY__":
            return ""

        # Step 2: Determine which character selection algorithm to use
        ByteUtils.update_seed(cache0)
        check_op = ByteUtils.zero_extend(ByteUtils.and_op(cache0[0], [0x01]), 2)
        alternate_char_getter = (ByteUtils.to_int16(check_op) != 0)
        ByteUtils.update_seed(cache0)

        # Step 3: Calculate how many additional characters to generate
        step1 = ByteUtils.add(cache1[2], [0x01])
        step2 = ByteUtils.sub(step1, cache1[1])
        step3 = ByteUtils.multiply(step2, cache0[0])
        step5 = ByteUtils.add(ByteUtils.shr(step3, 4), cache1[1])
        register0 = ByteUtils.sub(step5, [0x03])
        limit = ByteUtils.to_int16(ByteUtils.sign_extend(register0, 2))

        # Step 4: Generate additional characters based on weights
        if 0 < limit:
            i, safety = 0, 0
            while i < limit:
                ByteUtils.update_seed(cache0)
                # Get the last 3 characters as context for next character selection
                sub_str = name[i: i + 3]
                alphaset_idx = cache1[0][0] if cache1[0] else 0
                char_weights = ProceduralNameGenerator.get_string_weights(sub_str, alphaset_idx, data_source)

                # Generate random value for character selection
                val_u32 = ByteUtils.to_uint32(cache0[0])
                tiny_dbl = ByteUtils.to_double(ProceduralNameGenerator._TINY_DOUBLE_BYTES)
                target = float(val_u32 * tiny_dbl)

                if char_weights is None:
                    # No weights available, backtrack and try again
                    i -= 1
                    safety += 1
                    if safety > SAFETY_LIMIT:
                        # Prevent infinite loop if we can't find valid characters
                        break
                else:
                    safety = 0
                    if alternate_char_getter:
                        # Use alternate algorithm for character selection
                        target *= (len(char_weights) - 1)
                        b_tgt = list(struct.pack('<f', target))
                        op_and = ByteUtils.and_op(b_tgt, [0x00, 0x00, 0x00, 0x80])
                        op = ByteUtils.or_op(op_and, [0x00, 0x00, 0x00, 0x3F])
                        index = int(ByteUtils.to_single(op) + target)
                    else:
                        # Use primary algorithm: cumulative weight selection
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
                if len(name) > NAME_MAX_LENGTH - 1:
                    name = name[:NAME_MAX_LENGTH]
                i += 1

        if not name:
            return ""

        # Step 5: Apply vowel insertion rules for difficult consonant clusters

        # Check beginning of name for consonant clusters that need vowels
        first, second = name[0], name[1] if len(name) > 1 else ''
        skip_vowel_insertion = False
        if (first not in ProceduralNameGenerator.VOWELS) and (second not in ProceduralNameGenerator.VOWELS):
            cond1 = first != 's' or second not in "hklmnprtwy"
            if cond1:
                # Check specific consonant pairs that don't need vowels
                cond_list = [
                    (second == 'h' and first in "ctw"),
                    (second == 'l' and first in "bcfgps"),
                    (second == 'r' and first in "bcdfgkpt"),
                    (second == 'w' and first in "dgt"),
                    (second == 'y' and first in "hmr")
                ]
                if any(cond_list):
                    skip_vowel_insertion = True
                if not skip_vowel_insertion:
                    # Insert vowel between these two consonants
                    name = ProceduralNameGenerator.insert_vowel(name, cache0, 1)

        # Check end of name for consonant clusters that need vowels
        ult, penult = name[-1], name[-2] if len(name) > 1 else ''
        if len(name) > 1 and (penult != 'g' or ult in ProceduralNameGenerator.VOWELS):
            idx = len(name) - 1
            # Check specific ending consonant pairs that need vowels
            cond_list = [
                (ult == 'b' and penult in "gn"),
                (ult == 'd' and penult in "bdfghkmpst"),
                (ult == 'g' and penult == 'l'),
                (ult == 'p' and penult in "bdhkt"),
                (ult == 'r' and penult in "bfg"),
                (ult == 't' and penult == 'g'),
                (ult == 'w' and penult not in ProceduralNameGenerator.VOWELS)
            ]
            if any(cond_list):
                # Insert vowel before the last consonant
                name = ProceduralNameGenerator.insert_vowel(name, cache0, idx)

        # Step 6: Break up long consonant clusters
        consonant_cluster_index = ProceduralNameGenerator.get_consecutive_consonants(name)
        if consonant_cluster_index != -1:
            ByteUtils.update_seed(cache0)
            # Calculate where to insert the vowel (random offset)
            mult = ByteUtils.multiply(cache0[0], [0x03])
            shr = ByteUtils.shr(mult, 4)
            add = ByteUtils.add(shr, [0x01])
            offset = ByteUtils.to_int32(ByteUtils.zero_extend(add, 4))
            name = ProceduralNameGenerator.insert_vowel(name, cache0, consonant_cluster_index + offset)

        return name

    @staticmethod
    def get_characters_from_alphaset(cache0, cache1, data_source):
        """
        Gets the initial character triplet from the alphaset.

        Args:
            cache0 (list): Seed cache for random selection
            cache1 (list): Cache containing alphaset index
            data_source (NMSData): Data source containing alphasets

        Returns:
            str: 3-character starting string, or "__EMPTY__" if unavailable
        """
        ByteUtils.update_seed(cache0)
        # Get which alphaset to use (determined by game logic)
        idx = cache1[0][0] if cache1[0] else 0
        if not data_source.ALPHASETS:
            return "__EMPTY__"
        if idx >= len(data_source.ALPHASETS):
            idx = 0
        alphaset_str = data_source.ALPHASETS[idx]
        if not alphaset_str:
            return "__EMPTY__"

        # Calculate random starting position in the alphaset
        length_bytes = ByteUtils.get_bytes_uint32(len(alphaset_str) // 3)
        register0 = ByteUtils.multiply(cache0[0], length_bytes)
        shr_reg = ByteUtils.shr(register0, 4)
        register1 = ByteUtils.format_short(ByteUtils.multiply(shr_reg, [0x03]))
        start = ByteUtils.to_int16(register1)
        end = ByteUtils.to_int16(ByteUtils.add(register1, [0x03]))
        # Extract 3-character sequence
        return alphaset_str[start:end]

    @staticmethod
    def get_string_weights(s, alphaset, data_source):
        """
        Gets character weight data for a given string context.

        Args:
            s (str): String context (last few characters)
            alphaset (int): Which alphaset to use
            data_source (NMSData): Data source with letter mappings

        Returns:
            list or None: List of (character, weight) tuples, or None if not found
        """
        if not data_source.LETTER_MAP or alphaset not in data_source.LETTER_MAP:
            return None
        subset = data_source.LETTER_MAP[alphaset]
        if not s or s[0] not in subset:
            return None
        # Recursively search through the nested structure
        return ProceduralNameGenerator.recursive_search(subset[s[0]], s)

    @staticmethod
    def recursive_search(arr, s):
        """
        Recursively searches the letter map structure for weight data.

        Args:
            arr (list): Current level of the nested letter map
            s (str): String to search for

        Returns:
            list or None: Weight data if found, None otherwise
        """
        result, i = None, 0
        # Linear search through array until found or exhausted
        while result is None and i < len(arr):
            item = arr[i]
            if len(item) > 2:
                type_code, val = item[2], item[0]
                if type_code == "ja":
                    # Compare string bytes to determine which branch to follow
                    s_bytes = ByteUtils.zero_extend(list(s.encode('utf-8')), 4)
                    val_b = ByteUtils.zero_extend(list(str(val).encode('utf-8')), 4)
                    if ByteUtils.to_int32(s_bytes) > ByteUtils.to_int32(val_b):
                        result = ProceduralNameGenerator.recursive_search(item[1], s)
                elif type_code == "jz" and str(val) == s:
                    # Found exact match, extract weight data
                    return [(w.get("Item1"), float(w.get("Item2", 0))) for w in item[1]]
            i += 1
        return result

    @staticmethod
    def insert_vowel(name, seed, index):
        """
        Inserts a vowel at the specified position in the name.

        Args:
            name (str): Current name
            seed (list): Seed cache for random vowel selection
            index (int): Position to insert vowel

        Returns:
            str: Name with vowel inserted
        """
        ByteUtils.update_seed(seed)
        # Randomly select which vowel to insert (a, e, i, o, or u)
        calc = ByteUtils.shr(ByteUtils.multiply(seed[0], [0x05]), 4)
        if calc and calc[0] < 5:
            if index <= len(name):
                return name[:index] + "aeiou"[calc[0]] + name[index:]
        return name

    @staticmethod
    def get_consecutive_consonants(name):
        """
        Finds positions with too many consecutive consonants.

        Args:
            name (str): Name to check

        Returns:
            int: Index where consonant cluster starts, or -1 if none found
        """
        consonance = 0
        for i in range(len(name)):
            if consonance < CONSECUTIVE_CONSONANT_LIMIT:
                # Count consecutive consonants
                if name[i] not in ProceduralNameGenerator.VOWELS:
                    consonance += 1
                else:
                    consonance = 0
            else:
                # Found at least limit consonants, check if next is also consonant
                if name[i] not in ProceduralNameGenerator.VOWELS_Y:
                    # Need to break this cluster
                    return i - 3
                else:
                    consonance = 0
        return -1


class RegionNameGenerator:
    """
    Generates procedural region names from coordinates.

    This class uses the game's algorithm to generate region names (like
    "Sea of Abandoned Stars") based on galactic coordinates and portal glyphs.

    Class Attributes:
        PROC_ADORNMENTS (list): List of region name templates with placeholders
    """

    # Templates for region name adornments (suffixes/prefixes)
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
        Creates a seed value from coordinates for region name generation.

        Args:
            x (int): X coordinate (signed)
            y (int): Y coordinate (signed)
            z (int): Z coordinate (signed)
            galaxy (int): Galaxy index

        Returns:
            list: Byte list seed for name generation
        """
        # Format each component as hex with specific digit counts
        s_gal = HexFormatter.short_to_formatted_hex(galaxy, 2)
        s_y = HexFormatter.short_to_formatted_hex(y, 2)
        s_z = HexFormatter.short_to_formatted_hex(z, 3)
        s_x = HexFormatter.short_to_formatted_hex(x, 3)
        # Concatenate and parse as bytes
        return ByteUtils.parse(s_gal + s_y + s_z + s_x)

    @staticmethod
    def format_name(seed, data_source):
        """
        Generates a region name from a seed value.

        Args:
            seed (list): Byte list seed
            data_source (NMSData): Data source for name generation

        Returns:
            str: Generated region name, or "Unknown Region" on failure
        """
        # Initialize seed caches for the name generation algorithm
        cache0, cache1 = [[], []], [[0x00], [0x06], []]

        # Step 1: Transform seed through series of operations
        register0 = ByteUtils.shr(seed, 4)
        if register0:
            register0[0] //= 2
        xor_res = ByteUtils.xor(register0, seed)

        # Step 2: Apply multiplication and XOR transformations
        mult_arr_1 = [0xD7, 0x31, 0xBD, 0x2C, 0x48, 0x81, 0xDD, 0x64]
        register0 = ByteUtils.multiply(xor_res, mult_arr_1)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        xor2 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)

        # Step 3: Second transformation pass
        mult_arr_2 = [0x97, 0x29, 0x61, 0x13, 0xC6, 0xA5, 0x6A, 0xE3]
        register0 = ByteUtils.multiply(xor2, mult_arr_2)[:8]
        val_u32 = ByteUtils.to_uint32(ByteUtils.shr(register0, 4)) // 2
        register0 = ByteUtils.xor(ByteUtils.get_bytes_uint32(val_u32), register0)

        # Step 4: Final seed preparation
        shl4 = ByteUtils.shl(register0, 4)
        xor_mid = ByteUtils.xor(ByteUtils.rotate_left(shl4, 2), ByteUtils.shr(register0, 4))
        cache0[1] = ByteUtils.xor(xor_mid, shl4)
        cache0[0] = shl4

        # Ensure cache[0] is not zero
        if ByteUtils.to_int32(cache0[0]) == 0:
            cache0[0] = ByteUtils.add(cache0[0], [0x01])
        ByteUtils.update_seed(cache0)

        # Step 5: Set generation parameters
        calc_len = ByteUtils.shr(ByteUtils.multiply(cache0[0], [0x04]), 4)
        cache1[2] = ByteUtils.add(calc_len, [0x06])

        # Step 6: Generate base name
        name = ProceduralNameGenerator.generate_name(cache0, cache1, data_source)
        if not name:
            return "Unknown Region"
        if "[" in name:
            # Name already contains brackets (special case), return as-is
            return name
        # Capitalize first letter
        name = name[0].upper() + name[1:]

        # Step 7: Randomly decide whether to add adornment (suffix/prefix)
        ByteUtils.update_seed(cache0)
        mult_check = ByteUtils.multiply(cache0[0], [0x64])
        should_adorn = ByteUtils.shr(mult_check, 4)[0] < ADORN_THRESHOLD

        if should_adorn:
            ByteUtils.update_seed(cache0)
            # Randomly select which adornment template to use
            idx_cal = ByteUtils.multiply(cache0[0], [0x14])
            idx = ByteUtils.shr(idx_cal, 4)[0]
            if idx < len(RegionNameGenerator.PROC_ADORNMENTS):
                adornment = RegionNameGenerator.PROC_ADORNMENTS[idx]
                # Replace placeholder with generated name
                name = adornment.replace("%NAME%", name)
        return name


@dataclass
class AppWidgets:
    """
    Container class for all UI widgets in the application.

    This dataclass holds references to all interactive widgets, organized by
    their purpose (identity, physiology, location, etc.). Using a dataclass
    makes it easy to access widgets by name and ensures type hints work properly.

    Attributes:
        Each attribute represents a widget for a specific form field.
        Names correspond to the field names in the wiki template.
    """

    # Identity tab widgets
    name: Text = field(default=None)
    orgname: Text = field(default=None)
    genus: Dropdown = field(default=None)
    othername: Text = field(default=None)
    rarity: Dropdown = field(default=None)
    activity: Dropdown = field(default=None)

    # Physiology tab widgets
    diet: Combobox = field(default=None)
    diettype: Text = field(default=None)
    gender1: Dropdown = field(default=None)
    gender2: Dropdown = field(default=None)
    ecosystem: Dropdown = field(default=None)
    behaviour: Combobox = field(default=None)
    produces: Text = field(default=None)
    predator: Checkbox = field(default=None)
    notes: Combobox = field(default=None)
    appearance: Textarea = field(default=None)
    summarynote: Text = field(default=None)
    addnote: Textarea = field(default=None)
    discoveryinfo: Textarea = field(default=None)

    # Location tab widgets
    galaxy: Combobox = field(default=None)
    region: Text = field(default=None)
    system: Text = field(default=None)
    planet: Text = field(default=None)
    moon: Text = field(default=None)
    hemisphere: Dropdown = field(default=None)
    glyphs: Text = field(default=None)
    coordinates: Text = field(default=None)
    locnotes: Text = field(default=None)
    discovered: Text = field(default=None)
    discoveredlink: Text = field(default=None)
    discovered_on: DatePicker = field(default=None)
    agt_stardate: Text = field(default=None)
    civilized: Text = field(default=None)
    platform: Dropdown = field(default=None)

    # Biometrics tab widgets
    gender1w: FloatText = field(default=None)
    gender1h: FloatText = field(default=None)
    gender1note: Text = field(default=None)
    gender2w: FloatText = field(default=None)
    gender2h: FloatText = field(default=None)
    gender2note: Text = field(default=None)
    genderdw: FloatText = field(default=None)
    genderdh: FloatText = field(default=None)
    genderdnote: Text = field(default=None)

    # Media tab widgets
    image: Text = field(default=None)
    gallery_text: Textarea = field(default=None)
    footertitle: Text = field(default=None)
    footer: Textarea = field(default=None)
    release: Text = field(default=None)


class WikiDataModel(BaseModel):
    """
    Pydantic model for validating and structuring wiki data.

    This model defines the schema for all data fields that go into the wiki
    template. It ensures data types are correct and required fields are present
    before template rendering.

    Attributes:
        Each field corresponds to a variable in the wiki template with its
        default value. The 'name' field is required (must not be empty).
    """

    # Identity fields
    activity: str = ""
    addnote: str = ""
    agt_stardate: str = ""
    appearance: str = ""
    behaviour: str = ""
    civilized: str = ""
    coordinates: str = ""
    diet: str = ""
    diettype: str = ""
    discovered: str = ""
    discovered_on: str = ""
    discoveredlink: str = ""
    discoveryinfo: str = ""
    ecosystem: str = ""
    footer: str = ""
    footertitle: str = ""
    galaxy: str = ""
    gallery_text: str = ""
    gender1: str = ""
    gender1h: str = ""
    gender1note: str = ""
    gender1w: str = ""
    gender2: str = ""
    gender2h: str = ""
    gender2note: str = ""
    gender2w: str = ""
    genderdh: str = ""
    genderdnote: str = ""
    genderdw: str = ""
    genus: str = ""
    glyphs: str = ""
    heightmax: str = ""
    hemisphere: str = ""
    image: str = ""
    locnotes: str = ""
    moon: str = ""
    # Required field - creature name cannot be empty
    name: str = Field(..., min_length=1)
    notes: str = ""
    orgname: str = ""
    othername: str = ""
    planet: str = ""
    platform: str = ""
    predator: str = ""
    produces: str = ""
    rarity: str = ""
    region: str = ""
    release: str = ""
    summarynote: str = ""
    system: str = ""
    weightmax: str = ""


class NMSWikiFaunaGenerator:
    """
    Main application class for generating No Man's Sky fauna wiki pages.

    This class orchestrates the entire application:
    - Creates and manages the interactive UI
    - Handles user input and form logic
    - Performs coordinate calculations and region name generation
    - Validates data and renders the final wiki template
    - Provides example loading and form reset functionality

    Attributes:
        WIKI_TEMPLATE (str): Jinja2 template for the wiki page
        EXAMPLE_DATA (dict): Sample data for demonstration/testing
        data (NMSData): Instance of data manager
        widgets (AppWidgets): Container for all UI widgets
        generated_content (str): Last successfully generated wiki markup
        jinja_env (Environment): Jinja2 environment for template rendering
        status_text (HTML): Widget for displaying status messages
    """

    # Wiki template using Jinja2 syntax with double curly braces escaped
    WIKI_TEMPLATE = """{{ '{{' }}PAGECreatureEnhanced
| name = {{ name }}
| orgname = {{ orgname }}
| othername = {{ othername }}
| image = {{ image }}
| bait =
| gender1 = {{ gender1 }}
| gender2 = {{ gender2 }}
| behaviour = {{ behaviour }}
| hemisphere = {{ hemisphere }}
| ecosystem = {{ ecosystem }}
| activity = {{ activity }}
| rarity = {{ rarity }}
| diet = {{ diet }}
| weightmax = {{ weightmax }}
| heightmax = {{ heightmax }}
| notes = {{ notes }}
| genus = {{ genus }}
| produces = {{ produces }}
| catalogue =
| galaxy = {{ galaxy }}
| region = {{ region }}
| system = {{ system }}
| planet = {{ planet }}
| moon = {{ moon }}
| coordinates = {{ coordinates }}
| glyphs = {{ glyphs }}
| civstub = {{ '{{' }}AGT Notice}}
| civilized = {{ civilized }}
| discovered = {{ discovered }}
| discoveredlink = {{ discoveredlink }}
| discovered_on = {{ discovered_on }}
| agtstardate = {{ agt_stardate }}
| researchteam = {{ civilized }}
| platform = {{ platform }}
| mode = Normal
| release = {{ release }}
| civimage = AGT-Bzoological research.png
| civlabel = AGT Bureau of Zoological Research
| diettype= {{ diettype }}
| summarynote = {{ summarynote }}
| extinct =
| predator = {{ predator }}
| appearance = {{ appearance }}
| addnote = {{ addnote }}
| discoveryinfo = {{ discoveryinfo }}
| locnotes = {{ locnotes }}
| gender1w = {{ gender1w }}
| gender1h = {{ gender1h }}
| gender1note = {{ gender1note }}
| gender2w = {{ gender2w }}
| gender2h = {{ gender2h }}
| gender2note = {{ gender2note }}
| genderdw = {{ genderdw }}
| genderdh = {{ genderdh }}
| genderdnote = {{ genderdnote }}
| footertitle = {{ footertitle }}
| footer = {{ footer }}
}}
==Gallery==
<gallery>
{{ gallery_text }}
</gallery>

==AGT Galactic Archives==
{{ '{{' }}AGT Galactic Archive Sync}}"""

    # Example data that can be loaded into the form for demonstration
    EXAMPLE_DATA = {
        'name': 'B. Rookcairaleus',
        'orgname': 'B. Rookcairaleus',
        'othername': '',
        'image': 'AGT Ontiniangp Nexus-(BR)-Fauna-01a.jpg',
        'gender1': 'Unknown',
        'gender2': '',
        'behaviour': 'Submissive',
        'hemisphere': '',
        'ecosystem': 'Ground',
        'activity': 'Always Active',
        'rarity': 'Rare',
        'diet': 'Processes dirt',
        'notes': 'Asexual reproduction',
        'genus': 'Bos',
        'galaxy': 'Ontiniangp',
        'region': '',
        'system': 'AGT Ontiniangp Embassy',
        'planet': 'AGT Ontiniangp Nexus',
        'moon': '',
        'coordinates': '043D:0072:0D44:005F',
        'glyphs': '105FF3545C3E',
        'civilized': 'Alliance of Galactic Travellers',
        'discovered': '',
        'discoveredlink': 'celab99',
        'discovered_on': '3-Nov-2025',
        'platform': '',
        'release': 'Breach',
        'summarynote': '',
        'predator': False,
        'appearance': '',
        'addnote': '',
        'discoveryinfo': '',
        'locnotes': '',
        'gender1w': 276.8,
        'gender1h': 5.7,
        'gender1note': '',
        'gender2w': 0.0,
        'gender2h': 0.0,
        'gender2note': '',
        'genderdw': 273.2,
        'genderdh': 5.7,
        'genderdnote': '',
        'footertitle': '',
        'footer': '',
        'gallery_text': (
            "File:AGT Ontiniangp Nexus-(BR)-Fauna-01b.jpg| Discovery Menu\n"
            "File:AGT Ontiniangp Nexus-(BR)-Fauna-01c.jpg| Unknown gender scan"
        )
    }

    def __init__(self):
        """
        Initializes the application, sets up UI, and loads data.

        This constructor:
        1. Creates data manager instance
        2. Initializes widget container
        3. Sets up Jinja2 environment
        4. Defines UI styles and layouts
        5. Creates all UI components
        6. Connects event handlers
        7. Displays the application
        """
        self.data = NMSData()
        self.widgets = AppWidgets()
        self.generated_content = ""
        self.jinja_env = Environment(autoescape=False)
        self.status_text = None

        # Setup UI components
        self._define_styles_and_layouts()
        self._setup_ui()
        self._connect_events()

        # Initialize form state
        self._update_stardate_ui(None)
        self._handle_genus_change({'new': ''})

    def _define_styles_and_layouts(self):
        """
        Defines CSS styles and layout configurations for UI widgets.

        This method sets up consistent visual styling and spacing for
        all UI components to create a polished, user-friendly interface.
        """
        # Header style for section titles
        self.HEADER_STYLE = (
            "font-weight:bold; font-size:16px; margin-top:20px; "
            "border-bottom:2px solid #00ACC1; padding-bottom:5px; "
            "color:#006064;"
        )
        # Description style for explanatory text below headers
        self.DESC_STYLE = (
            "font-style:italic; font-size:12px; color:#555; "
            "margin-bottom:12px; line-height:1.4em; "
            "background-color:#E0F7FA; padding:8px; "
            "border-left:4px solid #00BCD4; border-radius:4px;"
        )
        # Label styling for form fields
        self.LABEL_STYLE = {'description_width': '140px'}
        # Main widget layout (most widgets use this)
        self.WIDGET_LAYOUT = Layout(width='98%')
        # Special layout for gallery textarea (taller)
        self.GALLERY_LAYOUT = Layout(width='100%', height='180px')
        # Layout for two-column rows (half width each)
        self.COL_LAYOUT = Layout(width='50%')
        # Layout for full-width rows
        self.FULL_ROW = Layout(width='100%', margin='5px 0')
        # Layout for grid input fields in biometrics tab
        self.GRID_INPUT_LAYOUT = Layout(width='90%')

    def _setup_ui(self):
        """
        Creates and assembles all UI components.

        This method builds the tabbed interface by creating each tab
        individually and then combining them into the main display.
        """
        # Create individual tabs
        self._create_tab_identity()
        self._create_tab_physiology()
        self._create_tab_location()
        self._create_tab_biometrics()
        self._create_tab_media()
        self._create_tab_generate()

        # Combine tabs into tab container
        tabs_children = [self.t1, self.t2, self.t3, self.t4, self.t5, self.t6]
        self.tabs = Tab(children=tabs_children)
        # Set tab titles
        titles = ['Identity', 'Physiology', 'Location', 'Measurements', 'Media', 'Generate']
        for i, t in enumerate(titles):
            self.tabs.set_title(i, t)

        # Display the entire application
        display(self.tabs)

    def _create_tab_identity(self):
        """
        Creates the 'Identity' tab with creature identification fields.

        This tab contains basic creature information: name, genus, rarity,
        activity period, and diet information.
        """
        self.t1 = VBox([
            self._header('Creature Identification'),
            self._desc("Enter basic identification details from the Discovery Menu. The <b>Genus</b> selection provides both scientific and common names for easier identification."),
            # Row 1: Current and original names
            self._two_col_row(
                self._create_widget(Text, 'name', 'Current Name', placeholder='e.g. B. Diplocea'),
                self._create_widget(Text, 'orgname', 'Original Name', placeholder='(Only if renamed)')
            ),
            # Row 2: Genus classification and alias
            self._two_col_row(
                self._create_widget(Dropdown, 'genus', 'Genus', options=self.data.GENUS_OPTIONS, placeholder='Select Genus Classification...'),
                self._create_widget(Text, 'othername', 'Alias/Nickname', placeholder='(Optional)')
            ),
            # Row 3: Rarity and activity period
            self._two_col_row(
                self._create_widget(Dropdown, 'rarity', 'Rarity', options=['Common', 'Uncommon', 'Rare'], placeholder='Select Rarity...'),
                self._create_widget(Dropdown, 'activity', 'Activity', options=['Diurnal', 'Nocturnal', 'Always Active', 'Mostly Diurnal', 'Mostly Nocturnal'], placeholder='Select Activity Period...')
            ),
            self._header('Diet & Nutrition'),
            self._desc("Choose the specific <b>Diet Source</b> observed. The generalized <b>Diet Class</b> will be automatically determined."),
            # Row 4: Diet source and auto-calculated diet class
            self._two_col_row(
                self._create_widget(Combobox, 'diet', 'Diet Source', options=self.data.DIET_OPTIONS, placeholder='Start typing to search...'),
                self._create_widget(Text, 'diettype', 'Diet Class', disabled=True, placeholder='(Auto-determined from Diet Source)')
            )
        ], layout=Layout(padding='20px'))

    def _create_tab_physiology(self):
        """
        Creates the 'Physiology' tab with biological trait fields.

        This tab contains information about creature biology: gender,
        ecosystem, behavior, produces, predator status, and descriptive notes.
        """
        self.t2 = VBox([
            self._header('Biological Analysis'),
            self._desc("Configure biological traits. Select the appropriate <b>Ecosystem</b> and <b>Behaviour</b>. Note: Gender options vary between biological and mechanical species."),
            # Row 1: Gender selections
            self._two_col_row(
                self._create_widget(Dropdown, 'gender1', 'Gender 1', options=self.data.GENDER_STANDARD, placeholder='Select Gender...'),
                self._create_widget(Dropdown, 'gender2', 'Gender 2', options=self.data.GENDER_STANDARD, placeholder='Select Gender (Optional)...')
            ),
            # Row 2: Ecosystem and behavior
            self._two_col_row(
                self._create_widget(Dropdown, 'ecosystem', 'Ecosystem', options=['Ground', 'Underground', 'Underwater', 'Flying'], placeholder='Select Ecosystem...'),
                self._create_widget(Combobox, 'behaviour', 'Behaviour', options=self.data.BEHAVIORS, placeholder='Type to search...')
            ),
            # Row 3: Produces (auto-determined) and predator checkbox
            self._two_col_row(
                self._create_widget(Text, 'produces', 'Produces', disabled=True, placeholder='(Auto-determined from Genus)'),
                self._create_widget(Checkbox, 'predator', 'Predator (Attacks player/creatures)')
            ),
            self._header('Observations'),
            self._desc("Input descriptive data. <b>Scan Notes</b> must be the specific flavor text from the analysis view."),
            # Descriptive fields
            self._create_widget(Combobox, 'notes', 'Scan Notes', options=self.data.NOTES_LIST, placeholder='Select or Type Scan Text...'),
            self._create_widget(Textarea, 'appearance', 'Visual Description', placeholder='Describe head, body, colors...'),
            self._create_widget(Text, 'summarynote', 'Summary Sentence', placeholder='Brief intro text...'),
            self._create_widget(Textarea, 'addnote', 'Additional Notes', placeholder='Enter any other details...'),
            self._create_widget(Textarea, 'discoveryinfo', 'Discovery Menu Text', placeholder='(Optional) Copy exact text...')
        ], layout=Layout(padding='20px'))

    def _create_tab_location(self):
        """
        Creates the 'Location' tab with galactic coordinate fields.

        This tab contains information about where the creature was discovered:
        galaxy, region, system, planet, moon, coordinates, glyphs, and
        discovery metadata.
        """
        self.t3 = VBox([
            self._header('Galactic Coordinates'),
            self._desc("Region and Star System details. Entering <b>Portal Glyphs</b> will automatically calculate Coordinates and the <b>Procedural Region Name</b>."),
            # Row 1: Galaxy and auto-generated region
            self._two_col_row(
                self._create_widget(Combobox, 'galaxy', 'Galaxy', options=self.data.GALAXIES, placeholder='Start typing...'),
                self._create_widget(Text, 'region', 'Region', placeholder='(Auto-Generated from Glyphs)', disabled=True)
            ),
            # Row 2: Star system and planet names
            self._two_col_row(
                self._create_widget(Text, 'system', 'System', placeholder='Star System Name'),
                self._create_widget(Text, 'planet', 'Planet', placeholder='Planet Name')
            ),
            # Row 3: Moon and hemisphere
            self._two_col_row(
                self._create_widget(Text, 'moon', 'Moon', placeholder='(Leave empty if on planet)'),
                self._create_widget(Dropdown, 'hemisphere', 'Hemisphere', options=['North', 'South', 'Equator'], placeholder='Select (Optional)...')
            ),
            # Row 4: Portal glyphs and auto-calculated coordinates
            self._two_col_row(
                self._create_widget(Text, 'glyphs', 'Portal Hex', placeholder='12-Char Hex (e.g. 10AF...)'),
                self._create_widget(Text, 'coordinates', 'Coordinates', disabled=True, placeholder='(Auto-calculated from Portal Hex)')
            ),
            # Location notes
            self._create_widget(Text, 'locnotes', 'Location Notes', placeholder='Specific landmarks...'),
            self._header('Discovery Data'),
            self._desc("Metadata regarding the discovery. <b>Discovery Date</b> and <b>AGT Stardate</b> are synced."),
            # Row 5: Discoverer information
            self._two_col_row(
                self._create_widget(Text, 'discovered', 'Discoverer Alias', placeholder='In-Game Username'),
                self._create_widget(Text, 'discoveredlink', 'Wiki Profile', placeholder='Wiki Username (optional)')
            ),
            # Row 6: Discovery date and auto-calculated stardate
            self._two_col_row(
                self._create_widget(DatePicker, 'discovered_on', 'Discovery Date', value=arrow.now().date()),
                self._create_widget(Text, 'agt_stardate', 'AGT Stardate', disabled=True, placeholder='(Synced with Discovery Date)')
            ),
            # Row 7: Civilization and platform
            self._two_col_row(
                self._create_widget(Text, 'civilized', 'Civilization', value=DEFAULT_CIVILIZATION),
                self._create_widget(Dropdown, 'platform', 'Platform', options=['PC', 'PS5', 'PS4', 'Xbox', 'Switch', 'Mac'], placeholder='Select Platform...')
            )
        ], layout=Layout(padding='20px'))

    def _create_tab_biometrics(self):
        """
        Creates the 'Measurements' tab with biometric data fields.

        This tab contains weight and height measurements for different
        gender variants, displayed in a grid format for easy data entry.
        """
        # Create header row for the measurement grid
        header = HBox([
            VBox([self._grid_header('Source')], layout=Layout(width='20%', margin='0 0 0 10px')),
            VBox([self._grid_header('Weight (kg)')], layout=Layout(width='20%')),
            VBox([self._grid_header('Height (m)')], layout=Layout(width='20%')),
            VBox([self._grid_header('Notes')], layout=Layout(width='40%'))
        ], layout=Layout(width='100%', padding='10px 5px', border_bottom='2px solid #ccc', margin='10px 0'))

        self.t4 = VBox([
            self._header('Biometrics & Measurements'),
            self._desc("Record exact measurements from analysis. <b>Max Weight</b> and <b>Max Height</b> are automatically calculated. Enter '0.0' if unknown."),
            # Grid header
            header,
            # Row 1: Gender 1 measurements
            self._grid_row("Gender 1",
                self._create_widget(FloatText, 'gender1w', ''),
                self._create_widget(FloatText, 'gender1h', ''),
                self._create_widget(Text, 'gender1note', '', placeholder='Gender 1...')
            ),
            # Row 2: Gender 2 measurements (with alternate background color)
            self._grid_row("Gender 2",
                self._create_widget(FloatText, 'gender2w', ''),
                self._create_widget(FloatText, 'gender2h', ''),
                self._create_widget(Text, 'gender2note', '', placeholder='Gender 2...'),
                bg_color="#f9f9f9"
            ),
            # Row 3: Discovery menu measurements
            self._grid_row("Discovery Menu",
                self._create_widget(FloatText, 'genderdw', ''),
                self._create_widget(FloatText, 'genderdh', ''),
                self._create_widget(Text, 'genderdnote', '', placeholder='Menu discrepancies...')
            )
        ], layout=Layout(padding='20px'))

    def _create_tab_media(self):
        """
        Creates the 'Media' tab with image and footer fields.

        This tab contains information about images, gallery entries,
        footer content, and game version.
        """
        self.t5 = VBox([
            self._header('Visuals & Metadata'),
            self._desc("<b>Main Image</b> filename must match the uploaded Wiki file. Use a clean screenshot from the Discovery Menu or Photo Mode."),
            # Main image filename
            self._create_widget(Text, 'image', 'Infobox Image', placeholder='File:CreatureName.jpg', value=DEFAULT_IMAGE),
            # Gallery images with captions
            self._create_widget(Textarea, 'gallery_text', 'Gallery Images', placeholder='File:Img1.jpg|Caption\nFile:Img2.jpg|Caption'),
            self._header('Footer & Release'),
            self._desc("Additional information for the page footer."),
            # Footer fields
            self._create_widget(Text, 'footertitle', 'Footer Title', placeholder='Section Header...'),
            self._create_widget(Textarea, 'footer', 'Footer Content', placeholder='Lore or extra details...'),
            # Game version/release
            self._create_widget(Text, 'release', 'Game Version', value=DEFAULT_RELEASE)
        ], layout=Layout(padding='20px'))

    def _create_tab_generate(self):
        """
        Creates the 'Generate' tab with action buttons and output display.

        This tab contains buttons for previewing, generating, copying,
        downloading, loading examples, and clearing the form, plus an
        output area for displaying generated wiki code.
        """
        # Create action buttons with different styles and icons
        self.btn_preview = Button(description='Preview Code', button_style='info', icon='eye')
        self.btn_gen = Button(description='Generate & Save', button_style='success', icon='code')
        self.btn_copy = Button(description='Copy to Clipboard', button_style='warning', icon='copy', disabled=True)
        self.btn_download = Button(description='Download File', button_style='primary', icon='download', disabled=True)
        self.btn_load_example = Button(description='Load Example', button_style='', icon='flask')
        self.btn_clear = Button(description='Reset Form', button_style='danger', icon='trash')

        # Output area for displaying generated wiki code
        self.output = Output(layout={'border': '1px solid #ccc', 'height': '400px', 'overflow_y': 'scroll', 'padding': '10px', 'margin-top': '10px'})
        # Status display for user feedback
        self.status_text = HTML(value="<i>Status: Ready</i>", layout=Layout(margin='10px 0 0 0'))

        # Arrange buttons in a horizontal box
        buttons = HBox(
            [self.btn_preview, self.btn_gen, self.btn_copy, self.btn_download, self.btn_load_example, self.btn_clear],
            layout=Layout(justify_content='center', margin='15px 0', flex_wrap='wrap')
        )

        # Assemble the entire tab
        self.t6 = VBox([
            self._header('Finalization'),
            self._desc("Generate the final wiki text. Use <b>Copy Code</b> to move it to the clipboard. The <b>Download</b> button also activates after generation."),
            buttons, self.status_text, self.output
        ], layout=Layout(padding='20px'))

    def _create_widget(self, widget_class, key, description, **kwargs):
        """
        Creates a widget and stores it in the widgets container.

        Args:
            widget_class (class): Type of widget to create (Text, Dropdown, etc.)
            key (str): Attribute name for storing the widget
            description (str): Label text for the widget
            **kwargs: Additional widget-specific parameters

        Returns:
            widget: The created widget instance
        """
        # Select appropriate layout based on widget type
        layout = self.WIDGET_LAYOUT
        if widget_class == FloatText:
            layout = self.GRID_INPUT_LAYOUT
        elif widget_class == Textarea and key == 'gallery_text':
            layout = self.GALLERY_LAYOUT
        elif widget_class == Checkbox:
            layout = Layout(width='auto')

        # Extract special parameters
        placeholder = kwargs.pop('placeholder', None)
        options = kwargs.pop('options', [])

        # Base parameters for all widgets
        params = {'description': description, 'style': self.LABEL_STYLE, 'layout': layout}

        # Widget-specific parameter handling
        if widget_class == Dropdown:
            # Dropdowns can have tuple options (display, value) or simple values
            is_tuple = options and isinstance(options[0], tuple)
            if is_tuple:
                # Add placeholder as first option with empty value
                opts = [(placeholder, '')] + list(options)
                val = ''
            else:
                opts = [placeholder] + list(options)
                val = placeholder
            params.update({'options': opts, 'value': val})
        elif widget_class == Combobox:
            # Comboboxes have searchable options
            params.update({'options': list(options), 'ensure_option': False})
            if placeholder:
                params['placeholder'] = placeholder
        elif widget_class in [Text, Textarea] and placeholder:
            # Text fields with placeholder text
            params['placeholder'] = placeholder

        # Handle widgets without descriptions (like grid inputs)
        if not description:
            params.pop('description', None)
            params['style'] = {'description_width': '0px'}

        # Apply any additional parameters
        params.update(kwargs)

        # Create widget and store in container
        widget_instance = widget_class(**params)
        setattr(self.widgets, key, widget_instance)
        return widget_instance

    def _header(self, text):
        """
        Creates a styled header HTML element.

        Args:
            text (str): Header text

        Returns:
            HTML: Widget displaying the header
        """
        return HTML(f"<div style='{self.HEADER_STYLE}'>{text}</div>")

    def _desc(self, text):
        """
        Creates a styled description HTML element.

        Args:
            text (str): Description text (can contain HTML)

        Returns:
            HTML: Widget displaying the description
        """
        return HTML(f"<div style='{self.DESC_STYLE}'>{text}</div>")

    def _two_col_row(self, w1, w2=None):
        """
        Creates a two-column row layout.

        Args:
            w1: Left column widget
            w2: Right column widget (optional)

        Returns:
            HBox: Horizontal box containing both widgets
        """
        c1 = VBox([w1], layout=self.COL_LAYOUT)
        c2 = VBox([w2] if w2 else [], layout=self.COL_LAYOUT)
        return HBox([c1, c2], layout=self.FULL_ROW)

    def _grid_header(self, text):
        """
        Creates a grid column header.

        Args:
            text (str): Header text

        Returns:
            HTML: Widget displaying the grid header
        """
        return HTML(f"<div style='font-weight:bold; font-size:13px; text-transform:uppercase; color:#006064;'>{text}</div>")

    def _grid_row(self, label_text, w1, w2, w3, bg_color="#ffffff"):
        """
        Creates a row in the measurements grid.

        Args:
            label_text (str): Row label (leftmost column)
            w1: First input widget (weight)
            w2: Second input widget (height)
            w3: Third input widget (notes)
            bg_color (str): Background color for the row

        Returns:
            HBox: Horizontal box containing all row widgets
        """
        return HBox([
            VBox([HTML(f"<b>{label_text}</b>")], layout=Layout(width='20%', display='flex', justify_content='flex-start', margin='5px 0 0 10px')),
            VBox([w1], layout=Layout(width='20%')),
            VBox([w2], layout=Layout(width='20%')),
            VBox([w3], layout=Layout(width='40%')),
        ], layout=Layout(width='100%', padding='5px', border_bottom='1px solid #eeeeee', background_color=bg_color, align_items='center'))

    def _connect_events(self):
        """
        Connects event handlers to widget events.

        This method sets up all the interactive behavior:
        - Field change handlers for auto-calculations
        - Button click handlers for actions
        """
        # Field change observers
        self.widgets.genus.observe(self._handle_genus_change, names='value')
        self.widgets.diet.observe(self._handle_diet_change, names='value')
        self.widgets.glyphs.observe(self._handle_glyphs_change, names='value')
        self.widgets.galaxy.observe(self._handle_galaxy_change, names='value')
        self.widgets.discovered_on.observe(self._update_stardate_ui, names='value')

        # Button click handlers
        self.btn_preview.on_click(lambda _button: self._generate_and_render('preview'))
        self.btn_gen.on_click(lambda _button: self._generate_and_render('full'))
        self.btn_clear.on_click(self._clear_form)
        self.btn_download.on_click(self._download)
        self.btn_copy.on_click(self._copy_to_clipboard)
        self.btn_load_example.on_click(self._load_example_data)

    def _update_stardate_ui(self, change):
        """
        Updates the AGT stardate field based on the selected discovery date.

        Args:
            change: Widget change event (ignored, value is read from widget)
        """
        dt = self.widgets.discovered_on.value
        if dt:
            # Convert date to arrow object for easy formatting
            d = arrow.get(dt)
            # AGT stardate format: (year + offset).day.month
            self.widgets.agt_stardate.value = f"{d.year + STARDATE_YEAR_OFFSET}.{d.day}.{d.month:02d}"

    def _handle_diet_change(self, change):
        """
        Updates diet type when diet source selection changes.

        Args:
            change: Widget change event with new value
        """
        self.widgets.diettype.value = self.data.DIET_DB.get(change.get('new', ''), "")

    def _handle_genus_change(self, change):
        """
        Updates dependent fields when genus selection changes.

        When a genus is selected, this updates:
        - Produces field (what the creature produces)
        - Ecosystem field (auto-sets if genus has only one habitat)
        - Gender options (different for biological vs robotic creatures)

        Args:
            change: Widget change event with new genus value
        """
        raw_genus = change.get('new', '')
        if not raw_genus or not (config := self.data.GENUS_DB.get(raw_genus)):
            return

        # Update produces field with genus products
        self.widgets.produces.value = ", ".join(config.get('products', []))

        # Auto-set ecosystem if genus has only one possible habitat
        habs = config.get('habitats', [])
        eco_w = self.widgets.ecosystem
        if len(habs) == 1 and habs[0] in eco_w.options:
            eco_w.value, eco_w.disabled = habs[0], True
        else:
            eco_w.disabled = False
            # Reset to placeholder if current value isn't valid for this genus
            if eco_w.value not in habs and 'Select Ecosystem...' in eco_w.options:
                eco_w.value = 'Select Ecosystem...'

        # Update gender options based on whether creature is robotic
        g_opts = self.data.GENDER_ROBOT if config.get('robot') else self.data.GENDER_STANDARD
        for w_name in ['gender1', 'gender2']:
            w = getattr(self.widgets, w_name)
            if w.options:
                ph, is_tuple = w.options[0], isinstance(w.options[0], tuple)
                # Replace options list with appropriate gender list
                w.options = [ph] + list(g_opts)
                w.value = "" if is_tuple else ph

    def _parse_glyphs_to_voxel_coords(self, glyph_hex):
        """
        Parses portal glyphs into voxel coordinates.

        Portal glyphs are 12 hexadecimal characters that encode:
        - Positions 1-3: Solar system index
        - Positions 4-6: Y coordinate
        - Positions 6-9: Z coordinate
        - Positions 9-12: X coordinate

        Args:
            glyph_hex (str): 12-character hexadecimal glyph string

        Returns:
            tuple or None: (x, y, z, system_index) or None if invalid
        """
        # Validate glyph format
        if not re.match(GLYPH_PATTERN, glyph_hex):
            return None
        try:
            # Extract coordinate components from glyph string
            x = int(glyph_hex[9:12], 16)  # Last 3 digits: X coordinate
            y = int(glyph_hex[4:6], 16)   # Digits 4-5: Y coordinate
            z = int(glyph_hex[6:9], 16)   # Digits 6-8: Z coordinate
            s = int(glyph_hex[1:4], 16)   # Digits 1-3: System index

            # Convert to signed coordinates using thresholds
            cx = x - X_THRESHOLD if x >= X_THRESHOLD else x + X_OFFSET
            cz = z - X_THRESHOLD if z >= X_THRESHOLD else z + X_OFFSET
            cy = y - Y_THRESHOLD if y >= Y_THRESHOLD else y + Y_OFFSET

            return cx, cy, cz, s
        except ValueError:
            # Handle invalid hex characters
            return None

    def _handle_glyphs_change(self, change):
        """
        Handles changes to the glyphs field, calculating coordinates and region.

        Args:
            change: Widget change event with new glyph value
        """
        glyph_hex = change.get('new', '').strip().upper()
        parsed = self._parse_glyphs_to_voxel_coords(glyph_hex)
        if parsed:
            cx, cy, cz, s = parsed
            # Format coordinates as hexadecimal with colons
            self.widgets.coordinates.value = f"{cx:04X}:{cy:04X}:{cz:04X}:{s:04X}"
            # Calculate region name from coordinates
            self._calculate_region(cx, cy, cz)
        else:
            self.widgets.coordinates.value = "Calculation Error"
            self.widgets.region.value = ""

    def _handle_galaxy_change(self, change):
        """
        Recalculates region name when galaxy selection changes.

        Args:
            change: Widget change event with new galaxy value
        """
        glyph_hex = self.widgets.glyphs.value.strip().upper()
        parsed = self._parse_glyphs_to_voxel_coords(glyph_hex)
        if parsed:
            cx, cy, cz, _ = parsed
            # Recalculate region with new galaxy
            self._calculate_region(cx, cy, cz)

    def _calculate_region(self, x_raw, y_raw, z_raw):
        """
        Calculates and displays the procedural region name.

        Args:
            x_raw (int): Raw X coordinate
            y_raw (int): Raw Y coordinate
            z_raw (int): Raw Z coordinate
        """
        gal_name = self.widgets.galaxy.value
        gal_index = self.data.GALAXY_MAP.get(gal_name, 0)

        # Convert to region-relative coordinates
        rx = x_raw - VOXEL_CENTER_XZ
        ry = y_raw - VOXEL_CENTER_Y
        rz = z_raw - VOXEL_CENTER_XZ

        try:
            # Generate region seed and name
            seed = RegionNameGenerator.create_region_seed(rx, ry, rz, gal_index)
            name = RegionNameGenerator.format_name(seed, self.data)
            self.widgets.region.value = name
        except Exception:
            # Handle any errors in region name generation
            self.widgets.region.value = "Gen Error"

    def _get_cleaned_value(self, widget):
        """
        Extracts and cleans the value from a widget.

        Converts special values (placeholders, None, etc.) to empty strings
        and handles different widget types appropriately.

        Args:
            widget: The widget to get value from

        Returns:
            str: Cleaned string value
        """
        v = widget.value
        if v is None:
            return ""
        if isinstance(v, (float, int)):
            # Convert numbers to string, but empty string for zero
            return str(v) if v != 0.0 else ""
        if isinstance(v, bool):
            # Convert boolean to 'Yes' or empty string
            return 'Yes' if v else ''
        s = str(v).strip()
        # Return empty string for placeholder values
        return "" if s in PLACEHOLDER_SENTINELS else s

    def _clear_form(self, _button):
        """
        Clears all form fields and resets to default state.

        Args:
            _button: Button click event (ignored)
        """
        # Reset each widget to its default/empty state
        for widget_field in fields(self.widgets):
            widget = getattr(self.widgets, widget_field.name)
            if not hasattr(widget, 'value'):
                continue

            if isinstance(widget, (Text, Textarea, Combobox)):
                widget.value = ""
            elif isinstance(widget, FloatText):
                widget.value = 0.0
            elif isinstance(widget, Dropdown) and widget.options:
                # Reset to first option (placeholder)
                widget.value = "" if isinstance(widget.options[0], tuple) else widget.options[0]
            elif isinstance(widget, DatePicker):
                widget.value = arrow.now().date()
            elif isinstance(widget, Checkbox):
                widget.value = False

        # Set specific default values
        self.widgets.image.value = DEFAULT_IMAGE
        self.widgets.civilized.value = DEFAULT_CIVILIZATION
        self.widgets.release.value = DEFAULT_RELEASE
        self.widgets.ecosystem.disabled = False

        # Update stardate and clear output
        self._update_stardate_ui(None)
        self.output.clear_output()
        if self.status_text:
            self.status_text.value = "<i>Status: Form cleared. Ready.</i>"

        # Disable action buttons until new content is generated
        self.btn_download.disabled = True
        self.btn_copy.disabled = True
        self.generated_content = ""

    def _generate_and_render(self, mode):
        """
        Generates wiki markup from form data and displays it.

        Args:
            mode (str): 'preview' to only display, 'full' to also save file
        """
        self.status_text.value = "<i>Status: Generating...</i>"

        # Step 1: Collect and clean all form data
        raw_data = {
            f.name: self._get_cleaned_value(getattr(self.widgets, f.name))
            for f in fields(self.widgets)
        }

        # Step 2: Calculate max weight and height from measurements
        try:
            # Convert weight strings to floats, filter out empty values
            weights = [float(x) for x in [raw_data['gender1w'], raw_data['gender2w'], raw_data['genderdw']] if x]
        except ValueError:
            weights = []
            self.status_text.value = "<i style='color:orange;'>Status: Invalid weight values, ignoring.</i>"
        try:
            # Convert height strings to floats, filter out empty values
            heights = [float(x) for x in [raw_data['gender1h'], raw_data['gender2h'], raw_data['genderdh']] if x]
        except ValueError:
            heights = []
            self.status_text.value = "<i style='color:orange;'>Status: Invalid height values, ignoring.</i>"

        # Set max weight and height (highest of all entered values)
        raw_data['weightmax'] = str(max(weights)) if weights else ""
        raw_data['heightmax'] = str(max(heights)) if heights else ""

        # Step 3: Clean gallery text (remove empty lines)
        raw_data['gallery_text'] = "".join([
            f"{line.strip()}\n" for line in self.widgets.gallery_text.value.split('\n') if line.strip()
        ])

        # Step 4: Format discovery date
        if self.widgets.discovered_on.value:
            try:
                raw_data['discovered_on'] = arrow.get(self.widgets.discovered_on.value).strftime('%d-%b-%Y')
            except arrow.parser.ParserError:
                raw_data['discovered_on'] = ""

        try:
            # Step 5: Validate data using Pydantic model
            model = WikiDataModel(**raw_data)

            # Step 6: Render template with validated data
            template = self.jinja_env.from_string(self.WIKI_TEMPLATE)
            markup = template.render(model.model_dump())

            # Step 7: Display generated markup
            with self.output:
                clear_output(wait=True)
                print(markup)

            # Step 8: Update application state
            self.generated_content = markup
            self.status_text.value = "<i style='color:green;'>Status: Generation successful.</i>"
            self.btn_copy.disabled = False

            # Step 9: Save to file if in full mode
            if mode == 'full':
                self.btn_download.disabled = False
                # Create safe filename from creature name
                safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', model.name.replace(' ', '_'))
                filename = f"{safe_name}_Creature.txt"
                try:
                    # Write markup to file
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(markup)
                    self.status_text.value += f" <b style='color:darkgreen'>Saved locally as '{filename}'</b>"
                except Exception as e:
                    self.status_text.value += f" <b style='color:red'>Local save failed: {e}</b>"

        except ValidationError as e:
            # Handle validation errors (missing/invalid fields)
            with self.output:
                clear_output(wait=True)
                errors = "\n".join([f"- Field '{err['loc'][0]}': {err['msg']}" for err in e.errors()])
                print(f"VALIDATION ERROR:\n{'=' * 20}\n{errors}")
            self.status_text.value = "<i style='color:red;'>Status: Validation failed. See errors above.</i>"
        except TemplateSyntaxError as e:
            # Handle template syntax errors (shouldn't happen with static template)
            with self.output:
                clear_output(wait=True)
                print(f"TEMPLATE ERROR:\n{'=' * 20}\nThere is a syntax error in the WIKI_TEMPLATE string.\nDetails: {e}")
            self.status_text.value = f"<i style='color:red;'>Status: Template Error: {e}</i>"

    def _download(self, _button):
        """
        Initiates download of generated wiki file.

        Note: Only works in Google Colab environment.

        Args:
            _button: Button click event (ignored)
        """
        if not self.generated_content or not self.widgets.name.value:
            return

        self.status_text.value = "<i>Status: Initiating download...</i>"
        try:
            # Google Colab-specific download functionality
            from google.colab import files

            # Create safe filename
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', self.widgets.name.value.replace(' ', '_'))
            filename = f"{safe_name}_Creature.txt"

            # Check file exists (should have been created by generate)
            if not os.path.exists(filename):
                self.status_text.value = "<i style='color:red;'>Status: File not found. Generate & Save first.</i>"
                return

            # Trigger browser download
            files.download(filename)
            self.status_text.value = "<i style='color:blue;'>Status: Download sent to browser.</i>"
        except ImportError:
            # Not running in Colab
            self.status_text.value = "<i style='color:orange;'>Status: Download is only available in Google Colab.</i>"
        except Exception as e:
            # Handle other download errors
            self.status_text.value = f"<i style='color:red;'>Status: Download failed: {e}</i>"

    def _copy_to_clipboard(self, _button):
        """
        Copies generated wiki markup to clipboard using JavaScript.

        Args:
            _button: Button click event (ignored)
        """
        if not self.generated_content:
            return

        # Escape special characters for JavaScript template literal
        escaped_content = self.generated_content.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$').replace('</', '<\\/')

        # Create JavaScript to copy to clipboard
        js_code = f"navigator.clipboard.writeText(`{escaped_content}`);"
        display(Javascript(js_code))
        self.status_text.value = "<i style='color:blue;'>Status: Copy command sent to browser.</i>"

    def _load_example_data(self, _button):
        """
        Loads example data into the form for demonstration.

        Args:
            _button: Button click event (ignored)
        """
        # Clear form first
        self._clear_form(None)
        self.status_text.value = "<i>Status: Loading example data...</i>"

        # Populate each field with example data
        for key, value in self.EXAMPLE_DATA.items():
            if hasattr(self.widgets, key):
                widget = getattr(self.widgets, key)

                # Handle different widget types
                if isinstance(widget, DatePicker):
                    try:
                        # Parse date string from example data
                        widget.value = arrow.get(value, 'D-MMM-YYYY').date()
                    except (arrow.parser.ParserError, TypeError):
                        # Fallback to current date on parse error
                        widget.value = arrow.now().date()
                elif isinstance(widget, Checkbox):
                    widget.value = bool(value)
                elif isinstance(widget, Dropdown):
                    # Check if value is in dropdown options
                    option_values = [opt[1] if isinstance(opt, tuple) else opt for opt in widget.options]
                    if value in option_values:
                        widget.value = value
                    else:
                        # Reset to placeholder if value not in options
                        is_tuple = isinstance(widget.options[0], tuple)
                        widget.value = '' if is_tuple else widget.options[0]
                else:
                    # For other widgets, just set the value
                    widget.value = value

        # Update dependent fields
        self._update_stardate_ui(None)
        self._handle_diet_change({'new': self.widgets.diet.value})
        self._handle_genus_change({'new': self.widgets.genus.value})
        self._handle_glyphs_change({'new': self.widgets.glyphs.value})

        self.status_text.value = "<i style='color:green;'>Status: Example data loaded.</i>"


# Application entry point
if __name__ == '__main__':
    # Create and run the application
    app = NMSWikiFaunaGenerator()