#!/usr/bin/env python3
import argparse
import sys
import os

CHIP_SIZE = 16 * 1024 * 1024  # 16MB

def main():
    parser = argparse.ArgumentParser(
        description='Extract BIOS from Synology PAT file and prepare for flashing'
    )
    parser.add_argument('-i', '--input', required=True, 
                        help='Input bios.ROM from PAT file')
    parser.add_argument('-o', '--output', default='flash.bin',
                        help='Output file (16MB, default: flash.bin)')
    args = parser.parse_args()

    # Read input file
    try:
        with open(args.input, 'rb') as f:
            data = f.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Input: {len(data)} bytes ({len(data)/(1024*1024):.2f} MB)\n")

    # Find header signature
    header_sig = b'$_IFLASH_BIOSIMG'
    header_idx = data.find(header_sig)
    if header_idx == -1:
        print("Error: $_IFLASH_BIOSIMG signature not found")
        sys.exit(1)
    print(f"Found header at: 0x{header_idx:X}")

    header_idx += len(header_sig)+8

    # Find footer signature
    footer_sig = b'$_IFLASH_INI_IMG'
    footer_idx = data.find(footer_sig)
    if footer_idx == -1:
        print("Error: $_IFLASH_INI_IMG signature not found")
        sys.exit(1)
    print(f"Found footer at: 0x{footer_idx:X}")

    # Extract BIOS data between signatures
    bios_data = data[header_idx:footer_idx]
    bios_size = len(bios_data)
    print(f"\nExtracted BIOS: {bios_size} bytes ({bios_size/(1024*1024):.2f} MB)")

    # Calculate padding
    pad_size = CHIP_SIZE - bios_size
    if pad_size < 0:
        print(f"Error: BIOS too large ({bios_size} > {CHIP_SIZE})")
        sys.exit(1)
    print(f"Padding needed: {pad_size} bytes ({pad_size/(1024*1024):.2f} MB)")

    # Show structure
    print(f"\nStructure:")
    print(f"  0x{0:08X} - 0x{bios_size-1:08X}: BIOS code")
    print(f"  0x{bios_size:08X} - 0x{CHIP_SIZE-1:08X}: Zero padding")

    # Write output: [BIOS] + [zeros]
    try:
        with open(args.output, 'wb') as f:
            # Write BIOS data
            f.write(bios_data)
            
            # Write zero padding
            f.write(b'\x00' * pad_size)
    except IOError as e:
        print(f"Error writing file: {e}")
        sys.exit(1)

    print(f"\n✓ Created: {args.output} ({CHIP_SIZE} bytes)")
    print("\nFlash with: CH341A + 1.8V adapter")
    print("IMPORTANT: Leave pins 3(WP) and 7(HOLD) disconnected")

if __name__ == '__main__':
    main()