---
title: "Fuzzing Tools and Techniques"
date: 2024-12-13
draft: false
category: "security"
tags: ["fuzzing", "security", "testing", "afl", "libfuzzer", "vulnerability"]
---

Fuzzing (fuzz testing) automatically generates and injects malformed or unexpected inputs to find bugs, crashes, and security vulnerabilities.

## Overview

**Fuzzing** is an automated software testing technique that provides invalid, unexpected, or random data as inputs to a program.

**Goals:**
- Find crashes and hangs
- Discover memory corruption bugs
- Identify security vulnerabilities
- Test error handling
- Improve code coverage

---

## Types of Fuzzing

### 1. Black-Box Fuzzing

No knowledge of internal structure.

```text
Input → [Program] → Monitor for crashes
```

**Tools:** Radamsa, zzuf, Peach Fuzzer

### 2. White-Box Fuzzing

Uses program analysis and symbolic execution.

```text
Input → [Analyze Code] → Generate targeted inputs → Test
```

**Tools:** KLEE, Driller, Mayhem

### 3. Grey-Box Fuzzing

Uses lightweight instrumentation for feedback.

```text
Input → [Instrumented Program] → Coverage feedback → Mutate input
```

**Tools:** AFL, AFL++, LibFuzzer, Honggfuzz

---

## AFL (American Fuzzy Lop)

Most popular coverage-guided fuzzer.

### Installation

```bash
# Install AFL++
git clone https://github.com/AFLplusplus/AFLplusplus
cd AFLplusplus
make
sudo make install

# Or via package manager
sudo apt-get install afl++  # Debian/Ubuntu
brew install afl++          # macOS
```

### Basic Usage

```bash
# 1. Compile target with AFL instrumentation
afl-gcc -o target target.c
# Or for C++
afl-g++ -o target target.cpp

# 2. Create input directory with seed files
mkdir input
echo "test" > input/seed1.txt
echo "hello" > input/seed2.txt

# 3. Create output directory
mkdir output

# 4. Run fuzzer
afl-fuzz -i input -o output -- ./target @@
# @@ is replaced with input filename

# 5. Monitor results
# AFL shows real-time stats in terminal
# Crashes saved in output/crashes/
# Hangs saved in output/hangs/
```

### Example Target Program

```c
// vulnerable.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void process_input(char *data) {
    char buffer[16];
    
    // Vulnerability: no bounds checking
    if (data[0] == 'A' && data[1] == 'B') {
        strcpy(buffer, data);  // Buffer overflow!
    }
    
    // Another bug: integer overflow
    int len = strlen(data);
    if (len > 0) {
        char *ptr = malloc(len - 1);  // Can underflow!
        free(ptr);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    
    char data[1024];
    size_t len = fread(data, 1, sizeof(data) - 1, fp);
    data[len] = '\0';
    fclose(fp);
    
    process_input(data);
    
    return 0;
}
```

```bash
# Compile with AFL
afl-gcc -o vulnerable vulnerable.c

# Fuzz
mkdir in out
echo "test" > in/seed
afl-fuzz -i in -o out -- ./vulnerable @@

# AFL will find the buffer overflow when input starts with "AB"
```

### Advanced AFL Options

```bash
# Multiple cores (parallel fuzzing)
afl-fuzz -i in -o out -M fuzzer1 -- ./target @@  # Master
afl-fuzz -i in -o out -S fuzzer2 -- ./target @@  # Slave
afl-fuzz -i in -o out -S fuzzer3 -- ./target @@  # Slave

# With dictionary (for structured formats)
afl-fuzz -i in -o out -x dict.txt -- ./target @@

# Dictionary example (dict.txt)
# keyword_http="GET"
# keyword_http="POST"
# keyword_http="HTTP/1.1"

# Persistent mode (faster)
# Requires modifying target to use __AFL_LOOP
afl-fuzz -i in -o out -- ./target_persistent

# QEMU mode (no source code needed)
afl-fuzz -Q -i in -o out -- ./binary @@
```

### Analyzing Crashes

```bash
# Reproduce crash
./target output/crashes/id:000000,sig:11,src:000000,op:havoc,rep:2

# With Address Sanitizer for better diagnostics
afl-gcc -fsanitize=address -o target_asan target.c
./target_asan output/crashes/id:000000*

# Minimize crashing input
afl-tmin -i crash_input -o minimized -- ./target @@

# Get crash statistics
afl-whatsup output/
```

---

## LibFuzzer

LLVM's in-process, coverage-guided fuzzer.

### Basic Example

```cpp
// fuzz_target.cpp
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Vulnerable function
void process_data(const uint8_t *data, size_t size) {
    if (size >= 4) {
        if (data[0] == 'F' &&
            data[1] == 'U' &&
            data[2] == 'Z' &&
            data[3] == 'Z') {
            // Trigger crash
            char *ptr = nullptr;
            *ptr = 0;  // Null pointer dereference
        }
    }
}

// LibFuzzer entry point
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    process_data(data, size);
    return 0;
}
```

```bash
# Compile with LibFuzzer
clang++ -g -fsanitize=fuzzer,address fuzz_target.cpp -o fuzz_target

# Run fuzzer
./fuzz_target

# With corpus directory
mkdir corpus
./fuzz_target corpus/

# With options
./fuzz_target -max_len=1024 -timeout=10 corpus/

# Reproduce crash
./fuzz_target crash-file
```

### Advanced LibFuzzer

```cpp
// Custom mutator
extern "C" size_t LLVMFuzzerCustomMutator(
    uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {
    // Custom mutation logic
    return Size;
}

// Custom crossover
extern "C" size_t LLVMFuzzerCustomCrossOver(
    const uint8_t *Data1, size_t Size1,
    const uint8_t *Data2, size_t Size2,
    uint8_t *Out, size_t MaxOutSize, unsigned int Seed) {
    // Custom crossover logic
    return 0;
}

// Initialize (runs once)
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    // Setup code
    return 0;
}
```

---

## Honggfuzz

Security-oriented fuzzer with hardware-assisted feedback.

### Installation

```bash
# Install
git clone https://github.com/google/honggfuzz
cd honggfuzz
make
sudo make install
```

### Usage

```bash
# Compile target
hfuzz-gcc target.c -o target
# Or
hfuzz-clang target.c -o target

# Fuzz
honggfuzz -i input_corpus -o output -- ./target ___FILE___

# With sanitizers
hfuzz-clang -fsanitize=address target.c -o target
honggfuzz -i input -o output -- ./target ___FILE___

# Persistent mode
honggfuzz -i input -o output -P -- ./target
```

---

## Protocol Fuzzing

### HTTP Fuzzing with Wfuzz

```bash
# Install
pip install wfuzz

# Fuzz URL parameters
wfuzz -z file,wordlist.txt http://target.com/page?param=FUZZ

# Fuzz POST data
wfuzz -z file,payloads.txt -d "username=admin&password=FUZZ" \
      http://target.com/login

# Fuzz headers
wfuzz -z file,xss.txt -H "User-Agent: FUZZ" http://target.com/

# Multiple injection points
wfuzz -z file,users.txt -z file,passes.txt \
      -d "user=FUZZ&pass=FUZ2Z" http://target.com/login

# Filter responses
wfuzz -z range,1-1000 --hc 404 http://target.com/page?id=FUZZ

# Common options:
# --hc: Hide responses with code
# --hl: Hide responses with lines
# --hw: Hide responses with words
# --hh: Hide responses with chars
```

### Ffuf (Fast Web Fuzzer)

```bash
# Install
go install github.com/ffuf/ffuf@latest

# Directory fuzzing
ffuf -w wordlist.txt -u http://target.com/FUZZ

# Virtual host fuzzing
ffuf -w vhosts.txt -u http://target.com -H "Host: FUZZ.target.com"

# POST data fuzzing
ffuf -w wordlist.txt -X POST -d "username=admin&password=FUZZ" \
     -u http://target.com/login

# Recursive fuzzing
ffuf -w wordlist.txt -u http://target.com/FUZZ -recursion

# Match/filter responses
ffuf -w wordlist.txt -u http://target.com/FUZZ \
     -mc 200,301,302 \
     -fs 1234  # Filter by size

# Rate limiting
ffuf -w wordlist.txt -u http://target.com/FUZZ -rate 100
```

### Boofuzz (Network Protocol Fuzzing)

```python
# Install
pip install boofuzz

# Example: Fuzz HTTP server
from boofuzz import *

def main():
    session = Session(
        target=Target(
            connection=TCPSocketConnection("127.0.0.1", 80)
        ),
    )
    
    # Define HTTP request
    s_initialize("http_get")
    s_string("GET", fuzzable=False)
    s_delim(" ", fuzzable=False)
    s_string("/", fuzzable=True)
    s_delim(" ", fuzzable=False)
    s_string("HTTP/1.1", fuzzable=False)
    s_static("\r\n")
    s_string("Host:", fuzzable=False)
    s_delim(" ", fuzzable=False)
    s_string("localhost", fuzzable=True)
    s_static("\r\n\r\n")
    
    session.connect(s_get("http_get"))
    session.fuzz()

if __name__ == "__main__":
    main()
```

### Radamsa (General-Purpose Fuzzer)

```bash
# Install
git clone https://gitlab.com/akihe/radamsa.git
cd radamsa
make
sudo make install

# Generate fuzzed inputs
echo "GET / HTTP/1.1" | radamsa

# Generate multiple
echo "test input" | radamsa -n 100 -o fuzz-%n.txt

# Fuzz file
radamsa input.txt -o output-%n.txt -n 1000

# Use in pipeline
cat valid_request.txt | radamsa | nc target.com 80
```

---

## Structure-Aware Fuzzing

### Protobuf Fuzzing

```cpp
// proto_fuzzer.cpp
#include <libprotobuf-mutator/libfuzzer/libfuzzer_macro.h>
#include "message.pb.h"

DEFINE_PROTO_FUZZER(const MyMessage& message) {
    // Process protobuf message
    ProcessMessage(message);
}
```

```bash
# Compile
clang++ -fsanitize=fuzzer,address proto_fuzzer.cpp \
        message.pb.cc -lprotobuf -o proto_fuzzer

# Run
./proto_fuzzer
```

### JSON Fuzzing

```python
# json_fuzzer.py
import json
import atheris
import sys

@atheris.instrument_func
def test_json_parser(data):
    try:
        parsed = json.loads(data)
        # Process parsed JSON
        process_json(parsed)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

def main():
    atheris.Setup(sys.argv, test_json_parser)
    atheris.Fuzz()

if __name__ == "__main__":
    main()
```

---

## Fuzzing Best Practices

### 1. Seed Corpus

```bash
# Good seed corpus:
# - Valid inputs that exercise different code paths
# - Minimal but diverse
# - Cover edge cases

# Example for HTTP fuzzer
mkdir corpus
echo "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n" > corpus/get.txt
echo "POST / HTTP/1.1\r\nContent-Length: 0\r\n\r\n" > corpus/post.txt
echo "OPTIONS / HTTP/1.1\r\n\r\n" > corpus/options.txt
```

### 2. Sanitizers

```bash
# Address Sanitizer (memory errors)
clang -fsanitize=address -g target.c -o target

# Undefined Behavior Sanitizer
clang -fsanitize=undefined -g target.c -o target

# Memory Sanitizer (uninitialized memory)
clang -fsanitize=memory -g target.c -o target

# Thread Sanitizer (data races)
clang -fsanitize=thread -g target.c -o target

# Combine multiple
clang -fsanitize=address,undefined -g target.c -o target
```

### 3. Coverage Analysis

```bash
# Generate coverage report with AFL
afl-cov -d output/ --live --coverage-cmd \
        "cat AFL_FILE | ./target" \
        --code-dir .

# With LLVM coverage
clang -fprofile-instr-generate -fcoverage-mapping target.c -o target
LLVM_PROFILE_FILE="target.profraw" ./target input
llvm-profdata merge -sparse target.profraw -o target.profdata
llvm-cov show ./target -instr-profile=target.profdata
```

### 4. Continuous Fuzzing

```yaml
# .github/workflows/fuzzing.yml
name: Continuous Fuzzing

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  fuzz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install AFL++
        run: |
          sudo apt-get update
          sudo apt-get install -y afl++
      
      - name: Build
        run: |
          afl-gcc -o target target.c
      
      - name: Fuzz
        run: |
          mkdir -p in out
          echo "seed" > in/seed
          timeout 3600 afl-fuzz -i in -o out -- ./target @@
      
      - name: Upload crashes
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: crashes
          path: out/crashes/
```

---

## Fuzzing Frameworks Comparison

| Tool | Type | Speed | Learning Curve | Best For |
|------|------|-------|----------------|----------|
| **AFL++** | Grey-box | Fast | Medium | C/C++ programs |
| **LibFuzzer** | Grey-box | Very Fast | Low | In-process fuzzing |
| **Honggfuzz** | Grey-box | Fast | Medium | Security research |
| **Boofuzz** | Black-box | Slow | Low | Network protocols |
| **Radamsa** | Black-box | Fast | Very Low | Quick tests |
| **OSS-Fuzz** | Platform | N/A | High | Open source projects |

---

## Common Vulnerabilities Found

### Buffer Overflows

```c
// AFL will find this
void vulnerable(char *input) {
    char buf[16];
    strcpy(buf, input);  // No bounds check
}
```

### Integer Overflows

```c
// Fuzzer can trigger with large values
void allocate(size_t size) {
    if (size > 0) {
        char *buf = malloc(size - 1);  // Underflow!
    }
}
```

### Format String Bugs

```c
// Fuzzer finds with %n%n%n...
void log_message(char *msg) {
    printf(msg);  // Should be printf("%s", msg)
}
```

### Use-After-Free

```c
// Fuzzer triggers with specific input sequence
void process(char *input) {
    char *ptr = malloc(100);
    free(ptr);
    if (input[0] == 'X') {
        strcpy(ptr, input);  // Use after free!
    }
}
```

---

## Further Reading

- [AFL++ Documentation](https://aflplus.plus/)
- [LibFuzzer Tutorial](https://llvm.org/docs/LibFuzzer.html)
- [Fuzzing Book](https://www.fuzzingbook.org/)
- [Google OSS-Fuzz](https://github.com/google/oss-fuzz)
- [Fuzzing Project](https://fuzzing-project.org/)
- [Awesome Fuzzing](https://github.com/secfigo/Awesome-Fuzzing)

