# TPU-Based Tiled Matrix Multiplication Accelerator

A hardware-level implementation of a programmable **Tensor Processing Unit (TPU)** built in **Logisim Evolution**, designed for tiled matrix multiplication.

This project contains **two independent TPU implementations**:

- **RnC Tiling Core**
- **Systolic Array Core**

Both implementations operate on **4×4 matrix tiles** and support smaller or dimensionally incompatible matrices through **zero-padding provided in the input RAMs**.

---

## 📐 Full TPU Architecture

The complete TPU consists of:

- Instruction RAM
- Matrix A RAM
- Matrix B RAM
- Tile loading and buffering logic
- 4×4 tile processing
- Matrix multiplication core
- Accumulator
- Result tile RAMs
- Tile alignment / reconstruction logic
- Result RAM
- Sequential instruction execution

### Full Architecture

![Full TPU Architecture](docs/full-tpu-architecture.png)

---

# 🧠 How the TPU Works

The TPU is controlled by a sequence of **24-bit instructions** stored in the Instruction RAM.

Each instruction contains:

- Operation / Opcode
- Matrix dimensions
- RAM address

The TPU executes the instructions **sequentially**.

```text
Instruction RAM
       │
       ▼
Instruction Decode
       │
       ▼
┌─────────────────────────────┐
│ Load A → Load B → Multiply  │
│              → Store Result │
└─────────────────────────────┘
       │
       ▼
Next Instruction
```

---

# 🧾 Instruction Set

| Opcode | Operation | Description |
|--------|-----------|-------------|
| `1` | Load Matrix A | Loads Matrix A from RAM A and divides it into 4×4 tiles |
| `2` | Load Matrix B | Loads Matrix B from RAM B and divides it into 4×4 tiles |
| `3` | Store Result | Aligns result tiles and stores the reconstructed matrix |
| `4` | Matrix Multiply | Performs multiplication of matrix tiles and accumulates partial results |

---

# 🔢 24-bit Instruction Format

Each instruction is **24 bits** wide.

```text
┌──────────┬──────────────────────┬─────────────────────┐
│  Opcode  │  Matrix Dimensions   │     RAM Address     │
└──────────┴──────────────────────┴─────────────────────┘
```

The instruction provides the information required by the TPU to locate and process a matrix.

---

# 🧩 Tiled Matrix Multiplication

The TPU processes matrices using **4×4 tiles** rather than processing an entire large matrix at once.

Each large matrix is decomposed into 4×4 tiles, which are then processed by the multiplication core.

The maximum supported matrix size in the current design is **64×64**.

---

# 🟦 Padding Support

The multiplication cores operate on **4×4 tiles**.

Therefore, smaller or dimensionally incompatible matrices can be handled using **zero-padding**.

For example, a 2×2 matrix:

```text
[ a b ]
[ c d ]
```

can be stored in RAM as a padded 4×4 matrix:

```text
[ a b 0 0 ]
[ c d 0 0 ]
[ 0 0 0 0 ]
[ 0 0 0 0 ]
```

### Important

**The TPU does not generate the padding automatically.**

The zero-padded matrix is prepared and stored in the input RAM.

The TPU then processes the padded 4×4 tiles normally.

After computation, the padded portions are excluded during result reconstruction so that the required matrix dimensions are obtained.

---

# ⚙️ Matrix Loading

## Opcode 1 — Load Matrix A

When Opcode `1` is executed:

```text
RAM A
  │
  ▼
Matrix A
  │
  ▼
4×4 Tile Decomposition
  │
  ▼
Internal Tile RAMs
```

Matrix A is read from RAM A and divided into 4×4 tiles.

---

## Opcode 2 — Load Matrix B

When Opcode `2` is executed:

```text
RAM B
  │
  ▼
Matrix B
  │
  ▼
4×4 Tile Decomposition
  │
  ▼
Internal Tile RAMs
```

Matrix B is read from RAM B and divided into 4×4 tiles.

---

# ➗ Matrix Multiplication

## Opcode 4 — Matrix Multiply

After both matrices have been loaded, Opcode `4` initiates the matrix multiplication.

```text
A Tiles ──────┐
              │
              ▼
        Multiplication Core
              ▲
              │
B Tiles ──────┘
              │
              ▼
        Partial Results
              │
              ▼
         Accumulator
              │
              ▼
       Result Tile RAMs
```

The multiplication core processes the 4×4 tiles.

When multiple tile products contribute to the same output tile, the accumulator combines the partial results.

---

# 💾 Result Reconstruction

After tile multiplication, the resulting tiles are stored in intermediate result RAMs.

When Opcode `3` is executed:

```text
Result Tile RAMs
       │
       ▼
Tile Alignment
       │
       ▼
Matrix Reconstruction
       │
       ▼
Result RAM
```

The result tiles are aligned and reconstructed into the final matrix.

---

# 🔬 Multiplication Cores

This project contains **two separate TPU implementations**.

Both use the same overall tiled matrix multiplication concept, but each uses a different multiplication architecture.

---

# 1. RnC Tiling Core

The first TPU implementation uses an **RnC Tiling Core**.

### Core Architecture

![RnC Tiling Core](docs/rnc-tiling-core.png)

The RnC Tiling Core performs multiplication on complete **4×4 tiles**.

```text
4×4 A Tile
     │
     ▼
┌───────────────┐
│ RnC Tile Core │
└───────┬───────┘
        │
        ▼
  Partial Result
        │
        ▼
   Accumulator
```

The complete tile multiplication is performed and the resulting partial matrix is sent to the accumulator.

---

# 2. Systolic Array Core

The second TPU implementation uses a **Systolic Array Core**.

### Core Architecture

![Systolic Array Core](docs/systolic-array-core.png)

The Systolic Array Core uses an array of Processing Elements (PEs).

The input data moves through the PE array while multiplication and accumulation take place inside the processing elements.

Conceptually:

```text
        B →     B →     B →
          ┌──────┬──────┬──────┐
A →       │ PE   │ PE   │ PE   │
          ├──────┼──────┼──────┤
A →       │ PE   │ PE   │ PE   │
          ├──────┼──────┼──────┤
A →       │ PE   │ PE   │ PE   │
          └──────┴──────┴──────┘
```

The Systolic Array Core also operates on **4×4 tiles**, but its internal computation mechanism is different from the RnC Tiling Core.

---

# 🔄 Processing Multiple Matrices

The TPU can process multiple matrix multiplication tasks sequentially.

For example, the Instruction RAM can contain:

```text
1 → Load Matrix A₁
2 → Load Matrix B₁
4 → Multiply A₁ × B₁
3 → Store C₁

1 → Load Matrix A₂
2 → Load Matrix B₂
4 → Multiply A₂ × B₂
3 → Store C₂

1 → Load Matrix A₃
2 → Load Matrix B₃
4 → Multiply A₃ × B₃
3 → Store C₃
```

The TPU continues executing instructions sequentially until the instruction sequence is complete.

---

# 🏗️ Overall Architecture

```text
                    ┌──────────────────┐
                    │  Instruction RAM │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Instruction Ctrl │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
         ┌──────────┐                  ┌──────────┐
         │   RAM A  │                  │   RAM B  │
         └────┬─────┘                  └────┬─────┘
              │                             │
              ▼                             ▼
         ┌──────────┐                  ┌──────────┐
         │ A Tiles  │                  │ B Tiles  │
         │   4×4    │                  │   4×4    │
         └────┬─────┘                  └────┬─────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Multiplication  │
                    │      Core       │
                    └────────┬────────┘
                             │
                             ▼
                       ┌───────────┐
                       │Accumulator│
                       └─────┬─────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Result Tile RAM│
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ Tile Alignment │
                    └───────┬────────┘
                            │
                            ▼
                       Result RAM
```

---

# 📊 TPU Implementations

| Feature | RnC Tiling TPU | Systolic Array TPU |
|---|---|---|
| Tile Size | 4×4 | 4×4 |
| Maximum Matrix Size | 64×64 | 64×64 |
| Matrix Tiling | ✓ | ✓ |
| Zero-Padded Input | ✓ | ✓ |
| Instruction Driven | ✓ | ✓ |
| Accumulation | ✓ | ✓ |
| Result Reconstruction | ✓ | ✓ |
| Multiplication Architecture | RnC Tiling | Systolic Array |
| Implementation | Logisim Evolution | Logisim Evolution |

---

# 🛠️ Technologies & Concepts

- **Logisim Evolution**
- Digital Logic Design
- Hardware Architecture
- Matrix Multiplication
- Tiled Matrix Multiplication
- Processing Elements
- Multiply-Accumulate (MAC)
- Systolic Array Architecture
- Instruction-Based Control
- Memory-Based Matrix Processing

---

# 📁 Repository Structure

```text
TPU-Matrix-Multiplication/
│
├── README.md
│
├── RnC-Tiling-TPU/
│   └── TPU-RnC-Tiling.circ
│
├── Systolic-Array-TPU/
│   └── TPU-Systolic-Array.circ
│
├── docs/
│   ├── full-tpu-architecture.png
│   ├── rnc-tiling-core.png
│   └── systolic-array-core.png
│
└── examples/
    ├── 2x2/
    └── 4x4/
```

---

# 🚀 Future Improvements

Possible future improvements include:

- Automatic padding generation
- Larger tile sizes
- More optimized Processing Elements
- Cycle-level performance comparison
- Hardware resource comparison
- Additional matrix operations
- Support for more flexible matrix dimensions
- Improved instruction encoding

---

# 📜 License

This project is intended for educational and research purposes.
