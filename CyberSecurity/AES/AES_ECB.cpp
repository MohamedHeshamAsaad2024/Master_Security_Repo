#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <bitset>


using namespace std;

// S-Box for SubBytes step
const uint8_t sbox[256] = {
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76, 
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0, 
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15, 
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75, 
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF, 
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08, 
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF, 
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
};
// Inverse S-Box for InverseSubBytes step
const uint8_t inv_sbox[256] ={
        0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
        0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
        0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
        0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
        0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
        0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
        0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
        0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
        0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
        0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
        0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
        0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
        0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
        0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
        0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
};
// Round constants (for AES key expansion)
const uint8_t rcon[10] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36 };

/********************************************************************
 * Function: GaloisFieldMultiplication
 * Description:
 *  This function performs Galois Field Multiplication in GF(2^8).
 *  In AES we perform operations in GF(2^8) ensuring that each number
 *  can be represented by a single byte. The irreducable polynomial
 *  defined by AES NIST is m(x) = x^8 + x^4 + x^3 + x + 1 which
 *  corresponds to {01}{1b} in hex format or 283 in decimal format.
 *  GF(2^8) Multiplication steps:
 *      1. Distribute the first operand over the second operand
 *      2. Reduce the result to modulo m(x)
 * Inputs:  firstOperand    - First operand to GF multiplication
 *          secondOperand   - Second operand to GF multiplication
 * Outputs: void
 * Returns: Result of multipling firstOperand by the secondOperand
 *          in GF(2^8)
 ********************************************************************/
uint8_t GaloisFieldMultiplication(uint8_t firstOperand, uint8_t secondOperand) 
{
    /* Initialize a varibale to hold the result incrementally */
    uint8_t result = 0;

    /* Initialize a variable to hold the MSB during computations */
    uint8_t high_bit = 0;

    /* Distibute the first operand on the second operand through looping over all its terms */
    while(secondOperand) 
    {
        /* If the LSB of the second operand is 1, then when distributed over the first operand a constant term will exist */
        if ((secondOperand & 1) == 1)
        {
            /* This will require mod-2 adding (XOR) the first operand to the previous result */
            result ^= firstOperand;
        }
        
        /* Maintain the MSB which will be checked to indicate whether the result exceeded GF(2^8) or not */
        high_bit = firstOperand & 0x80;

        /* Multiply the first Operand by x to consider the next term in the next iteration. Multiplication by x in GF is essentially a simple shift left by 1 */
        firstOperand <<= 1;

        /* Check if the result of multiplication by x exceeded the GF(2^8) which is indicated by the MSB being 1 before multiplication */
        if(high_bit == 1)
        {
            /* We need to reduce the result by the AES irreducable polynomial (x^8 + x^4 + x^3 + x + 1)
               This is performed by (firstOperand) mod (0x1b) which can be performed by a single division step
               as the highest term is x^8 in the firstOperand. This division step is essentially a mod-2 subtraction
               which is a simple XOR operation */
            firstOperand ^= 0x1b;
        }
        
        /* Move to the next term (bit) to be processed in the second operand */
        secondOperand >>= 1;
    }

    /* Return the final result of multiplication */
    return result;
}

void MixColumns(uint8_t state[4][4]) 
{
    /* Define a 4x4 temporary matrix for intermediate computations in order not to corrupt the input matrix */
    uint8_t temp[4][4];

    /* Initialize the fixed polynomial matrix defined by AES NIST for Mix Columns step */
    uint8_t fixedMatrix[4][4] = {
        {0x02, 0x03, 0x01, 0x01},
        {0x01, 0x02, 0x03, 0x01},
        {0x01, 0x01, 0x02, 0x03},
        {0x03, 0x01, 0x01, 0x02}
    };

    /* Apply GF Multiplication of the Fixed Polynomial Matrix (4x4) by the input State Matrix (4x4) */
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            temp[row][col] = 0;
            for (int k = 0; k < 4; ++k) {
                temp[row][col] ^= GaloisFieldMultiplication(fixedMatrix[row][k], state[k][col]);
            }
        }
    }

    /* Copy the result back from the temporary matrix to the output state matrix */
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            state[row][col] = temp[row][col];
        }
    }
}

/********************************************************************
 ********************** Decryption Functions ************************
 ********************************************************************/
/********************************************************************
 * Function: InvMixColumns
 * Description:
 *  This function performs AES Inverse Mix Columns steps by multipying
 *  the inverse of the fixed polynomial matrix defined by AES NIST 
 *  by the input state. Essentially, it performs 4x4 matrix 
 *  multiplication in GF(2^8)
 * Inputs:  state   - Refernece to Input State Matrix (4x4)
 * Outputs: state   - Refernece to Output State Matrix (4x4)
 * Returns: void
 ********************************************************************/
void InvMixColumns(uint8_t state[4][4]) 
{
    /* Define a 4x4 temporary matrix for intermediate computations in order not to corrupt the input matrix */
    uint8_t temp[4][4];

    /* Initialize the fixed polynomial matrix defined by AES NIST for Mix Columns step */
    uint8_t fixedMatrix[4][4] = {
        {0x0E, 0x0B, 0x0D, 0x09},
        {0x09, 0x0E, 0x0B, 0x0D},
        {0x0D, 0x09, 0x0E, 0x0B},
        {0x0B, 0x0D, 0x09, 0x0E}
    };

    /* Apply GF Multiplication of the Fixed Polynomial Matrix (4x4) by the input State Matrix (4x4) */
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            temp[row][col] = 0;
            for (int k = 0; k < 4; ++k) {
                temp[row][col] ^= GaloisFieldMultiplication(fixedMatrix[row][k], state[k][col]);
            }
        }
    }

    /* Copy the result back from the temporary matrix to the output state matrix */
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            state[row][col] = temp[row][col];
        }
    }
}


// AddRoundKey: XORs the state with the round key used for both encryption and decryption
void AddRoundKey(uint8_t state[4][4], uint8_t roundKey[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            state[i][j] ^= roundKey[i][j];
        }
    }
}

// SubBytes: Applies the S-Box to the state
void SubBytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            state[i][j] = sbox[state[i][j]];
        }
    }
}

// Inverse SubBytes: Applies the inverse S-Box to the state
void InvSubBytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            state[i][j] = inv_sbox[state[i][j]];
        }
    }
}

// ShiftRows: Performs the row shift in the AES process
void ShiftRows(uint8_t state[4][4]) {
    uint8_t temp;
    // Row 1: Left shift by 1
    temp = state[1][0];
    for (int i = 0; i < 3; ++i) {
        state[1][i] = state[1][i + 1];
    }
    state[1][3] = temp;

    // Row 2: Left shift by 2
    temp = state[2][0];
    state[2][0] = state[2][2];
    state[2][2] = temp;
    temp = state[2][1];
    state[2][1] = state[2][3];
    state[2][3] = temp;

    // Row 3: Left shift by 3
    temp = state[3][0];
    for (int i = 0; i < 3; ++i) {
        state[3][i] = state[3][i + 1];
    }
    state[3][3] = temp;
}

// Inverse ShiftRows: Performs the inverse row shift for decryption
void InvShiftRows(uint8_t state[4][4]) {
    uint8_t temp;
    // Row 1: Right shift by 1
    temp = state[1][3];
    for (int i = 3; i > 0; --i) {
        state[1][i] = state[1][i - 1];
    }
    state[1][0] = temp;

    // Row 2: Right shift by 2
    temp = state[2][3];
    state[2][3] = state[2][1];
    state[2][1] = temp;
    temp = state[2][2];
    state[2][2] = state[2][0];
    state[2][0] = temp;

    // Row 3: Right shift by 3
    temp = state[3][3];
    for (int i = 3; i > 0; --i) {
        state[3][i] = state[3][i - 1];
    }
    state[3][0] = temp;
}

/********************************************************************
 * Function: MixColumns
 * Description:
 *  This function performs AES Mix Columns steps by multipying
 *  the fixed polynomial matrix defined by AES NIST by the input
 *  state. Essentially, it performs 4x4 matrix multiplication in
 *  GF(2^8)
 * Inputs:  state   - Refernece to Input State Matrix (4x4)
 * Outputs: state   - Refernece to Output State Matrix (4x4)
 * Returns: void
 ********************************************************************/

// KeyExpansion to generate round keys
void KeyExpansion(uint8_t key[16], uint8_t roundKeys[11][4][4]) 
{
    uint8_t temp[4]; // Temporary storage for the column being processed
    int i = 0;

    // Copy the original key as the first round key
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            roundKeys[0][c][r] = key[i++];
        }
    }

    // Generate the rest of the round keys
    for (int round = 1; round <= 10; ++round) {
        // Step 1: Get the last column from the previous round key
        for (int row = 0; row < 4; ++row) {
            temp[row] = roundKeys[round - 1][3][row];
        }

        // Step 2: Rotate the column (Word rotation)
        uint8_t t = temp[0];
        temp[0] = temp[1];
        temp[1] = temp[2];
        temp[2] = temp[3];
        temp[3] = t;

        // Step 3: Apply SubBytes to the column
        for (int row = 0; row < 4; ++row) {
            temp[row] = sbox[temp[row]];
        }

        // Step 4: XOR the first element with the round constant
        temp[0] ^= rcon[round - 1];

        // Step 5: Compute the first column of the current round key
        for (int row = 0; row < 4; ++row) {
            roundKeys[round][0][row] = roundKeys[round - 1][0][row] ^ temp[row];
        }

        // Step 6: Compute the remaining columns of the current round key
        for (int col = 1; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                roundKeys[round][col][row] = roundKeys[round - 1][col][row] ^ roundKeys[round][col - 1][row];
            }
        }
    }
}

// AESEncrypt function
void AESEncrypt(uint8_t plaintext[16], uint8_t ciphertext[16], uint8_t roundKeys[11][4][4]) {
    uint8_t state[4][4];

    // Copy plaintext into state array (row-major to column-major format)
    int idx = 0;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            state[row][col] = plaintext[idx++];
        }
    }

    // Initial round key addition
    AddRoundKey(state, roundKeys[0]);

    // Main rounds
    for (int round = 1; round <= 9; ++round) {
        SubBytes(state);
        ShiftRows(state);
        MixColumns(state);
        AddRoundKey(state, roundKeys[round]);
    }

    // Final round (no MixColumns)
    SubBytes(state);
    ShiftRows(state);
    AddRoundKey(state, roundKeys[10]);

    // Copy state to ciphertext
    idx = 0;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            ciphertext[idx++] = state[row][col];
        }
    }
}
// AESDecrypt function
void AESDecrypt(uint8_t ciphertext[16], uint8_t plaintext[16], uint8_t roundKeys[11][4][4]) {
    uint8_t state[4][4];

    // Copy ciphertext into state array (row-major to column-major format)
    int idx = 0;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            state[row][col] = ciphertext[idx++];
        }
    }

    // Initial round key addition
    AddRoundKey(state, roundKeys[10]);



    // Main rounds
    for (int round = 9; round >= 1; --round) {
        InvShiftRows(state);
        InvSubBytes(state);
        AddRoundKey(state, roundKeys[round]);
        InvMixColumns(state);
    }

    
    InvShiftRows(state);
    InvSubBytes(state);
    // Final round (no InvMixColumns)
    AddRoundKey(state, roundKeys[0]);

    // Copy state to plaintext
    idx = 0;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            plaintext[idx++] = state[row][col];
        }
    }
}


// Main function to test AES encryption and decryption
int main() {

    uint8_t key[16] = {
        0x2b, 0x7e, 0x15, 0x16,
        0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0xcf, 0xa5,
        0x30, 0x8d, 0x31, 0x32
    };
    
    uint8_t plaintext[16] = {
        0x32, 0x88, 0x31, 0xe0,
        0x43, 0x5a, 0x31, 0x37,
        0xf6, 0x30, 0x98, 0x07,
        0xa8, 0x8d, 0xa2, 0x34
    };
    
    uint8_t ciphertext[16];
    uint8_t decryptedText[16];
    uint8_t roundKeys[11][4][4];
    // Expanded key schedule (NUM_COLUMN * (Nr + 1) words)
    uint32_t expandedKey[NUM_COLUMN * (Nr + 1)];
    // Perform key expansion
    KeyExpansion(key, 4, 10, expandedKey);
    cout << "Round keys";
    for(int i=0;i<11;i++)
    {
        for (int col = 0; col < 4; ++col)
        {
            for (int row = 0; row < 4; ++row) 
            {
                 cout << hex << setw(2) << setfill('0') << (int)plaintext[i][col][row] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }
    // Encrypt the plaintext
    AESEncrypt(plaintext, ciphertext, roundKeys);
    
    // Decrypt the ciphertext
    AESDecrypt(ciphertext, decryptedText, roundKeys);
    
    // Output the result
    cout << "Plaintext: ";
    for (int i = 0; i < 16; ++i) {
        cout << hex << setw(2) << setfill('0') << (int)plaintext[i] << " ";
    }
    cout << endl;
    
    cout << "Ciphertext: ";
    for (int i = 0; i < 16; ++i) {
        cout << hex << setw(2) << setfill('0') << (int)ciphertext[i] << " ";
    }
    cout << endl;
    
    cout << "Decrypted text: ";
    for (int i = 0; i < 16; ++i) {
        cout << hex << setw(2) << setfill('0') << (int)decryptedText[i] << " ";
    }
    cout << endl;

    return 0;
}