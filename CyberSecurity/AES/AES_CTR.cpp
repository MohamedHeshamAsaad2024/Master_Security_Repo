#include <cstdint>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <random>

/********************************************************************
 *********************** Configurations *****************************
 ********************************************************************/
#define CORES_NUMBER    (10U)


/*******************************TBD*************************************/
using namespace std;

/* Number of columns (32-bit words) in the state */
#define NUM_COLUMN 4
/* Number of Word Size */
#define WORD_SIZE  4

/* AES S-box */
const uint8_t sBox[256] = 
{
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
/* AES Rcon (Round constant) */
const uint8_t Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
/*******************************//************************************/



/********************************************************************
 ************************* Prototypes *******************************
 ********************************************************************/
/* Encryption Functions */
void AddRoundKey(uint8_t state[4][4], const std::vector<uint8_t>& roundKey);
void SubBytes(uint8_t state[4][4]);
void ShiftRows(uint8_t state[4][4]);
void MixColumns(uint8_t state[4][4]);

uint32_t SubWord(uint32_t word);
uint32_t RotWord(uint32_t word);
void KeyExpansion(const uint8_t* key, int Nk, int Nr, uint32_t* w) ;
void PrintExpandedKey(const uint32_t* w, int size);


/* Decryption functions */
void InvMixColumns(uint8_t state[4][4]);

/* Prining Functions */
void printState(uint8_t state[4][4]);

/* Utility Functions */
uint8_t GaloisFieldMultiplication(uint8_t firstOperand, uint8_t secondOperand);
void IncrementCounter(std::vector<uint8_t>& counter);
void TextPreprocessor(const std::string& strText, std::vector<std::vector<std::vector<uint8_t>>>& states);
void TextPostprocessor(const std::vector<std::vector<std::vector<uint8_t>>>& states, std::string& strText);

/* Counter Mode Functions */
void CounterModeInitializer(std::vector<uint8_t>& counter);
void EncryptionDispatcher(const std::vector<std::vector<std::vector<uint8_t>>>& states, std::vector<uint8_t> counter, std::vector<std::vector<std::vector<uint8_t>>>& encryptedStates);
void DecryptionDispatcher(const std::vector<std::vector<std::vector<uint8_t>>>& states, std::vector<uint8_t> counter, std::vector<std::vector<std::vector<uint8_t>>>& decryptedStates);
void EncryptionWorker(const std::vector<std::vector<uint8_t>>& state, const std::vector<uint8_t> counter, std::vector<std::vector<uint8_t>>& encryptedState);
void DecryptionWorker(const std::vector<std::vector<uint8_t>>& state, const std::vector<uint8_t> counter, std::vector<std::vector<uint8_t>>& decryptedState);

/********************************************************************
 ************************* Main Function ****************************
 ********************************************************************/
int main() 
{
    /* Declare a string to hold the plaintext */
    std::string plainText;

    /* Declare a string to hold the plaintext */
    std::string cipherText;

    /* Declare a string to hold the plaintext */
    std::string decryptedText;

    /* Plain States */
    std::vector<std::vector<std::vector<uint8_t>>> plainStates;

    /* Encrypted States */
    std::vector<std::vector<std::vector<uint8_t>>> encryptedStates;

    /* Encrypted States */
    std::vector<std::vector<std::vector<uint8_t>>> decryptedStates;

    /* Declare a vector to hold the original counter for the CTR Mode of Operation */
    std::vector<uint8_t> counter;

    /* Take input string from the user */
    std::cout << "Enter the Plain Text: ";
    std::cin >> plainText;

    /* Initialize the counter for AES CTR Mode encryption */
    CounterModeInitializer(counter);

    /* Transform the input */
    TextPreprocessor(plainText, plainStates);

    /* Reserve memory for the Encrypted states */
    encryptedStates.resize(plainStates.size());
    for (int i = 0; i < plainStates.size(); ++i) 
    {
        encryptedStates[i].resize(4);
        for (int j = 0; j < 4; ++j) 
        {
            encryptedStates[i][j].resize(4);
        }
    }


    /* Initiate the Encryption Dispatcher */
    EncryptionDispatcher(plainStates, counter, encryptedStates);

    /* Transform the Encrypted States from 3D into a 1D vector for printing */
    TextPostprocessor(encryptedStates, cipherText);

    /* Print the ciphertext */
    std::cout << "Cipher Text: " << cipherText << std::endl;

    /* Clear the encrypted states vector for decryption */
    encryptedStates.clear();

    /* Preprocess the ciphertext */
    TextPreprocessor(cipherText, encryptedStates);

    /* Reserve memory for the decrypted states */
    decryptedStates.resize(encryptedStates.size());
    for (int i = 0; i < encryptedStates.size(); ++i) 
    {
        decryptedStates[i].resize(4);
        for (int j = 0; j < 4; ++j) 
        {
            decryptedStates[i][j].resize(4);
        }
    }

    /* Initiate the Encryption Dispatcher */
    DecryptionDispatcher(encryptedStates, counter, decryptedStates);

    /* Transform the Encrypted States from 3D into a 1D vector for printing */
    TextPostprocessor(decryptedStates, decryptedText);

    /* Print the decryptedText */
    std::cout << "Decypted Text: " << decryptedText << std::endl;

    return 0;
}

/********************************************************************
 ********************** Encryption Functions ************************
 ********************************************************************/
/********************************************************************
 * Function: AddRoundKey
 * Description:
 *  This function performs AddRoundKey step in AES. It XORs the 
 *  state matrix with the round key
 * Inputs:  state       - Refernece to Input State Matrix (4x4)
 *          roundKey    - Round Key
 * Outputs: state       - Refernece to Output State Matrix (4x4)
 * Returns: void
 ********************************************************************/
void AddRoundKey(uint8_t state[4][4], const std::vector<uint8_t>& roundKey) 
{
    for (int row = 0; row < 4; ++row) {  // Iterate through each row of the state matrix
        for (int col = 0; col < 4; ++col) {  // Iterate through each column of the state matrix
            state[row][col] ^= roundKey[row * 4 + col]; // XOR the state byte with the corresponding round key byte
        }
    }
}


/********************************************************************
 * Function: SubBytes
 * Description:
 *  This function performs the SubBytes transformation step in AES.
 *  It substitutes each byte in the state matrix with a corresponding
 *  byte from the AES S-box.
 * Inputs:  state   - Refernece to Input State Matrix (4x4)
 * Outputs: state   - Refernece to Output State Matrix (4x4)
 * Returns: void
 ********************************************************************/
void SubBytes(uint8_t state[4][4]) {
    // The AES S-box (Substitution Box) is a predefined lookup table used in SubBytes.
    // This is a standard, constant table defined by the AES specification.
    // Declaring it as 'static const' helps optimization and avoids redundant copies.
    const uint8_t sbox[16][16] = {
        {0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
        {0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0},
        {0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15},
        {0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75},
        {0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84},
        {0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf},
        {0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8},
        {0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2},
        {0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73},
        {0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb},
        {0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79},
        {0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08},
        {0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a},
        {0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e},
        {0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf},
        {0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16}
    };

    // Iterate over each byte of the state matrix
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // Get the current byte from the state matrix
            uint8_t currentByte = state[row][col];
            // Use the high nibble (upper 4 bits) as the row index for the S-box
            uint8_t sboxRow = currentByte >> 4;
            // Use the low nibble (lower 4 bits) as the column index for the S-box
            uint8_t sboxCol = currentByte & 0x0f;
            // Substitute the current byte with the value from the S-box
            state[row][col] = sbox[sboxRow][sboxCol];
        }
    }
}



/********************************************************************
 * Function: ShiftRows
 * Description:
 *  This function performs the ShiftRows transformation in AES. It
 *  cyclically shifts the bytes in each row of the state matrix.
 * Inputs:  state   - Refernece to Input State Matrix (4x4)
 * Outputs: state   - Refernece to Output State Matrix (4x4)
 * Returns: void
 ********************************************************************/
void ShiftRows(uint8_t state[4][4])
{
    uint8_t rowTemp[4];
    // Row 0: No Shift

    // Row 1: Shift left by 1
    for (int col = 0; col < 4; col++) {
        rowTemp[col] = state[1][(col + 1) % 4]; // Store the shifted byte in a temporary variable
    }
    for (int col = 0; col < 4; col++) {
        state[1][col] = rowTemp[col];  // Copy the shifted bytes back to row 1
    }

    // Row 2: Shift left by 2
    for (int col = 0; col < 4; col++) {
        rowTemp[col] = state[2][(col + 2) % 4]; // Store the shifted byte in a temporary variable
    }
    for (int col = 0; col < 4; col++) {
        state[2][col] = rowTemp[col];  // Copy the shifted bytes back to row 2
    }
    
    // Row 3: Shift left by 3
    for (int col = 0; col < 4; col++) {
        rowTemp[col] = state[3][(col + 3) % 4]; // Store the shifted byte in a temporary variable
    }
    for (int col = 0; col < 4; col++) {
        state[3][col] = rowTemp[col];  // Copy the shifted bytes back to row 3
    }
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


/********************************************************************
 ************************ Utility Functions *************************
 ********************************************************************/
/********************************************************************
 * Function: printState
 * Description:
 *  Function to print the state (4x4) Matrix
 * Inputs:  state   - Refernece to Input State Matrix (4x4)
 * Returns: void
 ********************************************************************/
void printState(const uint8_t state[4][4])
{
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            std::cout << std::hex << static_cast<int>(state[row][col]) << " ";
        }
        std::cout << std::endl;
    }
}

/********************************************************************
 * Function: IncrementCounter
 * Description:
 *  Function to increment the 16 byte counter considering the carry
 *  propagation
 * Inputs:  counter   - Refernece to the counter array
 * Outputs: counter   - The counter after being incremented
 * Returns: void
 ********************************************************************/
void IncrementCounter(std::vector<uint8_t>& counter) 
{
    for (size_t i = 0; i < 16; ++i) { // Iterate from LSB to MSB
        if (++counter[i] != 0) {      // Increment the current byte
            break;                     // If no overflow, we're done
        }
    } // If the loop completes, the counter wrapped around
}


/********************************************************************
 * Function: TextPreprocessor
 * Description:
 *  Function to transform text into states for AES manipulation for
 *  either encryption or decryption
 * Inputs:  strText  - text as string either plaintext or ciphertext
 * Outputs: states   - Prepared states
 * Returns: void
 ********************************************************************/
void TextPreprocessor(const std::string& strText, std::vector<std::vector<std::vector<uint8_t>>>& states) 
{
    /* Create a copy vector to avoid modifying the original */
    std::vector<uint8_t> paddedText; 

    /* Transform the string into a vector for manipulation */
    paddedText.assign(strText.begin(), strText.end());
    
    /* Check that the plaintext is multiples of 16 bytes */
    if (paddedText.size() % 16 != 0) 
    {
        /* Pad the plain text by 0x00 until it's a multiple of 16 */
        paddedText.resize(paddedText.size() + (16 - (paddedText.size() % 16)), 0x00);
    }

    /* Store each 16 bytes in the plaintext in each element of the states vector */
    for (size_t i = 0; i < paddedText.size(); i += 16) 
    {
        /* Create a temporary state to hold the currently extracted state from the plaintext */
        std::vector<std::vector<uint8_t>> state(4, std::vector<uint8_t>(4));

        /* Extract 16 bytes from the plaintext and store them in the temporary state */
        for(int j = 0; j < 4; j++) 
        {
            for(int k = 0; k < 4; k++) 
            {
                state[j][k] = paddedText[i + (j * 4) + k];
            }
        }

        /* Append the state matrix to the states vector (Move Semantics) */
        states.push_back(std::move(state));
    }
}

/********************************************************************
 * Function: TextPreprocessor
 * Description:
 *  Function to transform states into text to be printed
 * Inputs:  states    - states
 * Outputs: strText   - text to be printed
 * Returns: void
 ********************************************************************/
void TextPostprocessor(const std::vector<std::vector<std::vector<uint8_t>>>& states, std::string& strText) 
{
    std::stringstream ss; // Use a stringstream for efficiency
    for (size_t i = 0; i < states.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(states[i][j][k]);
            }
        }
    }
    strText = ss.str();

}

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

/********************************************************************
 ********************** Counter Mode Functions **********************
 ********************************************************************/
/********************************************************************
 * Function: CounterModeInitializer
 * Description:
 *  Function to initialize a counter for the CTR mode of operation
 *  for AES. It generates an 8-byte random number and concatenate it
 *  to a zero-initialized counter. The output counter is then used
 *  for CTR Mode AES
 * Inputs:  counter   - Place holder to hold the output counter
 * Outputs: counter   - The initialized counter
 * Returns: void
 ********************************************************************/
void CounterModeInitializer(std::vector<uint8_t>& counter)
{
    /* Check if the counter is of size 16 bytes. Resize if its not of the correct size */
    if (counter.size() != 16)
    {
        counter.resize(16);
    }

    /* Obtain a random seed from the OS */
    std::random_device rd;
    std::mt19937_64 gen(rd());  
    std::uniform_int_distribution<> distrib(0, 255);

    for (int i = 0; i < 16; ++i) 
    {
        if(i < 8)
        {
            /* Set the remaining 8 bytes to 0 */
            counter[i] = 0x00;
        }
        else
        {
            /* Store the random number in the first 8 bytes of the counter */
            counter[i] = static_cast<uint8_t>(distrib(gen));
        }
    }
}

/********************************************************************
 * Function: EncryptionWorker
 * Description:
 *  Function to perform AES 128 encryption in counter mode
 *  for an input 4x4 state. The encrypted block is provided
 *  in the output encryptedState
 * Inputs:  state   - State to be encrypted
 *          counter - Counter to be used in encryption
 * Outputs: encryptedState   - Output state after encryption
 * Returns: void
 ********************************************************************/
void EncryptionWorker(const std::vector<std::vector<uint8_t>>& state, const std::vector<uint8_t> counter, std::vector<std::vector<uint8_t>>& encryptedState)
{
    /* Declare temporal array to be used during the process */    
    uint8_t stateArray[4][4];
    /*******************************TBD*************************************/
    uint8_t key[32] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                       0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                       0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                       0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};  // Example key
    /* Select AES parameters based on key size */
    int keySize = 32; // Use 16 for AES-128, 24 for AES-192, 32 for AES-256
    int Nk = keySize / 4;
    int Nr = Nk + 6;
    // Expanded key schedule (NUM_COLUMN * (Nr + 1) words)
    uint32_t expandedKey[NUM_COLUMN * (Nr + 1)];
    /*******************************//************************************/

    /* Transform the state vector into an array for efficient access */
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            stateArray[i][j] = state[i][j];
        }
    }

    /* Perform AES Encryption rounds */
    //Simplified encryption process
    KeyExpansion(key, Nk, Nr, expandedKey);
    // *** Convert expandedKey to a vector of bytes ***
    std::vector<uint8_t> roundKey(4 * NUM_COLUMN * (Nr+1)); // Or size: 4 * (NUM_COLUMN * (Nr + 1)
    for(int i = 0; i < NUM_COLUMN * (Nr + 1); ++i)
    {
        roundKey[4*i] = (expandedKey[i] >> 24) & 0xff;
        roundKey[4*i + 1] = (expandedKey[i] >> 16) & 0xff;
        roundKey[4*i + 2] = (expandedKey[i] >> 8) & 0xff;
        roundKey[4*i + 3] = expandedKey[i] & 0xff;
    }
    AddRoundKey(stateArray, roundKey);
    for(int round = 1; round < Nr; ++round)
    {
       SubBytes(stateArray);
       ShiftRows(stateArray);
       MixColumns(stateArray);  // mixColumns in all but the last round. Implement full matrix multiplication with GaloisField::mul
       KeyExpansion(key, Nk, Nr, expandedKey);
       // *** Convert expandedKey to a vector of bytes ***
        for(int i = 0; i < NUM_COLUMN * (Nr + 1); ++i)
        {
            roundKey[4*i] = (expandedKey[i] >> 24) & 0xff;
            roundKey[4*i + 1] = (expandedKey[i] >> 16) & 0xff;
            roundKey[4*i + 2] = (expandedKey[i] >> 8) & 0xff;
            roundKey[4*i + 3] = expandedKey[i] & 0xff;
        }
       AddRoundKey(stateArray, roundKey);
    }
    SubBytes(stateArray);
    ShiftRows(stateArray);
    KeyExpansion(key, Nk, Nr, expandedKey);
    // *** Convert expandedKey to a vector of bytes ***
    for(int i = 0; i < NUM_COLUMN * (Nr + 1); ++i)
    {
        roundKey[4*i] = (expandedKey[i] >> 24) & 0xff;
        roundKey[4*i + 1] = (expandedKey[i] >> 16) & 0xff;
        roundKey[4*i + 2] = (expandedKey[i] >> 8) & 0xff;
        roundKey[4*i + 3] = expandedKey[i] & 0xff;
    }
    AddRoundKey(stateArray, roundKey);

    /* Set the encrypted state array to the output vector parameter */
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            encryptedState[i][j] = stateArray[i][j];
        }
    }
    
}

/********************************************************************
 * Function: DecryptionWorker
 * Description:
 *  Function to perform AES 128 decryption in counter mode
 *  for an input 4x4 state. The decrypted block is provided
 *  in the output decryptedState
 * Inputs:  state   - State to be decrypted
 *          counter - Counter to be used in decryption
 * Outputs: decryptedState   - Output state after decryption
 * Returns: void
 ********************************************************************/
void DecryptionWorker(const std::vector<std::vector<uint8_t>>& state, const std::vector<uint8_t> counter, std::vector<std::vector<uint8_t>>& decryptedState)
{
    /* Declare temporal array to be used during the process */    
    uint8_t stateArray[4][4];

    /* Transform the state vector into an array for efficient access */
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            stateArray[i][j] = state[i][j];
        }
    }

    /* Perform AES decryption rounds */
    // //Simplified decryption process
    // addRoundKey(stateArray, keyExpansion.getRoundKey(/*Number of rounds*/));
    // invShiftRows(stateArray);
    // invSubBytes(stateArray);

    // for(int round = /*Number of rounds*/ -1; round > 0; --round)
    // {
    //     addRoundKey(stateArray, keyExpansion.getRoundKey(round));
    //     invMixColumns(stateArray);    // invMixColumns in all but the last round. Implement full matrix multiplication with GaloisField::mul
    //     invShiftRows(stateArray);
    //     invSubBytes(stateArray);
    // }
    // addRoundKey(stateArray, keyExpansion.getRoundKey(0));

    /* Set the decrypted state array to the output vector parameter */
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            decryptedState[i][j] = stateArray[i][j];
        }
    }
}

/********************************************************************
 * Function: EncryptionDispatcher
 * Description:
 *  Function to perform AES 128 encryption in counter mode
 *  for an input data. This function parallize the encryption
 *  over multiple working encryption threads to speed up the
 *  encryption process
 * Inputs:  states   - States to be encrypted
 *          counter - Counter to be used in encryption
 * Outputs: encryptedStates   - Output states after encryption
 * Returns: void
 ********************************************************************/
void EncryptionDispatcher(const std::vector<std::vector<std::vector<uint8_t>>>& states, std::vector<uint8_t> counter, std::vector<std::vector<std::vector<uint8_t>>>& encryptedStates)
{
    /* Declare a thread pool with a number of available threads of CORES_NUMBER */
    std::vector<std::thread> threadPool(CORES_NUMBER);

    /* Initialize an iterator to loop over the states */
    uint64_t statesIterator = 0;

    /* Initialize an variable to hold the number of threads currently dispatched */
    uint8_t threadIterator = 0;

    /* Loop over all the states in rounds */
    while(statesIterator < states.size())
    {
        /* Increment the counter */
        IncrementCounter(counter);

        /* Dispatch a thread to encrypt the current state */
        threadPool[threadIterator] = std::thread(EncryptionWorker, std::ref(states[statesIterator]), counter, std::ref(encryptedStates[statesIterator]));

        /* Increment the number of working threads */
        threadIterator++;
        
        /* If all allowed threads have been allocated or reached the last state. Wait for all the in progress thread to finish */
        if((threadIterator == CORES_NUMBER) || (statesIterator == (states.size() - 1)))
        {   
            /* Wait for running threads to finish before dispatching new threads */
            for(threadIterator; threadIterator > 0; threadIterator--) 
            {
                threadPool[threadIterator - 1].join();
            }
        }

        /* Increment the states iterator for the next iteration */
        statesIterator++;
    }
}

/********************************************************************
 * Function: DecryptionDispatcher
 * Description:
 *  Function to perform AES 128 decryption in counter mode
 *  for an input data. This function parallize the decryption
 *  over multiple working decryption threads to speed up the
 *  decryption process
 * Inputs:  states   - States to be decrypted
 *          counter - Counter to be used in decryption
 * Outputs: encryptedStates   - Output states after decryption
 * Returns: void
 ********************************************************************/
void DecryptionDispatcher(const std::vector<std::vector<std::vector<uint8_t>>>& states, std::vector<uint8_t> counter, std::vector<std::vector<std::vector<uint8_t>>>& decryptedStates)
{
    /* Declare a thread pool with a number of available threads of CORES_NUMBER */
    std::vector<std::thread> threadPool(CORES_NUMBER);

    /* Initialize an iterator to loop over the states */
    uint64_t statesIterator = 0;

    /* Initialize an variable to hold the number of threads currently dispatched */
    uint8_t threadIterator = 0;

    /* Loop over all the states in rounds */
    while(statesIterator < states.size())
    {
        /* Increment the counter */
        IncrementCounter(counter);

        /* Dispatch a thread to encrypt the current state */
        threadPool[threadIterator] = std::thread(DecryptionWorker, std::ref(states[statesIterator]), counter, std::ref(decryptedStates[statesIterator]));

        /* Increment the number of working threads */
        threadIterator++;
        
        /* If all allowed threads have been allocated or reached the last state. Wait for all the in progress thread to finish */
        if((threadIterator == CORES_NUMBER) || (statesIterator == (states.size() - 1)))
        {   
            /* Wait for running threads to finish before dispatching new threads */
            for(threadIterator; threadIterator > 0; threadIterator--) 
            {
                threadPool[threadIterator - 1].join();
            }
        }

        /* Increment the states iterator for the next iteration */
        statesIterator++;
    }
}


/*******************************TBD*************************************/
/* Function to perform the SubWord operation */
uint32_t SubWord(uint32_t word) 
{
    uint32_t result = 0;
    uint16_t Indx;
    for (Indx = 0; Indx < WORD_SIZE; ++Indx) 
    {   
        /* Extract byte */
        uint8_t byte = (word >> (24 - Indx * 8)) & 0xFF; 
        /* Substitute byte with S-box */
        result |= sBox[byte] << (24 - Indx * 8);         
    }
    return result;
}
/* Function to perform the RotWord operation */
uint32_t RotWord(uint32_t word) 
{
    /* Rotate left by 8 bits */
    return (word << 8) | (word >> 24); 
}

/* AES key expansion function */
void KeyExpansion(const uint8_t* key, int Nk, int Nr, uint32_t* w) 
{
    uint32_t temp;
    int i = 0;

    // Copy the original key into the first Nk words of w
    while (i < Nk) 
    {
        w[i] = (key[4 * i] << 24) | (key[4 * i + 1] << 16) |
               (key[4 * i + 2] << 8) | key[4 * i + 3];
        ++i;
    }

    // Generate the remaining words for the expanded key
    while (i < NUM_COLUMN * (Nr + 1)) 
    {
        temp = w[i - 1];
        if (i % Nk == 0) 
        {
            temp = SubWord(RotWord(temp)) ^ (Rcon[i / Nk - 1] << 24);
        } else if (Nk > 6 && i % Nk == 4) 
        {
            temp = SubWord(temp);
        }
        w[i] = w[i - Nk] ^ temp;
        ++i;
    }
}

// Inverse SubBytes: Applies the inverse S-Box to the state
void InvSubBytes(uint8_t state[4][4]) 
{
    for (int i = 0; i < 4; ++i) 
    {
        for (int j = 0; j < 4; ++j) 
        {
            state[i][j] = inv_sbox[state[i][j]];
        }
    }
}

/*******************************//************************************/