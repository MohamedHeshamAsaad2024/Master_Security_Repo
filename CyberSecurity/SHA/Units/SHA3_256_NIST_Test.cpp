/*********************************************************************
 * @file SHA3_256.cpp
 * @author Abdelrahman Gamiel, Eslam Khaled, Mohamed Alaa Eldin Eldehimy, Mohamed Hesham Kamaly
 * @date 2024-12-30
 * @brief 
 *        This file provides a C++ implementation of the Secure Hash Algorithm 3 (SHA3), 
 *        specifically SHA3-256.  
 ********************************************************************/
/********************************************************************
 ************************** Included ********************************
 ********************************************************************/
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

/********************************************************************
 *********************** Configurations *****************************
 ********************************************************************/

/********************************************************************
 *********************** SYMBOLIC NAMES *****************************
 ********************************************************************/
#define STATE_ROW_SIZE              5U
#define STATE_COLUMN_SIZE           5U
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
/********************************************************************
 **************************** GLOBALS *******************************
 ********************************************************************/
// Constants for SHA3-224
int RATE = 1088;
int CAPACITY = 512;
int BLOCK_SIZE = RATE + CAPACITY; // b = 1600
int OUTPUT_LENGTH = 256;
int ROUNDS = 24;
int STATE_SIZE = STATE_ROW_SIZE * STATE_COLUMN_SIZE * 64;  // 1600 bits
// Type aliases for clarity
using State = std::vector<uint64_t>;

// Round constants for Iota step (first 24 rounds)
const uint64_t RC[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};
/********************************************************************
 ************************* Prototypes *******************************
 ********************************************************************/
void KeccakF(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);
void SHA_ComputeTheta(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);
void SHA_ComputeRho(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);
void SHA_ComputePi(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);
void SHA_ComputeChi(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);
void SHA_ComputeIota(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE], int round);
std::vector<uint8_t> SHA_MultiRatePadding(const std::string& input, int rate);
void convertToStateArray(const std::vector<uint8_t>& paddedMessage, uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);
void printState(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);
std::vector<uint8_t> convertFromStateArray(const uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]);

/********************************************************************
 ************************* Main Function ****************************
 ********************************************************************/
int main() 
{
    // 1. Padding
    /* Exampl usage of padding scheme */
    // std::string message = "Hello, SHA-3!";
    // std::vector<uint8_t> paddedMessage = SHA_MultiRatePadding(message, RATE);


    // 2.1 Absorbing Phase
    uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE] = {0};
    // uint64_t paddedMessageArray[STATE_ROW_SIZE][STATE_COLUMN_SIZE] = {0};
    // convertToStateArray(paddedMessage,paddedMessageArray);
    uint64_t paddedMessageArray[STATE_ROW_SIZE][STATE_COLUMN_SIZE] =
    {
        {0x00000001997b5853, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000}
    };

    // XOR the padded message with the state (absorbing phase)
    for (int x = 0; x < STATE_ROW_SIZE; ++x) {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) {
            state[x][y] ^= paddedMessageArray[x][y]; 
        }
    }
    KeccakF(state);

    // 2.2 Squeezing Phase
    std::vector<uint8_t> outputHash;
    /*squezing implementation */

    outputHash = convertFromStateArray(state);
    outputHash.resize(OUTPUT_LENGTH / 8);

    // Print the hash 
    std::cout << "Output Hash: ";
    for (uint8_t byte : outputHash) {
        printf("%02X", byte); // Use printf for consistent formatting
    }
    std::cout << std::endl;

}


/********************************************************************
 **************************** Functions *****************************
 ********************************************************************/
/*************************************************************
 * Function Name: KeccakF
 * Description:
 *   This function implements the Keccak-f permutation, the core
 *   transformation of the SHA3 algorithm. It iteratively applies
 *   five internal step mappings (Theta, Rho, Pi, Chi, and Iota)
 *   for a fixed number of rounds (ROUNDS) to the given state.
 *
 * Arguments:
 *   state (uint64_t[STATE_ROW_SIZE][STATE_COLUMN_SIZE]):
 *     The state array, represented as a 2D array of 64-bit
 *     unsigned integers.  This array is modified in-place by
 *     the Keccak-f permutation.
 *
 * Return:
 *   void (The function modifies the state array directly)
 ************************************************************/
void KeccakF(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) 
{
    for (int round = 0; round < ROUNDS; ++round) 
    {
        std::cout << "Round " << round << ":" << std::endl; // Indicate the round number
        std::cout << "State before Theta:" << std::endl;
        printState(state);

        SHA_ComputeTheta(state);

        std::cout << "State after Theta & Before Rho:" << std::endl;
        printState(state);

        SHA_ComputeRho(state);

        std::cout << "State after Rho & Before Pi:" << std::endl;
        printState(state);

        SHA_ComputePi(state);

        std::cout << "State after Pi & Before Chi:" << std::endl;
        printState(state);

        SHA_ComputeChi(state);

        std::cout << "State after Chi & Before Iota:" << std::endl;
        printState(state);

        SHA_ComputeIota(state, round);
        
        std::cout << "State after Iota:" << std::endl;
        printState(state);
    }
}

/*************************************************************
 * Function Name: SHA_ComputeRho
 * Description:
 *  This function performs the Rho step mapping which performs
 *  bitwise left circular rotations on each lane of the state.
 *  The amount of rotation for each lane is pre-defined and
 *  differs from lane to lane according to the following
 *  formula:
 *    A'[x,y,z] = ROT(A[x,y,z], r[x,y])
 *  Where r[x,y] are rotation constants derived according to
 *  specific rules.  Rho step provides inter-lane diffusion.
 * Arguments:
 *  state (uint64_t[STATE_ROW_SIZE][STATE_COLUMN_SIZE]):
 *    Input-Output argument containing the state represented
 *    as a 2D array of 64-bit unsigned integers.
 * Return:
 *  void
 ************************************************************/
void SHA_ComputeRho(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) {
    // Define rotation constants r[x,y] as per the SHA3 specification
    // These constants determine the amount of left rotation for each lane.
    int r[STATE_ROW_SIZE][STATE_COLUMN_SIZE] = {
        { 0, 36,  3, 41, 18},
        { 1, 44, 10, 45,  2},
        {62,  6, 43, 15, 61},
        {28, 55, 25, 21, 56},
        {27, 20, 39,  8, 14}
    };

     /* Define a temprary state to hold the original states before computation */
    uint64_t tempState[STATE_ROW_SIZE][STATE_COLUMN_SIZE];

    /* Copy the current state into the temporary State */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            tempState[x][y] = state[x][y];
        }
    }

    /* Apply the Rho step mapping: Left-rotate each lane by r[x,y] bits */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) {
            state[x][y] = ROTL64(tempState[x][y], r[x][y]);
        }
    }
}

/*************************************************************
 * Function Name: SHA_ComputeIota
 * Descriotion:
 *  This function performs the Iota step mapping which adds
 *  round constants to the state. Round constants are added to
 *  the first lane (lane [0,0]) to break any symmetry and
 *  introduce more complexity.  The round constant is obtained
 *  from a precomputed array based on the round number
 * Arguments:
 *  state (uint64_t *): Input-Output argument containing the state
 *  round (int): the round number used to determine the constant
 * Return:
 *  void
 ************************************************************/
void SHA_ComputeIota(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE], int round) {
    state[0][0] ^= RC[round];
}

/*************************************************************
 * Function Name: SHA_ComputePi
 * Descriotion:
 *  This function performs the Pi step mapping which performs
 *  diffusion on slice level through the following formula:
 *    A'[x,y,z] = A[(x + 3y) mod 5, x, z]
 *  This Step provides more diffusion and helps in breaking
 *  positional patterns
 * Arguments:
 *  state (uint8_t *): Input-Output argument containing the state
 * Return:
 *  void
 ************************************************************/
void SHA_ComputePi(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) 
{
    /* Define a temprary state to hold the original states before computation */
    uint64_t tempState[STATE_ROW_SIZE][STATE_COLUMN_SIZE];

    /* Copy the current state into the temporary State */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            tempState[x][y] = state[x][y];
        }
    }

    /* Apply the Pi step mapping */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            state[x][y] = tempState[(x + 3 * y) % 5][x];
        }
    }
}

/*************************************************************
 * Function Name: SHA_ComputeTheta
 * Description:
 *  This function performs the Theta step mapping which ensures
 *  diffusion across all bits in the state. Each bit is XORed
 *  with a parity bit derived from the columns of the state.
 *  This is achieved through the following formula:
 *    C[x] = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4]
 *    D[x] = C[x-1] ^ ROT(C[x+1], 1)
 *    A'[x,y] = A[x,y] ^ D[x]
 * Arguments:
 *  state (uint64_t *): Input-Output argument containing the state
 * Return:
 *  void
 ************************************************************/
void SHA_ComputeTheta(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) 
{
    uint64_t C[STATE_ROW_SIZE];
    uint64_t D[STATE_ROW_SIZE];

    /* Compute the parity of each column */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        C[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];
    }

    /* Compute the D array */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        D[x] = C[(x + STATE_ROW_SIZE - 1) % STATE_ROW_SIZE] ^ ROTL64(C[(x + 1) % STATE_ROW_SIZE], 1);
    }

    /* Apply the Theta step */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            state[x][y] ^= D[x];
        }
    }
}

/*************************************************************
 * Function Name: SHA_ComputeChi
 * Descriotion:
 *  This function performs the Chi step mapping which performs
 *  diffusion on row level. Basically, each bit in a row is
 *  XORed with the result of NOT of the following bit ANDed
 *  with the one after it. This is achieved through the
 *  following formula:
 *    A'[x,y,z] = A[x,y,z] ^ (~A[(x + 1) mod 5,y,z] & A[(x + 2) mod 5, y, z])
 *  This Step introduces non-linearity in the permutation
 *  through the use of AND operation making the output of this
 *  step irreversable.
 * Arguments:
 *  state (uint8_t *): Input-Output argument containing the state
 * Return:
 *  void
 ************************************************************/
void SHA_ComputeChi(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) 
{
    /* Define a temporary state to hold the original states before computation */
    uint64_t tempState[STATE_ROW_SIZE][STATE_COLUMN_SIZE];

    /* Copy the current state into the temporary state */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            tempState[x][y] = state[x][y];
        }
    }

    /* Apply the Chi step mapping */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            state[x][y] = tempState[x][y] ^ 
                          ((~tempState[(x + 1) % STATE_ROW_SIZE][y]) & 
                           tempState[(x + 2) % STATE_ROW_SIZE][y]);
        }
    }
}

/*************************************************************
 * Function Name: SHA_MultiRatePadding
 * Descriotion:
 *  This function performs does two things:
 *  - Append the hash mode identifiers (01)
 *  - Padding rule pad10*1 which adds padding to the input
 *    message to make it multiples of the rate so that it 
 *    can be partioned during the SPONGE operation.
 * Note:
 *   - The hash identifier (01) and the starting padding bit (1) are byte-aligned as 0x06
 *   - The terminating padding bit (1) is byte-aligned as 0x80
 * Arguments:
 *  inputMessage (string): Input string to be hashed
 *  paddedMessage (vector): the input string after being padded
 * Return:
 *  vector: The padded message with the hash mode prefix
 ************************************************************/
std::vector<uint8_t> SHA_MultiRatePadding(const std::string& input, int rate) 
{
    /* Convert the input string to a vector of bytes */
    std::vector<uint8_t> paddedMessage(input.begin(), input.end());

    /* Determine the rate in bytes */
    size_t rateInBytes = rate / 8;

    /* Compute the length of the input in bytes */
    size_t m = paddedMessage.size();

    /* Calculate the number of padding bytes (q) */
    size_t q = rateInBytes - (m % rateInBytes);

    /* Append the padding bytes depending on the needed number of padding bytes (q) */
    if (q == 1) 
    {
        /* If the input message is multiple of the rate, then we only append one byte */
        paddedMessage.push_back(0x86);
    } 
    else if (q == 2) 
    {
        /* If two bytes are needed, we set the first byte 0x06 as the prefix byte */
        paddedMessage.push_back(0x06);
        /* We add the last byte which terminates the padding */
        paddedMessage.push_back(0x80);
    } else 
    {
        /* If more than 2 bytes are needed, we set the first byte 0x06 as the prefix byte */
        paddedMessage.push_back(0x06);

        /* Add intermediate zero bytes */
        for (size_t i = 0; i < q - 2; ++i) 
        {
            paddedMessage.push_back(0x00);
        }

        /* We add the last byte which terminates the padding */
        paddedMessage.push_back(0x80);
    }

    return paddedMessage;
}

/*************************************************************
 * Function Name: convertToStateArray
 * Description:
 *  This function converts a padded message (vector of bytes)
 *  into a state array (2D array of uint64_t).  It iterates
 *  through the padded message and distributes the bytes into
 *  the state array according to the Keccak layout.
 * Arguments:
 *  paddedMessage (const std::vector<uint8_t>&): The padded message.
 *  state (uint64_t[STATE_ROW_SIZE][STATE_COLUMN_SIZE]): The state array to populate.
 * Return:
 *  void
 ************************************************************/
void convertToStateArray(const std::vector<uint8_t>& paddedMessage, uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) {
    // Initialize the state array to all zeros
    for (int x = 0; x < STATE_ROW_SIZE; ++x) {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) {
            state[x][y] = 0;
        }
    }

    // Iterate through the padded message and populate the state array
    for (size_t i = 0; i < paddedMessage.size(); ++i) {
        size_t x = (i * 8 / 64) % 5;      // Calculate the x coordinate
        size_t y = (i * 8 / 64) / 5;      // Calculate the y coordinate
        size_t z = i * 8 % 64;          // Calculate the z coordinate


        for(int bit_in_byte = 0; bit_in_byte < 8; ++bit_in_byte)
        {   
            uint64_t bit = (paddedMessage[i] >> bit_in_byte) & 1;
            state[x][y] ^= (bit << z + bit_in_byte);
        }
    }
}

/*************************************************************
 * Function Name: convertFromStateArray
 * Description:
 * This function converts the state array (2D array of uint64_t)
 * back into a vector of bytes.  This is typically used after
 * the Keccak-f permutation to extract the resulting hash.
 * Arguments:
 *  state (const uint64_t[STATE_ROW_SIZE][STATE_COLUMN_SIZE]): The state array.
 * Return:
 * std::vector<uint8_t>: The message extracted from the state array.
 ************************************************************/
std::vector<uint8_t> convertFromStateArray(const uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) {
    std::vector<uint8_t> message;
    message.reserve(STATE_ROW_SIZE * STATE_COLUMN_SIZE * sizeof(uint64_t)); // Reserve space for efficiency

    for (int y = 0; y < STATE_COLUMN_SIZE; ++y) {
        for (int x = 0; x < STATE_ROW_SIZE; ++x) {
            // Get a pointer to the bytes of the current uint64_t
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&state[x][y]);

            // Append each byte to the message vector
            for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                message.push_back(bytes[i]);
            }
        }
    }

    return message;
}

/*************************************************************
 * Function Name: printState
 * Description:
 * This function prints the state array in a hexadecimal format.
 * It is primarily used for debugging and visualization.
 * Arguments:
 * state (uint64_t[STATE_ROW_SIZE][STATE_COLUMN_SIZE]): the state array to print
 * Return:
 * void
 ************************************************************/
void printState(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE])
{
    int byteCount = 0;

    /* Print the updated state */
    for (int x = 0; x < STATE_ROW_SIZE; ++x)
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y)
        {
            uint8_t *bytes = reinterpret_cast<uint8_t*>(&state[y][x]);
            for (int i = 0; i < sizeof(uint64_t); ++i)
            {
                printf("%02X ", bytes[i]);
                byteCount++;

                if (byteCount % 16 == 0) 
                {
                    printf("\n");
                }
            }
        }
    }
    printf("\n");
}