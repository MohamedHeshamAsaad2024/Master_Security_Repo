#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#define STATE_ROW_SIZE              5U
#define STATE_COLUMN_SIZE           5U
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

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

int main() 
{
    uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE] = 
    {
        {0x00000001997b5853, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000},
        {0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000}
    };

    std::cout << "Input State:" << std::endl;
    printState(state);

    SHA_ComputeTheta(state);

    std::cout << "After Theta Step:" << std::endl;
    printState(state);

    SHA_ComputeRho(state);
    std::cout << "After Rho Step:" << std::endl;
    printState(state);

    return 0;
}