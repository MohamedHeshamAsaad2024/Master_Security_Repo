#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#define STATE_ROW_SIZE              5U
#define STATE_COLUMN_SIZE           5U
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
        {0x00000001997B5852, 0x0000000000000000, 0x0000000332F6B0A6, 0x1997B58530000000, 0x2000000000000000},
        {0xB585300000001997, 0x00332F6B0A600000, 0x0000000000000020, 0x0000001000000000, 0x0000000000000000},
        {0x0000040000000000, 0x0000000000000008, 0x0000000000000000, 0x00000665ED614C00, 0x7B58530000000199},
        {0x0000000000000000, 0x6B0A70000000332F, 0x00000332F6B0A600, 0x0000000000004000, 0x0000020000000000},
        {0x0000CCBDAC298000, 0x1000000000000000, 0x0000000000040000, 0x0000000000000000, 0x0000000665ED614C}
    };
    
    std::cout << "Input State:" << std::endl;
    printState(state);

    SHA_ComputeChi(state);

    std::cout << "After Chi Step:" << std::endl;
    printState(state);

    SHA_ComputeIota(state,0);

    std::cout << "After iota Round1 Step:" << std::endl;
    printState(state);

    return 0;
}